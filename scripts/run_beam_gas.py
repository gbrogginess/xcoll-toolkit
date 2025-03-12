import os
import sys
import copy
import time
import yaml
import shutil
import random
import numpy as np
import pandas as pd
import subprocess
import merge_output
from schema import Schema, And, Or, Use, Optional, SchemaError
from copy import deepcopy
from warnings import warn
from collections import namedtuple
from pathlib import Path
from contextlib import contextmanager

import xtrack as xt
import xpart as xp
import xcoll as xc
import xgas as xg
from pylhc_submitter.job_submitter import main as htcondor_submit


ParticleInfo = namedtuple('ParticleInfo', ['name', 'pdgid', 'mass', 'A', 'Z','charge'])

# Note that YAML has inconsitencies when parsing numbers in scientific notation
# To avoid numbers parsed as strings in some configurations, always cast to float / int
to_float = lambda x: float(x)
to_int = lambda x: int(float(x))


XTRACK_TWISS_KWARGS = {}


BEAMGAS_OPTIONS_SCHEMA = Schema({'eBrem': Use(bool),
                                 Optional('eBrem_energy_cut'): Use(to_float),
                                 'CoulombScat': Use(bool),
                                 Optional('theta_min'): Use(to_float),
                                 Optional('theta_max'): Use(to_float)
                                 })

INPUT_SCHEMA = Schema({
    'machine': str,
    'xtrack_line': And(str, os.path.exists),
    'collimator_file': And(str, os.path.exists),
    'bdsim_config': And(str, os.path.exists),
    'gas_density_profile': And(str, os.path.exists),
    'beamgas_options': BEAMGAS_OPTIONS_SCHEMA,
    Optional('material_rename_map', default={}): Schema({str: str}),
})

BEAM_SCHEMA = Schema({'particle': str,
                      'momentum': Use(to_float),
                      'emittance': Or(Use(to_float), {'x': Use(to_float), 'y': Use(to_float)}),
                      'sigma_z': Use(to_float),
                      })

RUN_SCHEMA = Schema({'energy_cut': Use(to_float),
                     'seed': Use(to_int),
                     'turns': Use(to_int),
                     'nparticles': Use(to_int),
                     'max_particles': Use(to_int),
                     Optional('radiation', default='off'): And(str, lambda s: s in ('off', 'mean', 'quantum')),
                     Optional('beamstrahlung', default='off'): And(str, lambda s: s in ('off', 'mean', 'quantum')),
                     Optional('bhabha', default='off'): And(str, lambda s: s in ('off', 'mean', 'quantum')),
                     Optional('turn_rf_off', default=False): Use(bool),
                     Optional('compensate_sr_energy_loss', default=False): Use(bool),
                     Optional('sr_compensation_delta', default=None): Or(Use(to_float), None),
                     Optional('aperture_interp', default=None): Or(Use(to_float), None),
                     Optional('outputfile', default='part.hdf'): str,
                     Optional('batch_mode', default=True): Use(bool),
                     })

JOB_SUBMIT_SCHEMA = Schema({'mask': Or(os.path.exists, lambda s: s=='default'),
                            'working_directory': str,
                            'num_jobs': Use(to_int),
                            Optional('output_destination'): str,
                            Optional('replace_dict'): str,
                            Optional('append_jobs'): bool,
                            Optional('dryrun'): bool,
                            Optional('executable', default='bash'): str,
                            Optional('htc_arguments'): dict,
                            Optional('job_output_dir'): str,
                            Optional('jobflavour'): str,
                            Optional('jobid_mask'): str,
                            Optional('num_processes'): Use(to_int),
                            Optional('resume_jobs'): bool,
                            Optional('run_local'): bool,
                            Optional('script_arguments'): bool,
                            Optional('script_extension'): str,
                            Optional('ssh'): str,
                          })


LOSSMAP_SCHEMA = Schema({'norm': And(str, lambda s: s in ('none','total', 'max', 'max_coll', 'total_coll')),
                         'weights': And(str, lambda s: s in ('none', 'energy')),
                         'aperture_binwidth': Use(to_float),
                         Optional('make_lossmap', default=True): Use(bool),
                         })

CONF_SCHEMA = Schema({'input': INPUT_SCHEMA,
                      'beam': BEAM_SCHEMA,
                      'run': RUN_SCHEMA,
                      Optional('lossmap'): LOSSMAP_SCHEMA,
                      Optional('jobsubmission'): JOB_SUBMIT_SCHEMA,
                      Optional(object): object})  # Allow input flexibility with extra keys


def find_apertures(line):
    i_apertures = []
    apertures = []
    for ii, ee in enumerate(line.elements):
        if ee.__class__.__name__.startswith('Limit'):
            i_apertures.append(ii)
            apertures.append(ee)
    return np.array(i_apertures), np.array(apertures)


def insert_missing_bounding_apertures(line):
    print('Inserting missing bounding apertures...')
    # Place aperture defintions around all active elements in order to ensure
    # the correct functioning of the aperture loss interpolation
    # the aperture definitions are taken from the nearest neighbour aperture in the line
    tab = line.get_table()
    needs_aperture = ['ThickSliceQuadrupole', 'Bend', 'ThinSliceBendEntry',
                      'ThickSliceBend', 'ThinSliceBendExit', 'ThickSliceSextupole',
                      'ZetaShift', 'Cavity', 'BeamInteraction']

    s_pos = line.get_s_elements(mode='upstream')
    apert_idx, apertures = find_apertures(line)
    apert_s = np.take(s_pos, apert_idx)

    mask_misses_apertures = [False] * len(line.element_names)

    for ii, ee_type in enumerate(tab.element_type):
        if ee_type in needs_aperture:
            has_upstream_aperture = (
                np.any(tab.name[:ii] == tab.name[ii] + '_aper_start') or
                np.any(tab.name[:ii] == tab.name[ii] + '_aper_entry')
            )
            has_downstream_aperture = (
                np.any(tab.name[ii:] == tab.name[ii] + '_aper_end') or
                np.any(tab.name[ii:] == tab.name[ii] + '_aper_exit')
            )

            if not (has_upstream_aperture and has_downstream_aperture):
                # The element has no aperture upstream and/or downstream
                mask_misses_apertures[ii] = True

    idx = np.where(mask_misses_apertures)[0]
    elems = np.take(line.elements, idx)
    elem_names = np.take(line.element_names, idx)
    elems_s_start = np.take(s_pos, idx)

    # Find the nearest neighbour aperture in the line
    apert_idx_start = np.searchsorted(apert_s, elems_s_start, side='left')
    apert_idx_end = apert_idx_start + 1

    aper_start = apertures[apert_idx_start]
    # TODO: check if this is correct
    if apert_idx_end[-1] >= len(apertures):
        aper_end = apertures[apert_idx_end[:-1]]
        aper_end = np.concatenate((aper_end, np.array([apertures[0]])))
    else:
        aper_end = apertures[apert_idx_end]

    # TODO: fix this
    chicane_aper_names = []
    for (ii, jj), kk in zip(enumerate(idx), range(len(elems))):
        if elem_names[kk].startswith(('bp2nrp.', 'bp1nrp.', '-bp1nrp.', '-bp2nrp.')):
            # Chicane need separate treatment
            arc_aper = xt.LimitEllipse(a=0.45, b=0.45)
            line.insert_element(at=jj+1,
                                element=arc_aper.copy(),
                                name=elem_names[kk] + '_aper_end')

            line.insert_element(at=jj,
                                element=arc_aper.copy(),
                                name=elem_names[kk] + '_aper_start')
            
            chicane_aper_names.append(elem_names[kk]+'_aper_start')
            chicane_aper_names.append(elem_names[kk]+'_aper_end')
            
            # Update bg_idx after insertion of two elements
            idx[ii:] += 2
            continue

        line.insert_element(at=jj+1,
                    element=aper_end[kk].copy(),
                    name=elem_names[kk] + '_aper_end')

        line.insert_element(at=jj,
                            element=aper_start[kk].copy(),
                            name=elem_names[kk] + '_aper_start')
        
        # Update bg_idx after insertion of two elements
        idx[ii:] += 2

    # Shift missing chicane aperture
    line.build_tracker()
    tw = line.twiss()
    line.discard_tracker()
    x_shift = tw['x', chicane_aper_names]

    for ii, nn in enumerate(chicane_aper_names):
        if abs(x_shift[ii]) > 1e-2:
            line[nn].shift_x = x_shift[ii]
            line[nn].shift_x = x_shift[ii]



def deactivate_bg_elems(line):
    # Deactivate beam-gas elements replacing them with dummy markers
    line.discard_tracker()
    newLine = xt.Line(elements=[], element_names=[])

    for ee, nn in zip(line.elements, line.element_names):
        if nn.startswith('beam_gas_') and 'aper' not in nn:
            newLine.append_element(xt.Marker(), nn)
        else:
            newLine.append_element(ee, nn)

    # Update the line in place
    line.element_names = newLine.element_names
    line.element_dict.update(newLine.element_dict)
        

def _configure_tracker_radiation(line, radiation_model, beamstrahlung_model=None, bhabha_model=None, for_optics=False):
    mode_print = 'optics' if for_optics else 'tracking'

    print_message = f"Tracker synchrotron radiation mode for '{mode_print}' is '{radiation_model}'"

    _beamstrahlung_model = None if beamstrahlung_model == 'off' else beamstrahlung_model
    _bhabha_model = None if bhabha_model == 'off' else bhabha_model

    if line.tracker is None:
        line.build_tracker()

    if radiation_model == 'mean':
        if for_optics:
            # Ignore beamstrahlung and bhabha for optics
            line.configure_radiation(model=radiation_model)
        else:
            line.configure_radiation(model=radiation_model, 
                                     model_beamstrahlung=_beamstrahlung_model,
                                     model_bhabha=_bhabha_model)

    elif radiation_model == 'quantum':
        if for_optics:
            print_message = ("Cannot perform optics calculations with radiation='quantum',"
            " reverting to radiation='mean' for optics.")
            line.configure_radiation(model='mean')
        else:
            line.configure_radiation(model='quantum',
                                     model_beamstrahlung=_beamstrahlung_model,
                                     model_bhabha=_bhabha_model)

    elif radiation_model == 'off':
        pass
    else:
        raise ValueError('Unsupported radiation model: {}'.format(radiation_model))
    print(print_message)


def _save_particles_hdf(fpath, particles=None, lossmap_data=None, reduce_particles_size=False):
    if fpath.suffix != '.hdf':
        fpath = fpath.with_suffix('.hdf')

    # Remove a potential old file as the file is open in append mode
    if fpath.exists():
        fpath.unlink()

    if particles is not None:
        df = particles.to_pandas(compact=True)
        if reduce_particles_size:
            for dtype in ('float64', 'int64'):
                thistype_columns = df.select_dtypes(include=[dtype]).columns
                df[thistype_columns] = df[thistype_columns].astype(dtype.replace('64', '32'))

        df.to_hdf(fpath, key='particles', format='table', mode='a',
                  complevel=9, complib='blosc')

    if lossmap_data is not None:
        for key, lm_df in lossmap_data.items():
            lm_df.to_hdf(fpath, key=key, mode='a', format='table',
                         complevel=9, complib='blosc')


def _load_lossmap_hdf(filename):
    keys = ('lossmap_scalar', 'lossmap_aper', 'lossmap_coll')

    lm_dict = {}
    for key in keys:
        # Pandas HDF file table format doesn't save empty dataframes
        try:
            lm_dict[key] = pd.read_hdf(filename, key=key)
        except KeyError:
            lm_dict[key] = None
    return lm_dict
            

def load_config(config_file):
    with open(config_file, 'r') as stream:
        config_dict = yaml.safe_load(stream)
    return config_dict


def _compensate_energy_loss(line, delta0=0.):
    _configure_tracker_radiation(line, 'mean', for_optics=True)
    line.compensate_radiation_energy_loss(delta0=delta0)
    line.discard_tracker()


def _prepare_matched_beam(config_dict, line, twiss, ref_particle, element, emitt_x, emitt_y, num_particles, capacity):
    print(f'Preparing a matched Gaussian beam at {element}')
    sigma_z = config_dict['beam']['sigma_z']

    x_norm, px_norm = xp.generate_2D_gaussian(num_particles)
    y_norm, py_norm = xp.generate_2D_gaussian(num_particles)
    
    # The longitudinal closed orbit needs to be manually supplied for now
    element_index = line.element_names.index(element)
    zeta_co = twiss.zeta[element_index] 
    delta_co = twiss.delta[element_index] 

    assert sigma_z >= 0
    zeta = delta = 0
    if sigma_z > 0:
        print(f'Paramter sigma_z > 0, preparing a longitudinal distribution matched to the RF bucket')
        zeta, delta = xp.generate_longitudinal_coordinates(
                        line=line,
                        num_particles=num_particles, distribution='gaussian',
                        sigma_z=sigma_z, particle_ref=ref_particle)

    part = line.build_particles(
        _capacity=capacity,
        particle_ref=ref_particle,
        x_norm=x_norm, px_norm=px_norm,
        y_norm=y_norm, py_norm=py_norm,
        zeta=zeta + zeta_co,
        delta=delta + delta_co,
        nemitt_x=emitt_x,
        nemitt_y=emitt_y,
        at_element=element,
        **XTRACK_TWISS_KWARGS,
        )

    return part


def generate_xpart_particles(config_dict, line, twiss, ref_particle, start_elem, capacity):
    num_particles = config_dict['run']['nparticles']
    radiation_mode = config_dict['run'].get('radiation', 'off')

    emittance = config_dict['beam']['emittance']
    if isinstance(emittance, dict): # Normalised emittances
        ex, ey = emittance['x'], emittance['y']
    else:
        ex = ey = emittance

    particles = _prepare_matched_beam(config_dict, line, twiss, ref_particle, start_elem,  
                                      ex, ey, num_particles, capacity)

    # TODO: Add offsets here
    
    # Disable this option as the tracking from element is handled
    # separately for consistency with other distribution sources
    particles.start_tracking_at_element = -1

    return particles


def prepare_particles(config_dict, line, twiss, ref_particle, start_elem):
    capacity = config_dict['run']['max_particles']

    particles = generate_xpart_particles(config_dict, line, twiss, ref_particle, start_elem, capacity)

    return particles


def get_particle_info(particle_name):
    pdg_id = xp.pdg.get_pdg_id_from_name(particle_name)
    charge, A, Z, _ = xp.pdg.get_properties_from_pdg_id(pdg_id)
    mass = xp.pdg.get_mass_from_pdg_id(pdg_id)
    return ParticleInfo(particle_name, pdg_id, mass, A, Z, charge)


def load_colldb(colldb, emit_dict):
    if colldb.endswith('.json'):
        colldb = xc.CollimatorDatabase.from_json(colldb,
                                                    nemitt_x=emit_dict['x'],
                                                    nemitt_y=emit_dict['y'])
    elif colldb.endswith('.dat'):
        colldb = xc.CollimatorDatabase.from_SixTrack(colldb,
                                                nemitt_x=emit_dict['x'],
                                                nemitt_y=emit_dict['y'])
    else:
        raise ValueError('Unknown collimator database format: {}. Must be .json or .dat'.format(colldb))
    return colldb


def load_and_process_line(config_dict):
    beam = config_dict['beam']
    inp = config_dict['input']
    run = config_dict['run']

    emittance = beam['emittance']

    particle_name = config_dict['beam']['particle']
    particle_info = get_particle_info(particle_name)

    p0 = beam['momentum']
    mass = particle_info.mass
    q0 = particle_info.charge
    ref_part = xt.Particles(p0c=p0, mass0=mass, q0=q0, pdg_id=particle_info.pdgid)

    comp_eloss = run.get('compensate_sr_energy_loss', False)

    # Load the line and compute optics
    line = xt.Line.from_json(inp['xtrack_line'])
    line.particle_ref = ref_part
    
    rf_cavities = line.get_elements_of_type(xt.elements.Cavity)[0]

    if run.get('turn_rf_off', False):
        print('Turning RF cavities off (set voltage to 0)')
        for cav in rf_cavities:
            cav.voltage = 0

    if not any((cav.voltage > 0 for cav in rf_cavities)) or not any((cav.frequency > 0 for cav in rf_cavities)):
        assert not comp_eloss, 'Cannot compensate SR energy loss with cavities off'
        print('RF cavities have no voltage or frequency, Twiss will be 4D')
        XTRACK_TWISS_KWARGS['method'] = '4d'

    print('Using Xtrack-generated twiss table for collimator optics')
    radiation_mode = run['radiation']

    if comp_eloss:
        # If energy loss compensation is required, taper the lattice
        print('Compensating synchrotron energy loss (tapering mangets)')
        comp_eloss_delta0 = run.get('sr_compensation_delta', 0.0)
        _compensate_energy_loss(line, comp_eloss_delta0)
    else:
        # Build and discard the tracker. Needed to have all the element slices with ._parent attribute
        # TODO: possibly remove this when xsuite issue #551 (https://github.com/xsuite/xsuite/issues/551) is resolved
        # NOTE: if comp_eloss is True, the tracker is built and discarded in _compensate_energy_loss
        line.build_tracker()
        line.discard_tracker()    

    colldb = load_colldb(inp['collimator_file'], emittance)
    
    colldb.install_geant4_collimators(line=line, verbose=True)

    _configure_tracker_radiation(line, radiation_mode, for_optics=True)
    twiss = line.twiss(**XTRACK_TWISS_KWARGS)
    print('Assigning optics to collimators...')
    line.collimators.assign_optics(twiss=twiss)
    print('Done assigning optics to collimators.')
    line.discard_tracker()

    ##########################################################################################################
    # Beam-gas (via Xgas Python module)
    ##########################################################################################################

    # Beam-gas options dictionary
    beamgas_opt = inp['beamgas_options']

    # Insert beam-gas elements
    print('Loading gas density profile...')
    density_df = pd.read_csv(inp['gas_density_profile'], sep='\t')
    print('Done loading gas density profile.')
    # Initialise beam-gas manager
    print('Initialising beam-gas manager...')
    bgman = xg.BeamGasManager(density_df, q0, p0,
                              eBrem=beamgas_opt['eBrem'], # Bremsstrahlung on
                              eBrem_energy_cut=beamgas_opt['eBrem_energy_cut'], # Bremsstrahlung gammas low energy cut [eV]
                              CoulombScat=beamgas_opt['CoulombScat'], # Coulomb scattering off
                              theta_lim=(beamgas_opt['theta_min'], beamgas_opt['theta_max']) # Coulomb scattering angle limits
                              )
    print('Done initialising beam-gas manager.')
    # Install beam-gas elements
    print('Installing beam-gas elements...')
    bgman.install_beam_gas_elements(line)
    print('Done installing beam-gas elements.')
    # insert_bg_elems_bounding_apertures(line)

    # TODO: make this more general (seems to be required only when using thick lines)
    if 'superkekb' in inp['machine']:
        insert_missing_bounding_apertures(line)
    # aper_check = line.check_aperture()

    ##########################################################################################################

    # Twiss again to get the optics at the beam-gas elements
    _configure_tracker_radiation(line, radiation_mode, for_optics=True)
    twiss = line.twiss(**XTRACK_TWISS_KWARGS)

    # Randomly select one of the beam-gas elements to use it as start point
    # A unique seed is assigned to reduce probability of selecting the same element in different jobs
    # NOTE: when the script is ran through htcondor, the seed is the htcondor job id
    seed = run.get('seed')
    random.seed(seed)
    start_element = random.choice(bgman.bg_element_names)
    s0 = line.get_s_position(at_elements=start_element, mode='upstream')

    return line, twiss, ref_part, bgman, start_element, s0


def get_n_interactions_dict(line):
    df_line = line.to_pandas()

    mask_bg_elem = df_line['element_type'] == 'BeamGasElement'
    bg_elements_list = df_line[mask_bg_elem].name.values.tolist()

    n_interactions_dict = {}

    for bg_elem in bg_elements_list:
        n_interactions_dict[bg_elem] = int(line[bg_elem].n_interactions)

    return n_interactions_dict


def prepare_lossmap(particles, line, s0, binwidth, weights):
    lossmap_weights = ['none', 'energy']
    if weights not in lossmap_weights:
        raise ValueError('weights must be in [{}]'.format(', '.join(lossmap_weights)))

    s_ele = np.array(line.get_s_elements(mode='downstream'))
    max_s = max(s_ele)
    s_range = (0, max_s)  # Get the end point as assume the start point is zero

    coll_idx = []
    for idx, elem in enumerate(line.elements):
        if isinstance(elem, xc.Geant4Collimator):
            coll_idx.append(idx)
    coll_idx = np.array(coll_idx)
    coll_names = np.take(line.element_names, coll_idx)

    # Ignore unallocated array slots
    mask_allocated = particles.state > -9999
    particles = particles.filter(mask_allocated)

    # Count the number of primary particles for information
    mask_prim = particles.parent_particle_id == particles.particle_id
    n_prim = len(particles.filter(mask_prim).x)

    mask_lost = particles.state <= 0

    # If no losses return empty loss maps, but preserve structures
    if not np.any(mask_lost):
        warn('No losses found, loss map will be empty')
        particles = xp.Particles(x=[], **{kk[1]:getattr(particles,kk[1]) 
                                          for kk in particles.scalar_vars})
    else:
        particles = particles.filter(mask_lost) #(mask_part_type & mask_lost)

    # Get a mask for the collimator losses
    mask_losses_coll = np.isin(particles.at_element, coll_idx)

    if weights == 'energy':
        part_mass_ratio = particles.charge_ratio / particles.chi
        part_mom = (particles.delta + 1) * particles.p0c * part_mass_ratio
        part_mass = part_mass_ratio * particles.mass0
        part_tot_energy = np.sqrt(part_mom**2 + part_mass**2)
        histo_weights = part_tot_energy
    elif weights == 'none':
        histo_weights = np.full_like(particles.x, 1)
    else:
        raise ValueError('weights must be in [{}]'.format(', '.join(lossmap_weights)))

    # Collimator losses binned per element
    h_coll, edges_coll = np.histogram(particles.at_element[mask_losses_coll],
                                      bins=range(max(coll_idx)+2),
                                      weights=histo_weights[mask_losses_coll])

    # Process the collimator per element histogram for plotting
    coll_lengths = np.array([line.elements[ci].length for ci in coll_idx])
    # reduce the empty bars in the histogram
    coll_values = np.take(h_coll, coll_idx)

    coll_end = np.take(s_ele, coll_idx)
    coll_start = coll_end - coll_lengths

    # Aperture losses binned in S
    nbins_ap = int(np.ceil((s_range[1] - s_range[0])/binwidth))
    bins_ap = np.linspace(s_range[0], s_range[1], nbins_ap)

    aper_loss, _ = np.histogram(particles.s[~mask_losses_coll],
                                bins=bins_ap,
                                weights=histo_weights[~mask_losses_coll])

    # Prepare structures for optimal storage
    aper_loss_series = pd.Series(aper_loss)

    # Scalar variables go in their own DF to avoid replication
    # The bin edges can be re-generated with linspace, no need to store
    scalar_dict = {
        'binwidth': binwidth,
        'weights': weights,
        'nbins': nbins_ap,
        's_min': s_range[0],
        's_max': s_range[1],
        'n_primaries': n_prim,
    }

    # Drop the zeros while preserving the index
    aperloss_dict = {
        'aper_loss': aper_loss_series[aper_loss_series > 0],
    }

    coll_dict = {
        'coll_name': coll_names,
        'coll_element_index': coll_idx,
        'coll_start': coll_start,
        'coll_end': coll_end,
        'coll_loss': coll_values
    }

    scalar_df = pd.DataFrame(scalar_dict, index=[0])
    coll_df = pd.DataFrame(coll_dict)
    aper_df = pd.DataFrame(aperloss_dict)

    lm_dict = {'lossmap_scalar': scalar_df,
               'lossmap_aper': aper_df,
               'lossmap_coll': coll_df}

    return lm_dict


def run(config_file_path, config_dict, line, bgman, particles, start_element, s0):
    radiation_mode = config_dict['run']['radiation']
    beamstrahlung_mode = config_dict['run']['beamstrahlung']
    bhabha_mode = config_dict['run']['bhabha']

    nturns = config_dict['run']['turns']
    
    _configure_tracker_radiation(line, radiation_mode, beamstrahlung_mode, bhabha_mode, for_optics=False)
    if 'quantum' in (radiation_mode, beamstrahlung_mode, bhabha_mode):
        # Explicitly initialise the random number generator for the quantum mode
        seed = config_dict['run']['seed']
        if seed > 1e5:
            raise ValueError('The random seed is too large. Please use a smaller seed (<1e5).')
        seeds = np.full(particles._capacity, seed) + np.arange(particles._capacity)
        particles._init_random_number_generator(seeds=seeds)

    # Start interaction record
    impacts = xc.InteractionRecord.start(line=line)

    t0 = time.time()
    xc.Geant4Engine.start(line=line,
                          seed=config_dict['run']['seed'],
                          bdsim_config_file=config_dict['input']['bdsim_config'])
    
    line.scattering.enable()

    # Track
    for turn in range(nturns):
        print(f'\nStart turn {turn}, Survivng particles: {particles._num_active_particles}')

        if turn == 1:
            deactivate_bg_elems(line)
            _configure_tracker_radiation(line, radiation_mode, beamstrahlung_mode, bhabha_mode, for_optics=False)

        line.track(particles, ele_start=start_element, ele_stop=start_element, num_turns=1)

        if particles._num_active_particles == 0:
            print(f'All particles lost by turn {turn}, teminating.')
            break

    line.scattering.disable()
    xc.Geant4Engine.stop()

    impacts.stop()

    print(f'Tracking {nturns} turns done in: {time.time()-t0} s')

    output_file = Path(config_dict['run'].get('outputfile', 'part.hdf'))
    output_file = config_file_path.parent / output_file
    output_dir = output_file.parent
    output_dir = config_file_path.parent / output_dir
    if not os.path.exists(output_dir):
        # If the output directory does not exist, create it
        os.makedirs(output_dir)

    # Save impacts
    df_impacts = impacts.to_pandas()
    del impacts
    fpath = output_dir / 'impacts.csv'
    df_impacts.to_csv(fpath, index=False)
    del df_impacts

    # Save beam_gas_log
    beam_gas_log = bgman.df_interactions_log
    del bgman
    fpath = output_dir / 'beamgas_log.json'
    beam_gas_log.to_json(fpath, orient='records', indent=2)
    del beam_gas_log

    aper_interp = config_dict['run']['aperture_interp']
    # Make xcoll loss map
    LossMap = xc.LossMap(line,
                         part=particles,
                         line_is_reversed=False,
                         interpolation=aper_interp,
                         weights=None, # energy weights?
                         weight_function=None)
    particles = LossMap.part
    # # Save xcoll loss map
    # fpath = output_dir / 'lossmap.json'
    # LossMap.to_json(fpath)
    del LossMap

    # Make collimasim loss map
    # TODO: remove this at some point
    if ('lossmap' in config_dict
            and config_dict['lossmap'].get('make_lossmap', False)):
        binwidth = config_dict['lossmap']['aperture_binwidth']
        weights = config_dict['lossmap'].get('weights', 'none')
        lossmap_data = prepare_lossmap(
            particles, line, s0, binwidth=binwidth, weights=weights)
    else:
        lossmap_data = None

    _save_particles_hdf(output_file, particles, lossmap_data)


def execute(config_file_path, config_dict):
    config_dict = CONF_SCHEMA.validate(config_dict)

    line, twiss, ref_part, bgman, start_elem, s0 = load_and_process_line(config_dict)

    particles = prepare_particles(config_dict, line, twiss, ref_part, start_elem)

    bgman.initialise_particles(particles)

    run(config_file_path, config_dict, line, bgman, particles, start_elem, s0)


@contextmanager
def set_directory(path: Path):
    """
    Taken from: https://dev.to/teckert/changing-directory-with-a-python-context-manager-2bj8
    """
    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)


def dump_dict_to_yaml(dict_obj, file_path):
        with open(file_path, 'w') as yaml_file:
            yaml.dump(dict_obj, yaml_file, 
                      default_flow_style=False, sort_keys=False)
            

def resolve_and_cache_paths(iterable_obj, resolved_iterable_obj, cache_destination):
    if isinstance(iterable_obj, (dict, list)):
        for k, v in (iterable_obj.items() if isinstance(iterable_obj, dict) else enumerate(iterable_obj)):
            possible_path = Path(str(v))
            if not isinstance(v, (dict, list)) and possible_path.exists() and possible_path.is_file():
                shutil.copy(possible_path, cache_destination)
                resolved_iterable_obj[k] = possible_path.name
            resolve_and_cache_paths(v, resolved_iterable_obj[k], cache_destination)


def submit_jobs(config_dict, config_file):
    # Relative path from the config file should be relative to
    # the file itself, not to where the script is executed from
    if config_file:
        conf_path = Path(config_file).resolve()
        conf_dir = conf_path.parent
        conf_fname = conf_path.name
    else:
        conf_dir = Path().resolve()
        conf_fname = 'config_beam_gas.yaml'
        conf_path = Path(conf_dir, conf_fname)
        
    with set_directory(conf_dir):
        config_dict = CONF_SCHEMA.validate(config_dict)

        sub_dict = config_dict['jobsubmission']
        workdir = Path(sub_dict['working_directory']).resolve()
        num_jobs = sub_dict['num_jobs']
        replace_dict_in = sub_dict.get('replace_dict', {})
        executable = sub_dict.get('executable', 'bash')
        mask_abspath = Path(sub_dict['mask']).resolve()
        
        max_local_jobs = 10
        if sub_dict.get('run_local', False) and num_jobs > max_local_jobs:
            raise Exception(f'Cannot run more than {max_local_jobs} jobs locally,'
                            f' {num_jobs} requested.')
            
        # Make a directory to copy the files for the submission
        input_cache = Path(workdir, 'input_cache')
        os.makedirs(workdir)
        os.makedirs(input_cache)

        # Copy the files to the cache and replace the path in the config
        # Copy the configuration file
        if conf_path.exists():
            shutil.copy(conf_path, input_cache)
        else:
            # If the setup came from a dict a dictionary still dump it to archive
            dump_dict_to_yaml(config_dict, Path(input_cache, conf_path.name))
            
        exclude_keys = {'jobsubmission',} # The submission block is not needed for running
        # Preserve the key order
        reduced_config_dict = {k: config_dict[k] for k in 
                               config_dict.keys() if k not in exclude_keys}
        resolved_config_dict = copy.deepcopy(reduced_config_dict)
        resolve_and_cache_paths(reduced_config_dict, resolved_config_dict, input_cache)

        resolved_conf_file = f'for_jobs_{conf_fname}' # config file used to run each job
        dump_dict_to_yaml(resolved_config_dict, Path(input_cache, resolved_conf_file))

        # compress the input cache to reduce network traffic
        shutil.make_archive(input_cache, 'gztar', input_cache)
        # for fpath in input_cache.iterdir():
        #     fpath.unlink()
        # input_cache.rmdir()

        # Set up the jobs
        seeds = np.arange(num_jobs) + 1 # Start the seeds at 1
        replace_dict_base = {'seed': seeds.tolist(),
                             'config_file': resolved_conf_file,
                             'input_cache_archive': str(input_cache) + '.tar.gz'}

        # Pass through additional replace dict option and other job_submitter flags
        if replace_dict_in:
            replace_dict = {**replace_dict_base, **replace_dict_in}
        else:
            replace_dict = replace_dict_base
        
        processed_opts = {'working_directory', 'num_jobs', 'executable', 'mask'}
        submitter_opts = list(set(sub_dict.keys()) - processed_opts)
        submitter_options_dict = { op: sub_dict[op] for op in submitter_opts }
        
        # Send/run the jobs via the job_submitter interface
        htcondor_submit(
            mask=mask_abspath,
            working_directory=workdir,
            executable=executable,
            replace_dict=replace_dict,
            **submitter_options_dict)

        print('Done!')


def submit_local_jobs(config_file_path, config_dict):
    if subprocess.run("parallel --version", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        raise RuntimeError("GNU parallel is not installed or not available in the PATH.")

    config_dict = CONF_SCHEMA.validate(config_dict)

    sub_dict = config_dict['localjobsubmission']
    
    working_dir_path = config_file_path.parent / sub_dict['working_directory']
    os.makedirs(working_dir_path)

    n_jobs = sub_dict['num_jobs']
    # Create directories and copy config files for each job
    for ii in range(n_jobs):
        job_dir = working_dir_path / f"Job.{ii}"
        os.makedirs(job_dir)
        os.makedirs(job_dir / "Outputdata")
        # Copy the config file to each job directory
        config_file_path_job = job_dir / config_file_path.name
        shutil.copy(config_file_path, config_file_path_job)

    python_script = os.path.abspath(__file__)
    max_parallel_jobs = sub_dict['max_parallel_jobs']

    # Write a bash script to file
    with open(config_file_path.parent / 'run_jobs_local.sh', 'w') as f:
        f.write('#!/bin/bash\n\n')
        f.write('start_time=$(date +%s)\n')
        f.write(
            f"/usr/bin/time seq 0 {n_jobs - 1} | parallel -j {max_parallel_jobs} "
            f"'python {python_script} --run {working_dir_path}/Job.{{}}/{config_file_path.name} > {working_dir_path}/Job.{{}}/log.txt'\n"
        )
        f.write('end_time=$(date +%s)\n')
        f.write('echo "Total runtime: $((end_time-start_time)) s"\n')

    print('Local jobs have been set up.')
    print('Execute the run_jobs_local.sh script to run the jobs.')

    # subprocess.run(f"""seq 0 {n_jobs - 1} | parallel -j {max_parallel_jobs} \\
    #                'python {shlex.quote(python_script)} --run {shlex.quote(str(working_dir_path))}/Job.{{}}/{shlex.quote(config_file_path.name)} \\
    #                > {shlex.quote(str(working_dir_path))}/Job.{{}}/log.txt'""", shell=True)


def merge(directory, output_file, match_pattern='*part.hdf*', load_particles=True):
    output_file = Path(output_file)

    t0 = time.time()

    part_merged, lmd_merged, dirs_visited, files_loaded = merge_output.load_output(directory, output_file, match_pattern=match_pattern, load_particles=load_particles)

    _save_particles_hdf(output_file, part_merged, lmd_merged)

    print('Directories visited: {}, files loaded: {}'.format(
        dirs_visited, files_loaded))
    print(f'Processing done in {time.time() -t0} s')


def main():
    if len(sys.argv) != 3:
        raise ValueError(
            'The script only takes two inputs: the mode and the target')

    if sys.argv[1] == '--run':
        t0 = time.time()
        config_file = sys.argv[2]
        config_file_path = Path(config_file).resolve()
        config_dict = load_config(config_file)
        execute(config_file_path, config_dict)
        print(f'Done! Time taken: {time.time()-t0} s')
    elif sys.argv[1] == '--submit':
        config_file = sys.argv[2]
        config_dict = load_config(config_file)
        submit_jobs(config_dict, config_file)
    elif sys.argv[1] == '--submit_local':
        t0 = time.time()
        config_file = sys.argv[2]
        config_file_path = Path(config_file).resolve()
        config_dict = load_config(config_file)
        submit_local_jobs(config_file_path, config_dict)
    elif sys.argv[1] == '--merge':
        match_pattern = '*part.hdf*'
        output_file = 'part_merged.hdf'
        merge(sys.argv[2], output_file, match_pattern=match_pattern, load_particles=True)
    else:
        raise ValueError('The mode must be one of --run, --submit, --submit_local, --merge')


if __name__ == '__main__':
    main()