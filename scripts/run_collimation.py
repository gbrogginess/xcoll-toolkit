import os
import re
import sys
import time
import yaml
import shutil
import copy
import numpy as np
import pandas as pd
import subprocess
import merge_output

import xtrack as xt
import xpart as xp
import xfields as xf
import xobjects as xo
import xcoll as xc

from collections import namedtuple
from pathlib import Path
from warnings import warn
from schema import Schema, And, Or, Use, Optional, SchemaError
from contextlib import redirect_stdout, redirect_stderr, contextmanager
from pylhc_submitter.job_submitter import main as htcondor_submit
from memory_profiler import profile
from multiprocessing import Pool


ParticleInfo = namedtuple('ParticleInfo', ['name', 'pdgid', 'mass', 'A', 'Z','charge'])

# Note that YAML has inconsitencies when parsing numbers in scientific notation
# To avoid numbers parsed as strings in some configurations, always cast to float / int
to_float = lambda x: float(x)
to_int = lambda x: int(float(x))


XTRACK_TWISS_KWARGS = {}


INPUT_SCHEMA = Schema({'machine': str,
                       'xtrack_line': os.path.exists,
                       'collimator_file': os.path.exists,
                       'bdsim_config': os.path.exists,
                       Optional('material_rename_map', default={}): Schema({str: str}),
                       })

BEAM_SCHEMA = Schema({'particle': str,
                      'momentum': Use(to_float),
                      'emittance': Or(Use(to_float), {'x': Use(to_float), 'y': Use(to_float)}),
                      })

XSUITE_DIST_SCHEMA = Schema({'file': os.path.exists,
                             Optional('keep_ref_particle', default=False): Use(bool),
                             Optional('copy_file', default=False): Use(bool),
                             })
HALO_DIR_SCHM = Schema({'type': And(str, lambda s: s in ('halo_direct',)),
                        'pencil_spread': Use(to_float),
                        'side': And(str, lambda s: s in ('+', '-', '+-')),
                        'sigma_z': Use(to_float)
                        })
HALO_MDIR_SCHM = Schema({'type': And(str, lambda s: s in ('halo_direct_momentum',)),
                        'num_betatron_sigma': Use(to_float),
                        'pencil_spread': Use(to_float),
                        'side': And(str, lambda s: s in ('+', '-', '+-')),
                        'sigma_z': Use(to_float)
                        })
MATCHED_SCHM = Schema({'type': And(str, lambda s: s in ('matched_beam',)),
                        'sigma_z': Use(to_float)
                        })

DIST_SCHEMA = Schema({'source': And(str, lambda s: s in ('internal', 'xsuite')),
             Optional('start_element', default=None): Or(str.lower, None),
             Optional('weight', default=False): Use(bool),
             Optional('initial_store_file', default=None): Or(str.lower, None),
        'parameters': Or(XSUITE_DIST_SCHEMA,
                         MATCHED_SCHM,
                         HALO_DIR_SCHM,
                         HALO_MDIR_SCHM),
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

DYNC_ELE_SCHEMA = Schema({Optional('element_name'): str,
                          Optional('element_regex'): str,
                          'parameter': str,
                          'change_function': str
                         })

DYNC_SCHEMA = Schema({'element': Or(DYNC_ELE_SCHEMA, Schema([DYNC_ELE_SCHEMA])),
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
                      'dist': DIST_SCHEMA,
                      'run': RUN_SCHEMA,
                      Optional('dynamic_change'): DYNC_SCHEMA,
                      Optional('jobsubmission'): JOB_SUBMIT_SCHEMA,
                      Optional('lossmap'): LOSSMAP_SCHEMA,
                      Optional(object): object})  # Allow input flexibility with extra keys


CUSTOM_ELEMENTS = {} 
try:
    from xcain import laser_interaction as xcain
    print('XCain found, LaserInteraction will be available as a user element')
    CUSTOM_ELEMENTS['LaserInteraction'] = xcain.LaserInteraction
except ImportError:
    pass


def load_config(config_file):
    with open(config_file, 'r') as stream:
        config_dict = yaml.safe_load(stream)
    return config_dict


def get_particle_info(particle_name):
    pdg_id = xp.pdg.get_pdg_id_from_name(particle_name)
    charge, A, Z, _ = xp.pdg.get_properties_from_pdg_id(pdg_id)
    mass = xp.pdg.get_mass_from_pdg_id(pdg_id)
    return ParticleInfo(particle_name, pdg_id, mass, A, Z, charge)


def _configure_tracker_radiation(line, radiation_model, beamstrahlung_model=None, bhabha_model=None, for_optics=False):
    mode_print = 'optics' if for_optics else 'tracking'

    print_message = f"Tracker synchrotron radiation mode for '{mode_print}' is '{radiation_model}'"

    _beamstrahlung_model = None if beamstrahlung_model == 'off' else beamstrahlung_model
    _bhabha_model = None if bhabha_model == 'off' else bhabha_model

    if radiation_model == 'mean':
        if for_optics:
            # Ignore beamstrahlung and bhabha for optics
            line.configure_radiation(model=radiation_model)
        else:
            line.configure_radiation(model=radiation_model, 
                                     model_beamstrahlung=_beamstrahlung_model,
                                     model_bhabha=_bhabha_model)
        # The matrix stability tolerance needs to be relaxed for radiation and tapering
        # TODO: check if this is still needed
        line.matrix_stability_tol = 0.5

    elif radiation_model == 'quantum':
        if for_optics:
            print_message = ("Cannot perform optics calculations with radiation='quantum',"
            " reverting to radiation='mean' for optics.")
            line.configure_radiation(model='mean')
        else:
            line.configure_radiation(model='quantum',
                                     model_beamstrahlung=_beamstrahlung_model,
                                     model_bhabha=_bhabha_model)
        line.matrix_stability_tol = 0.5

    elif radiation_model == 'off':
        pass
    else:
        raise ValueError('Unsupported radiation model: {}'.format(radiation_model))
    print(print_message)


def _compensate_energy_loss(line, delta0=0.):
    _configure_tracker_radiation(line, 'mean')
    line.compensate_radiation_energy_loss(delta0=delta0)


def _insert_user_element(line, elem_def):
    elements = {**vars(xt.beam_elements.elements), **CUSTOM_ELEMENTS}
    # try a conversion to float, as because of the arbitraty yaml
    # inputs, no type enforcement can be made at input validation time
    # anything that can be cast to a number is likely a number
    parameters = {}
    for param, value in elem_def['parameters'].items():
        try:
            parameters[param] = float(value)
        except:
            parameters[param] = value
    print(parameters)
    elem_name = elem_def['name']
    #elem_obj = getattr(xt, elem_def['type'])(**elem_def['parameters'])
    elem_obj = elements[elem_def['type']](**parameters)
    s_position = elem_def['at_s']

    if not isinstance(s_position, list):
        print(f'Inserting {elem_name} ({elem_obj}) at s={s_position} m')
        line.insert_element(at_s=float(s_position), element=elem_obj, 
                            name=elem_name)
    else:
        for i, s_pos in enumerate(s_position):
            # TODO: Is a new instance really needed every time here?
            unique_name = f'{elem_name}_{i}'
            #unique_elem_obj = getattr(xt, elem_def['type'])(**elem_def['parameters'])
            unique_elem_obj = elements[elem_def['type']](**parameters)
            print(f'Inserting {unique_name} ({unique_elem_obj}) at s={s_pos} m')

            line.insert_element(at_s=float(s_pos), 
                                element=unique_elem_obj, 
                                name=unique_name)
            

def _make_bb_lens(nb, phi, sigma_z, alpha, n_slices, other_beam_q0,
                  sigma_x, sigma_px, sigma_y, sigma_py, beamstrahlung_on=False):
       
    slicer = xf.TempSlicer(n_slices=n_slices, sigma_z=sigma_z, mode="shatilov")

    el_beambeam = xf.BeamBeamBiGaussian3D(
            #_context=context,
            config_for_update = None,
            other_beam_q0=other_beam_q0,
            phi=phi, # half-crossing angle in radians
            alpha=alpha, # crossing plane
            # decide between round or elliptical kick formula
            min_sigma_diff = 1e-28,
            # slice intensity [num. real particles] n_slices inferred from length of this
            slices_other_beam_num_particles = slicer.bin_weights * nb,
            # unboosted strong beam moments
            slices_other_beam_zeta_center = slicer.bin_centers,
            slices_other_beam_Sigma_11    = n_slices*[sigma_x**2], # Beam sizes for the other beam, assuming the same is approximation
            slices_other_beam_Sigma_22    = n_slices*[sigma_px**2],
            slices_other_beam_Sigma_33    = n_slices*[sigma_y**2],
            slices_other_beam_Sigma_44    = n_slices*[sigma_py**2],
            # only if BS on
            slices_other_beam_zeta_bin_width_star_beamstrahlung = None if not beamstrahlung_on else slicer.bin_widths_beamstrahlung / np.cos(phi),  # boosted dz
            # has to be set
            slices_other_beam_Sigma_12    = n_slices*[0],
            slices_other_beam_Sigma_34    = n_slices*[0],
            compt_x_min                   = 1e-4,
        )
    el_beambeam.iscollective = True # Disable in twiss

    return el_beambeam


def find_apertures(line):
    i_apertures = []
    apertures = []
    for ii, ee in enumerate(line.elements):
        if ee.__class__.__name__.startswith('Limit'):
            i_apertures.append(ii)
            apertures.append(ee)
    return np.array(i_apertures), np.array(apertures)


def find_bb_lenses(line):
    i_apertures = []
    apertures = []
    for ii, ee in enumerate(line.elements):
        if ee.__class__.__name__.startswith('BeamBeamBiGaussian3D'):
            i_apertures.append(ii)
            apertures.append(ee)
    return np.array(i_apertures), np.array(apertures)


def insert_bb_lens_bounding_apertures(line):
    # Place aperture defintions around all beam-beam elements in order to ensure
    # the correct functioning of the aperture loss interpolation
    # the aperture definitions are taken from the nearest neighbour aperture in the line
    s_pos = line.get_s_elements(mode='upstream')
    apert_idx, apertures = find_apertures(line)
    apert_s = np.take(s_pos, apert_idx)

    bblens_idx, bblenses = find_bb_lenses(line)
    bblens_names = np.take(line.element_names, bblens_idx)
    bblens_s_start = np.take(s_pos, bblens_idx)
    bblens_s_end = np.take(s_pos, bblens_idx + 1)

    # Find the nearest neighbour aperture in the line
    bblens_apert_idx_start = np.searchsorted(apert_s, bblens_s_start, side='left')
    bblens_apert_idx_end = bblens_apert_idx_start + 1

    aper_start = apertures[bblens_apert_idx_start]
    aper_end = apertures[bblens_apert_idx_end]

    idx_offset = 0
    for ii in range(len(bblenses)):
        line.insert_element(name=bblens_names[ii] + '_aper_start',
                            element=aper_start[ii].copy(),
                            at=bblens_idx[ii] + idx_offset)
        idx_offset += 1

        line.insert_element(name=bblens_names[ii] + '_aper_end',
                            element=aper_end[ii].copy(),
                            at=bblens_idx[ii] + 1 + idx_offset)
        idx_offset += 1
            

def _insert_beambeam_elements(line, config_dict, twiss_table, emit):
    beamstrahlung_mode = config_dict['run'].get('beamstrahlung', 'off')
    beamstrahlung_mode = config_dict['run'].get('bhabha', 'off')
    # This is needed to set parameters of the beam-beam lenses
    beamstrahlung_on = beamstrahlung_mode != 'off'

    beambeam_block = config_dict.get('beambeam', None)
    if beambeam_block is not None:

        beambeam_list = beambeam_block
        if not isinstance(beambeam_list, list):
            beambeam_list = [beambeam_list, ]

        print('Beam-beam definitions found, installing beam-beam elements at: {}'
              .format(', '.join([dd['at_element'] for dd in beambeam_list])))
            
        for bb_def in beambeam_list:
            element_name = bb_def['at_element']
            # the beam-beam lenses are thin and have no effects on optics so no need to re-compute twiss
            element_twiss_index = list(twiss_table.name).index(element_name)
            # get the line index every time as it changes when elements are installed
            element_line_index = line.element_names.index(element_name)
            #element_spos = twiss_table.s[element_twiss_index]
            
            sigmas = twiss_table.get_betatron_sigmas(*emit if hasattr(emit, '__iter__') else (emit, emit))

            bb_elem = _make_bb_lens(nb=float(bb_def['bunch_intensity']), 
                                    phi=float(bb_def['crossing_angle']), 
                                    sigma_z=float(bb_def['sigma_z']),
                                    n_slices=int(bb_def['n_slices']),
                                    other_beam_q0=int(bb_def['other_beam_q0']),
                                    alpha=0, # Put it to zero, it is okay for this use case
                                    sigma_x=np.sqrt(sigmas['Sigma11'][element_twiss_index]), 
                                    sigma_px=np.sqrt(sigmas['Sigma22'][element_twiss_index]), 
                                    sigma_y=np.sqrt(sigmas['Sigma33'][element_twiss_index]), 
                                    sigma_py=np.sqrt(sigmas['Sigma44'][element_twiss_index]), 
                                    beamstrahlung_on=beamstrahlung_on)
            
            line.insert_element(index=element_line_index, 
                                element=bb_elem,
                                name=f'beambeam_{element_name}')
        
        insert_bb_lens_bounding_apertures(line)

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

    # # Check that all aperture markers are set
    # apercheck = line.check_aperture()
    # from IPython import embed; embed()
    
    rf_cavities = line.get_elements_of_type(xt.elements.Cavity)[0]

    if run.get('turn_rf_off', False):
        print('Turning RF cavities off (set voltage to 0)')
        for cv in rf_cavities:
            cv.voltage = 0

    if not any((cv.voltage > 0 for cv in rf_cavities)) or not any((cv.frequency > 0 for cv in rf_cavities)):
        assert not comp_eloss, 'Cannot compensate SR energy loss with cavities off'
        print('RF cavities have no voltage or frequency, Twiss will be 4D')
        XTRACK_TWISS_KWARGS['method'] = '4d'

    print('Using Xtrack-generated twiss table for collimator optics')
    # Use a clean tracker to compute the optics
    # TODO: reduce the copying here
    optics_line = line.copy()
    optics_line.build_tracker()
    radiation_mode = run['radiation']

    if comp_eloss:
        # If energy loss compensation is required, taper the lattice
        print('Compensating synchrotron energy loss (tapering mangets)')
        comp_eloss_delta0 = run.get('sr_compensation_delta', 0.0)
        _compensate_energy_loss(optics_line, comp_eloss_delta0)
        line = optics_line.copy()

    # Build and discard the tracker. Needed to have all the element slices with ._parent attribute
    # TODO: remove this when xsuite issue #551 (https://github.com/xsuite/xsuite/issues/551) is resolved
    line.build_tracker()
    line.discard_tracker()    

    # TODO: make this more elegant
    if inp['collimator_file'].endswith('.json'):
        colldb = xc.CollimatorDatabase.from_json(inp['collimator_file'],
                                                    nemitt_x=emittance['x'],
                                                    nemitt_y=emittance['y'])
    elif inp['collimator_file'].endswith('.dat'):
        colldb = xc.CollimatorDatabase.from_SixTrack(inp['collimator_file'],
                                                nemitt_x=emittance['x'],
                                                nemitt_y=emittance['y'])
    
    colldb.install_geant4_collimators(line=line, verbose=True)

    line.build_tracker()
    _configure_tracker_radiation(line, radiation_mode, for_optics=True)
    twiss = line.twiss(**XTRACK_TWISS_KWARGS)
    line.collimators.assign_optics(twiss=twiss)
    line.discard_tracker()

    s0 = 0
    start_element = config_dict['dist'].get('start_element', None)
    if start_element is not None:
        s0 = line.get_s_position(at_elements=start_element, mode='upstream')

    # Insert additional elements if any are specified:
    insert_elems = config_dict.get('insert_element', None)
    if insert_elems is not None:
        print('Inserting user-defined elements in the lattice')
        insert_elem_list = insert_elems
        if not isinstance(insert_elem_list, list):
            insert_elem_list = [insert_elem_list, ]
        
        for elem_def in insert_elem_list:
            _insert_user_element(line, elem_def)

    # Insert beam-beam lenses if any are specified:
    _insert_beambeam_elements(line, config_dict, twiss, (emittance['x'], emittance['y']))
    return line, ref_part, start_element, s0


def build_collimation_tracker(line):
    # Chose a context
    context = xo.ContextCpu()  # Only support CPU for Geant4 coupling TODO: maybe not needed anymore?
    # Transfer lattice on context and compile tracking code
    global_aper_limit = 1e3  # Make this large to ensure particles lost on aperture markers

    # compile the track kernel once and set it as the default kernel. TODO: a bit clunky, find a more elegant approach

    line.build_tracker(_context=context)

    tracker_opts=dict(track_kernel=line.tracker.track_kernel,
                      _buffer=line.tracker._buffer,
                      _context=line.tracker._context,
                      io_buffer=line.tracker.io_buffer)
    
    line.discard_tracker() 
    line.build_tracker(**tracker_opts)
    line.config.global_xy_limit=global_aper_limit


def load_xsuite_csv_particles(dist_file, ref_particle, line, element, num_part, capacity, keep_ref_particle=False, copy_file=False):

    orig_file_path = Path(dist_file)
    if copy_file:
        dest = Path.cwd() / f'copied_{orig_file_path.name}'
        shutil.copy2(orig_file_path, dest)
        dist_file_path = dest
    else:
        dist_file_path = orig_file_path

    part_df = pd.read_csv(dist_file_path, index_col=0)

    at_element = line.element_names.index(element)
    start_s = line.get_s_position(at_elements=at_element, mode="upstream")
    
    particles = line.build_particles(
                _capacity=capacity,
                particle_ref=ref_particle,
                mode='set',
                x = part_df['x'].values[:num_part],
                px = part_df['px'].values[:num_part],
                y = part_df['y'].values[:num_part],
                py = part_df['py'].values[:num_part],
                zeta = part_df['zeta'].values[:num_part],
                delta = part_df['delta'].values[:num_part],
                **XTRACK_TWISS_KWARGS,
            )
    
    particles.start_tracking_at_element = at_element
    particles.at_element = at_element
    particles.s = start_s
    
    return particles


def load_xsuite_particles(config_dict, line, ref_particle, capacity):
    dist_params = config_dict['dist']['parameters']
    num_particles = config_dict['run']['nparticles']
    element = config_dict['dist']['start_element']

    dist_file = dist_params['file']
    keep_ref_part = dist_params['keep_ref_particle']
    copy_file = dist_params['copy_file']

    part = load_xsuite_csv_particles(dist_file, ref_particle, line, element, num_particles, capacity, 
                                     keep_ref_particle=keep_ref_part, copy_file=copy_file)
    
    return part


def _prepare_matched_beam(config_dict, line, ref_particle, element, emitt_x, emitt_y, num_particles, capacity):
    print(f'Preparing a matched Gaussian beam at {element}')
    sigma_z = config_dict['dist']['parameters']['sigma_z']
    radiation_mode =  config_dict['run'].get('radiation', 'off')

    _configure_tracker_radiation(line, radiation_mode, for_optics=True)

    x_norm, px_norm = xp.generate_2D_gaussian(num_particles)
    y_norm, py_norm = xp.generate_2D_gaussian(num_particles)
    
    # The longitudinal closed orbit needs to be manually supplied for now
    twiss = line.twiss(**XTRACK_TWISS_KWARGS)
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

def generate_xpart_particles(config_dict, line, ref_particle, capacity):
    dist_params = config_dict['dist']['parameters']
    num_particles = config_dict['run']['nparticles']
    element = config_dict['dist']['start_element']
    dist_params = config_dict['dist']['parameters']

    emittance = config_dict['beam']['emittance']
    if isinstance(emittance, dict): # Normalised emittances
        ex, ey = emittance['x'], emittance['y']
    else:
        ex = ey = emittance

    particles = None
    dist_type = dist_params.get('type', '')
    if dist_type in ('halo_direct', 'halo_direct_momentum'):
        # Longitudinal distribution if specified
        assert dist_params['sigma_z'] >= 0
        longitudinal_mode = None
        if dist_params['sigma_z'] > 0:
            print(f'Paramter sigma_z > 0, preparing a longitudinal distribution matched to the RF bucket')
            longitudinal_mode = 'bucket'

        twiss = line.twiss(**XTRACK_TWISS_KWARGS)
        particles = xc.generate_pencil_on_collimator(line=line,
                                                     name=element,
                                                     num_particles=num_particles,
                                                     side=dist_params['side'],
                                                     pencil_spread=dist_params['pencil_spread'], # pencil spread is what I usually call impact parameter
                                                     impact_parameter=0,
                                                     sigma_z=dist_params['sigma_z'],
                                                     twiss=twiss,
                                                     longitudinal=longitudinal_mode,
                                                     longitudinal_betatron_cut=None,
                                                     _capacity=capacity)
    elif dist_type == 'matched_beam':
        particles = _prepare_matched_beam(config_dict, line, ref_particle, 
                                          element, ex, ey, num_particles, capacity)
    else:
        raise Exception('Cannot process beam distribution')

    # Add weights for ICS simulations with Xcain
    if config_dict['dist']['weight']:
        KB_FCCEE_Z = 2.14E+11 # FCC-ee Z bunch population
        weight = np.ones(capacity) * KB_FCCEE_Z / num_particles
        particles.weight = weight

    # TODO: Add offsets here
    
    # Disable this option as the tracking from element is handled
    # separately for consistency with other distribution sources
    particles.start_tracking_at_element = -1

    return particles


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


def prepare_particles(config_dict, line, ref_particle):
    dist = config_dict['dist']
    capacity = config_dict['run']['max_particles']

    _supported_dist = ['xsuite', 'internal']

    if dist['source'] == 'xsuite':
        particles = load_xsuite_particles(config_dict, line, ref_particle, capacity)
    elif dist['source'] == 'internal':
        particles = generate_xpart_particles(config_dict, line, ref_particle, capacity)
    else:
        raise ValueError('Unsupported distribution source: {}. Supported ones are: {}'
                         .format(dist['soruce'], ','.join(_supported_dist)))
    
    part_init_save_file=dist.get('initial_store_file', None)
    # if part_init_save_file is not None:
    #     _save_particles_hdf(particles=particles, lossmap_data=None,
    #                         filename=part_init_save_file)
    return particles


def _collect_element_names(line, match_string, regex_mode=False):
    match_re = re.compile(match_string) if regex_mode else re.compile(re.escape(match_string))
    names = [name for name in line.element_names if match_re.fullmatch(name)]
    if not names:
        raise Exception(f'Found no elements matching {match_string}')
    return names


def _compute_parameter(parameter, expression, turn, max_turn, extra_variables={}):
    # custom function handling - random numbers, special functions
    # populate the local variables with the computed values
    # TODO This is a bit wonky - may need a full parser later on
    if 'rand_uniform' in expression:
        rand_uniform = np.random.random()
    if 'rand_onoff' in expression:
        rand_onoff = np.random.randint(2)
    var_dict = {**locals(), **extra_variables}
    return type(parameter)(ne.evaluate(expression, local_dict=var_dict))


def _prepare_dynamic_element_change(line, twiss_table, gemit_x, gemit_y, change_dict_list, max_turn):
    if change_dict_list is None:
        return None
    
    tbt_change_list = []
    for change_dict in change_dict_list:
        if not ('element_name' in change_dict) != ('element_regex' in change_dict):
            raise ValueError('Element name for dynamic change not speficied.')

        element_match_string = change_dict.get('element_regex', change_dict.get('element_name'))
        regex_mode = bool(change_dict.get('element_regex', False))

        parameter = change_dict['parameter']
        change_function = change_dict['change_function']

        element_names = _collect_element_names(line, element_match_string, regex_mode)
        # Handle the optional parameter[index] specifiers
        # like knl[0]
        param_split = parameter.replace(']','').split('[')
        param_name = param_split[0]
        param_index = int(param_split[1]) if len(param_split)==2 else None

        #parameter_values = []
        ebe_change_dict = {}
        if Path(change_function).exists():
            turn_no_in, value_in = np.genfromtxt('tbt_params.txt', 
                                                 converters = {0: int, 1: float}, 
                                                 unpack=True, comments='#')
            parameter_values = np.interp(range(max_turn), turn_no_in, value_in).tolist()
            # If the the change function is loaded from file,
            # all of the selected elements in the block have the same values
            for ele_name in element_names:
                ebe_change_dict[ele_name] = parameter_values
        else:
            ebe_keys = set(twiss_table._col_names) - {'W_matrix', 'name'}
            scalar_keys = (set(twiss_table._data.keys()) 
                           - set(twiss_table._col_names) 
                           - {'R_matrix', 'values_at', 'particle_on_co'})

            # If the change is computed on the fly, iterative changes
            # are permitted, e.g a = a + 5, so must account for different starting values
            for ele_name in element_names:
                parameter_values = []

                elem = line.element_dict[ele_name]
                elem_index = line.element_names.index(ele_name)
                elem_twiss_vals = {kk: float(twiss_table[kk][elem_index]) for kk in ebe_keys}
                scalar_twiss_vals = {kk: twiss_table[kk] for kk in scalar_keys}
                twiss_vals = {**elem_twiss_vals, **scalar_twiss_vals}

                twiss_vals['sigx'] = np.sqrt(gemit_x * twiss_vals['betx'])
                twiss_vals['sigy'] = np.sqrt(gemit_y * twiss_vals['bety'])
                twiss_vals['sigxp'] = np.sqrt(gemit_x * twiss_vals['gamx'])
                twiss_vals['sigyp'] = np.sqrt(gemit_y * twiss_vals['gamy'])

                param_value = getattr(elem, param_name)
                if param_index is not None:
                    param_value = param_value[param_index]
                twiss_vals['parameter0'] = param_value # save the intial value for use too

                for turn in range(max_turn):
                    param_value = _compute_parameter(param_value, change_function, 
                                                     turn, max_turn, 
                                                     extra_variables=twiss_vals)
                    parameter_values.append(param_value)

                ebe_change_dict[ele_name] = parameter_values

        tbt_change_list.append([param_name, param_index, ebe_change_dict])
        print('Dynamic element change list: ', tbt_change_list)

    return tbt_change_list


def _set_element_parameter(element, parameter, index, value):
    if index is not None:
        getattr(element, parameter)[index] = value
    else:
        setattr(element, parameter, value)


def _apply_dynamic_element_change(line, tbt_change_list, turn):
    for param_name, param_index, ebe_change_dict in tbt_change_list:
        for ele_name in ebe_change_dict:
            element = line.element_dict[ele_name]
            param_val = ebe_change_dict[ele_name][turn]
            _set_element_parameter(element, param_name, param_index, param_val)


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
    mask_losses_coll = np.in1d(particles.at_element, coll_idx)

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
        conf_fname = 'config_collimation.yaml'
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


def run(config_file_path, config_dict, line, particles, ref_part, start_element, s0):
    radiation_mode = config_dict['run']['radiation']
    beamstrahlung_mode = config_dict['run']['beamstrahlung']
    bhabha_mode = config_dict['run']['bhabha']

    nturns = config_dict['run']['turns']

    # Look for changes to element parameters to apply every turn
    tbt_change_list = None
    if 'dynamic_change' in config_dict:
        dyn_change_dict =  config_dict['dynamic_change']

        _configure_tracker_radiation(line, radiation_mode, for_optics=True)
        twiss_table = line.twiss(**XTRACK_TWISS_KWARGS)

        emittance = config_dict['beam']['emittance']
        if isinstance(emittance, dict):
            emit = (emittance['x'], emittance['y'])
        else:
            emit = (emittance, emittance)

        gemit_x = emit[0]/ref_part.beta0[0]/ref_part.gamma0[0]
        gemit_y = emit[1]/ref_part.beta0[0]/ref_part.gamma0[0]

        if 'element' in dyn_change_dict:
            dyn_change_elem = dyn_change_dict.get('element', None)
            if dyn_change_elem is not None and not isinstance(dyn_change_elem, list):
                dyn_change_elem = [dyn_change_elem,]
            tbt_change_list = _prepare_dynamic_element_change(line, twiss_table, gemit_x, gemit_y, dyn_change_elem, nturns)
    
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
        print(f'Start turn {turn}, Survivng particles: {particles._num_active_particles}')
        if tbt_change_list is not None:
            _apply_dynamic_element_change(line, tbt_change_list, turn)

        line.track(particles, ele_start=start_element, ele_stop=start_element, num_turns=1)

        if particles._num_active_particles == 0:
            print(f'All particles lost by turn {turn}, teminating.')
            break

    line.scattering.disable()
    xc.Geant4Engine.stop()

    impacts.stop()

    print(f'Tracking {nturns} turns done in: {time.time()-t0} s')

    output_file = Path(config_dict['run'].get('outputfile', 'part.hdf'))
    output_dir = output_file.parent
    # This is new
    output_file = config_file_path.parent / output_file
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

    aper_interp = config_dict['run']['aperture_interp']
    # Make xcoll loss map
    LossMap = xc.LossMap(line,
                         part=particles,
                         line_is_reversed=False,
                         interpolation=aper_interp,
                         weights=None, # energy weights?
                         weight_function=None)
    particles = LossMap.part
    # Save xcoll loss map
    fpath = output_dir / 'lossmap.json'
    LossMap.to_json(fpath)
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
    
    line, ref_part, start_elem, s0 = load_and_process_line(config_dict)

    build_collimation_tracker(line)

    particles = prepare_particles(config_dict, line, ref_part)

    run(config_file_path, config_dict, line, particles, ref_part, start_elem, s0)


def merge(directory, output_file, match_pattern='*part.hdf*', load_particles=True):
    output_file = Path(output_file)

    t0 = time.time()

    if match_pattern == '*part.hdf*':
        part_merged, lmd_merged, dirs_visited, files_loaded = merge_output.load_output(directory,
                                                                                       output_file,
                                                                                       match_pattern=match_pattern,
                                                                                       load_particles=load_particles,
                                                                                       load_lossmap=True)
    elif match_pattern == '*photons.hdf*':
        part_merged, lmd_merged, dirs_visited, files_loaded = merge_output.load_output(directory,
                                                                                       output_file,
                                                                                       match_pattern=match_pattern,
                                                                                       load_particles=load_particles,
                                                                                       load_lossmap=False)

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
    elif sys.argv[1] == '--merge_photons':
        match_pattern = '*photons.hdf*'
        output_file = 'photons_merged.hdf'
        merge(sys.argv[2], output_file, match_pattern=match_pattern, load_particles=True)
    else:
        raise ValueError('The mode must be one of --run, --submit, --submit_local, --merge, --merge_photons')


if __name__ == '__main__':
    main()