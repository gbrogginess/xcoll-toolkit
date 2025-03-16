"""
Utilities for handling xtrack lines in different collimation simulation scenarios.
=============================================
Author(s): Giacomo Broggi, Andrey Abramov, Michael Hofer
Email:  giacomo.broggi@cern.ch
Date:   12-03-2025
"""
# ===========================================
# 🔹 Required modules
# ===========================================
import random
import numpy as np
import pandas as pd
import xobjects as xo
import xtrack as xt
import xfields as xf
import xcoll as xc
from warnings import warn

from .config import config
from .utils import get_particle_info
from .beamgas import xgas as xg
from .touschek import xtouschek as xtt


# ===========================================
# 🔹 Custom elements
# ===========================================
CUSTOM_ELEMENTS = {} 
try:
    from xcain import laser_interaction as xcain
    print('XCain found, LaserInteraction will be available as a user element')
    CUSTOM_ELEMENTS['LaserInteraction'] = xcain.LaserInteraction
except ImportError:
    pass


# ===========================================
# 🔹 Search element functions
# ===========================================
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

# ===========================================
# 🔹 Tracker and radiation
# ===========================================
def _configure_tracker_radiation(line, radiation_model, beamstrahlung_model=None, bhabha_model=None, for_optics=False):
    mode_print = 'optics' if for_optics else 'tracking'

    print_message = f"\nTracker synchrotron radiation mode for '{mode_print}' is '{radiation_model}'\n"

    _beamstrahlung_model = None if beamstrahlung_model == 'off' else beamstrahlung_model
    _bhabha_model = None if bhabha_model == 'off' else bhabha_model

    if line.tracker is None:
        print('\n')
        line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'))

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
            print_message = ("\nCannot perform optics calculations with radiation='quantum',"
            " reverting to radiation='mean' for optics.\n")
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

def compensate_energy_loss(line, delta0=0.):
    _configure_tracker_radiation(line, 'mean', for_optics=True)
    line.compensate_radiation_energy_loss(delta0=delta0)
    line.discard_tracker()

# ===========================================
# 🔹 Load CollDB in different formats
# ===========================================
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

# ===========================================
# 🔹 Insert user-defined elements
# ===========================================
def _insert_user_element(line, elem_def, CUSTOM_ELEMENTS):
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

# ===========================================
# 🔹 Beam-beam
# ===========================================
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

# ===========================================
# 🔹 Beam-gas
# ===========================================
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

# ===========================================
# 🔹 Apertures
# ===========================================
def insert_missing_bounding_apertures(line, machine):
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
        if machine == 'superkekb':
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

    if machine == 'superkekb':
        # Shift missing chicane aperture
        line.build_tracker()
        tw = line.twiss()
        line.discard_tracker()
        x_shift = tw['x', chicane_aper_names]

        for ii, nn in enumerate(chicane_aper_names):
            if abs(x_shift[ii]) > 1e-2:
                line[nn].shift_x = x_shift[ii]
                line[nn].shift_x = x_shift[ii]

# ===========================================
# 🔹 Load and process line
# ===========================================
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
    print('\n')
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
        config.XTRACK_TWISS_KWARGS['method'] = '4d'

    print('\nUsing Xtrack-generated twiss table for collimator optics\n')
    radiation_mode = run['radiation']

    if comp_eloss:
        # If energy loss compensation is required, taper the lattice
        print('Compensating synchrotron energy loss (tapering mangets)')
        comp_eloss_delta0 = run.get('sr_compensation_delta', 0.0)
        compensate_energy_loss(line, comp_eloss_delta0)
    else:
        # Build and discard the tracker. Needed to have all the element slices with ._parent attribute
        # TODO: possibly remove this when xsuite issue #551 (https://github.com/xsuite/xsuite/issues/551) is resolved
        # NOTE: if comp_eloss is True, the tracker is built and discarded in _compensate_energy_loss
        line.build_tracker()
        line.discard_tracker()

    # colldb = load_colldb(inp['collimator_file'], emittance)
    
    # colldb.install_geant4_collimators(line=line, verbose=True)

    _configure_tracker_radiation(line, radiation_mode, for_optics=True)
    twiss = line.twiss(**config.XTRACK_TWISS_KWARGS)
    # line.collimators.assign_optics(twiss=twiss)
    line.discard_tracker()

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

    if config.scenario == 'collimation':
        s0 = 0
        start_element = config_dict['dist'].get('start_element', None)
        if start_element is not None:
            s0 = line.get_s_position(at_elements=start_element, mode='upstream')

        return line, twiss, ref_part, start_element, s0

    elif config.scenario == 'beamgas':
        #########################################
        # Beam-gas (via Xgas module)
        #########################################
        beamgas_opt = inp['beamgas_options']
        # Load gas density profile
        print('Loading gas density profile...')
        density_df = pd.read_csv(inp['gas_density_profile'], sep='\t')
        print('Done loading gas density profile.')
        # Initialise beam-gas manager
        print('Initialising beam-gas manager...')
        bgman = xg.BeamGasManager(density_df, q0, p0,
                                  eBrem=beamgas_opt['eBrem'], # Bremsstrahlung
                                  eBrem_energy_cut=beamgas_opt['eBrem_energy_cut'], # Bremsstrahlung photons low energy cut [eV]
                                  CoulombScat=beamgas_opt['CoulombScat'], # Coulomb scattering
                                  theta_lim=(beamgas_opt['theta_min'], beamgas_opt['theta_max']) # Coulomb scattering angle limits
                                  )
        print('Done initialising beam-gas manager.')
        # Install beam-gas elements
        print('Installing beam-gas elements...')
        bgman.install_beam_gas_elements(line)
        print('Done installing beam-gas elements.')

        # TODO: make this more general (seems to be required only when using thick lines)
        if 'superkekb' or 'dafne' in inp['machine']:
            insert_missing_bounding_apertures(line, inp['machine'])
        # aper_check = line.check_aperture()

        # Twiss again to get the optics at the beam-gas elements
        _configure_tracker_radiation(line, radiation_mode, for_optics=True)
        twiss = line.twiss(**config.XTRACK_TWISS_KWARGS)

        # Randomly select one of the beam-gas elements to use it as start point
        # A unique seed is assigned to reduce probability of selecting the same element in different jobs
        # NOTE: when the script is ran through htcondor, the seed is the htcondor job id
        seed = run.get('seed')
        random.seed(seed)
        start_element = random.choice(bgman.bg_element_names)
        s0 = line.get_s_position(at_elements=start_element, mode='upstream')

        return line, twiss, ref_part, bgman, start_element, s0
    
    elif config.scenario == 'touschek':
        #########################################
        # Touschek (via Xtouschek Python module)
        #########################################
        touschek_opt = inp['touschek_options']

        seed = run.get('seed') # In sumbit mode seed is the job_id and starts from 1
        if seed > touschek_opt['n_elems']:
            raise ValueError(f"Seed {seed} is larger than the number of elements {touschek_opt['n_elems']}.\n \
                               In the Touschek simulations the seed is used to select one of the Touschek scattering centers.\n \
                               Please select a seed smaller than {touschek_opt['n_elems']}.")
        element = f"TMarker_{seed-1}" # Touschek scattering centers are counted from 0

        print('Initialising Touschek manager...')
        # Initialise Touschek manager
        local_momaper_fpath = touschek_opt.get('local_momentum_aperture', None)
        if local_momaper_fpath is not None:
            import json
            with open(local_momaper_fpath, 'r') as f:
                local_momaper = json.load(f)
        else:
            local_momaper = None

        touschek_manager = xtt.TouschekManager(line=line,
                                               local_momaper=local_momaper,
                                               n_elems=touschek_opt['n_elems'],
                                               nemitt_x=emittance['x'],
                                               nemitt_y=emittance['y'],
                                               sigma_z=beam['sigma_z'],
                                               sigma_delta=beam['sigma_delta'],
                                               kb=beam['bunch_population'], 
                                               n_part_mc=touschek_opt['n_part_mc'],
                                               fdelta=touschek_opt['fdelta'],
                                               )
        print('Done initialising Touschek manager.')

        touschek_manager.initialise_touschek(element=element)

        # TODO: make this more general (seems to be required only when using thick lines)
        if 'superkekb' or 'dafne' in inp['machine']:
            insert_missing_bounding_apertures(line, inp['machine'])

        return line, touschek_manager, element

    else:
        raise ValueError(f'Unknown scenario: {config.scenario}. The supported scenarios are: collimation, beamgas, touschek.')

