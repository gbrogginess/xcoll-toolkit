"""
Beam distribution generation utilities for xcoll_toolkit.
=============================================
Author(s): Giacomo Broggi, Andrey Abramov
Email:  giacomo.broggi@cern.ch
Date:   12-03-2025
"""
# ===========================================
# 🔹 Required modules
# ===========================================
import shutil
import pandas as pd
import xpart as xp
import xcoll as xc

from pathlib import Path
from .config import config
from .lattice import _configure_tracker_radiation


def load_xsuite_csv_particles(dist_file, ref_particle, line, element, num_part, capacity, copy_file=False):
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
                **config.XTRACK_TWISS_KWARGS,
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
        print(f'Paramter sigma_z > 0, preparing a longitudinal distribution matched to the RF bucket\n')
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
        **config.XTRACK_TWISS_KWARGS,
        )

    return part

def generate_xpart_particles(config_dict, line, twiss, ref_particle, capacity):
    dist_params = config_dict['dist']['parameters']
    num_particles = config_dict['run']['nparticles']
    element = config_dict['dist']['start_element']
    dist_params = config_dict['dist']['parameters']
    radiation_mode = config_dict['run'].get('radiation', 'off')

    emittance = config_dict['beam']['emittance']
    if isinstance(emittance, dict): # Normalised emittances
        ex, ey = emittance['x'], emittance['y']
    else:
        ex = ey = emittance

    _configure_tracker_radiation(line, radiation_mode, for_optics=True)

    particles = None
    dist_type = dist_params.get('type', '')
    if dist_type in ('halo_direct', 'halo_direct_momentum'):
        # Longitudinal distribution if specified
        assert dist_params['sigma_z'] >= 0
        longitudinal_mode = None
        if dist_params['sigma_z'] > 0:
            print(f'Paramter sigma_z > 0, preparing a longitudinal distribution matched to the RF bucket')
            longitudinal_mode = 'bucket'

        particles = xc.generate_pencil_on_collimator(line=line,
                                                     name=element,
                                                     num_particles=num_particles,
                                                     side=dist_params['side'],
                                                     pencil_spread=dist_params['pencil_spread'], # max impact_parameter
                                                     impact_parameter=0,
                                                     sigma_z=dist_params['sigma_z'],
                                                     twiss=twiss,
                                                     longitudinal=longitudinal_mode,
                                                     longitudinal_betatron_cut=None,
                                                     _capacity=capacity)
    elif dist_type == 'matched_beam':
        particles = _prepare_matched_beam(config_dict, line, twiss, ref_particle, 
                                          element, ex, ey, num_particles, capacity)
    else:
        raise Exception('Cannot process beam distribution')

    # TODO: Add offsets here
    
    # Disable this option as the tracking from element is handled
    # separately for consistency with other distribution sources
    particles.start_tracking_at_element = -1

    return particles

def prepare_particles(config_dict, line, twiss, ref_particle, start_elem=None):
    # TODO: Improve this
    capacity = config_dict['run']['max_particles']

    if config.scenario == 'collimation':
        dist = config_dict['dist']
        _supported_dist = ['xsuite', 'internal']

        if dist['source'] == 'xsuite':
            particles = load_xsuite_particles(config_dict, line, ref_particle, capacity)
        elif dist['source'] == 'internal':
            particles = generate_xpart_particles(config_dict, line, twiss, ref_particle, capacity)
        else:
            raise ValueError('Unsupported distribution source: {}. Supported ones are: {}'
                            .format(dist['soruce'], ','.join(_supported_dist)))
        
        part_init_save_file=dist.get('initial_store_file', None)
        # if part_init_save_file is not None:
        #     _save_particles_hdf(particles=particles, lossmap_data=None,
        #                         filename=part_init_save_file)
        return particles
    
    elif config.scenario == 'beamgas':
        num_particles = config_dict['run']['nparticles']

        emittance = config_dict['beam']['emittance']
        if isinstance(emittance, dict): # Normalised emittances
            ex, ey = emittance['x'], emittance['y']
        else:
            ex = ey = emittance

        particles = _prepare_matched_beam(config_dict, line, twiss, ref_particle, start_elem,
                                          ex, ey, num_particles, capacity)
        
        particles.start_tracking_at_element = -1
        
        return particles

    else:
        raise ValueError(f'Unknown scenario: {config.scenario}. The supported scenarios are: collimation, beamgas.')





