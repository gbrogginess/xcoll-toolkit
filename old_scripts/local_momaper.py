"""
Script to precompute local momentum aperture for Touschek scattering simulations.
=============================================
Author(s): Giacomo Broggi
Email:  giacomo.broggi@cern.ch
Date:   16-03-2025
"""
# ===========================================
# 🔹 Required modules
# ===========================================
import os
import sys
import json
import time

from pathlib import Path
from xcoll_toolkit import *
import xtrack as xt
import xcoll as xc
import numpy as np

from ..touschek.xtouschek import TouschekManager
from ..utils import get_particle_info
from ..lattice import compensate_energy_loss, load_colldb

# ===========================================================================
# 🔹 Run
# ===========================================================================
def run(config_file_path, config_dict):
    # ===========================================
    # 🔹 Define and validate the simulation scenario
    # ===========================================
    config.scenario = 'touschek'
    config_dict = TOUSCHEK_CONF_SCHEMA.validate(config_dict)

    # ===========================================
    # 🔹 Load variables from config_dict
    # ===========================================
    radiation_mode = config_dict['run']['radiation']
    nturns = config_dict['run']['turns']

    # ===========================================
    # 🔹 Load and process line
    # ===========================================
    beam = config_dict['beam']
    inp = config_dict['input']
    run = config_dict['run']

    emittance = beam['emittance']

    machine = inp['machine']

    output_file = Path(f'local_momentum_aperture/{machine}_local_momentum_aperture.json')

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

    colldb = load_colldb(inp['collimator_file'], emittance)
    
    colldb.install_geant4_collimators(line=line, verbose=True)

    _configure_tracker_radiation(line, radiation_mode, for_optics=True)
    twiss = line.twiss(**config.XTRACK_TWISS_KWARGS)
    line.collimators.assign_optics(twiss=twiss)
    line.discard_tracker()

    print('Initialising Touschek manager...')
    # Initialise Touschek manager
    touschek_opt = inp['touschek_options']
    touschek_manager = TouschekManager(line=line,
                                       local_momaper=None,
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

    s = touschek_manager._get_s_elements_to_insert()

    touschek_manager._install_touschek_markers(s)

    local_momentum_aperture = touschek_manager._compute_local_momentum_aperture(
                                    n_turns=nturns
                              )
    
    # ===========================================
    # 🔹 Setup output directory
    # ===========================================
    output_file = config_file_path.parent / output_file
    output_dir = output_file.parent
    output_dir = config_file_path.parent / output_dir
    if not os.path.exists(output_dir):
        # If the output directory does not exist, create it
        os.makedirs(output_dir)

    # ===========================================
    # 🔹 Save local momentum aperture
    # ===========================================
    with open(output_file, 'w') as f:
        json.dump(local_momentum_aperture, f, indent=4)

# ===========================================================================
# 🔹 Entry point
# ===========================================================================
def main():
    if len(sys.argv) != 3:
        raise ValueError(
            'The script only takes two inputs: the mode and the target')
    if sys.argv[1] == '--run':
        t0 = time.time()
        config_file = sys.argv[2]
        config_file_path = Path(config_file).resolve()
        config_dict = load_config(config_file)
        run(config_file_path, config_dict)
        print(f'Done! Time taken: {time.time()-t0} s')
    else:
        raise ValueError('The mode must be --run')
    
# ===============================================================================
# 🔹 Script mode
# ===============================================================================
if __name__ == '__main__':
    main()