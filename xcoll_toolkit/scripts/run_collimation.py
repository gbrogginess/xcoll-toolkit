"""
Run collimation simulation with the Xsuite-BDSIM(Geant4) coupling via Xcoll.
=============================================
Author(s): Giacomo Broggi, Andrey Abramov
Email:  giacomo.broggi@cern.ch
Date:   12-03-2025
"""
# ===========================================
# 🔹 Required modules
# ===========================================
import os
import sys
import time
import numpy as np
import xcoll as xc

from pathlib import Path
from xcoll_toolkit import *

# ===========================================================================
# 🔹 Run
# ===========================================================================
def run(config_file_path, config_dict):
    # ===========================================
    # 🔹 Define and validate the simulation scenario
    # ===========================================
    config.scenario = 'collimation'
    config_dict = COLLIMATION_CONF_SCHEMA.validate(config_dict)

    # ===========================================
    # 🔹 Load variables from config_dict
    # ===========================================
    radiation_mode = config_dict['run']['radiation']
    beamstrahlung_mode = config_dict['run']['beamstrahlung']
    bhabha_mode = config_dict['run']['bhabha']
    nturns = config_dict['run']['turns']
    output_file = Path(config_dict['run'].get('outputfile', 'part.hdf'))
    aper_interp = config_dict['run']['aperture_interp']
    
    # ===========================================
    # 🔹 Load and process line
    # ===========================================
    line, twiss, ref_part, start_elem, s0 = load_and_process_line(config_dict)

    # ===========================================
    # 🔹 Prepare particle distribution
    # ===========================================
    particles = prepare_particles(config_dict, line, twiss, ref_part)

    # ===========================================
    # 🔹 Look for changes to element parameters to apply every turn
    # ===========================================
    tbt_change_list = None
    if 'dynamic_change' in config_dict:
        tbt_change_list = setup_dynamic_element_change(config_dict, line, twiss, ref_part, nturns)
    
    # ===========================================
    # 🔹 Configure the tracker with radiation settings
    # ===========================================
    _configure_tracker_radiation(line, radiation_mode, beamstrahlung_mode, bhabha_mode, for_optics=False)

    # ===========================================
    # 🔹 Explicitly initialize the random number generator for the quantum mode
    # ===========================================
    if 'quantum' in (radiation_mode, beamstrahlung_mode, bhabha_mode):
        seed = config_dict['run']['seed']
        if seed > 1e5:
            raise ValueError('The random seed is too large. Please use a smaller seed (<1e5).')
        seeds = np.full(particles._capacity, seed) + np.arange(particles._capacity)
        particles._init_random_number_generator(seeds=seeds)

    # ===========================================
    # 🔹 Track!
    # ===========================================
    t0 = time.time()

    # Start interaction record
    impacts = xc.InteractionRecord.start(line=line)
    # Start the Geant4 engine
    xc.Geant4Engine.start(line=line,
                          seed=config_dict['run']['seed'],
                          bdsim_config_file=config_dict['input']['bdsim_config'])
    # Enable scattering
    line.scattering.enable()
    
    # Track particles
    for turn in range(nturns):
        print(f'\nStart turn {turn}, Survivng particles: {particles._num_active_particles}')
        if tbt_change_list is not None:
            _apply_dynamic_element_change(line, tbt_change_list, turn)

        line.track(particles, ele_start=start_elem, ele_stop=start_elem, num_turns=1)

        if particles._num_active_particles == 0:
            print(f'All particles lost by turn {turn}, teminating.')
            break

    # Disable scattering
    line.scattering.disable()
    # Stop the Geant4 engine
    xc.Geant4Engine.stop()
    # Stop interaction record
    impacts.stop()

    print(f'\nTracking {nturns} turns done in: {time.time()-t0} s\n')

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
    # 🔹 Save impacts table
    # ===========================================
    save_impacts(output_dir, impacts)

    # ===========================================
    # 🔹 Perform loss interpolation and make Xcoll loss map
    # ===========================================
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

    # ===========================================
    # 🔹 Make "collimasim" loss map
    # ===========================================
    # TODO: remove this at some point
    if ('lossmap' in config_dict
            and config_dict['lossmap'].get('make_lossmap', False)):
        binwidth = config_dict['lossmap']['aperture_binwidth']
        weights = config_dict['lossmap'].get('weights', 'none')
        lossmap_data = prepare_lossmap(
            particles, line, s0, binwidth=binwidth, weights=weights)
    else:
        lossmap_data = None

    # ===========================================
    # 🔹 Save particles and loss map
    # ===========================================
    _save_particles_hdf(output_file, particles, lossmap_data)

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
        raise ValueError('The mode must be one of --run, --submit, --submit_local, --merge, --merge_photons')
    
# ===============================================================================
# 🔹 Script mode
# ===============================================================================
if __name__ == '__main__':
    main()