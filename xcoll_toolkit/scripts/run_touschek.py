"""
Run Touschek simulation with the Xsuite-BDSIM(Geant4) coupling via Xcoll.
=============================================
Author(s): Giacomo Broggi
Email:  giacomo.broggi@cern.ch
Date:   13-03-2025
"""
# ===========================================
# 🔹 Required modules
# ===========================================
import os
import sys
import json
import time
import numpy as np
import xobjects as xo
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
    config.scenario = 'touschek'
    config_dict = TOUSCHEK_CONF_SCHEMA.validate(config_dict)

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
    line, touschek_manager, start_elem = load_and_process_line(config_dict)

    # ===========================================
    # 🔹 Take Touschek scattered distribution
    # ===========================================
    touschek_dict = touschek_manager.touschek_dict
    particles = touschek_dict[start_elem]['particles']

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

    # # Start interaction record
    # impacts = xc.InteractionRecord.start(line=line)
    # Start the Geant4 engine
    xc.Geant4Engine.start(line=line,
                          seed=config_dict['run']['seed'],
                          relative_energy_cut=config_dict['run']['energy_cut'],
                          bdsim_config_file=config_dict['input']['bdsim_config'])
    # Enable scattering
    line.scattering.enable()
    
    # Track particles
    print(f'\nTrack particles scattered at {start_elem}. \n')
    for turn in range(nturns):
        print(f'\nStart turn {turn}, Survivng particles: {particles._num_active_particles}')

        line.track(particles, ele_start=start_elem, ele_stop=start_elem, num_turns=1)

        if particles._num_active_particles == 0:
            print(f'All particles lost by turn {turn}, teminating.')
            break

    # Disable scattering
    line.scattering.disable()
    # Stop the Geant4 engine
    xc.Geant4Engine.stop()
    # # Stop interaction record
    # impacts.stop()

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

    # # ===========================================
    # # 🔹 Save impacts table
    # # ===========================================
    # save_impacts(output_dir, impacts)

    # ===========================================
    # 🔹 Save Touschek log
    # ===========================================
    tab = line.get_table()
    s_start_elem = tab.rows[tab.name == start_elem].s[0]
    touschek_log = {
        'element': start_elem,
        's': s_start_elem,
        'mc_rate': touschek_dict[start_elem]['mc_rate'],
        'piwinski_rate': touschek_dict[start_elem]['piwinski_rate'],
        'mc_to_piwinski_ratio': touschek_dict[start_elem]['mc_to_piwinski_ratio']
    }
    fpath = output_dir / 'touschek_log.json'
    with open(fpath, 'w') as f:
        json.dump(touschek_log, f, indent=4)

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
        config.scenario = 'touschek'
        config_file = sys.argv[2]
        config_dict = load_config(config_file)
        submit_jobs(config_dict, config_file)
    elif sys.argv[1] == '--submit_local':
        config.scenario = 'touschek'
        t0 = time.time()
        config_file = sys.argv[2]
        config_file_path = Path(config_file).resolve()
        config_dict = load_config(config_file)
        submit_local_jobs(config_file_path, config_dict)
    elif sys.argv[1] == '--merge':
        match_pattern = '*part.hdf*'
        output_file = 'part_merged.hdf'
        merge(sys.argv[2], output_file, match_pattern=match_pattern, load_lossmap=False, load_particles=True)
    else:
        raise ValueError('The mode must be one of --run, --submit, --submit_local, --merge, --merge_photons')
    
# ===============================================================================
# 🔹 Script mode
# ===============================================================================
if __name__ == '__main__':
    main()