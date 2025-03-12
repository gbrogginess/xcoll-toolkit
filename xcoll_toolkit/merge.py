"""
Module to merge collimation simulation outputs from different jobs.
=============================================
Author(s): Giacomo Broggi, Andrey Abramov
Email:  giacomo.broggi@cern.ch
Date:   12-03-2025
"""
# ===========================================
# 🔹 Required modules
# ===========================================
import time
import os
import glob
import numpy as np
import pandas as pd
import xpart as xp

from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

from .utils import _save_particles_hdf


def _read_particles_hdf(filename):        
    return pd.read_hdf(filename, key='particles')

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

def load_output(directory, match_pattern='*part.hdf*',
                imax=None, load_lossmap=True, load_particles=False):

    job_dirs = glob.glob(os.path.join(directory, 'Job.*')
                         )  # find directories to loop over

    job_dirs_sorted = []
    for i in range(len(job_dirs)):
        # Very inefficient, but it sorts the directories by their numerical index
        job_dir_idx = job_dirs.index(
            os.path.join(directory, 'Job.{}'.format(i)))
        job_dirs_sorted.append(job_dirs[job_dir_idx])

    part_hdf_files = []
    part_dataframes = []
    lossmap_dicts = []
    
    tqdm_ncols=100
    tqdm_miniters=10
    print(f'Parsing directories...')
    dirs_visited = 0
    files_loaded = 0
    for i, d in tqdm(enumerate(job_dirs_sorted), total=len(job_dirs_sorted), 
                     ncols=tqdm_ncols, miniters=tqdm_miniters):
        if imax is not None and i > imax:
            break

        #print(f'Processing {d}')
        dirs_visited += 1
        output_dir = os.path.join(d, 'Outputdata')
        output_files = glob.glob(os.path.join(output_dir, match_pattern))
        if output_files:
            of = output_files[0]
            part_hdf_files.append(of)
            files_loaded += 1
        else:
            print(f'No output found in {d}')

    part_merged = None
    if load_particles:
        print(f'Loading particles...')
        with Pool() as p:
            part_dataframes = list(tqdm(p.imap(_read_particles_hdf, part_hdf_files), total=len(part_hdf_files), 
                                        ncols=tqdm_ncols, miniters=tqdm_miniters))
        part_objects = [xp.Particles.from_pandas(pdf) for pdf in tqdm(part_dataframes, total=len(part_dataframes),
                                                                      ncols=tqdm_ncols, miniters=tqdm_miniters)]

        print('Particles load finished, merging...')
        part_merged = xp.Particles.merge(list(tqdm(part_objects, total=len(part_objects),
                                              ncols=tqdm_ncols, miniters=tqdm_miniters)))

    # Load the loss maps
    lmd_merged = None
    if load_lossmap:
        print(f'Loading loss map data...')
        with Pool() as p:
            lossmap_dicts = list(tqdm(p.imap(_load_lossmap_hdf, part_hdf_files), total=len(part_hdf_files), 
                                      ncols=tqdm_ncols, miniters=tqdm_miniters))

        print('Loss map load finished, merging..')

        num_tol = 1e-9
        lmd_merged = lossmap_dicts[0]
        for lmd in tqdm(lossmap_dicts[1:], ncols=tqdm_ncols, miniters=tqdm_miniters):
            # Scalar parameters
            # Ensure consistency
            identical_params = ('s_min', 's_max', 'binwidth', 'nbins')
            identical_strings = ('weights',)
            for vv in identical_params:
                assert np.isclose(lmd_merged['lossmap_scalar'][vv],
                                  lmd['lossmap_scalar'][vv],
                                  num_tol)
            for vv in identical_strings:
                assert np.all(lmd_merged['lossmap_scalar'][vv] == lmd['lossmap_scalar'][vv])

            lmd_merged['lossmap_scalar']['n_primaries'] += lmd['lossmap_scalar']['n_primaries']

            # Collimator losses
            # These cannot be empty dataframes even if there is no losses
            assert np.allclose(lmd_merged['lossmap_coll']['coll_start'],
                               lmd['lossmap_coll']['coll_start'],
                               atol=num_tol)

            assert np.allclose(lmd_merged['lossmap_coll']['coll_end'],
                               lmd['lossmap_coll']['coll_end'],
                               atol=num_tol)
            
            assert np.array_equal(lmd_merged['lossmap_coll']['coll_element_index'],
                                  lmd['lossmap_coll']['coll_element_index'])
            
            assert np.array_equal(lmd_merged['lossmap_coll']['coll_name'],
                                  lmd['lossmap_coll']['coll_name'])

            lmd_merged['lossmap_coll']['coll_loss'] += lmd['lossmap_coll']['coll_loss']

            # Aperture losses
            alm = lmd_merged['lossmap_aper']
            al = lmd['lossmap_aper']

            # If the aperture loss dataframe is empty, it is not stored on HDF
            if al is not None:
                if alm is None:
                    lmd_merged['lossmap_aper'] = al
                else:
                    lm = alm.aper_loss.add(al.aper_loss, fill_value=0)
                    lmd_merged['lossmap_aper'] = pd.DataFrame(
                        {'aper_loss': lm})
                    
    return part_merged, lmd_merged, dirs_visited, files_loaded

def merge(directory, output_file, match_pattern='*part.hdf*', load_particles=True):
    output_file = Path(output_file)
    t0 = time.time()
    part_merged, lmd_merged, dirs_visited, files_loaded = load_output(directory,
                                                                      match_pattern=match_pattern,
                                                                      load_particles=load_particles,
                                                                      load_lossmap=True)
    
    _save_particles_hdf(output_file, part_merged, lmd_merged)
    print('Directories visited: {}, files loaded: {}'.format(
        dirs_visited, files_loaded))
    print(f'Processing done in {time.time() -t0} s')