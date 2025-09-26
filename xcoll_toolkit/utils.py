"""
Common helper functions for xcoll_toolkit.

This module provides utility functions for file handling and particle properties.
=============================================
Author(s): Giacomo Broggi, Andrey Abramov
Email:  giacomo.broggi@cern.ch
Date:   12-03-2025
"""
# ===========================================
# 🔹 Required modules
# ===========================================
import yaml
import numpy as np
import pandas as pd
import xtrack as xt

from collections import namedtuple

# ===========================================
# 🔹 File Handling Functions
# ===========================================
def load_config(config_file):
    """Load a YAML configuration file."""
    with open(config_file, 'r') as stream:
        config_dict = yaml.safe_load(stream)
    return config_dict

def _save_particles_hdf(fpath, particles=None, lossmap_data=None, reduce_particles_size=False):
    """Save particles and loss map to an HDF file."""
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
            
def save_impacts(output_dir, impacts):
    """Save impacts to a CSV file."""
    fpath = output_dir / 'impacts.csv'
    df_impacts = impacts.to_pandas()
    df_impacts.to_parquet(fpath, index=False)

def save_beamgas_log(output_dir, bgman):
    """Save beam-gas log to a JSON file."""
    fpath = output_dir / 'beamgas_log.json'
    beam_gas_log = bgman.df_interactions_log
    beam_gas_log.to_json(fpath, orient='records', indent=2)


def load_lossmap_hdf(filename):
    """Loads lossmap data from an HDF file."""
    keys = ("lossmap_scalar", "lossmap_aper", "lossmap_coll")
    lm_dict = {}
    for key in keys:
        try:
            lm_dict[key] = pd.read_hdf(filename, key=key)
        except KeyError:
            lm_dict[key] = None
    return lm_dict

# ===========================================
# 🔹 Particle Properties
# ===========================================
ParticleInfo = namedtuple("ParticleInfo", ["name", "pdgid", "mass", "A", "Z", "charge"])

def get_particle_info(particle_name):
    pdg_id = xt.particles.pdg.get_pdg_id_from_name(particle_name)
    charge, A, Z, _ = xt.particles.pdg.get_properties_from_pdg_id(pdg_id)
    mass = xt.particles.pdg.get_mass_from_pdg_id(pdg_id)
    return ParticleInfo(particle_name, pdg_id, mass, A, Z, charge)

# ===========================================
# 🔹 Emittance tracking
# ===========================================
def _rms_emit(xsq, pxsq, xpxsq, no_particles):
    return np.sqrt((xsq/no_particles)*(pxsq/no_particles) - (xpxsq/no_particles)**2)

def _compute_second_order_moments(particles):
    xsq = np.sum(particles.x**2)
    pxsq = np.sum(particles.px**2)
    xpxsq = np.sum(particles.x*particles.px)

    ysq = np.sum(particles.y**2)
    pysq = np.sum(particles.py**2)
    ypysq = np.sum(particles.y*particles.py)

    no_particles = len(particles.x)

    return {
        'gemitt_x': _rms_emit(xsq, pxsq, xpxsq, no_particles),
        'gemitt_y': _rms_emit(ysq, pysq, ypysq, no_particles)
    }

def _emittance_tracking(particles):
    mask_alive = particles.state > 0
    pp = particles.filter(mask_alive)

    emit_dict = _compute_second_order_moments(pp)

    return emit_dict