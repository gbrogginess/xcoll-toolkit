"""
A package for collimation simulations with the Xsuite-BDSIM(Geant4) coupling via Xcoll.

Authors:
    - Giacomo Broggi (giacomo.broggi@cern.ch)
    - Andrey Abramov

License: Apache 2.0
Version: 0.1.0.dev1
"""

__version__ = "0.1.0.dev1"

from .config import config, COLLIMATION_CONF_SCHEMA, BEAMGAS_CONF_SCHEMA, TOUSCHEK_CONF_SCHEMA
from .utils import load_config, _save_particles_hdf, save_impacts, save_beamgas_log, _emittance_tracking
from .lattice import load_and_process_line, _configure_tracker_radiation, deactivate_bg_elems
from .beam import prepare_particles
from .dynamic_element_change import setup_dynamic_element_change, _apply_dynamic_element_change
from .submit import submit_jobs, submit_local_jobs
from .merge import merge
from .lossmap import prepare_lossmap

__all__ = [
    "config",
    "COLLIMATION_CONF_SCHEMA",
    "BEAMGAS_CONF_SCHEMA",
    "TOUSCHEK_CONF_SCHEMA",
    "load_config",
    "_save_particles_hdf",
    "save_impacts",
    "save_beamgas_log",
    "_emittance_tracking",
    "load_and_process_line",
    "_configure_tracker_radiation",
    "deactivate_bg_elems",
    "prepare_particles",
    "setup_dynamic_element_change",
    "_apply_dynamic_element_change",
    "submit_jobs",
    "submit_local_jobs",
    "merge",
    "prepare_lossmap"
]
