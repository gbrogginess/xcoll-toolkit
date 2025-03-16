"""
Configuration file for xcoll_toolkit.
=============================================
Author(s): Giacomo Broggi, Andrey Abramov
Email:  giacomo.broggi@cern.ch
Date:   12-03-2025
"""
# ===========================================
# 🔹 Required modules
# ===========================================
import os
from schema import Schema, And, Or, Use, Optional

# ===========================================
# 🔹 Package Metadata
# ===========================================
__version__ = "0.1.0"
__author__ = "Giacomo Broggi, Andrey Abramov"
__email__ = "giacomo.broggi@cern.ch"
__license__ = "Apache-2.0"

# ===========================================
# 🔹 Helper Functions for Config Validation
# ===========================================
# Note that YAML has inconsitencies when parsing numbers in scientific notation
# To avoid numbers parsed as strings in some configurations, always cast to float / int
to_float = lambda x: float(x)
to_int = lambda x: int(float(x))

# ===========================================
# 🔹 Global constants & default values
# ===========================================
DEFAULT_OUTPUT_FILE = "part.hdf"
DEFAULT_BATCH_MODE = True

VALID_RADIATION_MODELS = ("off", "mean", "quantum")
VALID_LOSSMAP_NORMALIZATIONS = ("none", "total", "max", "max_coll", "total_coll")
VALID_LOSSMAP_WEIGHTS = ("none", "energy")

# ===========================================
# 🔹 Schema definitions for config file validation
# ===========================================
COLLIMATION_INPUT_SCHEMA = Schema({'machine': str,
                                  'xtrack_line': os.path.exists,
                                  'collimator_file': os.path.exists,
                                  'bdsim_config': os.path.exists,
                                  Optional('material_rename_map', default={}): Schema({str: str}),
                                  })

BEAMGAS_OPTIONS_SCHEMA = Schema({'eBrem': Use(bool),
                                 Optional('eBrem_energy_cut'): Use(to_float),
                                 'CoulombScat': Use(bool),
                                 Optional('theta_min'): Use(to_float),
                                 Optional('theta_max'): Use(to_float)
                                 })
BEAMGAS_INPUT_SCHEMA = Schema({
                              'machine': str,
                              'xtrack_line': And(str, os.path.exists),
                              'collimator_file': And(str, os.path.exists),
                              'bdsim_config': And(str, os.path.exists),
                              'gas_density_profile': And(str, os.path.exists),
                              'beamgas_options': BEAMGAS_OPTIONS_SCHEMA,
                              Optional('material_rename_map', default={}): Schema({str: str}),
                              })

TOUSCHEK_OPTIONS_SCHEMA = Schema({'n_elems': Use(to_int),
                                 'n_part_mc': Use(to_int),
                                 Optional('local_momentum_aperture'): And(str, os.path.exists),
                                 'fdelta': Use(to_float),
                                 })
TOUSCHEK_INPUT_SCHEMA = Schema({
                              'machine': str,
                              'xtrack_line': And(str, os.path.exists),
                              'collimator_file': And(str, os.path.exists),
                              'bdsim_config': And(str, os.path.exists),
                              'touschek_options': TOUSCHEK_OPTIONS_SCHEMA,
                              Optional('material_rename_map', default={}): Schema({str: str}),
                              })

COLLIMATION_BEAM_SCHEMA = Schema({'particle': str,
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

BEAMGAS_BEAM_SCHEMA = Schema({'particle': str,
                      'momentum': Use(to_float),
                      'emittance': Or(Use(to_float), {'x': Use(to_float), 'y': Use(to_float)}),
                      'sigma_z': Use(to_float),
                      })

TOUSCHEK_BEAM_SCHEMA = Schema({'particle': str,
                      'momentum': Use(to_float),
                      'emittance': Or(Use(to_float), {'x': Use(to_float), 'y': Use(to_float)}),
                      'sigma_z': Use(to_float),
                      'sigma_delta': Use(to_float),
                      'bunch_population': Use(to_float)
                      })

DIST_SCHEMA = Schema({'source': And(str, lambda s: s in ('internal', 'xsuite')),
             Optional('start_element', default=None): Or(str.lower, None),
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

COLLIMATION_CONF_SCHEMA = Schema({'input': COLLIMATION_INPUT_SCHEMA,
                                'beam': COLLIMATION_BEAM_SCHEMA,
                                'dist': DIST_SCHEMA,
                                'run': RUN_SCHEMA,
                                Optional('dynamic_change'): DYNC_SCHEMA,
                                Optional('jobsubmission'): JOB_SUBMIT_SCHEMA,
                                Optional('lossmap'): LOSSMAP_SCHEMA,
                                Optional(object): object})  # Allow input flexibility with extra keys

BEAMGAS_CONF_SCHEMA = Schema({'input': BEAMGAS_INPUT_SCHEMA,
                      'beam': BEAMGAS_BEAM_SCHEMA,
                      'run': RUN_SCHEMA,
                      Optional('lossmap'): LOSSMAP_SCHEMA,
                      Optional('jobsubmission'): JOB_SUBMIT_SCHEMA,
                      Optional(object): object})  # Allow input flexibility with extra keys

TOUSCHEK_CONF_SCHEMA = Schema({'input': TOUSCHEK_INPUT_SCHEMA,
                      'beam': TOUSCHEK_BEAM_SCHEMA,
                      'run': RUN_SCHEMA,
                      Optional('lossmap'): LOSSMAP_SCHEMA,
                      Optional('jobsubmission'): JOB_SUBMIT_SCHEMA,
                      Optional(object): object})  # Allow input flexibility with extra keys

# ===========================================
# 🔹 Singleton class for shared configurations
# ===========================================
class Config:
    """ Singleton class to store global settings across the package."""
    XTRACK_TWISS_KWARGS = {}  # Twiss keyword arguments
    scenario = None # Simulation scenario (collimation, beamgas)

config = Config()
