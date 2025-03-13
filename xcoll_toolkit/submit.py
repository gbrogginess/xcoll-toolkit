"""
Module to submit jobs to HTCondor or run parallel jobs locally.
Submission to HTcondor is done using the pylhc_submitter package.
Alternatively, a bash script is produced to run parallel jobs locally using GNU parallel.
=============================================
Author(s): Giacomo Broggi, Andrey Abramov
Email:  giacomo.broggi@cern.ch
Date:   12-03-2025
"""
# ===========================================
# 🔹 Required modules
# ===========================================
import os
import yaml
import shutil
import copy
import subprocess
import numpy as np

from pathlib import Path
from contextlib import contextmanager
from pylhc_submitter.job_submitter import main as htcondor_submit
from .config import config, COLLIMATION_CONF_SCHEMA, BEAMGAS_CONF_SCHEMA, TOUSCHEK_CONF_SCHEMA


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
        if config.scenario == 'collimation':
            config_dict = COLLIMATION_CONF_SCHEMA.validate(config_dict)
        elif config.scenario == 'beamgas':
            config_dict = BEAMGAS_CONF_SCHEMA.validate(config_dict)
        elif config.scenario == 'touschek':
            config_dict = TOUSCHEK_CONF_SCHEMA.validate(config_dict)
        else:
            raise ValueError(f'Unknown scenario: {config.scenario}. The supported ones are collimation, beamgas and touschek.')

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

    if config.scenario == 'collimation':
        config_dict = COLLIMATION_CONF_SCHEMA.validate(config_dict)
    elif config.scenario == 'beamgas':
        config_dict = BEAMGAS_CONF_SCHEMA.validate(config_dict)
    elif config.scenario == 'touschek':
        config_dict = TOUSCHEK_CONF_SCHEMA.validate(config_dict)
    else:
        raise ValueError(f'Unknown scenario: {config.scenario}. The supported ones are collimation, beamgas and touschek.')

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