"""
Module to apply dynamic element changes during tracking simulations with Xsuite.
=============================================
Author(s): Giacomo Broggi, Andrey Abramov, Michael Hofer
Email:  giacomo.broggi@cern.ch
Date:   12-03-2025
"""
# ===========================================
# 🔹 Required modules
# ===========================================
import re
import numpy as np
import numexpr as ne

from pathlib import Path


def _collect_element_names(line, match_string, regex_mode=False):
    match_re = re.compile(match_string) if regex_mode else re.compile(re.escape(match_string))
    names = [name for name in line.element_names if match_re.fullmatch(name)]
    if not names:
        raise Exception(f'Found no elements matching {match_string}')
    return names

def _compute_parameter(parameter, expression, turn, max_turn, extra_variables={}):
    # custom function handling - random numbers, special functions
    # populate the local variables with the computed values
    # TODO This is a bit wonky - may need a full parser later on
    if 'rand_uniform' in expression:
        rand_uniform = np.random.random()
    if 'rand_onoff' in expression:
        rand_onoff = np.random.randint(2)
    var_dict = {**locals(), **extra_variables}
    return type(parameter)(ne.evaluate(expression, local_dict=var_dict))

def _prepare_dynamic_element_change(line, twiss, gemitt_x, gemitt_y, change_dict_list, max_turn):
    if change_dict_list is None:
        return None
    
    tbt_change_list = []
    for change_dict in change_dict_list:
        if not ('element_name' in change_dict) != ('element_regex' in change_dict):
            raise ValueError('Element name for dynamic change not speficied.')

        element_match_string = change_dict.get('element_regex', change_dict.get('element_name'))
        regex_mode = bool(change_dict.get('element_regex', False))

        parameter = change_dict['parameter']
        change_function = change_dict['change_function']

        element_names = _collect_element_names(line, element_match_string, regex_mode)
        # Handle the optional parameter[index] specifiers
        # like knl[0]
        param_split = parameter.replace(']','').split('[')
        param_name = param_split[0]
        param_index = int(param_split[1]) if len(param_split)==2 else None

        #parameter_values = []
        ebe_change_dict = {}
        if Path(change_function).exists():
            turn_no_in, value_in = np.genfromtxt('tbt_params.txt', 
                                                 converters = {0: int, 1: float}, 
                                                 unpack=True, comments='#')
            parameter_values = np.interp(range(max_turn), turn_no_in, value_in).tolist()
            # If the the change function is loaded from file,
            # all of the selected elements in the block have the same values
            for ele_name in element_names:
                ebe_change_dict[ele_name] = parameter_values
        else:
            ebe_keys = set(twiss._col_names) - {'W_matrix', 'name'}
            scalar_keys = (set(twiss._data.keys()) 
                           - set(twiss._col_names) 
                           - {'R_matrix', 'values_at', 'particle_on_co'})

            # If the change is computed on the fly, iterative changes
            # are permitted, e.g a = a + 5, so must account for different starting values
            for ele_name in element_names:
                parameter_values = []

                elem = line.element_dict[ele_name]
                elem_index = line.element_names.index(ele_name)
                elem_twiss_vals = {kk: float(twiss[kk][elem_index]) for kk in ebe_keys}
                scalar_twiss_vals = {kk: twiss[kk] for kk in scalar_keys}
                twiss_vals = {**elem_twiss_vals, **scalar_twiss_vals}

                twiss_vals['sigx'] = np.sqrt(gemitt_x * twiss_vals['betx'])
                twiss_vals['sigy'] = np.sqrt(gemitt_y * twiss_vals['bety'])
                twiss_vals['sigxp'] = np.sqrt(gemitt_x * twiss_vals['gamx'])
                twiss_vals['sigyp'] = np.sqrt(gemitt_y * twiss_vals['gamy'])

                param_value = getattr(elem, param_name)
                if param_index is not None:
                    param_value = param_value[param_index]
                twiss_vals['parameter0'] = param_value # save the intial value for use too

                for turn in range(max_turn):
                    param_value = _compute_parameter(param_value, change_function, 
                                                     turn, max_turn, 
                                                     extra_variables=twiss_vals)
                    parameter_values.append(param_value)

                ebe_change_dict[ele_name] = parameter_values

        tbt_change_list.append([param_name, param_index, ebe_change_dict])
        print('Dynamic element change list: ', tbt_change_list)

    return tbt_change_list

def setup_dynamic_element_change(config_dict, line, twiss, ref_part, nturns):
    dyn_change_dict = config_dict['dynamic_change']

    emittance = config_dict['beam']['emittance']
    if isinstance(emittance, dict):
        emit = (emittance['x'], emittance['y'])
    else:
        emit = (emittance, emittance)

    beta0 = ref_part.beta0[0]
    gamma0 = ref_part.gamma0[0]

    gemitt_x = emit[0] / beta0 / gamma0
    gemitt_y = emit[1] / beta0 / gamma0

    dyn_change_elem = dyn_change_dict.get('element', None)
    if dyn_change_elem is not None and not isinstance(dyn_change_elem, list):
        dyn_change_elem = [dyn_change_elem]

    return _prepare_dynamic_element_change(line, twiss, gemitt_x, gemitt_y, dyn_change_elem, nturns)


def _set_element_parameter(element, parameter, index, value):
    if index is not None:
        getattr(element, parameter)[index] = value
    else:
        setattr(element, parameter, value)

def _apply_dynamic_element_change(line, tbt_change_list, turn):
    for param_name, param_index, ebe_change_dict in tbt_change_list:
        for ele_name in ebe_change_dict:
            element = line.element_dict[ele_name]
            param_val = ebe_change_dict[ele_name][turn]
            _set_element_parameter(element, param_name, param_index, param_val)