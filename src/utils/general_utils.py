import os
import sys

from datetime import datetime, timedelta
from loguru import logger
import cProfile, pstats
from tqdm import tqdm

from typing import Optional, List, Tuple, Callable, Any

import numpy as np
import torch
import pandas as pd


# ============================
# Output and Result Management
# ============================

def verbosity_print(string : str, required_verbosity : int, set_verbosity : int, **kwargs):
    """
    Print function that only prints if the set verbosity exceeds the required_verbosity.

    Args:
        string (str):
            String to possibly print.
        required_verbosity (int):
            Required level of verbosity.
        set_verbosity (int):
            Level of verbosity set in current run.
        **kwargs:
            Key-word arguments for print
    """

    if set_verbosity >= required_verbosity:
        try:
            tqdm.write(string, file=sys.stdout, **kwargs)
        except TypeError:
            tqdm.write(str(string), file=sys.stdout, **kwargs)


def timer(func, *args, **kwargs):
    """
    Uses cProfile to time a function call and returns a pandas Dataframe with the results.
    
    Parameters
    ----------
    func : callable
        Function to profile.
    args
        Arguments for function.
    kwargs
        Keyword arguments for function.
    
    Returns
    -------
    Any, pd.DataFrame
        Output of function call, Dataframe of the cProfile
    """

    # Helper function that parses a cProfile file into a DataFrame
    def parse_profile(file_name : str):
        # Read file
        with open(file_name, 'r') as f:
            lines = f.readlines()
        ## Setup
        data=[]
        started=False
        # Parse line
        for l in lines:
            # Strip whitespace
            l = l.lstrip()
            # Find start of table
            if not started:
                if l == "ncalls  tottime  percall  cumtime  percall filename:lineno(function)\n":
                    started=True
            elif len(l) != 0 and l[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                entries = l.split()
                temp = []
                # First entry requires special attention in case of primitive calls
                first_entry_split = entries[0].split('/')
                if len(first_entry_split) == 1:
                    temp.append(int(first_entry_split[0]))
                    temp.append(0)
                elif len(first_entry_split) == 2:
                    temp.append(int(first_entry_split[0]))
                    temp.append(int(first_entry_split[1]))
                # Second to Fifth
                for i in range(1, 5):
                    temp.append(float(entries[i]))
                # Remaining entries need to be rejoined
                temp.append(' '.join(entries[5:]))
                data.append(tuple(temp))
        # Construct dataframe
        prof_df = pd.DataFrame(data, columns=["Total Calls", "Primitive Calls", "Total Time", "Total Time Average", "Cumulative Time", "Cumulative Time Average", "filename:lineno(function)"])
        # Return
        return prof_df

    # Setup and enable profile
    pr = cProfile.Profile()
    pr.enable()
    # Run function
    output = pr.runcall(func, *args, **kwargs)
    # Disable profile
    pr.disable()
    # Write profile to temp file
    with open('temp.profile', 'w') as f:
        p = pstats.Stats(pr, stream=f)
        p.strip_dirs()
        p.sort_stats('cumtime').print_stats()
    # Parse file into DataFrame
    df = parse_profile('temp.profile')
    # Clean-up temp file
    os.remove('temp.profile')
    # Return
    return output, df


class my_dict(dict):
    def __repr__(self):
        out = ''
        for key, value in self.items():
            if isinstance(value, np.ndarray):
                out += f"'{key}': np.ndarray(shape={value.shape}, dtype={value.dtype}), "
            elif isinstance(value, torch.Tensor):
                out += f"'{key}': np.ndarray(shape={value.shape}, dtype={value.dtype}), device={value.device}), "
            else:
                out += f"'{key}': {value}, "
        return f'{{{out[:-2]}}}'

# ============
# Streamlining
# ============

def tqdm_execute_over_list(parameter_list : List[Tuple[List[Any], str]], execute_function : Callable, input_check : bool = False, position_offset : int = 0):
    """
    A helper function for iterating over all combinations of parameters and running a function on those parameters. tqdm is used to monitor progress.

    Args:
        parameter_list (List[Tuple[List[Any], str]])
            A list of parameters to iterate over. Each parameter is represented as a tuple of the parameter options and the parameter's name.
        execute (Callable):
            A function with arguments with '{parameter_name}' and '{parameter_name}_idx' for each (parameter, parameter_name) tuple in parameter_list.
        input_check (bool, default=False):
            Whether to print the parameter list and wait for input before looping.
        position_offset (int, default=0):
            Offset for the generated tqdm bars for use when running method within already running tqdm bar(s).
    """
    def helper(current_index : int = 0, my_dict : Optional[dict] = None):
        # If first loop, initialise the dictionary
        if current_index == 0:
            my_dict = dict()
        
        # If final, set up loop and then execute
        if current_index == len(parameter_list) - 1:
            ls, name = parameter_list[current_index]
            bar = tqdm(enumerate(ls), total=len(ls), leave=False, dynamic_ncols=True, position=current_index+position_offset, file=sys.stdout)
            for i, val in bar:
                bar.set_description(f"{name} = {val}")
                my_dict[name] = val
                my_dict[f"{name}_idx"] = i
                execute_function(**my_dict)
        # Otherwise, set up loop and recurse
        else:
            ls, name = parameter_list[current_index]
            bar = tqdm(enumerate(ls), total=len(ls), leave=False, dynamic_ncols=True, position=current_index+position_offset)
            for i, val in bar:
                bar.set_description(f"{name} = {val}")
                my_dict[name] = val
                my_dict[f"{name}_idx"] = i
                helper(current_index=current_index+1, my_dict=my_dict)
    
    if input_check:
        print(parameter_list)
        input("Press enter to continue...")
    return helper()
