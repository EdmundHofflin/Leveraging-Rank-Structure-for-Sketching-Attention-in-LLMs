import os
from datetime import datetime, timedelta

import pickle as pk
from loguru import logger

from typing import  Optional, Union, List, Any
from jsonargparse.typing import restricted_number_type, Path_dw, Path_fr, PositiveInt, ClosedUnitInterval
from json import load, dump, JSONEncoder

import numpy as np
import torch
import pandas as pd
import xarray as xr

from utils.general_utils import my_dict


# =========
# CONSTANTS
# =========

_JSON_DUMPER_DICT = {
    "skipkeys": False,
    "ensure_ascii": True,
    "check_circular": True,
    "allow_nan": True,
    "indent": 2,
    "separators": (',', ': '),
    "sort_keys": False
}



# ================
# Helper Functions
# ================

class _my_encoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.dtype):
            return str(obj)
        elif isinstance(obj, xr.DataArray):
            return str(obj)
        elif isinstance(obj, pd.DataFrame):
            return str(obj)
        elif isinstance(obj, Union[datetime, timedelta]):
            return str(obj)
        elif isinstance(obj, Union[PositiveInt, ClosedUnitInterval, restricted_number_type, Path_fr, Path_dw]):
            return str(obj)
        else:
            super(_my_encoder, self).default(obj)


def _process_data_dict(input_dict : dict, output_path : str, logging : int = 0):
    """
    Parses a dictionary, possibly through deeper dictionaries, and converts any NumPy arrays or PyTorch Tensors into strings that point to file locations of their data.

    Args:
        input_dict (dict):
            The dictionary to parse.
        output_path (str):
            Path to the directory to save the NumPy arrays and/or PyTorch tensors. 
    """

    if logging > 0:
        logger.debug(f"Starting _process_data_dict: {input_dict.keys()=}, {output_path=}")
    else:
        logger.trace(f"Starting _process_data_dict: {input_dict.keys()=}, {output_path=}")
    new_dict = my_dict()
    for key, val in input_dict.items():
        if isinstance(val, (dict, my_dict)):
            if not os.path.exists(f"{output_path}/{key}/"):
                os.mkdir(f"{output_path}/{key}/")
            new_dict[key] = _process_data_dict(val, output_path=f"{output_path}/{key}", logging=0 if logging < 2 else logging)
            if len(os.listdir(f"{output_path}/{key}/")) == 0:
                os.rmdir(f"{output_path}/{key}/")
        elif isinstance(val, list):
            new_ls = []
            for i, entry in enumerate(val):
                if isinstance(entry, np.ndarray):
                    file_path = f"{output_path}/{key}_{i}.npy"
                    if logging > 0:
                        logger.trace(f"Starting saving of NumPy array to {file_path}")
                    with open(file_path, 'wb') as f:
                        np.save(f, entry)
                    new_ls.append(file_path)
                    if logging > 0:
                        logger.trace(f"Successful saving of NumPy array to {file_path}")
                elif isinstance(entry, torch.Tensor):
                    file_path = f"{output_path}/{key}_{i}.npy"
                    if logging > 0:
                        logger.trace(f"Starting saving of PyTorch tensor to {file_path}")
                    with open(file_path, 'wb') as f:
                        torch.save(f, entry)
                    new_ls.append(file_path)
                    if logging > 0:
                        logger.trace(f"Successful saving of PyTorch tensor to {file_path}")
                else:
                    new_ls.append(entry)
            new_dict[key] = new_ls
        else:
            if isinstance(val, np.ndarray):
                file_path = f"{output_path}/{key}.npy"
                if logging > 0:
                    logger.trace(f"Starting saving of NumPy array to {file_path}")
                with open(file_path, 'wb') as f:
                    np.save(f, val)
                new_dict[key] = file_path
                if logging > 0:
                    logger.trace(f"Successful saving of NumPy array to {file_path}")
            elif isinstance(val, torch.Tensor):
                file_path = f"{output_path}/{key}.npy"
                if logging > 0:
                    logger.trace(f"Starting saving of PyTorch tensor to {file_path}")
                with open(file_path, 'wb') as f:
                    torch.save(f, val)
                new_dict[key] = file_path
                if logging > 0:
                    logger.trace(f"Successful saving of PyTorch tensor to {file_path}")
            else:
                new_dict[key] = val
    if logging > 0:
        logger.debug(f"Successful _process_data_dict: {new_dict.keys()=}, {output_path=}")
    else:
        logger.trace(f"Successful _process_data_dict: {new_dict.keys()=}, {output_path=}")
    return new_dict


def _save_result(output_dict : dict, output_path : str, file_name : str, data_subdir : Optional[str]):
    """
    Saves the results.

    Args:
        output_dict (dict):
            Dictionary to save as JSON file.
        output_path (str):
            Path to the output directory.
        file_name (str):
            Name of the file.
        data_subdir (Optional[str]):
            Subdirectory within the output directory to save the data. If None, the dictionary isn't processed and large arrays not saved separately.
    """

    logger.debug(f"Starting _save_result: {output_path=}, {file_name=},\n{output_dict.keys()=}")
    # Update with datetime
    output_dict['write_datetime'] = datetime.now()
    if 'start_datetime' in output_dict:
        output_dict['elapsed_time'] = output_dict['write_datetime'] - output_dict['start_datetime']
    # Save NumPy arrays and PyTorch tensors in separate files
    if data_subdir:
        if not os.path.exists(f"{output_path}/{data_subdir}"):
            splits = data_subdir.split('/')
            current_path = ""
            for split in splits:
                current_path += f"{split}/"
                if not os.path.exists(f"{output_path}/{current_path}"):
                    os.mkdir(f"{output_path}/{current_path}")
        output_dict = _process_data_dict(output_dict, output_path=f"{output_path}/{data_subdir}", logging=1)
        if len(os.listdir(f"{output_path}/{data_subdir}")) == 0:
            os.rmdir(f"{output_path}/{data_subdir}")
    # Write to file
    with open(f"{output_path}/{file_name}", "w") as file:
        dump(output_dict, file, cls=_my_encoder, **_JSON_DUMPER_DICT)
    logger.debug(f"Successful _save_result")


def _read_data_dict(input_dict : dict, logging : int = 0, path : Optional[List[str]] = None, recursive : Optional[bool] = True, end_keys : Optional[List[Union[Any]]] = None, ignore_keys : Optional[List[Union[Any]]] = None):
    """
    Reads a dictionary, possibly through deeper dictionaries, and converts any file paths that point to NumPy arrays or PyTorch Tensors into their original data.

    Args:
        input_dict (dict):
            The dictionary to parse.
        logging (int):
            Logging level. 0 = no logging, 1 = debug, 2 = trace.
        path (Optional[List[str]]):
            Path of keys to follow. Once path is exhausted, it then follows usual behaviour. If None, reads the entire dictionary.
        recursive (Optional[bool]):
            If True, reads the directory recursively. If False, only gets top level information. If path is not None, this is ignored until the path is exhausted.
        end_keys (Optional[List[Union[Any]]]):
            End keys to save. If None, reads all end keys. If path is not None, this is ignored until the path is exhausted.
        ignore_keys (Optional[List[Union[Any]]]):
            Keys to ignore. If None, reads all keys. If path is not None, this is ignored until the path is exhausted.
    """

    if logging > 0:
        logger.debug(f"Starting _read_data_dict: {input_dict.keys()=}, {logging=}, {recursive=}, {path=}, {end_keys=}, {ignore_keys=}")
    else:
        logger.trace(f"Starting _read_data_dict: {input_dict.keys()=}, {logging=}, {recursive=}, {path=}, {end_keys=}, {ignore_keys=}")
    
    # Follow path if given
    if isinstance(path, list):
        assert len(path) > 0, f"_read_data_dict: If given, optional parameter {path=} must be a list of length > 0."
        d = input_dict
        for key in path:
            if key in d:
                d = d[key]
            else:
                raise KeyError(f"_read_data_dict: {key=} in optional parameter {path=} not found in dictionary.")
        return _read_data_dict(d, logging=logging, recursive=recursive, end_keys=end_keys, ignore_keys=ignore_keys)
    
    new_dict = my_dict()
    for key, val in input_dict.items():
        if ignore_keys is not None and key in ignore_keys:
            logger.trace(f"_read_data_dict: Skipping {key=}, {val=}, {ignore_keys=}")
            continue
        elif isinstance(val, (dict, my_dict)):
            if recursive:
                new_dict[key] = _read_data_dict(val, logging=0 if logging < 2 else logging, end_keys=end_keys, ignore_keys=ignore_keys)
            else:
                new_dict[key] = val
        elif end_keys is None or key in end_keys:
            if isinstance(val, list):
                new_ls = []
                for entry in val:
                    if isinstance(entry, str):
                        if entry[-4:] == ".npy":
                            new_ls.append(np.array(np.load(entry), dtype=float))
                        elif val[-3:] == ".pt":
                            new_ls.append(torch.load(entry))
                        else:
                            new_ls.append(entry)
                    else:
                        new_ls.append(entry)
                new_dict[key] = new_ls
            elif isinstance(val, str):
                if val[-4:] == ".npy":
                    new_dict[key] = np.array(np.load(val), dtype=float)
                elif val[-3:] == ".pt":
                    new_dict[key] = torch.load(val)
                else:
                    new_dict[key] = val
            else:
                new_dict[key] = val
    if logging > 0:
        logger.debug(f"Successful _read_data_dict: {new_dict.keys()=}")
    else:
        logger.trace(f"Successful _read_data_dict: {new_dict.keys()=}")
    return new_dict


def _read_result(output_path : str, name : str, logging : int = 1, path : Optional[List[str]] = None, recursive : bool = True, end_keys : Optional[List[Union[Any]]] = None, ignore_keys : Optional[List[Union[Any]]] = None):
    """
    Reads the result from a file.

    Args:.
        output_path (str):
            Path to the directory to save the file.
        name (str):
            Name for the result file. If None, uses the date and time.
        path (Optional[List[str]]):
            Path of keys to follow. Once path is exhausted, it then follows usual behaviour. If None, reads the entire dictionary.
        recursive (bool):
            If True, reads the directory recursively. If False, only gets top level information. If path is not None, this is ignored until the path is exhausted.
        end_keys (Optional[List[Union[Any]]]):
            End keys to save. If None, reads all keys. If path is not None, this is ignored until the path is exhausted.
        ignore_keys (Optional[List[Union[Any]]]):
            Keys to ignore. If None, reads all keys. If path is not None, this is ignored until the path is exhausted.
    """

    logger.debug(f"Initiating _read_result: {output_path=}, {name=}.")
    if os.path.exists(f"{output_path}/{name}"):
        with open(f"{output_path}/{name}", "r") as file:
            try:
                output = load(file)
            except:
                raise RuntimeError(f"Attempted to read file {output_path}/{name}, but it is not parseable as a .json file.")
            try:
                output = _read_data_dict(output, logging=logging, path=path, recursive=recursive, end_keys=end_keys, ignore_keys=ignore_keys)
            except:
                raise RuntimeError(f"Attempted to read file {output_path}/{name}, but it is not parseable as a data dictionary.")
            logger.debug(f"Successful _read_result")
            return output
    else:
        return RuntimeError(f"Attempted to read file {output_path}/{name}, but it didn't exist")
