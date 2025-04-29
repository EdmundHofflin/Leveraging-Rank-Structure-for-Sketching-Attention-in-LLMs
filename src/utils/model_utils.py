import os

from loguru import logger

from typing import Union, Optional, List
from jsonargparse.typing import PositiveInt, ClosedUnitInterval

import numpy as np
import torch as torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from modules.sketcher import Sketcher
from models.model_config import _model_manager



# ===================
# Data Parallel Class
# ===================

class AccessDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            getattr(self.module, name)



class AccessDistributed(torch.nn.parallel.DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            # print(f"Module Dict: {self.module.__dict__.keys()}")
            getattr(self.module, name)



# =========================
# Model and Tokeniser Setup
# =========================

def _load_model(model_name : str, data_path : str, trust_remote_code : bool, force_download : bool, prnt):
    """
    Loads Huggingface model either from data/models/{model_name} or Huggingface.
    
    Args:
        model_name (str):
            String defining the model.
        data_path (str):
            String path to the data folder.
        trust_remote_code (bool):
            Trust remote code flag for loading model from Huggingface.
        force_download (bool):
            Force the data to be redownloaded.
        prnt (Optional[function]):
            The verbosity_printing function set according to verbosity.
    """

    logger.debug(f"Starting _load_model: {model_name=}, {data_path=}, {trust_remote_code=}, {force_download=}.")
    # If downloading or file doesn't exist
    if force_download or (not os.path.exists(f"{data_path}/models/{model_name}.hf")):
        if not force_download:
            logger.debug(f"Model {model_name} is not local, so must download.", 2, end=" ")
        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        logger.debug(f"Model {model_name} downloaded.")
        # Save model
        model.save_pretrained(f"{data_path}/models/{model_name}.hf")
        logger.debug(f"Model {model_name} saved to {data_path}/models/{model_name}.hf.")
    # Otherwise load from file
    else:
        # Load model
        model = AutoModelForCausalLM.from_pretrained(f"{data_path}/models/{model_name}.hf", local_files_only=True, trust_remote_code=trust_remote_code)
        logger.debug(f"Model loaded locally from {data_path}/models/{model_name}.hf.")
    # Return
    return model


def _setup_model(model, device : Union[str, dict], dtype : torch.dtype):
    """
    Setups the model on the appropriate device and in eval model.
    
    Args:
        model:
            PyTorch model.
        device (Union[str, dict]):
            Compute device on which to evaluate the model.
        dtype (PyTorch.dtype):
            dtype used for storing all model values.
    """

    logger.debug(f"Starting _setup_model: {model=}, {device=}, {dtype=}.")
    if isinstance(device[1], dict) and torch.cuda.device_count() > 1:
        logger.debug(f"Setting up distributed model.")
        model = AccessDistributed(model, device_ids=[device["local_rank"]], output_device=device["local_rank"], broadcast_buffers=False)
    elif isinstance(device[1], list) and torch.cuda.device_count() > 1:
        logger.debug(f"Setting up parallel model.")
        model = AccessDataParallel(model, device_ids=device[1])
    else:
        logger.debug(f"Setting up standard model.")
        model.to(device=device[0] if device[1] is None else ("cuda:"+str(device[1][0]) if isinstance(device[1], list) else device[1]["local_device"]), dtype=dtype)
    model.eval()
    logger.debug(f"Successful _setup_model.")
    return model


def _load_tokeniser(model_name : str, data_path : str, trust_remote_code : bool, force_download : bool, prnt):
    r""" Loads Huggingface tokeniser either from local files or Huggingface.
    
    Args:
        model_name (str):
            String defining the model.
        data_path (str):
            String path to the data folder.
        trust_remote_code (bool):
            Trust remote code flag for loading model from Huggingface.
        force_download (bool):
            Force the data to be redownloaded.
        prnt (function):
            The verbosity_printing function set according to args.verbosity.
    """

    logger.debug(f"Starting _load_tokeniser: {model_name=}, {data_path=}, {trust_remote_code=}, {force_download=}")
    # If downloading or file doesn't exist
    if force_download or (not os.path.exists(f"{data_path}/tokenisers/{model_name}.hf")) or model_name == "THUDM/chatglm2-6b-32k":
        # Print update for downloading
        if not force_download:
            logger.debug(f"Tokeniser for {model_name} not local, so must download.")
        elif model_name == "THUDM/chatglm2-6b-32k":
            logger.debug(f"Model {model_name} doesn't support loading tokeniser from local files with latest packages, so must download.")
        # Load tokeniser
        if model_name == 'mistralai/Mistral-7B-v0.3':
            tokeniser = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code, use_fast=False)
        else:
            tokeniser = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        logger.debug(f"Tokeniser for {model_name} downloaded.")
        # Save tokeniser
        tokeniser.save_pretrained(f"{data_path}/tokenisers/{model_name}.hf")
        logger.debug(f"Tokeniser for {model_name} saved to {data_path}/tokenisers/{model_name}.hf.")
    # Otherwise load from file
    else:
        # Load tokeniser
        tokeniser = AutoTokenizer.from_pretrained(f"{data_path}/tokenisers/{model_name}.hf", local_files_only=True, trust_remote_code=trust_remote_code)
        logger.debug(f"Tokeniser for {model_name} loaded locally from {data_path}/tokenisers/{model_name}.hf.")
    # Return
    logger.debug("Successful _load_tokeniser.")
    prnt(f"Successfully loaded {model_name} tokeniser.", 2, end="\n")
    return tokeniser



# ========
# Patching
# ========

def _compute_patch_idxs(model_name : str, patch_pattern : Optional[str], patch_count : Optional[Union[PositiveInt, ClosedUnitInterval]], patch_idxs : Optional[List[PositiveInt]]):
    """
    Computes the indices that will be patched for a given model, provided the patching information.

    Args:
        model_name (str):
            The name of the model, used to determine how to patch the model.
        patch_pattern (Optional[str]):
            The layers within the model to patch.
        patch_count (Optional[Union[PositiveInt, ClosedUnitInterval]]):
            The number of layers to patch.
        patch_idxs (Optional[List[PositiveInt]]):
            Layers to patch. Layers are indexed from 1. If specified, overrules patch_pattern and patch_count.
    """

    logger.debug(f"Starting _comptue_patch_idxs: {model_name=}, {patch_pattern=}, {patch_count=}, {patch_idxs=}.")
    # Get model dict
    model_dict = _model_manager(model_name)
    # Get model num_layers
    num_layers = model_dict['NUM_LAYERS']

    # Get indices of layers to patch
    if patch_idxs is not None:
        pass
    elif patch_pattern is not None and patch_count is not None:
        if patch_pattern == 'all':
            patch_idxs = range(0, num_layers, 1)
        elif patch_pattern == 'first':
            patch_idxs = range(0, patch_count, 1)
        elif patch_pattern == 'last':
            patch_idxs = range(num_layers-1, num_layers-patch_count-1, -1)
        elif patch_pattern == 'even':
            patch_idxs = range(0, num_layers, 2)
        elif patch_pattern == 'odd':
            patch_idxs = range(1, num_layers, 2)
        else:
            raise RuntimeError(f"This should be impossible to reach! Parameter --patch_pattern is set to invalid value {patch_pattern}. Options are 'all', 'first', 'lst', 'even', and 'odd'.")
    else:
        raise RuntimeError(f"This should be impossible to reach! Parameters --patch_pattern, --patch_count, and --patch_idxs are all None, when either --patch_pattern and --patch_count or --patch_idxs must be specified.")
    logger.debug(f"Successful _compute patch_idxs.")
    return patch_idxs


def _patch_model(model, model_name : str, patch_idxs : Optional[List[PositiveInt]], sketcher : Sketcher, rng_generator : Optional[Union[np.random.Generator, torch.Generator]], sketching_info : bool, prnt):
    """
    Patches a given transformer model. Currently only supports patching the attention module by changing the AV multiplication.

    Args:
        model:
            Callable model to evaluate. Internal attention layers must be accessible.
        model_name (str):
            The name of the model, used to determine how to patch the model.
        patch_idxs (Optional[List[PositiveInt]]):
            Layers to patch. Layers are indexed from 1. If specified, overrules patch_pattern and patch_count.
        sketcher (Union[Sketcher, Callable]):
            An instance of a Sketcher class or a Callable. If a Callable, must take as input two batched matrices and a rng_generator and return the batched matrices product.
        rng_generator (Optional[Union[np.random.Generator, torch.Generator]]):
            Either a NumPy or PyTorch gnerator for random number sampling.
        sketching_info (bool):
            Whether to return diagnostic information about the sketch.
        prnt (Optional[function]):
            The verbosity_printing function set according to verbosity.
    """
    
    logger.debug(f"Starting _patch_model: {model=}, {model_name=}, {patch_idxs=}, {sketcher=}, {rng_generator=}")
    # Get model dict for patching function
    model_dict = _model_manager(model_name)
    # Isolate sketching function
    def sketch(A, B):
        return sketcher(A, B, rng_generator=rng_generator, sketching_info=sketching_info)
    # Patch attention
    model_dict['patch_attention'](model=model, sketch=sketch, patch_idxs=patch_idxs)
    logger.debug(f"Successful _patch_model.")
    prnt(f"Successfully patched attention: patch_idxs={list(patch_idxs)}, sketcher={sketcher}.", 2, end="\n")
