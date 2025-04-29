from typing import Optional, Union, List
from jsonargparse.typing import PositiveInt, NonNegativeInt, ClosedUnitInterval
from jsonargparse import ArgumentParser, Namespace

import numpy as np
import torch

from modules.default_initialiser import DefaultInitialiser
from models.model_config import MODEL_CONFIGS

from utils.model_utils import _load_model, _setup_model, _load_tokeniser, _compute_patch_idxs, _patch_model


# ======
# Module
# ======

class Model(DefaultInitialiser):
    @staticmethod
    def get_config_parser():
        parser = ArgumentParser()

        # General Model Arguments
        # -----------------------
        model_group = parser.add_argument_group('Model Options', description="Options that specify the model and its operation.")
        # Model
        model_group.add_argument(
           "--model",
            dest="model_name", type=str, choices=MODEL_CONFIGS.keys(), default='distilbert/distilgpt2',
            help="Huggingface model to evaluate.")
        # Context Length
        model_group.add_argument(
            "--context_length",
            type=Optional[PositiveInt], default=None,
            help="Context length used for evaluation. If None, defaults to the max context length of the model.")
        # Block Size
        # model_group.add_argument(
        #     "--block_size",
        #     type=PositiveInt, default=256,
        #     help="FIXME: What is this?")
        # Sample Size
        # model_group.add_argument(
        #     "--sample_size",
        #     type=PositiveInt, default=256,
        #     help="FIXME: What is this?")

        # Patching Arguments
        # ------------------
        patching_group = parser.add_argument_group('Patching Options', description="Options that specify the layers within the model to patch.")
        # Patch Config
        patching_group.add_argument(
            "--patch_pattern",
            type=Optional[str], choices=['first', 'last', 'even', 'odd', 'all'], default="all",
            help="Pattern used when patching the layers of the model with the attention variant.")
        # Patch Count
        patching_group.add_argument(
            "--patch_count",
            type=Optional[Union[PositiveInt, ClosedUnitInterval]], default=None,
            help="Number of layers to patch. Counting depends upin the patch_pattern used. If not set or None, all layers in the given pattern are patched.")
        # Patch Idxs
        patching_group.add_argument(
            "--patch_idxs",
            type=Optional[List[NonNegativeInt]], default=None,
            help="Indices of the layers to patch. Layers are indexed from 0. If specified, overrules patch_pattern and patchJcount parameters.")
        return parser


    def __init__(self, cfg: Optional[Namespace] = None):
        # Super to initialise self.cfg and get defaults
        super().__init__(cfg=cfg)

        # Parse Config Namespace
        # ----------------------
        # context_length: ensure compatible with MAX_CONTEXT_LENGTH of model
        if self.cfg.context_length is None:
            self.cfg.context_length = MODEL_CONFIGS[self.cfg.model_name]['MAX_CONTEXT_LENGTH']
        else:
            assert self.cfg.context_length <= MODEL_CONFIGS[self.cfg.model_name]['MAX_CONTEXT_LENGTH'], f"Parameter --context_len set to {self.cfg.context_length}, which exceeds the MAX_CONTEXT_LENGTH={MODEL_CONFIGS[self.cfg.model_name]['MAX_CONTEXT_LENGTH']} of model={self.cfg.model_name}."
        # patch_count: ensure compatible with NUM_LAYERS of model 
        if self.cfg.patch_count is None:
            self.cfg.patch_count = MODEL_CONFIGS[self.cfg.model_name]['NUM_LAYERS']
        else:
            assert self.cfg.patch_count <= MODEL_CONFIGS[self.cfg.model_name]['NUM_LAYERS'], f"Parameter --self.cfg.patch_count set to {self.cfg.patch_count}, which exceeds the NUM_LAYERS={MODEL_CONFIGS[self.cfg.model_name]['NUM_LAYERS']} of model={self.cfg.model_name}."
        # patch parameters: ensure enough info to determine layers to patch
        assert self.cfg.patch_idxs is not None or (self.cfg.patch_pattern is not None and self.cfg.patch_count is not None), f"Parameters --patch_pattern, --patch_count, and --patch_idxs are all None, when either --patch_pattern and --patch_count or --patch_idxs must be specified."

        
        # patch_idxs: compute using parameters
        self.cfg.patch_idxs = _compute_patch_idxs(self.cfg.model_name, patch_pattern=self.cfg.patch_pattern, patch_count=self.cfg.patch_count, patch_idxs=self.cfg.patch_idxs)
        
        # Set model to None
        self.model = None

        # Get model config
        self.model_config = MODEL_CONFIGS[self.cfg.model_name]

    
    def __call__(self, *args, **kwargs):
        """
        Calls the model.
        """
        return self.model(*args, **kwargs)
    

    def __str__(self):
        return f"Model({self.cfg.as_dict()})"
        
    
    def load_model(self, data_path : str, trust_remote_code : bool, force_download : bool, prnt):
        """ Loads Huggingface model either from local files or Huggingface.
    
        Args:
            data_path (str):
                String path to the data folder.
            trust_remote_code (bool):
                Trust remote code flag for loading model from Huggingface.
            force_download (bool):
                Force the data to be redownloaded.
            prnt (function, default: None):
                The verbosity_printing function set according to args.verbosity.
        """

        # Set model
        self.model = _load_model(model_name=self.cfg.model_name, data_path=data_path, trust_remote_code=trust_remote_code, force_download=force_download, prnt=prnt)
    

    def setup_model(self, device : str, dtype : torch.dtype):
        """
        Setups the model on the appropriate device and in eval model.
        
        Args:
            device (str):
                Compute device on which to evaluate the model.
            dtype (PyTorch.dtype):
                dtype used for storing all model values.
        """

        self.model = _setup_model(self.model, device=device, dtype=dtype)
    

    def load_tokeniser(self, data_path : str, trust_remote_code : bool, force_download : bool, prnt):
        r""" Loads Huggingface tokeniser either from local files or Huggingface.
    
        Args:
            data_path (str):
                String path to the data folder.
            trust_remote_code (bool):
                Trust remote code flag for loading model from Huggingface.
            force_download (bool):
                Force the data to be redownloaded.
            prnt (function):
                The verbosity_printing function set according to args.verbosity.
        """

        self.tokeniser = _load_tokeniser(model_name=self.cfg.model_name, data_path=data_path, trust_remote_code=trust_remote_code, force_download=force_download, prnt=prnt)
    

    def patch_model(self, sketcher, rng_generator : Optional[Union[np.random.Generator, torch.Generator]], sketching_info : bool, prnt):
        """
        Patches a given transformer model provided a Sketcher instance and rng generator. Currently only supports patching the attention module by changing the AV multiplication.

        Args:
            sketcher (Union[Sketcher, Callable]):
                An instance of a Sketcher class or a Callable. If a Callable, must take as input two batched matrices and a rng_generator and return the batched matrices product.
            rng_generator (Optional[Union[np.random.Generator, torch.Generator]])
                Either a NumPy or PyTorch gnerator for random number sampling.
            sketching_info (bool):
                Whether to return diagnostic information about the sketch.
            prnt (Optional[function]):
                The verbosity_printing function set according to verbosity.
        """
        
        # Ensure model is initialised
        assert self.model is not None, f"Model {self.cfg.model_name} is not setup yet. Run model.load_model(data_path, trust_remote_code, force_download, prnt) to initialise model."
        # Patch model
        _patch_model(model=self.model, model_name=self.cfg.model_name, patch_idxs=self.cfg.patch_idxs, sketcher=sketcher, rng_generator=rng_generator, sketching_info=sketching_info, prnt=prnt)
    

    def clear_model(self):
        """
        Deletes the model to free data and ensure no patches are kept.
        """

        del self.model
        self.model = None
