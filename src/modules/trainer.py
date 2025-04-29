import sys

from typing import Union, Optional, List, Tuple

import torch.distributed
from jsonargparse.typing import PositiveFloat

from jsonargparse import ArgumentParser, Namespace

import numpy as np
import torch as torch

from modules.default_initialiser import DefaultInitialiser

from utils.config_utils import InputParameterError

# ==============
# Module Helpers
# ==============

split_default = (0.8, 0.2)

def splitType(input: Optional[Tuple[PositiveFloat, PositiveFloat]]):
    if input is None:
        return split_default
    elif isinstance(input, Tuple) and len(input) == 2 and isinstance(input[0], PositiveFloat) and isinstance(input[1], PositiveFloat):
        return (input[0]/sum(input), input[1]/sum(input))
    else:
        raise InputParameterError(f"Parameter --split for the Trainer module is invalid: Valid parameters are 2-tuples of positive floats, e.g. (0.5, 0.5), or None, which defaults to (0.8, 0.2). Use --help for more information.")




# =======
# Modules
# =======

class Trainer(DefaultInitialiser):
    @staticmethod
    def get_config_parser():
        parser = ArgumentParser()

        # Trainer Management
        # ------------------
        trainer_group = parser.add_argument_group("Trainer Management", description="Management of the trainer processes.")
        # Training
        trainer_group.add_argument(
            "--training",
            type=bool,
            default=False,
            help="Train the model.")
        # Evaluation
        trainer_group.add_argument(
            "--evaluation",
            type=bool,
            default=True,
            help="Evaluate the model.")
        # Split
        trainer_group.add_argument(
            "--data_split",
            type=splitType,
            default=None,
            help=f"Split between training and evaluation dataset. If the sum is greater than 1.0, then the values are normalised. If unspecified or set to None, defaults to {split_default}.")
        
        # 

    def __init__(self, cfg: Optional[Namespace] = None):
        # Super to initialise self.cfg and get defaults
        super().__init__(cfg=cfg)

        # Parse Config Namespace
        # ----------------------