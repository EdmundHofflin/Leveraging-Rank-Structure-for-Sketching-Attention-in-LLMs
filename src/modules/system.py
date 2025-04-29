from loguru import logger

from types import MethodType
from typing import Optional
from jsonargparse.typing import PositiveInt
from jsonargparse import ArgumentParser, Namespace

import numpy as np
import torch

from utils.system_utils import _dtypeParser, _deviceParser, _seed_everything
from modules.default_initialiser import DefaultInitialiser



# ======
# Module
# ======

class System(DefaultInitialiser):
    @staticmethod
    def get_config_parser():
        parser = ArgumentParser()

        # System Management
        # -----------------
        system_group = parser.add_argument_group("System Management", description="Management of the system processes.")
        # seed
        system_group.add_argument(
            "--seed", 
            type=Optional[PositiveInt], default=None,
            help="Seed to initialise random generators. If not set or None, a random seed is generated.")
        # dtype
        system_group.add_argument(
            "--dtype",
            type=_dtypeParser, choices=[torch.bfloat16, torch.float16, torch.float32, torch.float64],
            default=torch.float32,
            help="dtype to use for Torch. Note: Some models and methods are incompatible with certain types.")
        # device
        system_group.add_argument(
            "--device",
            type=_deviceParser, default=None,
            help="Device on which to load and run the model. If unspecified or None, cuda is used if available.")
        return parser
    

    def __init__(self, cfg : Optional[Namespace] = None):
        # Super to initialise self.cfg and get defaults
        super().__init__(cfg=cfg)

        # Parse Config Namespace
        # ----------------------
        # seed
        if self.cfg.seed == None:
            self.cfg.seed = np.random.default_rng().integers(1e9)
        # dtype
        torch.set_default_dtype(self.cfg.dtype)
        # device
        if isinstance(self.cfg.device[1], dict):
            # Set default devices
            torch.cuda.device(self.cfg.device[1]["local_rank"])
            torch.set_default_device(self.cfg.device[1]["local_device"])
            # Setup distributed system
            torch.distributed.init_process_group(backend="nccl", rank=self.cfg.device[1]["global_rank"], world_size=self.cfg.device[1]["world_size"])
            logger.info(f"Distributed system initiated: local_rank={self.cfg.device[1]["local_rank"]}")
        elif isinstance(self.cfg.device[1], list):
            torch.cuda.device('cuda:'+str(self.cfg.device[1][0]))
            torch.set_default_device('cuda:'+str(self.cfg.device[1][0]))
            logger.info(f"Parallel system setup: primary_device=cuda:{self.cfg.device[1][0]}, device_ids={self.cfg.device[1]}.")
        elif self.cfg.device[1] is None:
            if self.cfg.device[0] == "cpu":
                pass
            else:
                torch.cuda.device(self.cfg.device[0])
            torch.set_default_device(self.cfg.device[0])
            logger.info(f"Single device system setup: device={self.cfg.device[0]}.")

        # Construct random generators
        self.np_rng, self.torch_rng = _seed_everything(self.cfg.seed, self.cfg.device)
        # Master method
        if not isinstance(self.cfg.device[1], dict) or self.cfg.device[1]["global_rank"] == 0:
            self.master = MethodType(lambda x: True, self)
        else:
            self.master = MethodType(lambda x: False, self)
        logger.debug(f"Setup system.master method according to device.")


    def __str__(self):
        return f"System({self.cfg.as_dict()})"
    
    
    def master(self):
        raise NotImplementedError(f"This method is formed only at the instance level.")
    

    # def distributed(self):
    #     # FIXME: Log this RuntimeError(f"Method 'distributed' of System module is only initialised if multi-gpu setup is required. However, the current device setup is {self.cfg.device}.")
    #     return None
