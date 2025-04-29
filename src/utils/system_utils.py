import os

from loguru import logger

from typing import Optional, Union, Tuple, List

import numpy as np
import torch

from utils.config_utils import InputParameterError



# ============
# Type Parsers
# ============

def _dtypeParser(input : Union[str, torch.dtype]):
    if isinstance(input, torch.dtype):
        return input
    else:
        if "bfloat16" in input:
            return torch.bfloat16
        elif "float16" in input:
            return torch.float16
        elif "float32" in input:
            return torch.float32
        elif "float64" in input:
            return torch.float64
        else:
            raise InputParameterError("This should be impossible to reach.")


def _deviceParser(input : Optional[Union[str, Tuple[str, Optional[List[int]]]]]):
    if input is None:
        return ("cuda", None) if torch.cuda.is_available() else ("cpu", None)
    elif isinstance(input, tuple):
        return input
    elif isinstance(input, str):
        if input == "cpu":
            return (input, None)
        elif "cuda" in input:
            if not torch.cuda.is_available():
                raise InputParameterError(f"Parameter --device set to str '{input}', but cuda is not available. Please install cuda and/or fix your environment.")
            if input == "cuda":
                return (input, None)
            elif input[:5] == "cuda:":
                device_ids = list(map(lambda c: int(c), input[5:].split(',')))
                if not (len(device_ids) > 0):
                    raise InputParameterError(f"Parameter --device set to str '{input}', which is an invalid option. Valid device options are 'cpu', 'cuda', 'cuda:X' (where X is a comma-separated list of valid GPU ids, e.g. 'cuda:1' or 'cuda:0,3'), and 'torchrun'. Note that valid device ids are [0, ..., {torch.cuda.device_count()-1}].")
                for device_id in device_ids:
                    if device_id < 0 or device_id >= torch.cuda.device_count():
                        raise InputParameterError(f"Parameter --device set to '{input}', which has invalid device id {device_id}. Valid device ids are [0, ..., {torch.cuda.device_count()-1}].")
                if len(device_ids) == 1:
                    return (input, None)
                else:
                    return (input, device_ids)
            else:
                raise InputParameterError(f"Parameter --device set to str '{input}', which is an invalid option. Valid device options are 'cpu', 'cuda', 'cuda:X' (where X is a GPU id, e.g. 'cuda:1'), and 'torchrun'. Note that valid device ids are [0, ..., {torch.cuda.device_count()-1}].")
        elif input == "torchrun":
            device_dict = {}
            try:
                device_dict["global_rank"] = int(os.environ['RANK'])
                device_dict["local_rank"] = int(os.environ["LOCAL_RANK"])
                device_dict["world_size"] = int(os.environ["LOCAL_WORLD_SIZE"])
                device_dict["local_device"] = f"cuda:{device_dict["local_rank"]}"
            except KeyError as e:
                raise InputParameterError(f"Parameter --device set to str '{input}', but unable to read local environment variables:\n{e}\nEnsure that the 'torchrun' command, rather than 'python', is being used to initiate the program. Consult https://pytorch.org/docs/stable/elastic/run.html for further details.")
            return (input, device_dict)
        else:
            raise InputParameterError(f"Parameter --device set to str '{input}', which is an invalid option. Valid device options are 'cpu', 'cuda', 'cuda:X' (where X is a GPU id, e.g. 'cuda:1'), and 'torchrun'. Note that valid device ids are [0, ..., {torch.cuda.device_count()-1}].")
    else:
        raise InputParameterError(f"Parameter --device set to str '{input}', which is an invalid option. Valid device options are 'cpu', 'cuda', 'cuda:X' (where X is a GPU id, e.g. 'cuda:1'), and 'torchrun'. Note that valid device ids are [0, ..., {torch.cuda.device_count()-1}].")



# ================
# Helper Functions
# ================

def _seed_everything(seed, device):
    logger.debug(f"Starting _seed_everything: {seed=}, {device=}.")
    np.random.seed(seed)
    np_rng = np.random.default_rng(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch_rng = torch.Generator(device=device[0] if device[1] is None else ("cuda:"+str(device[1][0]) if isinstance(device[1], list) else device[1]["local_device"])).manual_seed(int(seed))
    logger.debug(f"Successful _seed_everything.")
    return np_rng, torch_rng
