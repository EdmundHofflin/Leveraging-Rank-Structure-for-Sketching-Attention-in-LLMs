from loguru import logger

from typing import  Optional
from jsonargparse.typing import restricted_number_type, Path_dw
from jsonargparse import ArgumentParser, Namespace, ActionConfigFile, set_dumper
from json import dumps

from utils.inout_utils import _JSON_DUMPER_DICT, _my_encoder, _save_result, _read_result

from modules.default_initialiser import DefaultInitialiser


# ======
# Module
# ======

class InOutManagement(DefaultInitialiser):
    @staticmethod
    def get_config_parser():
        parser = ArgumentParser()

        # InOutManagement
        # ---------------
        config_group = parser.add_argument_group('General Configuration', description="General configuration and management.")
        # Config File
        config_group.add_argument(
            "--config_file",
            action=ActionConfigFile)
        # Save Config
        config_group.add_argument(
            "--save_config",
            type=bool, default=True,
            help="Whether to save the final config.")
        # Data Directory
        config_group.add_argument(
            "--data_path",
            type=Path_dw, default="data",
            help="Path to the data directory.")
        # Output Directory
        config_group.add_argument(
            "--output_path",
            type=Path_dw, default="out",
            help="Path to the output directory.")
        # Output Name
        config_group.add_argument(
            "--name",
            type=str, required=True,
            help="Name to identify the test in the output directory.")
        # Verbosity
        config_group.add_argument(
            "--verbosity",
            type=restricted_number_type("[0,1,2]", int, [(">=", 0), ("<=", 2)]), default=0,
            help="Printing verbosity: 0=silent, 1=summary, 2=warnings and debugging.")
        # Force Download
        config_group.add_argument(
            "--force_download",
            type=bool, default=False,
            help="Force (re)download of data and models.")
        # Trust Remote Code
        config_group.add_argument(
            "--trust_remote_code",
            type=bool, default=True,
            help="Whether to trust remote code (used when downloading 3rd party data and models from HuggingFace).")
        return parser
    

    def __init__(self, cfg : Optional[Namespace] = None):
        # Super to initialise self.cfg and get defaults
        super().__init__(cfg=cfg)

        # Parse Config Namespace
        # ----------------------
        # No setup required

        # dumper
        def _my_dumper(x, *kwargs):
            return dumps(x, cls=_my_encoder, *kwargs, **_JSON_DUMPER_DICT)
        set_dumper("_my_dumper", _my_dumper) # Save using custom JSONEncoder
        logger.debug("Setup custom JSONencoder and dumper.")


    def __str__(self):
        return f"System({self.cfg.as_dict()})"
    

    def save_result(self, output_dict : dict, file_name : str, data_subdir : Optional[str] = None):
        """
        Saves a dictionary to the output directory with a given title.
        
        Args:
            output_dict (dict):
                Dictionary to save as JSON file.
            file_name (str):
                The name of the file.
            data_subdir : (Optional[str], default: None):
                The subdirectory within the output directory to save the data. If None, the dictionary isn't processed and large arrays not saved separately.
        """

        _save_result(output_dict=output_dict, output_path=f"{self.cfg.output_path}/{self.cfg.name}", file_name=file_name, data_subdir=data_subdir)
