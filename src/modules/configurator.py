import os

from loguru import logger

from typing import Optional
from jsonargparse import ArgumentParser, Namespace, ActionParser

import numpy as np

from modules.default_initialiser import DefaultInitialiser
from modules.system import System
from modules.inout import InOutManagement
from modules.dataset import Dataset
from modules.model import Model
from modules.sketcher import Sketcher_Producer
from modules.evaluator import Evaluator

from utils.general_utils import verbosity_print


# =======
# Modules
# =======

class Configurator(DefaultInitialiser):
    @staticmethod
    def get_config_parser():
        parser = ArgumentParser()

        # Module Manamgement
        # ------------------
        module_group = parser.add_argument_group("Configuration Components", description="Configuration of modules.")
        # InOutManagement
        module_group.add_argument(
            "--inout",
            action=ActionParser(parser=InOutManagement.get_config_parser()),
            help="Module specifying configuration files and in/out management.")
        # System
        module_group.add_argument(
            "--system",
            action=ActionParser(parser=System.get_config_parser()),
            help="Module specifying system processes.")
        # Dataset
        module_group.add_argument(
            "--dataset",
            action=ActionParser(parser=Dataset.get_config_parser()),
            help="Module specifying the dataset and its operation.")
        # Model
        module_group.add_argument(
            "--model",
            action=ActionParser(parser=Model.get_config_parser()),
            help="Module specifying the model and its operation.")
        # Sketchers
        module_group.add_argument(
            "--sketching",
            action=ActionParser(parser=Sketcher_Producer.get_config_parser()),
            help="Module specifying sketching methods used.")
        # Evaluator
        module_group.add_argument(
            "--eval",
            action=ActionParser(parser=Evaluator.get_config_parser()),
            help="Module specifying evaluation and its operation.")
        return parser
    

    def __init__(self, cfg : Optional[Namespace] = None, temp_log : str = 'temp.log'):
        # Super to initialise self.cfg and get defaults
        super().__init__(cfg=cfg)

        # Parse Config Namespace
        # ----------------------
        self.system = System(self.cfg.system)
        self.inout = InOutManagement(self.cfg.inout)
        self.dataset = Dataset(self.cfg.dataset)
        self.model = Model(self.cfg.model)
        self.sketching = Sketcher_Producer(self.cfg.sketching)
        self.eval = Evaluator(self.cfg.eval)

        # Setup directories
        if self.system.master():
            # Out directory
            if os.path.exists(f"{self.inout.cfg.output_path}/{self.inout.cfg.name}"):
                logger.info(f"{self.inout.cfg.output_path}/{self.inout.cfg.name} folder already exists. Results will overwrite existing data.\n")
            else:
                os.mkdir(f"{self.inout.cfg.output_path}/{self.inout.cfg.name}")
            # Logs Directory
            if not os.path.exists(f"{self.inout.cfg.output_path}/{self.inout.cfg.name}/logs/"):
                os.mkdir(f"{self.inout.cfg.output_path}/{self.inout.cfg.name}/logs/")
        logger.debug(f"Setup {self.inout.cfg.output_path}/{self.inout.cfg.name} directory tree")  

        # Setup logger
        old_logs = "Pre-Configuration Logs:"
        reg = r"(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2}) (?P<time>[0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]{1,3}) \| (?P<level>[A-Za-z]+\s\s+)\| (?P<log>.+)"
        for log in logger.parse(temp_log, reg):
            old_logs += f"\n{str(log)}"
        logger.remove()
        os.remove(temp_log)
        if self.system.master():
            logger.add(open(f"{self.inout.cfg.output_path}/{self.inout.cfg.name}/logs/TRACE.log", 'w'), level=5)
            logger.add(open(f"{self.inout.cfg.output_path}/{self.inout.cfg.name}/logs/DEBUG.log", 'w'), level=10)
            logger.add(open(f"{self.inout.cfg.output_path}/{self.inout.cfg.name}/logs/INFO.log", 'w'), level=20)
            logger.add(open(f"{self.inout.cfg.output_path}/{self.inout.cfg.name}/logs/SUCCESS.log", 'w'), level=25)
            logger.add(open(f"{self.inout.cfg.output_path}/{self.inout.cfg.name}/logs/WARNING.log", 'w'), level=30)
            logger.add(open(f"{self.inout.cfg.output_path}/{self.inout.cfg.name}/logs/ERROR.log", 'w'), level=40)
            logger.add(open(f"{self.inout.cfg.output_path}/{self.inout.cfg.name}/logs/CRITICAL.log", 'w'), level=50)
        logger.info("Setup Logger")
        logger.debug(old_logs)

        # Setup print
        logger.debug("Setting up prnt.")
        def prnt(string, required_verbosity, **kwargs):
            if self.system.master():
                verbosity_print(string=string, required_verbosity=required_verbosity, set_verbosity=self.inout.cfg.verbosity, **kwargs)
        self.prnt = prnt
        logger.debug("Setup prnt.")

        # Saving Namespace
        if self.inout.cfg.save_config and self.system.master():
            logger.debug(f"save_config={self.inout.cfg.save_config}: Attempting to save config at {self.inout.cfg.output_path}/{self.inout.cfg.name}/config.json.")
            self.get_config_parser().save(cfg=self.cfg, path=f"{self.inout.cfg.output_path}/{self.inout.cfg.name}/config.json", format='_my_dumper', skip_none=True, skip_check=False, overwrite=True)
            logger.debug(f"save_config={self.inout.cfg.save_config}: Successfully saved config at {self.inout.cfg.output_path}/{self.inout.cfg.name}/config.json.")
        else:
            logger.debug(f"save_config={self.inout.cfg.save_config}: No config saving necessary.")
        
        # Printing Namespace
        if self.system.master():
            logger.debug("Setting up configuration printing.")
            namespace_str = ""
            namespace_str += "Configuration:\n==============\n"
            cutoff = 20
            for cfg_name, cfg_var in cfg.__dict__.items():
                for cfg_cfg_name, cfg_cfg_var in cfg_var.__dict__.items():
                    name = f"{cfg_name}.{cfg_cfg_name}"
                    if name == "sketching.sketchers" and isinstance(cfg_cfg_var, list):
                        namespace_str += f"{name:<{cutoff}} : [\n"
                        padding = " " * (cutoff+4)
                        for sketcher in cfg_cfg_var:
                            namespace_str += f"{padding}({sketcher}, {type(sketcher)}),\n"
                        padding = " " * (cutoff+3)
                        namespace_str += f"{padding}]\n"
                    else: 
                        namespace_str += f"{name:<{cutoff}} : {cfg_cfg_var}, {type(cfg_cfg_var)}\n"
            logger.info(namespace_str)
            self.prnt(namespace_str, 1)
        
        # Conclude
        logger.info("Configuration Complete")


    def __str__(self):
        return f"Configurator({self.cfg.as_dict()})"
