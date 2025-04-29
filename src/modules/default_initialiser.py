from abc import abstractmethod

from typing import Optional

from jsonargparse import Namespace

# Based On: https://medium.com/omnius/compositional-configuration-in-python-with-jsonargparse-b60b12738342
class DefaultInitialiser:
    @staticmethod
    @abstractmethod
    def get_config_parser():
        pass

    def __init__(self, cfg : Optional[Namespace] = None):
        # If no config provided, get defaults
        if cfg is None:
            self.cfg = (self.__class__.get_config_parser().get_defaults())
        else:
            if isinstance(cfg, dict):
                self.cfg = self.__class__.get_config_parser().parse_object(cfg)
            else:
                self.cfg = cfg