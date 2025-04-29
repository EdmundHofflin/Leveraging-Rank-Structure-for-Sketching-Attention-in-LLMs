from typing import Union, Optional, Callable
from jsonargparse.typing import PositiveInt, ClosedUnitInterval

from jsonargparse import ArgumentParser, Namespace

import numpy as np
import torch

from modules.default_initialiser import DefaultInitialiser
from utils.config_utils import DictExtend, HelpPasser
from utils.functional_utils import _np_sketching, _torch_sketching


# =======
# Modules
# =======

class Sketcher(DefaultInitialiser):
    @staticmethod
    def get_config_parser():
        parser = ArgumentParser()

        # Sketching Management
        # --------------------
        management_group = parser.add_argument_group("Sketching Management", description="Management of Matrix Sketching Object.")
        # Type
        management_group.add_argument(
            "--type",
            type=Optional[str], choices=["NumPy", "PyTorch"], default="PyTorch", required=True,
            help="Datatype on which the object functions.")
        
        # Sketching Operation
        # -------------------
        sketcher_group = parser.add_argument_group("Sketching Operation", description="Operation and Function of Matrix Sketching.")
        # Sketching Variant
        sketcher_group.add_argument(
            "--variant",
            type=str, choices=['standard', 'importance', 'random', 'max'], default="standard", required=True,
            help="Variant of matrix sketching used.")
        # Sample Proportion
        sketcher_group.add_argument(
            "--sample_proportion",
            type=ClosedUnitInterval, default=0.5,
            help="Number of samples used for sketching in proportion to the number of rows/cols.")
        # Replacement
        sketcher_group.add_argument(
            "--replacement",
            type=Optional[bool], default=None,
            help="Whether to sample with or without replacement for matrix sketching. If not set or None, replacement is only used for importance variant.")
        # Normalise
        sketcher_group.add_argument(
            "--normalise",
            type=Optional[bool], default=None,
            help="Whether to normalise the rows/columns by their probability. If not set or None, only normalises for the importance variant.")
        # Leverage Score Selection
        sketcher_group.add_argument(
            "--leverage_equation",
            type=Optional[str], choices=["L", "R", "LR", "L+R"], default=None,
            help="How to compute the leverage scores based on the row and column norms of L (the left matrix) and R (the right matrix). If not set or None, defaults to 'LR' for importance and max, but remains None for 'random' (as it has no impact).")
        # Forced History
        sketcher_group.add_argument(
            "--distribution_sampling",
            type=Optional[Union[ClosedUnitInterval, str]], default=None,
            help="Either a float in [0,1] or a str representing a function of the form 'lambda idx, shared_dimension: eval(distribution_sampling)', where idx is the index of the row/col (0 is oldest) and shared_dimension is the dimension shared between A and B. Both define the proportion of the most recent rows/cols to always sample, regardless of the sketching variant and leverage scores. The former type does so as a proportion, while the latter does so as by returning a boolean for each row/col. If None, no rows/cols are forced to be sampled. Note that these forced samples are included in the total sample proportion and so can't be larger. An error is raised if this is not the case.")
        # Linear Combinations
        sketcher_group.add_argument(
            "--linear_combination",
            type=PositiveInt, default=1,
            help="How many rows/columns to sum together when forming the rank-1 matrices used for sketching.")
        return parser
    

    def __init__(self, cfg : Optional[Namespace] = None, prnt : Optional[Callable] = None):
        # Super to initialise self.cfg and get defaults
        super().__init__(cfg=cfg)

        # Parse Config Namespace
        # ----------------------
        # If standard variant, simplify namespace
        if self.cfg.variant == 'standard':
            self.cfg = Namespace(type=self.cfg.type, variant='standard', sample_proportion=None, replacement=None, normalise=None, leverage_equation=None, distribution_sampling=None, linear_combination=None)
        else:
            # Replacement
            if self.cfg.replacement is None:
                self.cfg.replacement = self.cfg.variant == 'importance'
            # Normalise
            if self.cfg.normalise is None:
                self.cfg.normalise = self.cfg.variant == 'importance'
            # Leverage Equation
            if self.cfg.variant == 'random':
                self.cfg.leverage_equation = None
            elif self.cfg.leverage_equation is None:
                self.cfg.leverage_equation = "LR"
            # Forced History
            if isinstance(self.cfg.distribution_sampling, str):
                try:
                    f = lambda idx, shared_dimension: eval(self.cfg.distribution_sampling)
                except:
                    raise ValueError(f"Parameter --distribution_sampling for Sketcher instance set to {self.cfg.distribution_sampling}. which is not a valid function of the form 'lambda idx, shared_dimension: eval(distribution_sampling)'.")
                try:
                    out = f(0, 1)
                except:
                    raise ValueError(f"Parameter --distribution_sampling for Sketcher instance set to {self.cfg.distribution_sampling}. which is not a valid function of the form 'lambda idx, shared_dimension: eval(distribution_sampling)'.")
                try:
                    for idx in range(512):
                        out = f(idx, 512)
                        if not (isinstance(out, bool) or isinstance(out, np.bool_) or isinstance(out, torch.bool)):
                            raise ValueError(f"Parameter --distribution_sampling for Sketcher instance set to {self.cfg.distribution_sampling}, which is a valid function of the form 'lambda idx, shared_dimension: eval(distribution_sampling)' but does not return the required boolean output e.g. if idx = {idx} and shared_dimension = 512, then the function returns {out}.")
                except Exception as e:
                    raise e

        # Initialise sketch function for calling
        if self.cfg.type == 'NumPy':
            self.sketch = self.np_sketching
        elif self.cfg.type == 'PyTorch':
            self.sketch = self.torch_sketching
        else:
            raise ValueError(f"This should be impossible to reach. Parameter --type for Sketcher instance set to {self.cfg.type} when only parsable options are 'NumPy' or 'PyTorch'.")


    def __call__(self, A : Union[np.ndarray, torch.Tensor], B : Union[np.ndarray, torch.Tensor], **kwargs):
        """
        Sketches the matrix multiplication AB using the parameters of the sketcher instance.

        Args:
            A (np.ndarray or torch.Tensor):
                Left matrix.
            B (np.ndarray or torch.Tensor):
                Right matrix.
        """
        return self.sketch(A, B, **kwargs)
    

    def __str__(self):
        return f"Sketcher({self.cfg.as_dict()})"


    def np_sketching(self, A : np.ndarray, B : np.ndarray, **kwargs):
        """
        Batch sketches two NumPy tensors using a given sketching variant. 

        Args:
            A (torch.Tensor):
                Left matrix.
            B (torch.Tensor):
                Right matrix.
        """

        return _np_sketching(A, B, variant=self.cfg.variant, sample_proportion=self.cfg.sample_proportion, replacement=self.cfg.replacement, normalise=self.cfg.normalise, leverage_equation=self.cfg.leverage_equation, distribution_sampling=self.cfg.distribution_sampling, linear_combination=self.cfg.linear_combination, **kwargs)


    def torch_sketching(self, A : torch.Tensor, B : torch.Tensor, **kwargs):
        """
        Batch sketches two PyTorch tensors using a given sketching variant. 

        Args:
            A (torch.Tensor):
                Left matrix.
            B (torch.Tensor):
                Right matrix.
        """

        return _torch_sketching(A, B, variant=self.cfg.variant, sample_proportion=self.cfg.sample_proportion, replacement=self.cfg.replacement, normalise=self.cfg.normalise, leverage_equation=self.cfg.leverage_equation, distribution_sampling=self.cfg.distribution_sampling, linear_combination=self.cfg.linear_combination, **kwargs)



class Sketcher_Producer(DefaultInitialiser):
    @staticmethod
    def get_config_parser():
        parser = ArgumentParser()
        
        # Management
        # ----------
        # management_group = parser.add_argument_group("Sketcher_Producer Management", description="Management of the Sketcher Instances")

        # Sketcher Producer Operation
        # ---------------------------
        sketcher_producer_group = parser.add_argument_group("Sketcher_Producer Operation", description="Operation and Function of the Producer of Matrix Sketcher Instances.")
        sketcher_producer_group.add_argument(
            "--sketchers",
            action=DictExtend, nargs='*', default=[],
            help="Parameters given as dictionaries for the sketchers to be produced.")
        sketcher_producer_group.add_argument(
            "--sketchers.help",
            action=HelpPasser, help_parser=Sketcher.get_config_parser(), help_intro="usage: --sketchers [SKETCHERS ...]\nexample: --sketchers \"{'variant':'standard'}\", \"{'variant':'importance', 'sample_proportion':0.5}\"\n\nSKETCHERS must be a dictionary using the following options:\n",
            help="Outlines the usage of the --sketchers parameter.")
        return parser
    

    def __init__(self, cfg: Namespace):
        # Super to initialise self.cfg and get defaults
        super().__init__(cfg=cfg)

        # Parse Config Namespace
        # ----------------------
        # sketchers: construct Sketchers without duplicates
        cfgs = list(map(lambda x: Sketcher(x).cfg.as_dict(), self.cfg.sketchers))
        self.cfg.sketchers = []
        [self.cfg.sketchers.append(x) for x in cfgs if x not in self.cfg.sketchers]
        self.sketchers = list(map(lambda x: Sketcher(x), self.cfg.sketchers))
    
    
    def __str__(self):
        return f"Sketching_Producer({self.cfg.as_dict()})"
    

    def add(self, sketcher : Optional[Sketcher] = None, sketcher_dict : Optional[dict] = None):
        """
        Adds a sketcher. A Sketcher instance can be added or a Sketcher generated through a dictionary of its parameters. If both a Sketcher and dictionary are passed, the dictionary is not used. Returns True if the sketcher is added, False if the Producer already contained the Sketcher.

        Args:
            sketcher (Optional[Sketcher], default: None):
                Sketcher instance to be added.
            sketcher_dict (Optiona[dict], default: None):
                Dictionary of parameters to generate the Sketcher to be added.
        """

        if sketcher is None and sketcher_dict is None:
            raise RuntimeError(f"Method add_sketcher of Sketcher_Producer requires at least one non-None argument, but received arguments Sketcher={sketcher} and sketcher_dicit={sketcher_dict}.")
        
        # Initialise config properly
        if not sketcher is None:
            cfg = sketcher.cfg
        elif not sketcher_dict is None:
            cfg = Sketcher(sketcher_dict).cfg
        else:
            raise RuntimeError(f"This should be impossible to reach! Method add_sketcher of Sketcher_Producer received arguments Sketcher={sketcher} and sketcher_dicit={sketcher_dict}.")
        # Action dependent upon if cfg already included
        if cfg in list(map(lambda x: x.cfg, self.sketchers)):
            return False
        else:
            self.cfg.sketchers.append(cfg.as_dict())
            self.sketchers.append(Sketcher(cfg))
            return True
    

    def match(self, sketcher : Optional[Sketcher] = None, sketcher_dict : Optional[dict] = None):
        """
        Returns all Sketchers whose configs match either with the given Sketcher or a (partial) dictionary of parameters. If both a Sketcher and dictionary are passed, the dictionary is not used.

        Args:
            sketcher (Optional[Sketcher], default: None):
                Sketcher instance to be matched.
            sketcher_dict (Optiona[dict], default: None):
                Dictionary of parameters to match.
        """

        output = []
        if sketcher is None and sketcher_dict is None:
            raise RuntimeError(f"Method match of Sketcher_Producer requires at least one non-None argument, but received arguments Sketcher={sketcher} and sketcher_dicit={sketcher_dict}.")
        if not sketcher is None:
            # Overwrite sketcher_dict
            sketcher_dict = sketcher.cfg.as_dict()
        # Check that all viable parameters in config match
        warnings = [key for key in sketcher_dict if not key in Sketcher({'variant':'importance'}).cfg.as_dict()]
        if warnings:
            raise RuntimeWarning(f"Method match of Sketcher_Producer was passed argument sketcher_dict={sketcher_dict} which contains keys={warnings} not used as parameters for Sketcher. They will be ignored.")
        for sketcher in self.sketchers:
            checks = list(map(lambda key: sketcher.cfg[key] == sketcher_dict[key] if key in sketcher.cfg else True, sketcher_dict))
            if sum(checks) == len(checks):
                output.append(sketcher)
        return output
    

    def remove(self, sketcher : Optional[Sketcher] = None, sketcher_dict : Optional[dict] = None):
        """
        Removes Sketchers based on their config parameters. If a Sketcher is passed, then all Sketchers with matching config are removed. If a (partial) dictionary of parameters is passed, then all Sketchers that agree with all parameters in the dictionary are removed, e.g. {'variant': 'importance'} will remove all Sketchers with the 'importance' variant. If both a Sketcher and dictionary are passed, the dictionary is not used. Returns the list of Sketchers that are removed.
        
        Args:
            sketcher (Optional[Sketcher], default: None):
                Sketcher instance to be removed.
            sketcher_dict (Optiona[dict], default: None):
                Dictionary of parameters to generate the Sketcher to be added.
        """
        
        if sketcher is None and sketcher_dict is None:
            raise RuntimeError(f"Method remove_sketcher of Sketcher_Producer requires at least one non-None argument, but received arguments Sketcher={sketcher} and sketcher_dicit={sketcher_dict}.")
        # Get all matches
        matches = self.match(sketcher=sketcher, sketcher_dict=sketcher_dict)
        for sketcher in matches:
            self.sketchers.remove(sketcher)
        return matches
