from typing import Union, Optional, List, Tuple
from jsonargparse.typing import PositiveInt

from jsonargparse import ArgumentParser, Namespace

import numpy as np
import torch as torch

from modules.default_initialiser import DefaultInitialiser

from utils.eval_utils import _evaluate_model_on_data_over_batch_sizes, _process_internal_metrics, _collect_internal_metrics


# =======
# Modules
# =======

class Evaluator(DefaultInitialiser):
    @staticmethod
    def get_config_parser():
        parser = ArgumentParser()
        
        # Evaluation Arguments
        # --------------------
        eval_group = parser.add_argument_group('Evaluation Options', description="How to run and evaluate the models.")
        # Metric
        eval_group.add_argument(
            "--metrics",
            type=str, choices=['cross_entropy_loss', 'perplexity'], default=['perplexity'], nargs='+',
            help="Metrics used to evalaute the model.")
        # Batch Size
        eval_group.add_argument(
            "--batch_size",
            type=PositiveInt, nargs="+", default=[1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
            help="Batch sized used during evaluation. If multiple batch sizes are given, evaluation is attempted on each value in turn.")
        # Runs
        eval_group.add_argument(
            "--runs",
            type=PositiveInt, default=2,
            help="Number of repeat runs to average over.")

        # Timing and Profiler Arguments
        # -----------------------------
        timing_group = parser.add_argument_group('Timing and Profiling Options', description="How to time and profile the models.")
        # Time Metric
        timing_group.add_argument(
            "--time_metrics",
            type=str, choices=["total", "flops", "gpu_time", "cpu_time"], default=["total"], nargs='*',
            help="Timing metrics to report.")
        # Profiler Dump
        timing_group.add_argument(
            "--profiler_dump",
            type=Optional[dict], default=None,
            help="Takes a dictionary of arguments for the Torch.profiler class. If set, each run's profile will be dumped to a {output}/{name}/{run_settings}.profiler file. If not set or None, no timing profiles are saved.")
        
        # Internal Metrics
        # ----------------
        internal_group = parser.add_argument_group("Forward and Backward Hook Options.", description="What metrics are conducted on the internal mechanism of the model.")
        # Internal Metrics
        internal_group.add_argument(
            "--internal_metrics",
            type=str, choices=["sketching_info"], default=[], nargs='*',
            help="Whether to return diagnostic information about the sketch.")
        return parser


    def __init__(self, cfg: Optional[Namespace] = None):
        # Super to initialise self.cfg and get defaults
        super().__init__(cfg=cfg)

        # Parse Config Namespace
        # ----------------------
        # hooks
        self.hook_dict = {}
    

    def __str__(self):
        return f"Evaluator({self.cfg.as_dict()})"


    def evaluate_model_on_data(self, tokenised_data : Union[torch.Tensor, np.ndarray, List[torch.Tensor], List[np.ndarray]], attn_masks : Optional[Union[torch.Tensor, np.ndarray, List[torch.Tensor], List[np.ndarray]]], model, device : Optional[Union[str, Tuple[str, Optional[List[int]]]]]):
        """
        Evaluates the model on tokenised data using set metrics.

        Args:
            tokenised_data (Union[torch.Tensor, np.ndarray, List[torch.Tensor], List[np.ndarray]]):
                The tokenised data. Either a torch.tensor, a list of torch.Tensors, a list of numpy.ndarrays.
            attn_masks (Optional[Union[torch.Tensor, np.ndarray, List[torch.Tensor], List[np.ndarray]]]):
                Attention masks for the encoded data. If None, masks default to ones.
            model:
                Callable model to evaluate. Should output logits given a (batched) tokenised input.
            device (Optional[Union[str, Tuple[str, Optional[List[int]]]]])
                Device to use for computation.
        """

        return _evaluate_model_on_data_over_batch_sizes(tokenised_data=tokenised_data, attn_masks=attn_masks, model=model, metrics=self.cfg.metrics, batch_size=self.cfg.batch_size, device=device)
    

    def process_internal_metrics(self, model):
        """
        Processes the internal metrics collected during model evaluation.
    
        Args:
            model (torch.nn.module):
                The model on which the internal metrics were computed.
        """

        return _process_internal_metrics(internal_metrics=self.cfg.internal_metrics, model=model)
    

    def collect_internal_metrics(self, internal_metrics_dict):
        """
        Collects and processes the internal metrics gathered for one run of model evaluation.
    
        Args:
            model (torch.nn.module):
                The model on which the internal metrics were computed.
        """

        return _collect_internal_metrics(internal_metrics=self.cfg.internal_metrics, runs=self.cfg.runs, internal_metrics_dict=internal_metrics_dict)
