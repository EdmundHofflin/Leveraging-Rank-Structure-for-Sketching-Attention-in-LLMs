import os
import sys
import multiprocessing as mp

from loguru import logger
from tqdm import tqdm
from jsonargparse import ArgumentParser

from typing import Optional

import numpy as np
import pandas as pd
import torch

import plotly.express as px
import datashader as ds

from utils.inout_utils import _read_result
from utils.general_utils import my_dict
from utils.inmetric_plotting_utils import _plot_internal_metrics
from utils.plotting_utils import _process_results, _plot_results


# ================
# Plotting Options
# ================

DEFAULT_PLOTTING_OPTIONS = {
    "col_norms_A": {
        'plot_type': 'strip',
        'xaxis_title': 'Sorted Index',
        'yaxis_title': 'Norm',
        'title': None, # 'Column Norms of A',
        'name': 'Attention Head',
        'sort': True,
        'html': False,
        'marker': dict(
            size=1.5,                 # Size of the markers
            opacity=(0.95,0.75),    # Opacity of the markers
            symbol='x-thin-open',   # Symbol for the markers
        ),
        'show_median': False
    },
    "row_norms_B": {
        'plot_type': 'strip',
        'xaxis_title': 'Sorted Index',
        'yaxis_title': 'Norm',
        'title': None, # 'Row Norms of V',
        'name': 'Attention Head',
        'sort': True,
        'html': False,
        'marker': dict(
            size=1.5,                 # Size of the markers
            opacity=(0.95,0.75),    # Opacity of the markers
            symbol='x-thin-open',   # Symbol for the markers
        ),
        'show_median': False
    },
    "abs_error": {
        'plot_type': 'strip',
        'xaxis_title': 'Layer',
        'yaxis_title': 'Error',
        'title': None, # 'Absolute Frobenius Error',
        'sort': False,
        'log_y': False,
        'html': False,
    },
    "rel_error": {
        'plot_type': 'strip',
        'xaxis_title': 'Layer',
        'yaxis_title': 'Error',
        'title': None, # 'Relative Frobenius Error',
        'sort': False,
        'log_y': False,
        'html': False
    },
    "paper_error": {
        'plot_type': 'strip',
        'xaxis_title': 'Layer',
        'yaxis_title': 'Error',
        'title': None, # 'Normalised Frobenius Error',
        'sort': False,
        'log_y': False,
        'html': False
    },
    "orthogonality_A": {
        'plot_type': 'heatmap',
        'xaxis_title': 'Index',
        'yaxis_title': 'Index',
        'title': None, # 'Pair-Wise Orthogonality of A',
        'sort': False,
        'html': False
    },
    "orthogonality_B": {
        'plot_type': 'heatmap',
        'xaxis_title': 'Index',
        'yaxis_title': 'Index',
        'title': None, # 'Pair-Wise Orthogonality of V',
        'sort': False,
        'html': False
    },
    "stable_rank_A": {
        'plot_type': 'strip',
        'xaxis_title': 'Layer',
        'yaxis_title': 'Stable Rank',
        'title': None, # 'Stable Rank of A',
        'name': 'Attention Head',
        'sort': False,
        'log_y': True,
        'html': False,
        'show_median': True
    },
    "stable_rank_B": {
        'plot_type': 'strip',
        'xaxis_title': 'Layer',
        'yaxis_title': 'Stable Rank',
        'title': None, # 'Stable Rank of V',
        'name': 'Attention Head',
        'sort': False,
        'log_y': True,
        'html': False,
        'show_median': True,
        'legend': {
            'x': 0.0625,
            'y': 0.05,
        }
    }
}

# ============
# Setup Parser
# ============

@logger.catch
def startup():
    # Parser
    parser = ArgumentParser(prog="plot.py", description="Plot internal metrics of the model")
    parser.add_argument("--config", type=str, default="gpt", help="Configuration to use. Default: gpt.")
    parser.add_argument("--num_processes", type=int, default=1, help="Number of processes to use. Default: 1.")
    parser.add_argument("--result_params",
        type=str, choices=["variant", "sample_proportion", "replacement", "normalise", "leverage_equation", "distribution_sampling", "linear_combination", "forced_history"], default=["variant", "sample_proportion"], nargs='+',
        help="Coordinates to use for the results. Default: ['variant', 'sample_proportion'].")
    parser.add_argument("--metrics",
        type=str, choices=['sketching_info'], default=['sketching_info'], nargs='+',
        help="Metrics to plot. Default: ['sketching_info'].")
    parser.add_argument("--sketching_metrics",
        type=str, choices=['col_norms_A', 'row_norms_B', 'paper_error', 'abs_error', 'rel_error', 'orthogonality_A', 'orthogonality_B', 'stable_rank_A', 'stable_rank_B', 'leverage_idxs', 'leverage_scores', 'n_leverage_samples', 'n_distribution_samples', 'n_total_samples'], default=['col_norms_A', 'row_norms_B', 'abs_error', 'rel_error', 'paper_error', 'stable_rank_A', 'stable_rank_B'], nargs='+',
        help="Submetrics from sketching_info to plot. Default: ['col_norms_A', 'row_norms_B', 'paper_error', 'stable_rank'].")
    parser.add_argument("--plotting_options",
        type=dict, default=DEFAULT_PLOTTING_OPTIONS,
        help="Plotting options to use.")
    parser.add_argument("--just_standard",
        type=bool, default=True,
        help="If True, plot inmetrics for only standard variants. Otherwise, plot all variants. Only applied to internal metric plotting. Default: True.")
    cfg = parser.parse_args()

    # Logger
    logger.remove()
    log = f"logs/{cfg.config}"
    logger.add(open(f"{log}_TRACE.log", 'w'), level=5)
    logger.add(open(f"{log}_DEBUG.log", 'w'), level=10)
    logger.info(f"Logging to {log}")

    # Log config
    logger.info(f"Plotting Arguments: {cfg}")

    return cfg


# ====
# Main
# ====

if __name__ == "__main__":
    cfg = startup()
    if cfg.num_processes > mp.cpu_count():
        raise RuntimeError(f"Number of processes ({cfg.num_processes}) is larger than the number of CPUs ({mp.cpu_count()}). Please reduce the number of processes to below {mp.cpu_count()}.")

    # Read complete config
    logger.trace(f"Config={cfg.config}: Reading keys")
    config_dict = _read_result(output_path=f"/scratch/hofflin/out/{cfg.config}", name="config.json")
    # Create plots dir
    if not os.path.exists(f"{config_dict["inout"]["output_path"]}/{config_dict["inout"]["name"]}/plots/"):
        os.mkdir(f"{config_dict["inout"]["output_path"]}/{config_dict["inout"]["name"]}/plots/")

    # Process and plot internal metrics
    _plot_internal_metrics(config_dict, cfg.metrics, cfg.sketching_metrics, cfg.plotting_options, cfg.just_standard)

    # Process and plot resutls
    result_array, sketcher_coords, sketcher_info = _process_results(config_dict, cfg.result_params, cfg.plotting_options)
    output = _plot_results(result_array, sketcher_coords, sketcher_info, f"{config_dict['inout']['output_path']}/{config_dict['inout']['name']}/plots/", cfg.plotting_options)
            
        # with mp.Pool(processes=cfg.num_processes) as mp_pool:
            # _ = mp_pool.apply_async(_plot_submetrics, args=(config_dict, sketcher_id, sketcher, metric, key, key_dict, cfg.plotting_options, 4))
            # mp_pool.close()
            # mp_pool.join()
    
        # print("Waiting for multiprocessing of plots to finish.")
        # # Close and join multiprocessing pool
        # logger.debug(f"Closing multiprocessing pool.")
        # mp_pool.close()
        # logger.debug(f"Joining multiprocessing pool.")
        # mp_pool.join()
        # logger.debug(f"Joining multiprocessing pool finished.")

        # if 'results' in cfg.metrics:
        #     logger.debug(f"Plotting results")
            # _ = mp_pool.apply_async(_plot_results, args=(config_dict, sketcher_id, sketcher, metric, key, run, val, DEFAULT_NORM_PLOT_PARAMS))
