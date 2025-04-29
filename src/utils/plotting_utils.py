import os
import sys
import multiprocessing as mp

from loguru import logger
from tqdm import tqdm

from typing import Optional, Union, List

import numpy as np
import pandas as pd
import torch
import xarray as xr

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import datashader as ds

from utils.inout_utils import _read_result
from utils.general_utils import my_dict


# ==============================
# Results Reading and Organising
# ==============================

def _process_results(config_dict : dict, result_params : Union[str, List[str]], plotting_options : Optional[dict] = None) -> None:
    """
    Plot the results from the config_dict.

    Args:
        config_dict (dict):
            Dictionary containing the configuration and results.
        result_params (Union[str, List[str]]):
            Coordinates to plot.
    """

    logger.trace(f"Starting _read_results: {config_dict.keys()=}, {result_params=}")

    # Get sketchers
    config_key_dict = _read_result(output_path=f"{config_dict["inout"]["output_path"]}/{config_dict["inout"]["name"]}", name="results.json", recursive=False)
    sketchers = [sketcher for sketcher in config_key_dict if sketcher[:10] == "Sketcher({"]

    # Setup result dictionary coordinates
    metric_coords = config_dict["eval"]["metrics"]
    sketcher_coords = {
        'variant': [],
        'sample_proportion': [],
        'replacement': [],
        'normalise': [],
        'leverage_equation': [],
        'distribution_sampling': [],
        'linear_combination': [],
        'forced_history': [],
        'linear_combination': []
    }
    sketcher_info = {
        'variant': [],
        'sample_proportion': [],
        'replacement': [],
        'normalise': [],
        'leverage_equation': [],
        'distribution_sampling': [],
        'linear_combination': [],
        'forced_history': [],
        'linear_combination': []
    }
    for sketcher in sketchers:
        sketcher_dict = eval(sketcher[9:-1])
        for key, val in sketcher_dict.items():
            if key in sketcher_coords:
                sketcher_info[key].append(val)
                if not val in sketcher_coords[key]:
                    sketcher_coords[key].append(val)
    if isinstance(result_params, str):
        result_params = [result_params]
    for key in result_params:
        val = sketcher_coords[key]
        if len(val) == 0 or (val == [None]) or (len(val) == 2 and val[0] == None):
            logger.error(f"_plot_results: Current run configuration does not contain {key} coordinates. Please check the configuration.")
            return Error(f"_plot_results: Current run configuration does not contain {key} coordinates. Please check the configuration.")
    result_dims = ['metrics'] + result_params
    result_dims_coords = [metric_coords] + list(map(lambda k: sketcher_coords[k], result_params))
    result_dims_lens = [len(metric_coords)] + list(map(lambda k: len(sketcher_coords[k]), result_params))

    # Setup and populate result array
    result_array = xr.DataArray(
        np.full(result_dims_lens, np.nan),
        coords=result_dims_coords,
        dims=result_dims)
    for idx, sketcher in enumerate(sketchers):
        out = _read_result(output_path=f"{config_dict["inout"]["output_path"]}/{config_dict["inout"]["name"]}", name="results.json", recursive=True, path=[sketcher, "results"])
        # input(f"{list(map(lambda k: sketcher_info[k][idx], result_params))=}")
        for metric in metric_coords:
            result_array.loc[metric, *map(lambda k: sketcher_info[k][idx], result_params)] = xr.DataArray.from_dict(out).sel(metrics=metric, runs="average")
    
    # Return
    logger.trace(f"Successful _read_results: {result_array=}, {sketcher_coords=}, {sketcher_info=}")
    return result_array, sketcher_coords, sketcher_info


# ================
# Results Plotting
# ================

def _plot_results(result_array: xr.DataArray, sketcher_coords: dict, sketcher_info: dict, output_path : str, plotting_options: Optional[dict] = None) -> None:
    """
    Plot the result_array.

    Args:
        result_array (xr.DataArray):
            Array containing the results.
        sketcher_coords (dict):
            Dictionary containing the coordinates of the sketchers.
        sketcher_info (dict):
            Dictionary containing the information of the sketchers.
        output_path (str):
            Path to save the plots.
        plotting_options (Optional[dict]):
            Dictionary containing the plotting options.
    """

    logger.trace(f"Starting _plot_results: {result_array=}, {sketcher_coords=}, {sketcher_info=}, {plotting_options.keys()=}")

    # Setup plotting options
    colours = {
        'standard': 'black',
        'random': 'green',
        'importance': 'blue',
        'max': 'red'
    }

    # Iterate over metrics
    for metric in result_array.coords['metrics'].values:
        logger.debug(f"Plotting metric: {metric}")

        # Setup plot
        fig = go.Figure()

        # Standard Transformer (if available)
        if 'standard' in sketcher_coords['variant']:
            standard_value = result_array.sel(metrics=metric, variant='standard', sample_proportion=None).values
            fig.add_trace(
                go.Scatter(
                    x=[np.nanmin(np.array(result_array.coords['sample_proportion'].values, dtype=float)), np.nanmax(np.array(result_array.coords['sample_proportion'].values, dtype=float))],
                    y=standard_value * np.ones(2),
                    mode="lines",
                    name="Standard" + '   ',
                    line=dict(color=colours['standard'], width=4, dash='dot'),
                    marker=dict(size=0)
                )
            )
        
        # Other variants
        for variant in sketcher_coords['variant']:
            if variant not in ['standard', 'random']:
                variant_values = result_array.sel(metrics=metric, variant=variant).values
                fig.add_trace(
                    go.Scatter(
                        x=result_array.coords['sample_proportion'].values,
                        y=variant_values,
                        mode="lines",
                        name=("ApproxMatrixMultiply" if variant == 'importance' else "Greedy") + '   ',
                        line=dict(color=colours[variant], width=4, dash='solid'),
                        marker=dict(size=0)
                    )
                )

        # Random Transformer (if available)
        if 'random' in sketcher_coords['variant']:
            random_values = result_array.sel(metrics=metric, variant='random').values
            fig.add_trace(
                go.Scatter(
                    x=result_array.coords['sample_proportion'].values,
                    y=random_values,
                    mode="lines",
                    name="Random",
                    line=dict(color=colours['random'], width=4, dash='dash'),
                    marker=dict(size=0)
                )
            )

        # Format plot
        fig.update_layout(
            height=450, width=800,
            title_text=None, # f"Sample proportion vs {metric}",
            template="plotly_white",
            xaxis = dict(
                title=f"Sample Proportion",
                showgrid=True,
                type="linear",
                showexponent = 'all',
                exponentformat = 'e',
                tick0=0,
                dtick=0.1,
                tickfont=dict(
                    size=18,
                )
            ),
            yaxis = dict(
                title=f"{metric.capitalize()}",
                showgrid=True,
                type="log" if metric == 'perplexity' else "linear",
                showexponent = 'all',
                exponentformat = 'e',
                tick0=1,
                dtick=1,
                tickfont=dict(
                    size=18,
                )
            ),
            font=dict(
                size=24,
            ),
            legend=dict(
                yanchor="top",      # y position of the legend
                xanchor="right",    # x position of the legend
                font=dict(
                    size=20,        # Font size of the legend
                ),
            ),
            margin=dict(l=5, r=5, t=5, b=5),
        )

        # Save plots
        fig.write_image(f"{output_path}{metric}.png")
        fig.write_image(f"{output_path}{metric}.svg")
        # if plotting_options.get('html', False):
        #     fig.write_html(f"{output_path}.html")

        # Clean up
        del fig

    logger.trace(f"Successful _plot_results")