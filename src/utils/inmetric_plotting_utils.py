import os
import sys
import multiprocessing as mp

from loguru import logger
from tqdm import tqdm

from typing import Optional, List

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


# ======================================
# Internal Metric Reading and Organising
# ======================================

def _plot_internal_metrics(config_dict : dict, inmetrics : Optional[List[str]] = None, sketching_submetrics : Optional[List[str]] = None, plotting_options : Optional[dict] = None, just_standard : bool = True) -> None:
    """
    This function reads internal metrics from the results.json file and organizes them into a dictionary structure. It also sets up the output directory for plots.

d    Args:
        config_dict (dict):
            Configuration dictionary containing paths and options.
        inmetrics (Optional[List[str]], default=None):
            List of internal metrics to read. If None, all internal metrics are read.
        sketching_submetrics (Optional[List[str]], default=None):
            List of sketching submetrics to read. If None, all sketching submetrics are read.
        plotting_options (Optional[dict], default=None):
            Dictionary containing plotting options for each internal metric.
        just_standard (bool, default=True):
            If True, plot inmetrics for only standard variants. Otherwise, plot all variants.
    Returns:
        None: The function does not return any value but creates directories and files for plotting.
    """
    
    # Read keys of results
    config_key_dict = _read_result(output_path=f"{config_dict["inout"]["output_path"]}/{config_dict["inout"]["name"]}", name="results.json", recursive=False)
    # Get sketcher keys
    sketchers = [sketcher for sketcher in config_key_dict if sketcher[:10] == "Sketcher({"]

    # Iterate over sketchers
    for (sketcher_id, sketcher) in enumerate(tqdm(sketchers, leave=False, dynamic_ncols=True, position=0, file=sys.stdout, desc="Sketchers")):
        if just_standard and eval(sketcher[9:-1])['variant'] != 'standard':
            logger.debug(f"Sketcher={sketcher_id} / Diagnostic={sketcher}: Skipping as variant is not standard and {just_standard=}.")
            continue
        # Read sketcher data
        logger.trace(f"Sketcher={sketcher_id}: Reading keys")
        sketcher_key_dict = _read_result(output_path=f"{config_dict["inout"]["output_path"]}/{config_dict["inout"]["name"]}", name="results.json", recursive=False, path=[sketcher])
        # Setup output directory for plots
        if not os.path.exists(f"{config_dict["inout"]["output_path"]}/{config_dict["inout"]["name"]}/plots/sketcher_{sketcher_id}"):
            os.mkdir(f"{config_dict["inout"]["output_path"]}/{config_dict["inout"]["name"]}/plots/sketcher_{sketcher_id}")

        # Iterate over internal metrics
        for inmetric in tqdm(sketcher_key_dict, leave=False, dynamic_ncols=True, position=1, file=sys.stdout, desc="Metrics"):
            # Check if metric is in config
            if not (inmetric in config_dict['eval']['internal_metrics'] and inmetric in inmetrics):
                if not inmetric in config_dict['eval']['internal_metrics']:
                    logger.debug(f"Sketcher={sketcher_id} / Diagnostic={inmetric}: Skipping as {inmetric} not in {config_dict['eval']['internal_metrics']=}")
                else:
                    logger.debug(f"Sketcher={sketcher_id} / Diagnostic={metric}: Skipping as {inmetric} not in {inmetrics=}")
                continue
            # Read metric data
            logger.trace(f"Sketcher={sketcher_id} / Diagnostic={inmetric}: Reading data")
            inmetric_dict = _read_result(output_path=f"{config_dict["inout"]["output_path"]}/{config_dict["inout"]["name"]}", name="results.json", recursive=True, path=[sketcher, inmetric], ignore_keys=['orthogonality_A', 'orthogonality_B', 'leverage_scores', 'leverage_idxs', 'n_leverage_samples', 'n_distribution_samples', 'n_total_samples', 'batches'])
            # Check if metric is a dictionary
            if not isinstance(inmetric_dict, dict):
                logger.debug(f"Sketcher={sketcher_id} / Diagnostic={inmetric}: Skipping as {inmetric} not a dictionary")
                continue
            # Create output directory for plots
            if not os.path.exists(f"{config_dict["inout"]["output_path"]}/{config_dict["inout"]["name"]}/plots/sketcher_{sketcher_id}/{inmetric}"):
                    os.mkdir(f"{config_dict["inout"]["output_path"]}/{config_dict["inout"]["name"]}/plots/sketcher_{sketcher_id}/{inmetric}")

            # FIXME: Make separate func for each inmetric
            # Iterate over layers
            sk_inmet_dict = my_dict()
            for layer, layer_dict in tqdm(inmetric_dict.items(), leave=False, dynamic_ncols=True, position=2, file=sys.stdout, desc="Reading Layers"):
                logger.debug(f"Sketcher={sketcher_id} / Diagnostic={inmetric}[{layer}]: Reading data")
                assert isinstance(layer_dict, dict)
                for key, key_dict in tqdm(layer_dict.items(), leave=False, dynamic_ncols=True, position=3, file=sys.stdout, desc="Reading Diagnostics"):
                    if inmetric == 'sketching_info' and not (key in sketching_submetrics):
                        logger.debug(f"Sketcher={sketcher_id} / Diagnostic={inmetric}[{layer}][{key}]: Skipping")
                        continue
                    else:
                        logger.debug(f"Sketcher={sketcher_id} / Diagnostic={inmetric}[{layer}][{key}]: Reading data")
                        assert isinstance(key_dict, dict)
                    if key not in sk_inmet_dict:
                        sk_inmet_dict[key] = my_dict()
                    if layer not in sk_inmet_dict[key]:
                        sk_inmet_dict[key][layer] = my_dict()
                    for run, val in tqdm(key_dict.items(), leave=False, dynamic_ncols=True, position=4, file=sys.stdout, desc="Reading Runs"):
                        if run not in sk_inmet_dict:
                            sk_inmet_dict[key][layer][run] = my_dict()
                        if isinstance(val, str):
                            if val[-4:] == ".npy":
                                sk_inmet_dict[key][layer][run] = np.array(np.load(val), dtype=float)
                            elif val[-3:] == ".pt":
                                sk_inmet_dict[key][layer][run] = torch.load(val)
                            else:
                                sk_inmet_dict[key][layer][run] = val
                        else:
                            sk_inmet_dict[key][layer][run] = val
            
            # Clean up before plotting
            del inmetric_dict
            # Plotting
            logger.debug(f"Sketcher={sketcher_id} / Diagnostic={inmetric}: Deploying async multiprocess plotting")
            out = _plot_submetrics(sketcher_id, sketcher, inmetric, sk_inmet_dict, output_path=f"{config_dict['inout']['output_path']}/{config_dict['inout']['name']}/plots/sketcher_{sketcher_id}/{inmetric}/", plotting_options=plotting_options)
            logger.debug(f"Sketcher={sketcher_id} / Diagnostic={inmetric}[{key}]: error={out}")


# =======================================
# Internal Metric Processing and Plotting
# =======================================

def _plot_submetrics(sketcher_id : int, sketcher : str, internal_metric : str, sk_inmet_dict : dict, output_path : str, plotting_options : dict, tqdm_position : Optional[int] = 0) -> None:
    """
    This function processes and visualizes submetrics for a specific diagnostic key of an internal metric. It supports various types of submetrics such as norms, errors, and orthogonality metrics, and generates plots based on the provided plotting options.

    Args:
        sketcher_id (int): Identifier for the sketcher being processed.
        sketcher (str): Name of the sketcher.
        internal_metric (str): Name of the internal metric being analyzed.
        sk_inmet_dict (dict): Dictionary containing the submetric data for the internal metric.
        output_path (str): Path where the plots will be saved.
        plotting_options (dict): Dictionary specifying plotting options for each submetric key.
        tqdm_position (Optional[int], default=0): Position for the tqdm progress bar.
    Raises:
        ValueError: If the data type of the submetric entries is not a NumPy array or if required norms are unavailable.
        NotImplementedError: If a submetric key or plot type is not implemented.
    Returns:
        None: The function performs plotting and does not return any value.
    """

    outputs = []
    for key, key_dict in tqdm(sk_inmet_dict.items(), leave=False, dynamic_ncols=True, position=tqdm_position+0, file=sys.stdout, desc=f"Sketcher={sketcher_id} / Diagnostic={internal_metric}: Processing"):
        logger.debug(f"Sketcher={sketcher_id} / Diagnostic={internal_metric}[{key}]: Processing")
        # Construct data_dict based on key
        if key in ['col_norms_A', 'row_norms_B', 'leverage_scores']:
            plotting_data = my_dict()
            for layer, layer_dict in key_dict.items():
                entry_ls = []
                for run, entry in layer_dict.items():
                    if isinstance(entry, np.ndarray):
                        logger.trace(f"Sketcher={sketcher_id} with Diagnostic={internal_metric}[{key}][{layer}]: entry.shape={entry.shape}")
                        if plotting_options[key]['sort']:
                            entry = np.sort(entry, axis=-1)[..., ::-1]  # Sort along the last axis in descending order
                        entry_ls.append(entry)
                    else:
                        raise ValueError(f"In Sketcher={sketcher_id} with Diagnostic={internal_metric}[{key}][{layer}], value has type={type(vector)}. Expected NumPy array.")
            plotting_data[layer] = np.mean(entry_ls, axis=(0, 1)) # Mean over runs and batches, keeping heads
            plotting_data = np.concatenate([np.reshape(value, (value.shape[0], -1)) for value in plotting_data.values()], axis=0)

            # Compute ratios
            ratio_data = plotting_data[:, 0:1] / plotting_data
            m, n = ratio_data.shape
            key_indices = [int(0.01*n), int(np.floor(np.sqrt(n))), int(0.9*n)]
            ratio_data = np.mean(ratio_data[:, key_indices], axis=0)  # Select along the last axis and then take mean
            logger.success(f"Sketcher={sketcher_id} with Diagnostic={internal_metric}[{key}][{layer}]: {key_indices=}, {ratio_data=}")
        elif key in ['abs_error', 'rel_error', 'paper_error', 'stable_rank_A', 'stable_rank_B']:
            plotting_data = my_dict()
            for layer, layer_dict in key_dict.items():
                entry_ls = []
                for run, entry in layer_dict.items():
                    if isinstance(entry, np.ndarray):
                        logger.trace(f"Sketcher={sketcher_id} with Diagnostic={internal_metric}[{key}][{layer}]: entry.shape={entry.shape}")
                        if plotting_options[key]['sort']:
                            logger.trace(f"Sketcher={sketcher_id} with Diagnostic={internal_metric}[{key}][{layer}]: Plotting_Params[{key}]['sort'] set to {plotting_options[key]['sort']} but no sorting option is available for this key. Proceeding without sorting.")
                        entry_ls.append(entry) # No need to sort
                    else:
                        raise ValueError(f"Sketcher={sketcher_id} with Diagnostic={internal_metric}[{key}][{layer}]: value has type={type(entry)}. Expected NumPy array.")
                logger.trace(f"Sketcher={sketcher_id} with Diagnostic={internal_metric}[{key}][{layer}]: entry_ls.shape={np.array(entry_ls).shape}")
                plotting_data[layer] = np.mean(entry_ls, axis=(0, 1)) # Mean over runs and batches, keeping heads
            logger.trace(f"Sketcher={sketcher_id} with Diagnostic={internal_metric}[{key}][{layer}]: {plotting_data.keys()=}, plotting_data[layer].shape={plotting_data[layer].shape}")
            plotting_data = np.stack([value for value in plotting_data.values()], axis=0)
            logger.trace(f"Sketcher={sketcher_id} with Diagnostic={internal_metric}[{key}][{layer}]: {plotting_data.shape=}")
        elif key in ['orthogonality_A', 'orthogonality_B']:
            plotting_data = []
            if plotting_options[key]['sort']:
                norms = None
                if key == 'orthogonality_A':
                    norms = sk_inmet_dict['col_norms_A'] if 'col_norms_A' in sk_inmet_dict else False
                elif key == 'orthogonality_B':
                    norms = sk_inmet_dict['row_norms_B'] if 'row_norms_B' in sk_inmet_dict else False
                else:
                    raise ValueError(f"This should be impossible to reach! Sketcher={sketcher_id} with Diagnostic={internal_metric}[{key}]: Unknown key={key}.")
                if norms is None:
                    raise ValueError(f"Sketcher={sketcher_id} with Diagnostic={internal_metric}[{key}]: Received unknown matrix={key[14:]} for orthogonality metric.")
                elif not norms:
                    raise ValueError(f"Sketcher={sketcher_id} with Diagnostic={internal_metric}[{key}]: Sorting is not possible because norms for matrix={key[14:]} are not available.")
                else:
                    sorted_idx = np.argsort(norms, axis=-1)  # Sort along the last axis
            for layer, layer_dict in key_dict.items():
                for run, entry in layer_dict.items():
                    if isinstance(entry, np.ndarray):
                        if plotting_options[key]['sort']:
                            entry = np.take_along_axis(np.take_along_axis(entry, sorted_idx[..., np.newaxis], axis=-1), sorted_idx[..., np.newaxis, :], axis=-2)  # Sort along the last two axes
                        plotting_data.append(entry)
                    else:
                        raise ValueError(f"Sketcher={sketcher_id} with Diagnostic={internal_metric}[{key}][{layer}]: Value has type={type(vector)}. Expected NumPy array.")
            plotting_data = np.mean(plotting_data, axis=(0, 1, 2))  # Mean over layers+runs, batches, and heads
        else:
            raise NotImplementedError(f"Sketcher={sketcher_id} with Diagnostic={internal_metric}[{key}]: Key=[{key}] not implemented for plotting.")
        
        # Check
        logger.trace(f"Sketcher={sketcher_id} / Diagnostic={internal_metric}[{key}]: plotting_data{'' if isinstance(plotting_data, np.ndarray) else f"[{list(plotting_data.keys())[0]}]"}.shape={plotting_data.shape if isinstance(plotting_data, np.ndarray) else plotting_data[list(plotting_data.keys())[0]].shape}, plot_type={plotting_options[key]['plot_type']}")

        # Plot
        logger.debug(f"Sketcher={sketcher_id} / Diagnostic={internal_metric}[{key}]: Plotting")
        output = False
        _plot_func = plotting_funcs[plotting_options[key]['plot_type']]
        if _plot_func is None:
            raise NotImplementedError(f"Sketcher={sketcher_id} with Diagnostic={internal_metric}[{key}]: Plot type {plotting_options[key]['plot_type']} not implemented.")
        else:
            try:
                output = _plot_func(data=plotting_data, plotting_options=plotting_options[key], output_path=f"{output_path}{key}")
            except Exception as e:
                logger.error(f"Sketcher={sketcher_id} / Diagnostic={internal_metric}[{key}]: Error in plotting, {e}")
                output = e
        
        # Output
        if output:
            logger.debug(f"Sketcher={sketcher_id} / Diagnostic={internal_metric}[{key}]: Plotting complete")
            outputs.append(output)
        else:
            logger.warning(f"Sketcher={sketcher_id} / Diagnostic={internal_metric}[{key}]: Plotting failed with error={output}")
            outputs.append(output)
    
    logger.debug(f"Sketcher={sketcher_id} / Diagnostic={internal_metric}: Plotting complete with {outputs=}")
    return outputs


# ================
# Heatmap Plotting
# ================

DEFAULT_HEATMAP_OPTIONS = {
    'color_scale': 'Viridis',       # Color scale for the heatmap
    'color_range': None,            # Range for the color scale (e.g., [min, max])
    'show_colorbar': True,          # Whether to display the colorbar
    'width': 800,                   # Width of the plot
    'height': 600,                  # Height of the plot
    'margin': dict(l=40, r=40, t=40, b=40),  # Margins for the plot
    'title': None,                  # Title of the heatmap
    'xaxis_title': "X-axis",        # X-axis label
    'yaxis_title': "Y-axis",        # Y-axis label
    'template': 'plotly_white',     # Template for the plot
}

def _plot_heatmap(data: np.ndarray, plotting_options: dict, output_path: str) -> None:
    """
    Plot a heatmap for the given data.
        Args:
            data (np.ndarray): A 2D numpy array containing the data to be visualized in the heatmap.
            plotting_options (dict): A dictionary containing configuration options for the plot, such as titles, axis labels, and plot type.
            output_path (str): The file path (excluding extension) where the heatmap will be saved.
        Returns:
            None: The function saves the heatmap as `.html`, `.png`, and `.svg` files at the specified `output_path` and does not return any value.
    """

    logger.trace(f"Starting _plot_heatmap: {data.shape=}, {plotting_options=}, {output_path=}")
    assert plotting_options['plot_type'] == 'heatmap', f"This should be impossible to reach! Plot type is not heatmap."

    # Extract heatmap options
    heatmap_options = DEFAULT_HEATMAP_OPTIONS.copy()
    heatmap_options.update(plotting_options)

    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=data,
        colorscale=heatmap_options['color_scale'],
        zmin=heatmap_options['zmin'] if 'zmin' in heatmap_options else np.min(data),
        zmax=heatmap_options['zmax'] if 'zmax' in heatmap_options else np.max(data),
        colorbar=dict(showticklabels=heatmap_options['show_colorbar']),
    ))

    # Update layout
    fig.update_layout(
        title=heatmap_options['title'] if heatmap_options['title'] else f"Heatmap",
        xaxis_title=heatmap_options['xaxis_title'],
        yaxis_title=heatmap_options['yaxis_title'],
        width=heatmap_options['width'],
        height=heatmap_options['height'],
        margin=heatmap_options['margin'],
        template=heatmap_options['template'],
    )

    # Saving figure
    if 'html' in plotting_options and plotting_options['html']:
        fig.write_html(f"{output_path}.html")
    fig.write_image(f"{output_path}.png")
    fig.write_image(f"{output_path}.svg")

    # Output
    logger.trace(f"Successful plot_heatmap: {output_path=}.")
    return True


# ===============
# Violin Plotting
# ===============

DEFAULT_VIOLIN_OPTIONS = {
    'color_scale': 'Viridis',   # Color scale for the plot
    'show_box': True,           # Show box plot inside the violin
    'show_points': False,       # Show all points ('all', 'outliers', or False)
    'log_y': False,             # Whether to use a logarithmic scale for the y-axis
    'width': 800,               # Width of the plot
    'height': 600,              # Height of the plot
    'margin': dict(l=40, r=40, t=40, b=40),  # Margins for the plot
    'title': None,              # Title of the plot
    'xaxis_title': "Category",  # X-axis label
    'yaxis_title': "Value",     # Y-axis label
    'template': 'plotly_white', # Template for the plot
}

def _plot_violin(data: dict, plotting_options: dict, output_path: str) -> None:
    """
    Plot a violin plot for the given data.
        Args:
            data (dict): A dictionary where keys are categories and values are arrays of values for each category.
            output_path (str): The file path (excluding extension) where the violin plot will be saved.
            sketcher_id (int): An identifier for the sketcher associated with the plot.
            sketcher (str): The name or description of the sketcher.
            internal_metric (str): The internal metric being visualized.
            key (str): The submetric that is being plotted.
        Returns:
            None: The function saves the violin plot as `.html`, `.png`, and `.svg` files at the specified `output_path` and does not return any value.
    """

    logger.trace(f"Starting _plot_heatmap: {data.shape=}, {plotting_options=}, {output_path=}")
    assert plotting_options['plot_type'] == 'heatmap', f"This should be impossible to reach! Plot type is not heatmap."

    # Extract violin options
    violin_options = DEFAULT_VIOLIN_OPTIONS.copy()
    violin_options.update(plotting_options)

    # Prepare data for plotting
    categories = []
    values = []
    for category, value_array in data.items():
        categories.extend([category] * len(value_array))
        values.append(value_array)

    # Create the violin plot
    fig = go.Figure(data=go.Violin(
        x=categories,
        y=np.concatenate(values),
        box=violin_options['show_box'],
        points=violin_options['show_points'],
        line_color=violin_options['color_scale'],
    ))

    # Update layout
    fig.update_layout(
        title=violin_options['title'] if violin_options['title'] else f"Violin Plot",
        xaxis_title=violin_options['xaxis_title'],
        # yaxis_title=violin_options['yaxis_title'],
        xaxis_type="category",
        # yaxis_type="linear" if not violin_options['log_y'] else "log",
        width=violin_options['width'],
        height=violin_options['height'],
        margin=violin_options['margin'],
        template=heatmap_options['template'],
        legend=dict(
            yanchor="top",
            xanchor="right",
        ),
        yaxis = dict(
            title=violin_options['yaxis_title'],
            type="log" if violin_options['log_y'] else "linear",
            showexponent = 'all',
            exponentformat = 'e'
        ),
    )

    # Saving figure
    if 'html' in plotting_options and plotting_options['html']:
        fig.write_html(f"{output_path}.html")
    fig.write_image(f"{output_path}.png")
    fig.write_image(f"{output_path}.svg")

    # Output
    logger.trace(f"Successful _plot_violin: {output_path=}.")
    return True


# ===============
# Strip Plotting
# ===============

DEFAULT_STRIP_OPTIONS = {
    'color_scale': 'Viridis',   # Color scale for the plot
    'show_mean': True,          # Whether to compute and overlay the mean of each category
    'show_median': True,        # Whether to compute and overlay the median of each category
    'log_x': False,             # Whether to use a logarithmic scale for the x-axis
    'log_y': True,              # Whether to use a logarithmic scale for the y-axis
    'width': 800,               # Width of the plot
    'height': 450,              # Height of the plot
    'margin': dict(l=5, r=5, t=5, b=5),  # Margins for the plot
    'title': None,              # Title of the plot
    'xaxis_title': "Category",  # X-axis label
    'yaxis_title': "Value",     # Y-axis label
    'template': 'plotly_white', # Template for the plot
    'marker': dict(
        size=5,                 # Size of the markers
        opacity=(0.95,0.75),    # Opacity of the markers
        symbol='x-thin-open',   # Symbol for the markers
    ),
    'legend': dict(
        yanchor="top",          # yanchor for the legend
        xanchor="right",        # xanchor for the legend
        font=dict(
            size=20,            # Font size of the legend
        ),
        itemsizing='constant',  # Legend items sizing
    ),
    'font': dict(
        size=24,
    )

}

def _plot_strip(data: np.ndarray, plotting_options: dict, output_path: str) -> None:
    """
    Plot a strip plot for the given data.
        Args:
            data (np.ndarray): A 2D NumPy array where rows represent multiple entries for each x-axis category, and columns represent the x-axis categories.
            plotting_options (dict): A dictionary containing configuration options for the plot, such as titles, axis labels, and plot type.
            output_path (str): The file path (excluding extension) where the strip plot will be saved.
        Returns:
            None: The function saves the strip plot as `.html`, `.png`, and `.svg` files at the specified `output_path` and does not return any value.
    """

    logger.trace(f"Starting _plot_strip: {data.shape=}, {plotting_options=}, {output_path=}")
    assert plotting_options['plot_type'] == 'strip', f"This should be impossible to reach! Plot type is not strip."

    # Extract strip options
    strip_options = DEFAULT_STRIP_OPTIONS.copy()
    strip_options.update(plotting_options)

    logger.trace(f"_plot_strip: {strip_options['name']=}")

    # Prepare data for plotting
    num_rows, num_columns = data.shape
    categories = np.tile(np.arange(num_columns), num_rows)  # Repeat column indices for each row
    values = data.flatten()  # Flatten the 2D array into a 1D array
    means = []
    if strip_options['show_mean']:
        means = np.mean(data, axis=0)  # Compute the mean for each column (x-axis category)
        medians = np.median(data, axis=0)  # Compute the median for each column (x-axis category)

    # Create the strip plot
    fig = go.Figure(data=go.Scatter(
        x=categories,
        y=values,
        marker=dict(
            size=strip_options['marker']['size'],  # Adjust size based on number of entries
            opacity=strip_options['marker']['opacity'][1] if strip_options['show_mean'] or strip_options['show_median'] else strip_options['marker']['opacity'][0],
            colorscale=strip_options['color_scale'],
        ),
        mode='markers',
        name=(strip_options['name'] if 'name' in strip_options else 'Data Points') + '   ',
    ))

    # Overlay mean line plot if enabled
    if strip_options['show_mean']:
        fig.add_trace(go.Scatter(
            x=np.arange(num_columns),
            y=means,
            mode='lines',
            name='Mean' + '   ',
            line=dict(color='black', width=3)
        ))
    
    # Overlay mean line plot if enabled
    if strip_options['show_median']:
        fig.add_trace(go.Scatter(
            x=np.arange(num_columns),
            y=medians,
            mode='lines',
            name='Median' + '   ',
            line=dict(color='red', width=3, dash='dash')
        ))

    # Update layout
    fig.update_layout(
        xaxis = dict(
            title=strip_options['xaxis_title'],
            type="log" if strip_options['log_x'] else "linear",
            showexponent = 'all',
            exponentformat = 'e',
            tickfont=dict(
                size=18,
            ),
            tick0=1,
            dtick=np.log10(50) if strip_options['log_x'] else None,
        ),
        yaxis = dict(
            title=strip_options['yaxis_title'],
            type="log" if strip_options['log_y'] else "linear",
            showexponent = 'all',
            exponentformat = 'e',
            tickfont=dict(
                size=18,
            ),
            tick0=1,
            dtick=1 if strip_options['log_x'] else None,
        ),
        legend=strip_options['legend'],
        title=strip_options['title'] if 'title' in strip_options else f"Strip Plot",
        width=strip_options['width'],
        height=strip_options['height'],
        margin=strip_options['margin'],
        template=strip_options['template'],
        font=strip_options['font'],
    )

    # Apply logarithmic scale to axes if enabled
    if strip_options['log_x']:
        fig.update_layout(xaxis_type="log")
    if strip_options['log_y']:
        fig.update_layout(yaxis_type="log")

    # Saving figure
    if 'html' in plotting_options and plotting_options['html']:
        fig.write_html(f"{output_path}.html")
    fig.write_image(f"{output_path}.png")
    fig.write_image(f"{output_path}.svg")

    # Output
    logger.trace(f"Successful _plot_strip: {output_path=}.")
    return True


# ===============
# Plotting Config
# ===============

plotting_funcs = {
    'heatmap': _plot_heatmap,
    'violin': _plot_violin,
    'strip': _plot_strip,
    'scatter': None,
    'line': None,
    'histogram': None,
}