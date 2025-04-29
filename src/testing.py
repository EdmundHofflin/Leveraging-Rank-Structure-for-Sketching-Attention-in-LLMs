import os
import sys
from datetime import datetime

from loguru import logger
from tqdm import tqdm, trange

import numpy as np
import torch as torch
import xarray as xr

from modules.configurator import Configurator

@logger.catch
@torch.no_grad()
def main():
    # ================
    # Config and Setup
    # ================

    # Temporary log for initialising the modules
    randint = np.random.randint(0, int(1e5))
    temp_log = f"temp_{randint}.log"
    logger.add(open(temp_log, 'w'), level=0)

    # Parse and process configs
    parser = Configurator.get_config_parser()
    cfg = parser.parse_args()
    mdls = Configurator(cfg, temp_log=temp_log)
    
    # Setup output dictionary for saving results
    output_dict = dict()
    output_dict['start_datetime'] = datetime.now()
    output_dict['write_datetime'] = None
    output_dict['elapsed_time'] = None
    output_dict['device'] = mdls.system.cfg.device
    for sketcher in mdls.sketching.sketchers:
        output_dict[str(sketcher)] = None
    logger.debug(f"Setup output directory.")

    # Initiate result saving
    if mdls.system.master():
        mdls.inout.save_result(output_dict, file_name="results.json", data_subdir="compressed_data/results")

    logger.info("Setup complete.")
    mdls.prnt("Setup complete.", 2, end="\n")

    # ===================================
    # Setup Tokeniser, Dataset, and Model
    # ===================================

    # Load tokeniser from local files or Huggingface
    mdls.model.load_tokeniser(data_path=mdls.inout.cfg.data_path, trust_remote_code=mdls.inout.cfg.trust_remote_code, force_download=mdls.inout.cfg.force_download, prnt=mdls.prnt)
    # Set tokeniser max context length
    mdls.model.tokeniser.model_max_length = mdls.model.cfg.context_length

    # Load dataset from local files or Huggingface
    mdls.dataset.load(data_path=mdls.inout.cfg.data_path, force_download=mdls.inout.cfg.force_download, prnt=mdls.prnt)

    # Load tokeniser from local files or Huggingface
    mdls.dataset.tokenise(mdls.model.tokeniser, mdls.model.cfg.context_length, prnt=mdls.prnt)

    # Download model from local files or Huggingface
    mdls.model.load_model(data_path=mdls.inout.cfg.data_path, trust_remote_code=mdls.inout.cfg.trust_remote_code, force_download=mdls.inout.cfg.force_download, prnt=mdls.prnt)

    logger.info("Download and setup of tokeniser, dataset, and model complete.")
    mdls.prnt("Download and setup of tokeniser, dataset, and model complete.", 2, end="\n")

    # ============
    # Testing Loop
    # ============

    mdls.prnt("Beginning Testing:", 2, end="\n")
    sk_bar = tqdm(mdls.sketching.sketchers, leave=False, dynamic_ncols=True, position=0, file=sys.stdout)
    for sketcher_id, sketcher in enumerate(sk_bar):
        sk_bar.set_description(f"sketcher={sketcher}")
        # Setup result array
        standard_result_array = np.full((mdls.eval.cfg.runs+1, len(mdls.eval.cfg.metrics)+len(mdls.eval.cfg.time_metrics)+1), np.nan)
        # Setup internal metric dict
        internal_metrics_dict = dict()

        # Load model, patch, and setup
        mdls.model.load_model(data_path=mdls.inout.cfg.data_path, trust_remote_code=mdls.inout.cfg.trust_remote_code, force_download=False, prnt=mdls.prnt)
        mdls.model.patch_model(sketcher=sketcher, rng_generator=mdls.system.torch_rng, sketching_info=("sketching_info" in mdls.eval.cfg.internal_metrics), prnt=mdls.prnt)
        mdls.model.setup_model(device=mdls.system.cfg.device, dtype=mdls.system.cfg.dtype)

        # Testing loop
        run_bar = trange(mdls.eval.cfg.runs, leave=False, dynamic_ncols=True, position=1, desc="Evaluating Model: runs")
        for run in run_bar:
            # Subsample data
            encoded_texts = mdls.dataset.shuffle_and_subsample(mdls.system.torch_rng)
            # Evaluate model
            (btch_size, eval_output) = mdls.eval.evaluate_model_on_data(tokenised_data=encoded_texts, attn_masks=None, model=mdls.model.model, device=mdls.system.cfg.device) # TODO: Set attention mask from data
            del encoded_texts
            # If evaluation failed, skip to next sketcher
            if btch_size is None:
                output_dict[str(sketcher)] = eval_output # Save error dict
                # Write results
                if mdls.system.master():
                    mdls.inout.save_result(output_dict, file_name=f"results.json", data_subdir="compressed_data/results")
                # Clean Up
                mdls.model.clear_model()
                del standard_result_array
                break
            # Otherwise, average values across data
            for i5, metric in enumerate(mdls.eval.cfg.metrics):
                standard_result_array[run, i5] = np.nanmean(np.array(eval_output[metric]))
            # Record timing
            if len(mdls.eval.cfg.time_metrics) > 0:
                standard_result_array[run, len(mdls.eval.cfg.metrics):] = 0.0 # FIXME: Record timing results here
            # Record batch size
                standard_result_array[run, -1] = btch_size
            # Record internal metrics
            internal_metrics_dict[run] = dict()
            processed_internal_metrics = mdls.eval.process_internal_metrics(mdls.model.model)
            logger.trace(f"Processed Internal Metrics: {[name for (name, _) in processed_internal_metrics]}")
            for (name, processed_dict) in processed_internal_metrics:
                internal_metrics_dict[run][name] = processed_dict
        else:
            # Average values across runs
            for i5 in range(standard_result_array.shape[1]):
                standard_result_array[mdls.eval.cfg.runs, i5] = standard_result_array[:mdls.eval.cfg.runs, i5].sum()/mdls.eval.cfg.runs

            # Convert results to dict in xarray format and record in output_dict
            output_dict[str(sketcher)] = {}
            output_dict[str(sketcher)]['results'] = xr.DataArray(
                standard_result_array,
                coords=[
                    ("runs", list(range(1,mdls.eval.cfg.runs+1)) + ['average']),
                    ("metrics", mdls.eval.cfg.metrics + list(map(lambda x: f"time: {x}", mdls.eval.cfg.time_metrics)) + ['batch_size'])
                ]).to_dict()
            del standard_result_array
            output_dict[str(sketcher)]['id'] = sketcher_id

            # Process and pickle internal metrics:
            collected_internal_metrics = mdls.eval.collect_internal_metrics(internal_metrics_dict)
            # Save internal metrics dict
            for (name, processed_dict) in collected_internal_metrics:
                output_dict[str(sketcher)][name] = processed_dict
            
            # Write results
            if mdls.system.master():
                mdls.inout.save_result(output_dict, file_name=f"results.json", data_subdir="compressed_data")

            # Clean Up
            mdls.model.clear_model()


if __name__ == "__main__":
    main()
