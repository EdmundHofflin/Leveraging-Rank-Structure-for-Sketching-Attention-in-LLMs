import sys
import traceback

from loguru import logger

from typing import List, Dict, Union, Optional, Tuple
from jsonargparse.typing import PositiveInt

from tqdm import trange

import numpy as np
import torch as torch
from torch.nn import CrossEntropyLoss

from utils.general_utils import my_dict



# ==================
# Overall Evaluation
# ==================

def _evaluate_model_on_data_over_batch_sizes(tokenised_data : Union[torch.Tensor, np.ndarray, List[torch.Tensor], List[np.ndarray]], attn_masks : Union[torch.Tensor, np.ndarray, List[torch.Tensor], List[np.ndarray], None], model, metrics : List[str], batch_size : List[PositiveInt], device : Optional[Union[str, Tuple[str, Optional[List[int]]]]]):
    """
    Evaluates the model on tokenised data using set metrics, attempted different batch sizes until success or all fail.

    Args:
        tokenised_data (torch.Tensor, numpy.ndarray, list[torch.Tensor], or list(numpy.ndarray)):
            The tokenised data. Either a torch.tensor, a list of torch.Tensors, a list of numpy.ndarrays.
        attn_masks (torch.Tensor, numpy.ndarray, list[torch.Tensor], or list(numpy.ndarray)):
            Attention masks for the encoded data. If None, masks default to ones.
        model:
            Callable model to evaluate. Should output logits given a (batched) tokenised input.
        metrics (List[str]):
            A list of metrics to evaluate the model on. Each metric must take the output logits, label, and attention_mask for a given input
        batch_size (List[PositiveInt]):
            List of batch sizes. Evaluation is attempted on each size in turn. If all fail, returns a tuple of None and a dictionary of the errors that prevented evaluation by batch_size value.
        device (Optional[Union[str, Tuple[str, Optional[List[int]]]]])
            Device to use for computation.
    """

    logger.trace(f"Starting _evaluate_model_on_data_over_batch_sizes: {tokenised_data.shape=}, {attn_masks if attn_masks is None else attn_masks.shape=}, {model=}, {metrics=}, {batch_size=}, {device=}", 2, end="\n")

    # Setup error output
    error_dict = {}
    # Iterate over batch sizes
    for btch_sz in batch_size:
        # Attempt to evaluate
        try:
            output = _evaluate_model_on_data(tokenised_data=tokenised_data, attn_masks=attn_masks, model=model, metrics=metrics, batch_size=btch_sz, device=device)
            # If successful, delete error_dict (very data intensive) and return
            del error_dict
            logger.trace(f"Successful _evaluate_model_on_data_over_batch_sizes: ({btch_sz=}, {output.keys()=})", 2, end="\n")
            return (btch_sz, output)
        # If unsuccessful, record error and try again
        except RuntimeError as e:
            error_dict[btch_sz] = (str(e), traceback.format_exc())
    
    logger.trace(f"Unsuccessful _evaluate_model_on_data_over_batch_sizes.", 2, end="\n")
    return (None, error_dict)
        

def _evaluate_model_on_data(tokenised_data : Union[torch.Tensor, np.ndarray, List[torch.Tensor], List[np.ndarray]], attn_masks : Union[torch.Tensor, np.ndarray, List[torch.Tensor], List[np.ndarray], None], model, metrics : List[str], batch_size : PositiveInt, device : Optional[Union[str, Tuple[str, Optional[List[int]]]]]):
    """
    Evaluates the model on tokenised data using set metrics.

    Args:
        tokenised_data (torch.Tensor, numpy.ndarray, list[torch.Tensor], or list(numpy.ndarray)):
            The tokenised data. Either a torch.tensor, a list of torch.Tensors, a list of numpy.ndarrays.
        attn_masks (torch.Tensor, numpy.ndarray, list[torch.Tensor], or list(numpy.ndarray)):
            Attention masks for the encoded data. If None, masks default to ones.
        model:
            Callable model to evaluate. Should output logits given a (batched) tokenised input.
        metrics (List[str]):
            A list of metrics to evaluate the model on. Each metric must take the output logits, label, and attention_mask for a given input
        batch_size (PositiveInt):
            Batch size.
        device (Optional[Union[str, Tuple[str, Optional[List[int]]]]])
            Device to use for computation.
    """

    logger.trace(f"Starting _evaluate_model_on_data: {tokenised_data.shape=}, {attn_masks if attn_masks is None else attn_masks.shape=}, {model=}, {metrics=}, {batch_size=}, {device=}", 2, end="\n")

    # Setup attention masks
    if attn_masks is None:
        if torch.is_tensor(tokenised_data) or isinstance(tokenised_data, np.ndarray):
            attn_masks = torch.ones(tokenised_data.shape)
        elif isinstance(tokenised_data, list):
            attn_masks = torch.ones((len(tokenised_data), *tokenised_data[0].shape))
        else:
            RuntimeError(f"Tokenised data has unexpected type: {type(tokenised_data)=}. Exepected type torch.Tensor, numpy.ndarray, or list.")
    # Setup result dict
    eval_dict = {}
    for metric in metrics:
        eval_dict[metric] = []

    # Compute number of batches
    if torch.is_tensor(tokenised_data) or isinstance(tokenised_data, np.ndarray):
        n_batches = -(tokenised_data.shape[0] // -batch_size)
    elif isinstance(tokenised_data, list):
        n_batches = -(len(tokenised_data) // -batch_size)
    else:
        raise RuntimeError(f"Tokenised data has unexpected type: {type(tokenised_data)=}. Exepected type torch.Tensor, numpy.ndarray, or list.")
    
    # Helper function
    def isolate_and_process_batch(input, bid):
        # Isolate batch
        batch = input[bid * batch_size : (bid+1) * batch_size]
        # Process batch by type
        if torch.is_tensor(batch):
            batch = batch
        elif isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch)
        elif isinstance(batch, list):
            if torch.is_tensor(batch[0]):
                batch = torch.cat(batch, dim=0)
            elif isinstance(batch[0], np.ndarray):
                batch = torch.from_numpy(np.concatenate(batch), axis=0)
            else:
                raise RuntimeError(f"This should be impossible to reach! Tokenised data is a list {type(tokenised_data)=}, but elements have type {type(batch[0])=} when they can only have type Tensor or numpy.ndarray.")
        else:
            raise RuntimeError(f"This should be impossible to reach! Tokenised data has unexpected type: {type(tokenised_data)=}. Expected type torch.Tensor, numpy.ndarray, or list.")
        batch.to(device=device[0] if device[1] is None else ("cuda:"+str(device[1][0]) if isinstance(device[1], list) else device[1]["local_device"]))
        return batch

    # Evaluation Loop
    pbar = trange(n_batches, leave=False, dynamic_ncols=True, desc=f"nan_count = 0", file=sys.stdout)
    for bid in pbar:       
        # Isolate batch for data and attn masks
        tokenised_data_batch = isolate_and_process_batch(tokenised_data, bid)
        attn_masks_batch = isolate_and_process_batch(attn_masks, bid)
        # Compute logits
        out_logits_batch = model(tokenised_data_batch[...,:,:]).logits
        # Labels
        labels_batch = tokenised_data_batch[...,:,:] #FIXME: labels from data?
        # WARNING: If model can't compute, then change final range to :context_length for tokenised_data_batch

        # Evaluate models
        for metric in metrics:
            eval_dict[metric] += METRIC_DICT[metric](out_logits_batch, labels_batch, attn_masks_batch).tolist()
        
        # Update progress bar
        desc = f"nan_count = FIXME" # FIXME
        for metric in metrics:
            desc += f", avg {metric} = {np.mean(np.array(eval_dict[metric])[~np.isnan(np.array(eval_dict[metric]))]):.4f}"
        pbar.set_description(desc)
        
        # Clean up
        del tokenised_data_batch, attn_masks_batch, out_logits_batch, labels_batch

    # Return
    logger.trace(f"Successful _evaluate_model_on_data: {eval_dict.keys()=}", 2, end="\n")
    return eval_dict



# =======
# Metrics
# =======

def _cross_entropy_loss(logits, labels, attn_mask):
    """
    Computes the perplexity given the logits, labels, and mask, via the cross entropy loss.
    
    Args:
        logits (torch.Tensor):
            A torch tensor of the logits.
        labels (torch.Tensor):
            A torch tensor of the true labels.
        attn_mask (torch.Tensor):
            A torch tensor of the attention mask.
    """

    # Shift logits, labels, and mask
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    # Compute cross entropy loss
    cel_batch = CrossEntropyLoss(reduction='none')(shift_logits.transpose(1, 2), shift_labels).float()
    # Clean
    del logits, labels, attn_mask, shift_logits, shift_labels
    # Return
    return cel_batch
    

def _perplexity(logits, labels, attn_mask):
    """
    Computes the perplexity given the logits, labels, and mask, via the cross entropy loss.
    
    Args:
        logits (torch.Tensor):
            A torch tensor of the logits.
        labels (torch.Tensor):
            A torch tensor of the true labels.
        attn_mask (torch.Tensor):
            A torch tensor of the attention mask.
    """

    # Shift mask
    shift_attn_mask = attn_mask[..., 1:]
    # Compute cross entropy loss
    loss_batch = _cross_entropy_loss(logits, labels, attn_mask)
    # Compute perplexity
    perplexity_batch = torch.exp2(
        (loss_batch * shift_attn_mask).sum(1) / shift_attn_mask.sum(1)
    )
    # Clean
    del logits, labels, attn_mask, shift_attn_mask
    # Return
    return perplexity_batch


def _glue(logits, labels, attn_mask):
    """
    Computes the glue metric shrg
    """
    return None



# =====================
# Internal Metric Hooks
# =====================

def _sketching_info_hook():
    pass



# ==========================
# Internal Metric Processing
# ==========================


def _process_internal_metrics(internal_metrics : List[str], model):
    """
    Processes the internal metrics collected during one run of model evaluation.
    
    Args:
        internal_metrics (List[str]):
            A list of internal metrics to process.
        model (torch.nn.module):
            The model on which the internal metrics were computed.
    """

    logger.debug(f"Starting _process_internal_metrics: {internal_metrics=}, {model=}", 2, end="\n")
    output = []
    for internal_metric in internal_metrics:
        output.append((internal_metric, INTERNAL_METRIC_DICT[internal_metric][1](model=model)))
    logger.debug(f"Successful _process_internal_metrics", 2, end="\n")
    return output


def _sketching_info_process(model):
    """
    Finds and processes the sketching_info_dicts from an evaluated model.

    Args:
        model (torch.nn.module):
            The model on which the sketching_info internal metric was computed. 
    """

    logger.debug(f"Starting _sketching_info_process: {model=}", 2, end="\n")
    # Setup summary dict
    processed_sketching_info = {}
    # Find the modules that contain sketching_info_dicts
    for name, module in model.named_modules():
        if hasattr(module, 'sketching_info_dicts'):
            logger.trace(f"{name} has attr sketching_info_dicts")
            # Get layer
            layer = [int(val) for val in name.split('.') if val.isdigit()][0]
            # Setup process dict
            processed_dict = my_dict()
            processed_dict['batches'] = len(module.sketching_info_dicts)
            # For each key and val pair in each sketching_info_dict
            for sketching_info_dict in module.sketching_info_dicts:
                for key, val in sketching_info_dict.items():
                    # If a str, return dict with count of how often each str appears
                    if isinstance(val, str):
                        if key in processed_dict:
                            if val in processed_dict[key]:
                                processed_dict[key][val] += 1
                            else:
                                processed_dict[key][val] = 1
                        else:
                            processed_dict[key] = {val: 1}
                    # If a int, return (mean, dict with count of how often each appears)
                    elif isinstance(val, int):
                        if key in processed_dict:
                            old_count = sum(processed_dict[key][1].values())
                            if val in processed_dict[key][1]:
                                processed_dict[key][1][val] += 1
                            else:
                                processed_dict[key][1][val] = 1
                            processed_dict[key][0] = ((processed_dict[key][0]*old_count) + val)/(old_count+1)
                        else:
                            processed_dict[key] = [val, {val: 1}]
                    # If a numpy array, stack by first dimension
                    elif isinstance(val, np.ndarray):
                        if key in processed_dict:
                            logger.trace(f"{key}: {processed_dict[key].shape=}, {val.shape=}")
                            processed_dict[key] = np.concatenate((processed_dict[key], val), axis=0) # np.stack([*processed_dict[key], val])
                        else:
                            processed_dict[key] = val # np.stack([val])
                    # Otherwise just add all results to a list
                    else:
                        if key in processed_dict:
                            processed_dict[key].append(val)
                        else:
                            processed_dict[key] = [val]
            # Log saving
            processed_sketching_info[layer] = processed_dict
            logger.debug(f"Processed sketching_info_dict for {layer=}")
            logger.trace(f"processed_sketching_info[{layer}]={processed_sketching_info[layer]}")
            # Remove existing diagnostics
            delattr(module, 'sketching_info_dicts')
            logger.trace(f"Removed sketching_info_dicts attr from {module}: {hasattr(module, 'sketching_info_dicts')=}")
    logger.debug(f"Successful _sketching_info_process", 2, end="\n")
    return processed_sketching_info


def _collect_internal_metrics(internal_metrics : List[str], runs : int, internal_metrics_dict : Dict):
    """
    Collects and processes the internal metrics gathered for one run of model evaluation.
    
    Args:
        internal_metrics (List[str]):
            A list of internal metrics to process.
        runs (int):
            Number of runs.
        internal_metrics_dict (Dict):
            A dictionary mapping each run to a dictionary of internal_metrics and their results.
    """

    logger.debug(f"Starting _collect_internal_metrics: {internal_metrics=}, {runs=}", 2, end="\n")
    # Assert runs match
    assert runs == max(internal_metrics_dict) + 1, f"Number of runs recorded for internal metrics {max(internal_metrics_dict) + 1} is not equal to number of runs from config {runs}."
    output = []
    for internal_metric in internal_metrics:
        dicts = dict()
        for run, run_dict in internal_metrics_dict.items():
            dicts[run] = run_dict[internal_metric]
        output.append((internal_metric, INTERNAL_METRIC_DICT[internal_metric][2](dicts=dicts)))
    logger.debug(f"Successful _collect_internal_metrics", 2, end="\n")
    return output


def _sketching_info_collect(dicts : List[Dict]):
    """
    Collect and processes the processed _sketching_info_dicts from each run of an evaluated model.

    Args:
        dicts (List[Dict]):
            List of processed _sketching_info_dicts. 
    """

    logger.debug(f"Starting _sketching_info_collect", 2, end="\n")
    # Setup summary dict
    sketching_info_summary = {}
    for run, run_dict in dicts.items():
        for layer, layer_dict in run_dict.items():
            if not layer in sketching_info_summary:
                sketching_info_summary[layer] = dict()
            for key, val in layer_dict.items():
                if key in sketching_info_summary[layer]:
                    sketching_info_summary[layer][key][run] = val
                else:
                    sketching_info_summary[layer][key] = dict()
                    sketching_info_summary[layer][key][run] = val
    logger.debug(f"Successful _sketching_info_collect", 2, end="\n")
    return sketching_info_summary


# ==========
# Management
# ==========

METRIC_DICT = {
    'cross_entropy_loss': _cross_entropy_loss,
    'perplexity': _perplexity,
    'glue': _glue
}

INTERNAL_METRIC_DICT = {
    'sketching_info': (_sketching_info_hook, _sketching_info_process, _sketching_info_collect),
}
