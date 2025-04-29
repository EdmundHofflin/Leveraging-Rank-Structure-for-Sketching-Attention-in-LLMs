import sys

from loguru import logger
from tqdm import tqdm

from typing import Union

import numpy as np
import torch as torch

from datasets import Dataset
from tokenizers import Tokenizer


# =================
# Dataset Functions
# =================

def _shuffle_and_subsample(data : Union[list, torch.tensor], dataset_size : Union[int, float], rng_generator : Union[np.random.Generator, torch.Generator]):
    """
    Shuffles and subsamples data of various types.
    
    Args:
        data (Union[list, torch.tensor]):
            Data to be shuffled and subsampled.
        dataset_size (Union[int, float]):
            Represents the desired size of the dataset. An integer represents the maximum size; a float represents the proportion of the total.
        rng_generator (Union[np.random.Generator, torch.Generator])
            A NumPy generator or PyTorch generator for random number sampling.
    """
    
    logger.debug(f"Starting _shuffle_and_subsample: {data=}, {len(data)=}, {dataset_size=}, {rng_generator=}.")
    # Compute size
    size = min(dataset_size, len(data)) if isinstance(dataset_size, int) else max(1, int(dataset_size * len(data)))
    if isinstance(data, np.ndarray) or isinstance(data, list):
        # Shuffle
        rng_generator.shuffle(data)
        # Return subsample
        return data[:size]
    elif torch.is_tensor(data):
        # Get shuffle indices
        rand_idx = torch.randperm(len(data), generator=rng_generator)
        # Output
        output = data[rand_idx[:size]]
        logger.debug("Successful _shuffle_and_subsample.")
        return output
    else:
        raise RuntimeError(f"Tokenised data has unexpected type: {type(data)=}. Expected list, np.ndarray, or torch.tensor.")


def _tokenise(data : Dataset, tokeniser : Tokenizer, context_length : int, prnt):
    r"""
    Tokenises the data, only accepting those sequences that exceed the context length.
    
    Args:
        data (Dataset):
            Huggingface dataset.
        tokeniser (Tokenizer):
            Huggingface tokeniser.
        context_length (int):
            context length.
        prnt (function):
            The verbosity_printing function set according to args.verbosity.
    """

    logger.debug(f"Starting _tokenise: {data=}, {tokeniser=}, {context_length=}.")
    # Setup
    encoded_texts = []
    pbar = tqdm(data, leave=False, dynamic_ncols=True, file=sys.stdout)
    # Iterate over data
    for i, data_i in enumerate(pbar):
        encoded_text = tokeniser.encode(data_i['context'], return_tensors='pt', truncation=True)
        pbar.set_description(f"Tokenising: context_length: {len(encoded_text[0])}, n_data: {len(encoded_texts)}")
        # Discard encoded text that is too short
        if len(encoded_text[0]) < context_length:
            continue
        encoded_texts.append(encoded_text)
    # Output
    output = torch.cat(encoded_texts, dim=0)
    logger.debug(f"Successful _tokenise: context_length={len(encoded_text[0])}, n_data={len(encoded_texts)}.")
    prnt(f"Successfully tokenised data: context_length={len(encoded_text[0])}, n_data={len(encoded_texts)}.", 2, end="\n")
    return output