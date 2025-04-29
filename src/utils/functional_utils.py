from loguru import logger

from typing import Union, Optional, List
from jsonargparse.typing import PositiveInt, ClosedUnitInterval

import numpy as np
import torch

from utils.general_utils import my_dict


# =========
# Sketching
# =========

def _sketching(A : Union[np.ndarray, torch.Tensor], B : Union[np.ndarray, torch.Tensor], variant : str, sample_proportion : float, replacement : bool, normalise : bool, leverage_equation : str, distribution_sampling : Optional[Union[str, ClosedUnitInterval]], linear_combination : PositiveInt, np_rng : Optional[np.random.Generator], torch_rng : Optional[torch.Generator]):
    """
    Sketches the matrix multiplication AB using a set variant and sample proportion.

    Args:
        A (np.ndarray or torch.Tensor):
            Left matrix.
        B (np.ndarray or torch.Tensor):
            Right matrix.
        variant (str):
            The variant for sketching. Options are 'standard', 'importance', 'random', and 'max'.
        sample_proportion (float):
            The proportion of rows/columns to sample when sketching.
        replacement (bool):
            Whether to sample with or without replacement.
        normalise (bool):
            Whether to normalise the rows/cols by their probability.
        leverage_equation (str):
            The matrices used to compute the leverage score.
        distribution_sampling (Optional[Union[str, ClosedUnitInterval]]):
            How many of the most recent rows/cols in history are used for sketching. Can either be a str representing a function of the form 'lambda x: eval(distribution_sampling)', or a float representing the proportion. If not set or None, defaults to 0.
        linear_combination (PositiveInt):
            How many rows/cols to sum together when forming the rank-1 matrices used for sketching.
        np_rng (torch.Generator)
            A NumPy Generator for random number sampling.
        torch_rng (torch.Generator)
            A Torch Generator for random number sampling.
    """

    # Type check matrices to determine numpy or torch
    if isinstance(A, np.ndarray) and isinstance(B, np.ndarray):
        return _np_sketching(A, B, variant=variant, sample_proportion=sample_proportion, replacement=replacement, normalise=normalise, leverage_equation=leverage_equation, distribution_sampling=distribution_sampling, linear_combination=linear_combination, np_rng=np_rng)
    elif isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return _torch_sketching(A, B, variant=variant, sample_proportion=sample_proportion, replacement=replacement, normalise=normalise, leverage_equation=leverage_equation, torch_rng=torch_rng)
    else:
        raise RuntimeError(f"Function sketching applied to matrices A and B with types {type(A)} and {type(B)}, respectively. The type of these matrices must be np.ndarray or torch.Tensor, and must match.")


def _torch_sketching(A : torch.Tensor, B : torch.Tensor, variant : str, sample_proportion : float, replacement : bool, normalise : bool, leverage_equation : str, distribution_sampling : Optional[Union[str, ClosedUnitInterval]], linear_combination : PositiveInt, rng_generator : Optional[torch.Generator], sketching_info : bool):
    """
    Batch sketches the matrix multiplication AB using a set variant and sample proportion.

    Args:
        A (torch.Tensor):
            Left matrix.
        B (torch.Tensor):
            Right matrix.
        variant (str):
            The variant for sketching. Options are 'standard', 'importance', 'random', and 'max'.
        sample_proportion (float):
            The proportion of rows/columns to sample when sketching.
        replacement (bool):
            Whether to sample with or without replacement.
        normalise (bool):
            Whether to normalise the rows/cols by their probability.
        leverage_equation (List[str]):
            The matrices used to compute the leverage score.
        distribution_sampling (Optional[Union[str, ClosedUnitInterval]]):
            Either a float in [0,1] or a str representing a function of the form 'lambda idx, shared_dimension: eval(distribution_sampling)', where idx is the index of the row/col (0 is oldest) and shared_dimension is the dimension shared between A and B. Both define the proportion of the most recent rows/cols to always sample, regardless of the sketching variant and leverage scores. The former type does so as a proportion, while the latter does so as by returning a boolean for each row/col. If None, no rows/cols are forced to be sampled. Note that these forced samples are included in the total sample proportion and so can't be larger. An error is raised if this is not the case.
        linear_combination (PositiveInt):
            How many rows/cols to sum together when forming the rank-1 matrices used for sketching.
        rng_generator (Optional[torch.Generator]):
            A torch Generator for random number sampling.
        sketching_info (bool):
            Whether to return diagnostic information about the sketch.
    """

    logger.trace(f"Starting _torch_sketching({variant=}): {A.shape=}, {B.shape=}")

    # Ensure dimensions match
    assert A.shape[-1] == B.shape[-2]
    # Ensure no NaNs
    assert torch.all(torch.isnan(A) == False), f"NaN A: {A.shape=}, {A[torch.isnan(A)].shape=}"
    assert torch.all(torch.isnan(B) == False), f"NaN B: {B.shape=}, {B[torch.isnan(B)].shape=}"

    # Setup info_dict
    if sketching_info:
        info_dict = my_dict()
    # Number of dimensions
    num_batch_dims = len(A.shape) - 2

    # Standard variant
    if variant == 'standard':
        if sketching_info:
            temp_A = torch.linalg.vector_norm(A, ord=2, dim=-2)
            temp_B = torch.linalg.vector_norm(B, ord=2, dim=-1)
            info_dict['col_norms_A'] = torch.pow(temp_A, 2).numpy(force=True).astype(np.float16)
            info_dict['row_norms_B'] = torch.pow(temp_B, 2).numpy(force=True).astype(np.float16)
            # info_dict['orthogonality_A'] = orthogonality(A, col_norms_X=temp_A, num_batch_dims=num_batch_dims).numpy(force=True).astype(np.float16)
            # info_dict['orthogonality_B'] = orthogonality(torch.transpose(B, -1, -2), col_norms_X=temp_B, num_batch_dims=num_batch_dims).numpy(force=True).astype(np.float16)
            info_dict['stable_rank_A'] = stable_rank(A).numpy(force=True).astype(np.float16)
            info_dict['stable_rank_B'] = stable_rank(B).numpy(force=True).astype(np.float16)

            output = torch.matmul(A,B)
            logger.trace(f"Successful _torch_sketching({variant=}): {output.shape=}, {info_dict.keys()=}")
            return output, info_dict
        else:
            output = torch.matmul(A,B)
            logger.trace(f"Successful _torch_sketching({variant=}): {output.shape=}")
            return output
    # Sketching variants
    elif variant in ['random', 'importance', 'max']:
        # Total number of samples
        n_total_samples = min(max(1, int(np.round(sample_proportion*A.shape[-1]))), A.shape[-1]) # Ensure it is neither 0 nor over shared dimension
        if sketching_info:
            info_dict['n_total_samples'] = n_total_samples

        # Compute distribution sampling indices
        if distribution_sampling is not None:
            # If a float, just take most recent rows/cols
            if isinstance(distribution_sampling, float):
                distr_bools = torch.zeros(A.shape[-1], dtype=torch.bool)
                n_distribution_samples = int(np.round(distribution_sampling*A.shape[-1]))
                if n_distribution_samples == 0:
                    # raise RuntimeWarning(f"Parameter {distribution_sampling=} is invalid: given this inputs and the shared dimension of the matrices being {A.shape[-1]}, the number of forced samples {n_distribution_samples} is 0. It will be increased to 1.")
                    n_distribution_samples = 1
                distr_bools[-n_distribution_samples:] = True
            # If a str, evaluate the function for each row/col
            elif isinstance(distribution_sampling, str):
                distr_bools = torch.tensor(list(map(lambda x: (lambda idx, shared_dimension: eval(distribution_sampling))(x, A.shape[-1]), range(A.shape[-1]))), dtype=torch.bool)
                n_distribution_samples = sum(distr_bools).item()
            # If invalid type, raise error
            else:
                raise RuntimeError(f"Parameter {distribution_sampling=} is invalid type {type(distribution_sampling)=}: must be either a float in [0,1] or a str representing a function of the form 'lambda idx, shared_dimension: eval(distribution_sampling)'.")
        else:
            # If no distribution sampling, set distr_bools to all False
            distr_bools = torch.zeros(A.shape[-1], dtype=torch.bool)
            n_distribution_samples = 0
        # Determine number of samples via leverage scores
        n_leverage_samples = n_total_samples - n_distribution_samples
        if n_leverage_samples <= 0:
                raise RuntimeError(f"Parameters {sample_proportion=} and {distribution_sampling=} are invalid: given these inputs and the shared dimension of the matrices being {A.shape[-1]}, the number of forced samples {n_distribution_samples} equals or exceeds the total number of samples {n_total_samples}.")
        if sketching_info:
            info_dict['n_distribution_samples'] = n_distribution_samples
            info_dict['n_leverage_samples'] = n_leverage_samples
        assert torch.all(torch.isnan(distr_bools) == False), f"NaN distr_bools: {distr_bools.shape=}, {distr_bools[torch.isnan(distr_bools)].shape=}"

        # Compute leverage scores
        if variant == 'random':
            # Random corresponds to equal leverage scores
            unnormalised_leverage_scores = torch.ones((*A.shape[:-2], A.shape[-1]))
            if sketching_info:
                info_dict['col_norms_A'] = torch.pow(torch.linalg.vector_norm(A, ord=2, dim=-2), 2).numpy(force=True).astype(np.float16)
                info_dict['row_norms_B'] = torch.pow(torch.linalg.vector_norm(B, ord=2, dim=-1), 2).numpy(force=True).astype(np.float16)
        elif variant in ['importance', 'max']:
            # Leverage score according to setting
            if leverage_equation == "LR":
                col_norms_A = torch.pow(torch.linalg.vector_norm(A, ord=2, dim=-2), 2)
                row_norms_B = torch.pow(torch.linalg.vector_norm(B, ord=2, dim=-1), 2)
                unnormalised_leverage_scores = col_norms_A * row_norms_B
                if sketching_info:
                    info_dict['col_norms_A'] = col_norms_A.numpy(force=True).astype(np.float16)
                    info_dict['row_norms_B'] = row_norms_B.numpy(force=True).astype(np.float16)
                del col_norms_A, row_norms_B
            elif leverage_equation == "L+R":
                col_norms_A = torch.pow(torch.linalg.vector_norm(A, ord=2, dim=-2), 2)
                row_norms_B = torch.pow(torch.linalg.vector_norm(B, ord=2, dim=-1), 2)
                unnormalised_leverage_scores = col_norms_A + row_norms_B
                if sketching_info:
                    info_dict['col_norms_A'] = col_norms_A.numpy(force=True).astype(np.float16)
                    info_dict['row_norms_B'] = row_norms_B.numpy(force=True).astype(np.float16)
                del col_norms_A, row_norms_B
            elif leverage_equation == "L":
                unnormalised_leverage_scores = torch.linalg.vector_norm(A, ord=2, dim=-2)
                if sketching_info:
                    info_dict['col_norms_A'] = torch.pow(unnormalised_leverage_scores, 2).numpy(force=True).astype(np.float16)
                    info_dict['row_norms_B'] = torch.pow(torch.linalg.vector_norm(B, ord=2, dim=-1), 2).numpy(force=True).astype(np.float16)
            elif leverage_equation == "R":
                unnormalised_leverage_scores = torch.linalg.vector_norm(B, ord=2, dim=-1)
                if sketching_info:
                    info_dict['col_norms_A'] = torch.pow(torch.linalg.vector_norm(A, ord=2, dim=-2), 2).numpy(force=True).astype(np.float16)
                    info_dict['row_norms_B'] = torch.pow(unnormalised_leverage_scores, 2).numpy(force=True).astype(np.float16)
            else:
                raise RuntimeError(f"This should be impossible to reach. Input variable leverage_equation set to {leverage_equation} when only options accept by parser are 'L', 'R', 'LR' and L+R'.")
        # Normalise leverage scores
        if num_batch_dims == 0:
            leverage_scores = unnormalised_leverage_scores / unnormalised_leverage_scores.sum()
        else:
            leverage_scores = _batch_torch_func(lambda x, y: x[..., :] / y, num_batch_dims=num_batch_dims, randomness='error')(unnormalised_leverage_scores[..., :], unnormalised_leverage_scores.sum(dim=-1))
        del unnormalised_leverage_scores
        leverage_scores[leverage_scores <= 0] = 1e-9 # Ensure all values are positive
        if sketching_info:
            info_dict['leverage_scores'] = leverage_scores.numpy(force=True).astype(np.float16)
        assert torch.all(torch.isnan(leverage_scores) == False), f"NaN leverage_scores: {leverage_scores.shape=}, {leverage_scores[torch.isnan(leverage_scores)].shape=}"

        # Normalise matrices using leverage scores
        if normalise:
            normalised_A = (1/np.sqrt(n_total_samples)) * _batch_torch_func(lambda x, y: x[..., :, :] / y[..., :], num_batch_dims=num_batch_dims, randomness='error')(A, torch.sqrt(leverage_scores))
            normalised_B = (1/np.sqrt(n_total_samples)) * _batch_torch_func(lambda x, y: x[..., :, :] / y[..., :], num_batch_dims=num_batch_dims, randomness='error')(B.mT, torch.sqrt(leverage_scores)).mT
        else:
            normalised_A, normalised_B = A, B

        # Remove indices used in distribution sampling from leverage scores
        no_replacement_leverage_scores = leverage_scores[..., ~distr_bools] # _batch_torch_func(lambda x: x[~distr_bools], num_batch_dims=num_batch_dims, randomness='error')(leverage_scores)
        assert torch.all(torch.isnan(no_replacement_leverage_scores) == False), f"NaN no_replacement_leverage_scores: {no_replacement_leverage_scores.shape=}, {no_replacement_leverage_scores[torch.isnan(no_replacement_leverage_scores)].shape=}"
        del leverage_scores
        # Compute leverage score indices
        if variant in ['random', 'importance']:
            leverage_idxs = _batch_torch_func(lambda x: torch.multinomial(x, num_samples=n_leverage_samples, replacement=replacement, generator=rng_generator), num_batch_dims=num_batch_dims, randomness='different')(no_replacement_leverage_scores)
        else:
            _, leverage_idxs = _batch_torch_func(lambda x: torch.topk(x[..., :], k=n_leverage_samples), num_batch_dims=num_batch_dims, randomness='error')(no_replacement_leverage_scores)
        del no_replacement_leverage_scores
        assert torch.all(torch.isnan(leverage_idxs) == False), f"NaN leverage_idxs: {leverage_idxs.shape=}, {leverage_idxs[torch.isnan(leverage_idxs)].shape=}"
        if sketching_info:
            info_dict['leverage_idxs'] = leverage_idxs.numpy(force=True).astype(np.float16)

        # Get rows of A and cols of B using idxs
        sampled_A = _batch_torch_func(lambda x, y: torch.index_select(x[..., :, :], dim=1, index=y[..., :]), num_batch_dims=num_batch_dims, randomness='error')(normalised_A, leverage_idxs)
        sampled_B = _batch_torch_func(lambda x, y: torch.index_select(x[..., :, :], dim=0, index=y[..., :]), num_batch_dims=num_batch_dims, randomness='error')(normalised_B, leverage_idxs)
        del leverage_idxs
        # if sketching_info:
        #     info_dict['stable_rank_sampled_A'] = stable_rank(sampled_A).numpy(force=True).astype(np.float16)
        #     info_dict['stable_rank_sampled_B'] = stable_rank(sampled_B).numpy(force=True).astype(np.float16)
        #     info_dict['orthogonality_sampled_A'] = orthogonality(sampled_A, col_norms_X=info_dict['col_norms_A']).numpy(force=True).astype(np.float16) # FIXME: col_norms needs to be re-indexed to agree with sampled_A
        #     info_dict['orthogonality_sampled_B'] = orthogonality(sampled_B, col_norms_X=info_dict['row_norms_B']).numpy(force=True).astype(np.float16) # FIXME: col_norms needs to be re-indexed to agree with sampled_B

        # Compute output
        output = torch.matmul(sampled_A, sampled_B)
        del sampled_A, sampled_B
        if distribution_sampling is not None:
            output += torch.matmul(normalised_A[..., :, distr_bools], normalised_B[..., distr_bools, :])
        del distr_bools, normalised_A, normalised_B
        if sketching_info:
            true = torch.matmul(A,B)
            logger.trace(f"sketching end: {true.shape=}, {output.shape=}")
            info_dict['abs_error'] = torch.linalg.matrix_norm(output - true, ord='fro').numpy(force=True).astype(np.float16)
            info_dict['rel_error'] = info_dict['abs_error'] / torch.linalg.matrix_norm(true, ord='fro').numpy(force=True).astype(np.float16)
            info_dict['paper_error'] = info_dict['abs_error'] / torch.pow(torch.linalg.matrix_norm(A, ord='fro') * torch.linalg.matrix_norm(B, ord='fro'), 2).numpy(force=True).astype(np.float16)
            
            logger.trace(f"Successful _torch_sketching({variant=}): {output.shape=}, {info_dict.keys()=}")
            return output, info_dict
        else:
            logger.trace(f"Successful _torch_sketching({variant=}): {output.shape=}")
            return output
    else:
        raise RuntimeError(f"This should be impossible to reach! Parameter --sketching_variant is set to invalid value {variant}. Options are 'standard', 'importance', 'random', and 'max'.")


def _np_sketching(A : np.ndarray, B : np.ndarray, variant : str, sample_proportion : float, replacement : bool, normalise : bool, leverage_equation : str, distribution_sampling : Optional[Union[str, ClosedUnitInterval]], linear_combination : PositiveInt, rng_generator : Optional[np.random.Generator]):
    """
    Sketches the matrix multiplication AB using a set variant and sample proportion.

    Args:
        A (np.ndarray or torch.Tensor):
            Left matrix.
        B (np.ndarray or torch.Tensor):
            Right matrix.
        variant (str):
            The variant for sketching. Options are 'standard', 'importance', 'random', and 'max'.
        sample_proportion (float):
            The proportion of rows/columns to sample when sketching.
        replacement (bool):
            Whether to sample with or without replacement.
        normalise (bool):
            Whether to normalise the rows/cols by their probability. If None, only normalises for importance.
        leverage_equation (str):
            The matrices used to compute the leverage score.
        distribution_sampling (Optional[Union[str, ClosedUnitInterval]]):
            Either a float in [0,1] or a str representing a function of the form 'lambda idx, shared_dimension: eval(distribution_sampling)', where idx is the index of the row/col (0 is oldest) and shared_dimension is the dimension shared between A and B. Both define the proportion of the most recent rows/cols to always sample, regardless of the sketching variant and leverage scores. The former type does so as a proportion, while the latter does so as by returning a boolean for each row/col. If None, no rows/cols are forced to be sampled. Note that these forced samples are included in the total sample proportion and so can't be larger. An error is raised if this is not the case.
        linear_combination (PositiveInt):
            How many rows/cols to sum together when forming the rank-1 matrices used for sketching.
        rng_generator (torch.Generator)
            A NumPy Generator for random number sampling.
    """

    # Ensure dimensions match
    assert A.shape[-1] == B.shape[-2]

    # Standard variant
    if variant == 'standard':
        return np.matmul(A, B)
    elif variant in ['random', 'importance', 'max']:
        raise RuntimeError(f"Batched sketching for NumPy arrays not yet implemented.") # TODO
    else:
        raise RuntimeError(f"This should be impossible to reach! Parameter --sketching_variant is set to invalid value {variant}. Options are 'standard', 'importance', 'random', and 'max'.")



# ======================
# Batching Functionality
# ======================

def _batch_torch_func(func, num_batch_dims : int, randomness : str = 'error'):
    """
    Batches a torch function

    Args:
        func (function):
            The function that needs to be called across multiple batches.
        num_batch_dims (list[int]):
            The list of dimensions that are to be batched.
        randomness (str, default: 'error')
            How torch.vmap handles randomness across batches. See https://pytorch.org/docs/stable/generated/torch.vmap.html for details.
    """

    for _ in range(num_batch_dims):
        func = torch.vmap(func, in_dims=0, randomness=randomness)
    return func



# =======================
# Other Utility Functions
# =======================

def stable_rank(X : Union[np.ndarray, torch.Tensor]):
    """
    Computes the stable rank of a matrix.

    Args:
        X (np.ndarray or torch.Tensor):
            The matrix of which to compute the stable rank.
    """

    logger.trace(f"Starting stable_rank: {X.shape=}")

    if isinstance(X, np.ndarray):
        output = np.pow(np.linalg.norm(X, ord='fro'), 2) / np.pow(np.linalg.norm(X, ord=2), 2)
    elif isinstance(X, torch.Tensor):
        output = torch.pow(torch.linalg.matrix_norm(X, ord='fro'), 2) / torch.pow(torch.linalg.matrix_norm(X, ord=2), 2)
    else:
        raise RuntimeError(f"Function stable_rank applied to matrix X with type {type(X)}. The type of this matrix must be np.ndarray or torch.Tensor.")

    logger.trace(f"Successful stable_rank: {output=}")
    return output


def orthogonality(X : Union[np.ndarray, torch.Tensor], num_batch_dims : int, col_norms_X : Optional[Union[np.ndarray, torch.Tensor]] = None):
    """
    Computes the orthogonality of two matrices.

    Args:
        X (np.ndarray or torch.Tensor):
            Matrix of which to compute the orthogonality.
        num_batch_dims (int):
            The number of batch dimensions in the input matrix X.
        col_norms_X (Optional[Union[np.ndarray, torch.Tensor]]):
            The column norms of the matrix X. If not provided, they are computed.
    """

    logger.trace(f"Starting orthogonality: {X.shape=}, {num_batch_dims=}, {col_norms_X if col_norms_X is None else col_norms_X.shape=}")

    if isinstance(X, np.ndarray):
        prod = np.matmul(np.transpose(X, -1, -2), X)
        if col_norms_X is None:
            col_norms_X = np.linalg.norm(X, ord=2, axis=-2)
        output = np.zeros_like(prod)
        # FIXME: vectorise in same way as torch
        for i in range(X.shape[-1]):
            for j in range(X.shape[-1]):
                if i <= j:
                    output[..., i, j] = prod[..., i, j] / (col_norms_X[..., i] * col_norms_X[..., j])
    elif isinstance(X, torch.Tensor):
        prod = torch.matmul(torch.transpose(X, -1, -2), X)
        if col_norms_X is None:
            col_norms_X = torch.linalg.vector_norm(X, ord=2, dim=-2)
        denom = _batch_torch_func(lambda x: torch.outer(x, x), num_batch_dims=num_batch_dims)(col_norms_X)
        output = torch.triu(prod / denom)
    else:
        raise RuntimeError(f"Function orthogonality applied to matrix X with invalid type={type(X)}. The type must be np.ndarray or torch.Tensor.")

    logger.trace(f"Successful orthogonality: {output.shape=}")
    return output