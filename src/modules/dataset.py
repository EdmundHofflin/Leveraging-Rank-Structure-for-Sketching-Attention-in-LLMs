import os
from abc import abstractmethod

from loguru import logger

from typing import Optional, Union
from jsonargparse.typing import PositiveInt, ClosedUnitInterval
from jsonargparse import ArgumentParser, Namespace

import numpy as np
import torch

from datasets import load_dataset, load_from_disk, concatenate_datasets
from tokenizers import Tokenizer

from modules.default_initialiser import DefaultInitialiser

from utils.dataset_utils import _shuffle_and_subsample, _tokenise


# =======
# Modules
# =======

class Dataset(DefaultInitialiser):
    @staticmethod
    def get_config_parser():
        parser = ArgumentParser()

        # Dataset Arguments
        # -----------------
        dataset_group = parser.add_argument_group('Dataset Options', description="Options that specify the dataset and its properties.")
        # Dataset
        dataset_group.add_argument(
            "--dataset",
            type=str, choices=['LongBench'], default='LongBench',
            help="The HuggingFace dataset on which to evaluate the model.")
        # Size
        dataset_group.add_argument(
            "--size",
            type=Union[ClosedUnitInterval, PositiveInt], default=1.0,
            help="Size of the datasets used to evaluate the model. Can either be a proportion, e.g 0.5 for half the dataset, or an absolute number, e.g. 1000 entries.")
        # Dataset Options
        dataset_group.add_argument(
            "--dataset_options",
            type=dict, default=None,
            help="Additional options to be used by the specific loader of the chosen dataset.")
        return parser
    
    def __init__(self, cfg: Optional[Namespace] = None):
        # Super to initialise self.cfg and get defaults
        super().__init__(cfg=cfg)

        # Parse Config Namespace
        # ----------------------
        # EMPTY


        # Initialise dataset specific module
        if self.cfg.dataset == "LongBench":
            self.dataset_module = LongBench(self.cfg.dataset_options)
        else:
            raise RuntimeError(f"This should be impossible to reach! Parameter --dataset is set to invalid value {self.cfg.dataset}. Use --help to see supported datasets.")
    

    def __str__(self):
        return f"Dataset({self.cfg.as_dict()})"
    

    def load(self, data_path : str, force_download : bool, prnt):
        """
        Loads Huggingface dataset either from local files or Huggingface.
    
        Args:
            data_path (str):
                String path to the data folder.
            trust_remote_code (bool):
                Trust remote code flag for loading model from Huggingface.
            force_download (bool):
                Force the data to be redownloaded.
            prnt (function):
                The verbosity_printing function set according to args.verbosity.
        """

        self.data = self.dataset_module.load(data_path, force_download, prnt)
    

    def tokenise(self, tokeniser : Tokenizer, context_length : int, prnt):
        """
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

        self.tokenised_data = _tokenise(data=self.data, tokeniser=tokeniser, context_length=context_length, prnt=prnt)


    def shuffle_and_subsample(self, rng_generator : Union[np.random.Generator, torch.Generator]):
        """
        Shuffles and subsamples data of various types.
        
        Args:
            rng_generator (Union[np.random.Generator, torch.Generator])
                A NumPy generator or PyTorch generator for random number sampling.
        """

        return _shuffle_and_subsample(data=self.tokenised_data, dataset_size=self.cfg.size, rng_generator=rng_generator)
    


# ========================
# Dataset Specific Modules
# ========================

class Dataset_Specific(DefaultInitialiser):
    @staticmethod
    @abstractmethod
    def load(self, data_path : str, force_download : bool, prnt, *args):       
        pass



class LongBench(Dataset_Specific):
    @staticmethod
    def get_config_parser():
        parser = ArgumentParser()

        # Dataset Arguments
        # -----------------
        dataset_option_group = parser.add_argument_group('Additional Dataset Options for LongBench Dataset', description="Options that specify the Longbench dataset.")
        dataset_option_group.add_argument(
            "--languages",
            type=str, choices=['en', 'ch', 'code'], default=['en'], nargs='+',
            help="The languages to include in the LongBench dataset.")
        return parser
    
    
    def __init__(self, cfg: Optional[Namespace] = None):
        # Super to initialise self.cfg and get defaults
        super().__init__(cfg=cfg)

        # Parse Config Namespace
        # ----------------------
        # EMPTY

    
    def __str__(self):
        return f"LongBench({self.cfg.as_dict()})"
    

    def load(self, data_path : str, force_download : bool, prnt):
        """
        Loads LongBench data either from data/datasets/ or Huggingface.
    
        Args:
            data_path (str):
                String path to the datafolder to load and save datasets.
            force_download (bool):
                Force the data to be redownloaded.
            prnt (function):
                The verbosity_printing function set according to args.verbosity.
        """

        logger.debug(f"Starting LongBench.load: {data_path=}, {force_download=}")            
        # Language tasks of the LongBench dataset
        en_tasks = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en"]
        ch_tasks = ["dureader", "multifieldqa_zh", "vcsum", "lsht", "passage_retrieval_zh"]
        code_tasks = ["lcc", "repobench-p"]
        language_dict = {'en': en_tasks, 'ch': ch_tasks, 'code': code_tasks}

        # Helper to download and save each task individually
        def helper(tsk, lng, frcdwnld):
            # If downloading or file doesn't exist
            if frcdwnld or (not os.path.exists(f"{data_path}/datasets/LongBench/{lng}/{tsk}.hf")):
                if not frcdwnld:
                    logger.debug(f"LongBench task {tsk} in language {lng} is not local, so must download.")
                # Load dataset from Huggingface
                dtst = load_dataset('THUDM/LongBench', f"{tsk}", split='test')
                logger.debug(f"LongBench task {tsk} downloaded.")
                # Save dataset
                dtst.save_to_disk(f"{data_path}/datasets/LongBench/{lng}/{tsk}.hf")
                logger.debug(f"LongBench task {tsk} saved to {data_path}/datasets/LongBench/{lng}/{tsk}.hf.")
            # Otherwise load from disk
            else:
                dtst = load_from_disk(f"{data_path}/datasets/LongBench/{lng}/{tsk}.hf")
                logger.debug(f"LongBench task loaded locally from {data_path}/datasets/LongBench/{lng}/{tsk}.hf.")
            # Return dataset
            return dtst
        
        # Setup list to store dataset of each task loaded
        dataset_ls = []
        # Iterate over languages
        for language in self.cfg.languages:
            logger.debug(f"Loading LongBench language {language}.")
            # Isolate tasks
            tasks = language_dict[language]
            # Iterate over tasks in language
            for task in tasks:
                # Load and store dataset for task
                dtst = helper(task, language, force_download)
                if len(dtst) > 0:
                    dataset_ls.append(dtst)
        
        # Output
        output = concatenate_datasets(dataset_ls)
        logger.debug("Successful LongBench.load.")
        prnt(f"Successfully loaded LongBench dataset.", 2, end="\n")
        return output
