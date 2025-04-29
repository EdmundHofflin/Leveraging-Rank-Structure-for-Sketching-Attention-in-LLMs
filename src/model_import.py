import numpy as np
import torch as torch

from modules.configurator import Configurator

@torch.no_grad()
def main():
    # ================
    # Config and Setup
    # ================

    # Parse and process configs
    parser = Configurator.get_config_parser()
    cfg = parser.parse_args()
    mdls = Configurator(cfg) 

    # # Load model and setup
    # mdls.model.load_model(data_path=mdls.inout.cfg.data_path, trust_remote_code=mdls.inout.cfg.trust_remote_code, force_download=mdls.inout.cfg.force_download, prnt=mdls.inout.prnt)
    # mdls.model.model.to(device=mdls.system.cfg.device, dtype=mdls.system.cfg.dtype)
    # mdls.model.model.eval()

    # print(mdls.model.model)
    # print(mdls.model.model.config)

    # mdls.model.patch_model(sketcher=mdls.sketching.sketchers[0], rng_generator=mdls.system.torch_rng, prnt=mdls.inout.prnt)

    # ===================================
    # Setup Model
    # ===================================

    # Load model and setup
    mdls.model.load_model(data_path=mdls.inout.cfg.data_path, trust_remote_code=mdls.inout.cfg.trust_remote_code, force_download=mdls.inout.cfg.force_download, prnt=mdls.inout.prnt)
    mdls.model.setup_model(device=mdls.system.cfg.device, dtype=mdls.system.cfg.dtype)

    # Print
    print(mdls.model.model)
    print(mdls.model.model.config)

    val = input("Enter 'k' to kill:")
    if val == 'k':
        return None

    # ===========================
    # Setup Tokeniser and Dataset
    # ===========================

    # Load tokeniser from local files or Huggingface
    mdls.model.load_tokeniser(data_path=mdls.inout.cfg.data_path, trust_remote_code=mdls.inout.cfg.trust_remote_code, force_download=mdls.inout.cfg.force_download, prnt=mdls.inout.prnt)
    # Set tokeniser max context length
    mdls.model.tokeniser.model_max_length = mdls.model.cfg.context_length

    # Load dataset from local files or Huggingface
    mdls.dataset.load(data_path=mdls.inout.cfg.data_path, force_download=mdls.inout.cfg.force_download, prnt=mdls.inout.prnt)

    # Load tokeniser from local files or Huggingface
    mdls.dataset.tokenise(mdls.model.tokeniser, mdls.model.cfg.context_length, prnt=mdls.inout.prnt)
    
    # ==================
    # Test Functionality
    # ==================

    print("Standard Model")

    # Subsample data
    encoded_texts = mdls.dataset.shuffle_and_subsample(mdls.system.torch_rng)
    # Evaluate model
    btch_size, eval_output = mdls.eval.evaluate_model_on_data(tokenised_data=encoded_texts, attn_masks=None, model=mdls.model.model, device=mdls.system.cfg.device) # TODO: Set attention mask from data
    print(f"Evaluation Successful at Batch_Size={btch_size}:")
    if btch_size is not None:
        for metric in mdls.eval.cfg.metrics:
            print(f"{metric}: {np.nanmean(np.array(eval_output[metric]))}")
    else:
        print("Evaluation failed")
        for bs in eval_output:
            print(f"{bs}: {eval_output[bs]}")

    # =============
    # Test Patching
    # =============

    print("Patched Model")

    for sketcher in mdls.sketching.sketchers:
        print(f"Sketcher: {sketcher}")

        # Load model and setup
        mdls.model.load_model(data_path=mdls.inout.cfg.data_path, trust_remote_code=mdls.inout.cfg.trust_remote_code, force_download=mdls.inout.cfg.force_download, prnt=mdls.inout.prnt)
        mdls.model.setup_model(device=mdls.system.cfg.device, dtype=mdls.system.cfg.dtype)
        mdls.model.patch_model(sketcher=sketcher, rng_generator=mdls.system.torch_rng, prnt=mdls.inout.prnt)

        # Subsample data
        encoded_texts = mdls.dataset.shuffle_and_subsample(mdls.system.torch_rng)
        # Evaluate model
        btch_size, eval_output = mdls.eval.evaluate_model_on_data(tokenised_data=encoded_texts, attn_masks=None, model=mdls.model.model, device=mdls.system.cfg.device) # TODO: Set attention mask from data
        print(f"Evaluation Successful at Batch_Size={btch_size}:")
        if btch_size is not None:
            for metric in mdls.eval.cfg.metrics:
                print(f"{metric}: {np.nanmean(np.array(eval_output[metric]))}")
        else:
            print("Evaluation failed")
            for bs in eval_output:
                print(f"{bs}: {eval_output[bs]}")
        
        # Clean up
        mdls.model.clear_model()


if __name__ == "__main__":
    main()