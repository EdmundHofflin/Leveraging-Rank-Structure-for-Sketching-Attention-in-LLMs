# ====================================
# Managing Files and Config for Models
# ====================================

def _model_manager(model_name : str):
    """
    Uses the model name to import and fetch the model architecture and methods.

    Args:
        model_name (str):
            Name of the model.
    """

    # Get model architecture
    model_dict = MODEL_CONFIGS[model_name]
    # Get model methods
    if model_dict['ARCH'] == 'gpt2':
        from models.gpt2 import patch_attention
    elif model_dict['ARCH'] == 'bert':
        from models.bert import patch_attention
    elif model_dict['ARCH'] == 'opt':
        from models.opt import patch_attention
    elif model_dict['ARCH'] == 'xglm':
        from models.xglm import patch_attention
    elif model_dict['ARCH'] == 'qwen2':
        from models.qwen2 import patch_attention
    elif model_dict['ARCH'] == 'llama':
        from models.llama import patch_attention
    elif model_dict['ARCH'] == 'mistral':
        from models.mistral import patch_attention
    else:
        raise RuntimeError(f"This should be impossible to reach! Parameter --model is set to invalid value {model_name}. Use --help to see supported models.")
    model_dict['patch_attention'] = patch_attention
    
    return model_dict


# ============
# Model Config
# ============

MODEL_CONFIGS = {
    # DistilBert: https://huggingface.co/docs/transformers/model_doc/distilbert, https://arxiv.org/pdf/1910.01108
    'distilbert/distilgpt2': { # 88.2M
        'MAX_CONTEXT_LENGTH': 512,
        'NUM_LAYERS': 6,
        'ARCH': 'gpt2',
    },
    # GPT2: https://huggingface.co/openai-community/gpt2, https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
    'openai-community/gpt2': { # 137M
        'MAX_CONTEXT_LENGTH': 1024,
        'NUM_LAYERS': 12,
        'ARCH': 'gpt2',
    },
    # DialoGPT: https://github.com/microsoft/DialoGPT, https://arxiv.org/pdf/1911.00536 # MOVE TO GODEL: https://github.com/microsoft/GODEL
    'microsoft/DialoGPT-small': { # 176M
        'MAX_CONTEXT_LENGTH': 1024,
        'NUM_LAYERS': 12,
        'ARCH': 'gpt2',
    },
    'microsoft/DialoGPT-medium': { # 345M
        'MAX_CONTEXT_LENGTH': 1024,
        'NUM_LAYERS': 24,
        'ARCH': 'gpt2',
    },
    'microsoft/DialoGPT-large': { # 762M
        'MAX_CONTEXT_LENGTH': 1024,
        'NUM_LAYERS': 36,
        'ARCH': 'gpt2',
    },
    # OPT: https://huggingface.co/collections/facebook/opt-66ed00e15599f02966818844, https://arxiv.org/pdf/2205.01068
    'facebook/opt-125M': {
        'MAX_CONTEXT_LENGTH': 768,
        'NUM_LAYERS': 12,
        'ARCH': 'opt',
    },
    'facebook/opt-350M': {
        'MAX_CONTEXT_LENGTH': 1024,
        'NUM_LAYERS': 24,
        'ARCH': 'opt',
    },
    # XGLM: https://huggingface.co/facebook/xglm-564M, https://arxiv.org/pdf/2112.10668
    'facebook/xglm-564M': {
        'MAX_CONTEXT_LENGTH': 2048,
        'NUM_LAYERS': 24,
        'ARCH': 'xglm',
    },
    # QWEN2: https://huggingface.co/collections/Qwen/qwen2-6659360b33528ced941e557f, https://arxiv.org/pdf/2407.10671
    'Qwen/Qwen2-0.5B': {
        'MAX_CONTEXT_LENGTH': 896,
        'NUM_LAYERS': 24,
        'ARCH': 'qwen2',
    },
    # LLaMa: https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf
    'meta-llama/Llama-3.2-1B': {
        'MAX_CONTEXT_LENGTH': 2048, # 128000,
        'NUM_LAYERS': 16,
        'ARCH': 'llama',
    },
    'meta-llama/Llama-3.2-3B': {
        'MAX_CONTEXT_LENGTH': 4096, #128000
        'NUM_LAYERS': 28,
        'ARCH': 'llama',
    },
    # MistralAI:
    'mistralai/Mistral-7B-v0.3': {
        'MAX_CONTEXT_LENGTH': 2048,
        'NUM_LAYERS': 100,
        'ARCH': 'mistral'
    }
    # THDUM: https://huggingface.co/THUDM/chatglm2-6b-32k, https://arxiv.org/pdf/2210.02414
    # 'THUDM/chatglm2-6b-32k': {
    #     'MAX_CONTEXT_LENGTH': None,
    #     'NUM_LAYERS': 28,
    #     'ARCH': 'TODO',
    # }
}