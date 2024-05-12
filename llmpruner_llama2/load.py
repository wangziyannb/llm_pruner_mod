import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from LLMPruner.peft import PeftModel

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
torch_version = int(torch.__version__.split('.')[1])

def load(model_type: str = 'pruneLLM', base_model: str = 'llama2-7b', ckpt: str = '', lora_ckpt: str = ''):
    if model_type == 'pretrain':
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            low_cpu_mem_usage=True if torch_version >= 9 else False
        )
    elif model_type == 'pruneLLM':
        pruned_dict = torch.load(ckpt, map_location='cpu')
        model = pruned_dict['model']
    elif model_type == 'tune_prune_LLM':
        pruned_dict = torch.load(ckpt, map_location='cpu')
        model = pruned_dict['model']
        model = PeftModel.from_pretrained(
            model,
            lora_ckpt,
            torch_dtype=torch.float16,
        )
    else:
        raise NotImplementedError

    if device == "cuda":
        model.half()
        model = model.cuda()

    # # unwind broken decapoda-research config
    # model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    # model.config.bos_token_id = 1
    # model.config.eos_token_id = 2
    return model
