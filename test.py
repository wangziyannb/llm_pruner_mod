import argparse
import json
from pathlib import Path

import torch
import random
import numpy as np
import lm_eval

from LLMPruner.evaluator.ppl import PPLMetric
from config import Config


def main(args):

    set_random_seed(args.seed)
    model_dict = torch.load('/root/llm_pruner_mod/prune_log/$prune_ckpt_path/pytorch_model.bin')
    model = model_dict['model']
    tokenizer = model_dict['tokenizer']
    config = Config(args.config_path)
    c = config.get_config()
    ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], 128, device='cuda')
    if c.evaluation.lm_eval:
        lm_simple_eval(c, model, tokenizer, "result.json", '/root/llm_pruner_mod')


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lm_simple_eval(c, model, tokenizer, result_name, dir):
    wrapped_model = lm_eval.models.huggingface.HFLM(model, tokenizer=tokenizer, batch_size='auto')
    results = lm_eval.simple_evaluate(  # call simple_evaluate
        model=wrapped_model,
        tasks=c.evaluation.lm_eval_options.tasks,
        # tasks=["openbookqa", "arc_easy", "winogrande", "hellaswag", "arc_challenge", "piqa", "boolq"],
        # tasks=["openbookqa"],
        num_fewshot=0,
        log_samples=False,
    )

    def _handle_non_serializable(o):
        if isinstance(o, np.int64) or isinstance(o, np.int32):
            return int(o)
        elif isinstance(o, set):
            return list(o)
        else:
            return str(o)

    path = Path(dir)
    # check if file or 'dir/results.json' exists
    if path.is_file():
        raise FileExistsError(f"File already exists at {path}")
    output_path_file = path.joinpath(f"{result_name}.json")
    if path.suffix in (".json", ".jsonl"):
        output_path_file = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path = path.parent
    else:
        path.mkdir(parents=True, exist_ok=True)
    dumped = json.dumps(
        results, indent=2, default=_handle_non_serializable, ensure_ascii=False
    )
    output_path_file.open("w", encoding="utf-8").write(dumped)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')
    parser.add_argument('--config_path', type=str, help='config', default='/root/llm_pruner_mod/prune_llama2_7b.yml')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    args = parser.parse_args()

    torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    args.torch_version = torch_version
    main(args)
