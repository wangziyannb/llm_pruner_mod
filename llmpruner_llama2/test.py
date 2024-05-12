import torch
from transformers import AutoTokenizer

from llmpruner_llama2.LLMPruner.datasets.ppl import PPLMetric
from llmpruner_llama2.load import load

if __name__ == '__main__':
    base_model = 'pytorch_model.bin'
    for i in ['results']:
        for j in ['0_llmpruner_llama-2-7b_0.2', '0_llmpruner_llama-2-7b_0.4', '0_llmpruner_llama-2-7b_0.6',
                  '1_llmpruner_llama-2-7b_0.2', '1_llmpruner_llama-2-7b_0.4', '1_llmpruner_llama-2-7b_0.6']:
            model = load("pruneLLM", ckpt=i + '/' + j + '/' + base_model)
            tokenizer = AutoTokenizer.from_pretrained('llama2-7b')
            model.eval()
            ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], 128, device='cuda')
            print(f"pruneLLM from {i + '/' + j}", " PPL after pruning: {}".format(ppl))
            del model

            model = load("tune_prune_LLM", ckpt=i + '/' + j + '/' + base_model, lora_ckpt=i + '/' + j)
            model.eval()
            ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], 128, device='cuda')
            print(f"tune_prune_LLM from {i + '/' + j}", " PPL after pruning: {}".format(ppl))
            del model
