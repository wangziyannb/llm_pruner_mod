#!/bin/bash
export PYTHONPATH='.'

base_model=$1 # e.g., decapoda-research/llama-7b-hf
prune_ckpt=$2
prune_id="${prune_ckpt##*/}"
#python lm-evaluation-harness/main.py --model hf-causal-experimental --model_args checkpoint=$prune_ckpt/pytorch_model.bin,config_pretrained=$base_model --tasks  --device cuda:0 --output_path results/${prune_id}1.json --no_cache
#python lm-evaluation-harness/main.py --batch_size auto --model hf-causal-experimental --model_args checkpoint=$prune_ckpt/pytorch_model.bin,config_pretrained=$base_model --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --output_path results/${prune_id}.json --no_cache
python lm-evaluation-harness/main.py --batch_size auto --model hf-causal-experimental --model_args checkpoint=$prune_ckpt/pytorch_model.bin,config_pretrained=$base_model --tasks arc_easy,arc_challenge --device cuda:0 --output_path results/${prune_id}.json --no_cache