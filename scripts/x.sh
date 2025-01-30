#!/bin/bash
export PYTHONPATH='.'
CUDA_VISIBLE_DEVICES=1 python hf_prune.py --dtype float16 --base_model meta-llama/Llama-2-7b-hf --pruning_ratio 0.25 --device cuda --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 --save_ckpt_log_name '$prune_ckpt_path' --pruner_type taylor --taylor param_first --config_path /root/llm_pruner_mod/prune_llama2_7b.yml
CUDA_VISIBLE_DEVICES=1 python hf_prune.py --dtype float16 --base_model meta-llama/Llama-2-7b-hf --pruning_ratio 0.38 --device cuda --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 --save_ckpt_log_name '$prune_ckpt_path' --pruner_type taylor --taylor param_first --config_path /root/llm_pruner_mod/prune_llama2_7b.yml
CUDA_VISIBLE_DEVICES=1 python hf_prune.py --dtype float16 --base_model meta-llama/Llama-2-7b-hf --pruning_ratio 0.5 --device cuda --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 --save_ckpt_log_name '$prune_ckpt_path' --pruner_type taylor --taylor param_first --config_path /root/llm_pruner_mod/prune_llama2_7b.yml
CUDA_VISIBLE_DEVICES=1 python hf_prune.py --dtype float16 --base_model meta-llama/Llama-2-7b-hf --pruning_ratio 0.6 --device cuda --eval_device cuda --block_wise --block_mlp_layer_start 3 --block_mlp_layer_end 30 --block_attention_layer_start 3 --block_attention_layer_end 30 --save_ckpt_log_name '$prune_ckpt_path' --pruner_type taylor --taylor param_first --config_path /root/llm_pruner_mod/prune_llama2_7b.yml

CUDA_VISIBLE_DEVICES=1 python hf_prune.py --dtype bfloat16 --base_model meta-llama/Llama-2-7b-hf --pruning_ratio 0.25 --device cuda --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 --save_ckpt_log_name '$prune_ckpt_path' --pruner_type taylor --taylor param_first --config_path /root/llm_pruner_mod/prune_llama2_7b.yml
CUDA_VISIBLE_DEVICES=1 python hf_prune.py --dtype bfloat16 --base_model meta-llama/Llama-2-7b-hf --pruning_ratio 0.38 --device cuda --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 --save_ckpt_log_name '$prune_ckpt_path' --pruner_type taylor --taylor param_first --config_path /root/llm_pruner_mod/prune_llama2_7b.yml
CUDA_VISIBLE_DEVICES=1 python hf_prune.py --dtype bfloat16 --base_model meta-llama/Llama-2-7b-hf --pruning_ratio 0.5 --device cuda --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 --save_ckpt_log_name '$prune_ckpt_path' --pruner_type taylor --taylor param_first --config_path /root/llm_pruner_mod/prune_llama2_7b.yml
CUDA_VISIBLE_DEVICES=1 python hf_prune.py --dtype bfloat16 --base_model meta-llama/Llama-2-7b-hf --pruning_ratio 0.6 --device cuda --eval_device cuda --block_wise --block_mlp_layer_start 3 --block_mlp_layer_end 30 --block_attention_layer_start 3 --block_attention_layer_end 30 --save_ckpt_log_name '$prune_ckpt_path' --pruner_type taylor --taylor param_first --config_path /root/llm_pruner_mod/prune_llama2_7b.yml


CUDA_VISIBLE_DEVICES=1 python hf_prune.py --dtype float16 --base_model meta-llama/Llama-2-7b-hf --pruning_ratio 0.21 --device cuda --eval_device cuda --block_wise --block_mlp_layer_start 0 --block_mlp_layer_end 32 --block_attention_layer_start 0 --block_attention_layer_end 32 --save_ckpt_log_name '$prune_ckpt_path' --pruner_type taylor --taylor param_first --config_path /root/llm_pruner_mod/prune_llama2_7b.yml
CUDA_VISIBLE_DEVICES=1 python hf_prune.py --dtype float16 --base_model meta-llama/Llama-2-7b-hf --pruning_ratio 0.31 --device cuda --eval_device cuda --block_wise --block_mlp_layer_start 0 --block_mlp_layer_end 32 --block_attention_layer_start 0 --block_attention_layer_end 32 --save_ckpt_log_name '$prune_ckpt_path' --pruner_type taylor --taylor param_first --config_path /root/llm_pruner_mod/prune_llama2_7b.yml
CUDA_VISIBLE_DEVICES=1 python hf_prune.py --dtype float16 --base_model meta-llama/Llama-2-7b-hf --pruning_ratio 0.41 --device cuda --eval_device cuda --block_wise --block_mlp_layer_start 0 --block_mlp_layer_end 32 --block_attention_layer_start 0 --block_attention_layer_end 32 --save_ckpt_log_name '$prune_ckpt_path' --pruner_type taylor --taylor param_first --config_path /root/llm_pruner_mod/prune_llama2_7b.yml
CUDA_VISIBLE_DEVICES=1 python hf_prune.py --dtype float16 --base_model meta-llama/Llama-2-7b-hf --pruning_ratio 0.51 --device cuda --eval_device cuda --block_wise --block_mlp_layer_start 0 --block_mlp_layer_end 32 --block_attention_layer_start 0 --block_attention_layer_end 32 --save_ckpt_log_name '$prune_ckpt_path' --pruner_type taylor --taylor param_first --config_path /root/llm_pruner_mod/prune_llama2_7b.yml

CUDA_VISIBLE_DEVICES=1 python hf_prune.py --dtype bfloat16 --base_model meta-llama/Llama-2-7b-hf --pruning_ratio 0.21 --device cuda --eval_device cuda --block_wise --block_mlp_layer_start 0 --block_mlp_layer_end 32 --block_attention_layer_start 0 --block_attention_layer_end 32 --save_ckpt_log_name '$prune_ckpt_path' --pruner_type taylor --taylor param_first --config_path /root/llm_pruner_mod/prune_llama2_7b.yml
CUDA_VISIBLE_DEVICES=1 python hf_prune.py --dtype bfloat16 --base_model meta-llama/Llama-2-7b-hf --pruning_ratio 0.31 --device cuda --eval_device cuda --block_wise --block_mlp_layer_start 0 --block_mlp_layer_end 32 --block_attention_layer_start 0 --block_attention_layer_end 32 --save_ckpt_log_name '$prune_ckpt_path' --pruner_type taylor --taylor param_first --config_path /root/llm_pruner_mod/prune_llama2_7b.yml
CUDA_VISIBLE_DEVICES=1 python hf_prune.py --dtype bfloat16 --base_model meta-llama/Llama-2-7b-hf --pruning_ratio 0.41 --device cuda --eval_device cuda --block_wise --block_mlp_layer_start 0 --block_mlp_layer_end 32 --block_attention_layer_start 0 --block_attention_layer_end 32 --save_ckpt_log_name '$prune_ckpt_path' --pruner_type taylor --taylor param_first --config_path /root/llm_pruner_mod/prune_llama2_7b.yml
CUDA_VISIBLE_DEVICES=1 python hf_prune.py --dtype bfloat16 --base_model meta-llama/Llama-2-7b-hf --pruning_ratio 0.51 --device cuda --eval_device cuda --block_wise --block_mlp_layer_start 0 --block_mlp_layer_end 32 --block_attention_layer_start 0 --block_attention_layer_end 32 --save_ckpt_log_name '$prune_ckpt_path' --pruner_type taylor --taylor param_first --config_path /root/llm_pruner_mod/prune_llama2_7b.yml