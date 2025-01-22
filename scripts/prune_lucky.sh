#!/bin/bash
SCRIPT_A='scripts/a.sh'
SCRIPT_B='scripts/b.sh'
#BASE_MODEL='baffo32/decapoda-research-llama-7B-hf'
BASE_MODEL='/home/zwang53/llama2-7b'
prune0='0_llmpruner_llama-2-7b_0.2'
prune1='0_llmpruner_llama-2-7b_0.4'
prune2='0_llmpruner_llama-2-7b_0.6'
#prune1='llama_prune_without_tune_block_param_first_4_32_0.57_c4'
#prune2='llama_prune_without_tune_block_param_first_4_30_0.25_global'

#tune0='llmpruner_prune_tune_block_param1_3_31_0.23_0.2_c4'
#tune1='llmpruner_prune_tune_block_param1_3_31_0.46_0.2_c4'
#tune2='llmpruner_prune_tune_block_param1_3_31_0.68_0.2_c4'
#tune1='llama_prune_with_tune_block_param_first_4_32_0.57_c4_2ep'
#tune2='llama_prune_with_tune_block_param_first_4_32_0.57_c4_10ep'
#tune1='llama_prune_with_tune_block_param_first_0_32_0.2_local'
#tune2='llama_prune_with_tune_block_param_first_4_30_0.25_global'


task(){
#  CUDA_VISIBLE_DEVICES=$5 python hf_prune.py --test_before_train --base_model $4 --pruning_ratio 0.57 --device cuda  --eval_device cuda --block_wise --block_mlp_layer_start $1 --block_mlp_layer_end $2 --block_attention_layer_start $1 --block_attention_layer_end $2 --save_ckpt_log_name $3 --pruner_type taylor --test_after_train --taylor param_first --save_model
  CUDA_VISIBLE_DEVICES=$7 python hf_prune.py --base_model $1 --pruning_ratio $2 --device cuda  --eval_device cuda --block_wise --block_mlp_layer_start $3 --block_mlp_layer_end $4 --block_attention_layer_start $3 --block_attention_layer_end $4 --save_ckpt_log_name $5 --pruner_type taylor --taylor param_first --save_model --seed $6
}

task_post_training(){

  CUDA_VISIBLE_DEVICES=$4 python post_training.py --prune_model prune_log/"$1"/pytorch_model.bin --data_path c4 --output_dir tune_log/"$2" --wandb_project "$2" --lora_r 8 --num_epochs $3 --learning_rate 1e-4 --batch_size 64 --train_on_inputs
}

task_eval(){
  export PYTHONPATH='.'
  CUDA_VISIBLE_DEVICES=$4 python lm-evaluation-harness/main.py --batch_size auto --model hf-causal-experimental --model_args checkpoint=prune_log/"$1"/pytorch_model.bin,peft=tune_log/"$2",config_pretrained=$3 --tasks arc_easy,arc_challenge --device cuda:0 --output_path results/"$2".json --no_cache
}

task "$BASE_MODEL" 0.22 3 32 "$prune0" 0 1 &
task "$BASE_MODEL" 0.44 3 32 "$prune1" 0 2 &
task "$BASE_MODEL" 0.66 3 32 "$prune2" 0 3 &
#task 0.6 3 31 "$prune1" 2 &
#task 0.6 3 32 "$prune2" 3 &
wait

#CUDA_VISIBLE_DEVICES=1 bash $SCRIPT_B $BASE_MODEL prune_log/"$prune0" &
#CUDA_VISIBLE_DEVICES=2 bash $SCRIPT_B $BASE_MODEL prune_log/"$prune1" &
#CUDA_VISIBLE_DEVICES=3 bash $SCRIPT_B $BASE_MODEL prune_log/"$prune2" &
#CUDA_VISIBLE_DEVICES=0 bash $SCRIPT_B $BASE_MODEL prune_log/"$prune3"
wait

#task_post_training "$prune0" "$tune0" 1 &
#task_post_training "$prune1" "$tune1" 2 &
#task_post_training "$prune2" "$tune2" 3 &
#task_post_training "$prune3" "$tune3" 0

wait
#task_eval "$prune0" "$tune0" $BASE_MODEL 1
#CUDA_VISIBLE_DEVICES=0 bash $SCRIPT_A $BASE_MODEL tune_log/"$tune0" prune_log/"$prune0" 1400 &
#CUDA_VISIBLE_DEVICES=1 bash $SCRIPT_A $BASE_MODEL tune_log/"$tune1" prune_log/"$prune0" 1400 &
#CUDA_VISIBLE_DEVICES=2 bash $SCRIPT_A $BASE_MODEL tune_log/"$tune2" prune_log/"$prune0" 1400 &
#CUDA_VISIBLE_DEVICES=0 bash $SCRIPT_A $BASE_MODEL tune_log/"$tune3" prune_log/"$prune3" 1400

wait

#CUDA_VISIBLE_DEVICES=0 python generate.py --model_type tune_prune_LLM --ckpt prune_log/"$prune0"/pytorch_model.bin --lora_ckpt tune_log/"$tune0"/ --base_model "$BASE_MODEL" &
#CUDA_VISIBLE_DEVICES=1 python generate.py --model_type tune_prune_LLM --ckpt prune_log/"$prune0"/pytorch_model.bin --lora_ckpt tune_log/"$tune1"/checkpoint-1400 --base_model "$BASE_MODEL" &
#CUDA_VISIBLE_DEVICES=2 python generate.py --model_type tune_prune_LLM --ckpt prune_log/"$prune0"/pytorch_model.bin --lora_ckpt tune_log/"$tune2"/checkpoint-1400 --base_model "$BASE_MODEL" &
#CUDA_VISIBLE_DEVICES=0 python generate.py --model_type tune_prune_LLM --ckpt prune_log/"$prune3"/pytorch_model.bin --lora_ckpt tune_log/"$tune3"/checkpoint-1400 --base_model "$BASE_MODEL"
wait
echo "所有任务完成"