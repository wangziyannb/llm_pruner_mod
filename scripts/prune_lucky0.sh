#!/bin/bash
#BASE_MODEL='baffo32/decapoda-research-llama-7B-hf'
BASE_MODEL='/home/zwang53/Downloads/llama2-13b'
prune0='0_llmpruner_llama-2-13b_0.2'
prune1='0_llmpruner_llama-2-13b_0.4'
prune2='0_llmpruner_llama-2-13b_0.6'
prune3='1_llmpruner_llama-2-13b_0.2'
prune4='1_llmpruner_llama-2-13b_0.4'
prune5='1_llmpruner_llama-2-13b_0.6'
prune6='2_llmpruner_llama-2-13b_0.2'
prune7='2_llmpruner_llama-2-13b_0.4'
prune8='2_llmpruner_llama-2-13b_0.6'

BASE_MODEL_7B='/home/zwang53/Downloads/llama2-7b'
prune0_7b='0_llmpruner_llama-2-7b_0.2'
prune1_7b='0_llmpruner_llama-2-7b_0.4'
prune2_7b='0_llmpruner_llama-2-7b_0.6'
prune3_7b='1_llmpruner_llama-2-7b_0.2'
prune4_7b='1_llmpruner_llama-2-7b_0.4'
prune5_7b='1_llmpruner_llama-2-7b_0.6'
prune6_7b='2_llmpruner_llama-2-7b_0.2'
prune7_7b='2_llmpruner_llama-2-7b_0.4'
prune8_7b='2_llmpruner_llama-2-7b_0.6'

data(){
  python datas.py --base_model $1 --seed $2 --seq_len $3 --size $4
}

task(){
  CUDA_VISIBLE_DEVICES=$7 python hf_prune.py --seed $6 --base_model $1 --pruning_ratio $2 --block_wise --block_mlp_layer_start $3 --block_mlp_layer_end $4 --block_attention_layer_start $3 --block_attention_layer_end $4 --save_ckpt_log_name $5 --save_model
}

task_post_training(){
  CUDA_VISIBLE_DEVICES=$3 python post_training.py --seed $2 --prune_model prune_log/"$1"/pytorch_model.bin --output_dir tune_log/"$1" --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64 --train_on_inputs
}

#data "$BASE_MODEL_7B" 0 256 20000 &
#data "$BASE_MODEL_7B" 1 256 20000 &
#data "$BASE_MODEL_7B" 2 256 20000
wait
#
#task "$BASE_MODEL" 0.22 3 40 "$prune0" 0 0
#task "$BASE_MODEL" 0.44 3 40 "$prune1" 0 1
#task "$BASE_MODEL" 0.66 3 40 "$prune2" 0 2
#task "$BASE_MODEL" 0.22 3 40 "$prune3" 1 3
#task "$BASE_MODEL" 0.44 3 40 "$prune4" 1 4
#task "$BASE_MODEL" 0.66 3 40 "$prune5" 1 5
#task "$BASE_MODEL" 0.22 3 40 "$prune6" 2 6
#task "$BASE_MODEL" 0.44 3 40 "$prune7" 2 7
#task "$BASE_MODEL" 0.66 3 40 "$prune8" 2 8
wait
#
#task_post_training "$prune0" 0 0 &
#task_post_training "$prune1" 0 1 &
#task_post_training "$prune2" 0 2 &
#task_post_training "$prune3" 1 3 &
#task_post_training "$prune4" 1 4 &
#task_post_training "$prune5" 1 5 &
#task_post_training "$prune6" 2 6 &
#task_post_training "$prune7" 2 7 &
#task_post_training "$prune8" 2 8
#wait
#
task "$BASE_MODEL_7B" 0.22 3 32 "$prune0_7b" 0 0
#task "$BASE_MODEL_7B" 0.44 3 32 "$prune1_7b" 0 1
#task "$BASE_MODEL_7B" 0.66 3 32 "$prune2_7b" 0 2
#task "$BASE_MODEL_7B" 0.22 3 32 "$prune3_7b" 1 3
#task "$BASE_MODEL_7B" 0.44 3 32 "$prune4_7b" 1 4
#task "$BASE_MODEL_7B" 0.66 3 32 "$prune5_7b" 1 5
#task "$BASE_MODEL_7B" 0.22 3 32 "$prune6_7b" 2 6
#task "$BASE_MODEL_7B" 0.44 3 32 "$prune7_7b" 2 7
#task "$BASE_MODEL_7B" 0.66 3 32 "$prune8_7b" 2 8
wait
#
#task_post_training "$prune0_7b" 0 0 &
#task_post_training "$prune1_7b" 0 1 &
#task_post_training "$prune2_7b" 0 2 &
#task_post_training "$prune3_7b" 1 3 &
#task_post_training "$prune4_7b" 1 4 &
#task_post_training "$prune5_7b" 1 5 &
#task_post_training "$prune6_7b" 2 6 &
#task_post_training "$prune7_7b" 2 7 &
#task_post_training "$prune8_7b" 2 8
wait

echo '所有任务完成'