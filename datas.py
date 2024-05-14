import argparse
import json
import os
import random

import numpy as np
import torch
from datasets import load_dataset
import pickle

import tqdm
from transformers import LlamaTokenizer


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_c4(samples, cutoff_len, tokenizer, seed, size):
    file_name = f"data/c4_{seed}_{cutoff_len}_{size}.json"
    if os.path.exists(file_name):
        dataset = load_dataset("json", data_files=file_name)
        if len(dataset['train']) == samples:
            print("load c4 from {}".format(file_name))
            return dataset

    with open(f'sampled_dataset_seed{seed}_seqlen{cutoff_len}_size{size}.pkl', 'rb') as file:
        dataset = pickle.load(file)
    # dataset = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    print(f"Sampling {samples} data from sampled_dataset_seed{seed}_seqlen{cutoff_len}_size{size}.pkl")
    subdata, history = [], []
    for i in tqdm.tqdm(range(samples)):
        # while True:
        #     i = random.randint(0, len(dataset) - 1)
        #     trainenc = tokenizer(dataset[i]['text'], return_tensors='pt')
        #     if trainenc.input_ids.shape[1] > cutoff_len and i not in history:
        #         history.append(i)
        #         break
        subdata.append({"inputs": dataset[i]['text']})
    with open(file_name, 'w') as f:
        f.writelines(json.dumps(subdata))
    return load_dataset("json", data_files=file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate sub-dataset')

    # Model Type&Path
    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--seed', type=int, help='seed')
    parser.add_argument('--seq_len', type=int, help='length of cut_off seq')
    parser.add_argument('--size', type=int, help='sampel size')
    args = parser.parse_args()
    set_random_seed(args.seed)
    # base_model = 'baffo32/decapoda-research-llama-7B-hf'
    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    # 加载数据集
    data = load_dataset(
        'json',
        data_files={'train': 'c4-train.00000-of-01024.json.gz'},
        split='train'
    )

    # 随机打乱数据集
    data = data.shuffle(seed=args.seed)


    # 定义一个过滤函数，该函数检查文本长度是否大于cutoff_len
    def filter_by_length(example):
        # 假设tokenizer已经被定义并且可以使用
        # 注意：根据你的实际需求调整tokenizer的调用和文本长度的判断逻辑
        tokenized_length = len(tokenizer(example['text'], add_special_tokens=False).input_ids)
        return tokenized_length > args.seq_len


    # 过滤数据集，保留符合条件的样本
    filtered_data = data.filter(filter_by_length)

    # 检查过滤后的数据集是否有足够的样本
    if len(filtered_data) < args.size:
        print(f"过滤后的数据集样本数不足{args.size}个")
    else:
        # 选择过滤后的前2000个样本
        data_sampled = filtered_data.select(range(args.size))

        # 将选中的样本保存到文件
        with open(f'sampled_dataset_seed{args.seed}_seqlen{args.seq_len}_size{args.size}.pkl', 'wb') as file:
            pickle.dump(data_sampled, file)
    get_c4(args.size, args.seq_len, tokenizer, args.seed, args.size)
