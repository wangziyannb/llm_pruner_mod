import random

import numpy as np
import torch
from datasets import load_dataset
import pickle

from transformers import LlamaTokenizer


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    seed = 1
    seq_len = 256
    size = 20000
    set_random_seed(seed)
    # base_model = 'baffo32/decapoda-research-llama-7B-hf'
    base_model = 'llama2-7b'
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    # 加载数据集
    data = load_dataset(
        'json',
        data_files={'train': 'c4-train.00000-of-01024.json.gz'},
        split='train'
    )

    # 随机打乱数据集
    data = data.shuffle(seed=seed)


    # 定义一个过滤函数，该函数检查文本长度是否大于cutoff_len
    def filter_by_length(example):
        # 假设tokenizer已经被定义并且可以使用
        # 注意：根据你的实际需求调整tokenizer的调用和文本长度的判断逻辑
        tokenized_length = len(tokenizer(example['text'], add_special_tokens=False).input_ids)
        return tokenized_length > seq_len


    # 过滤数据集，保留符合条件的样本
    filtered_data = data.filter(filter_by_length)

    # 检查过滤后的数据集是否有足够的样本
    if len(filtered_data) < size:
        print(f"过滤后的数据集样本数不足{size}个")
    else:
        # 选择过滤后的前2000个样本
        data_sampled = filtered_data.select(range(size))

        # 将选中的样本保存到文件
        with open(f'sampled_dataset_seed{seed}_seqlen{seq_len}_size{size}.pkl', 'wb') as file:
            pickle.dump(data_sampled, file)
