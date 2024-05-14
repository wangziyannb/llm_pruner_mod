import os
import gc
import sys
import time
import json
import copy
import random
import argparse
from typing import Tuple

import torch
import numpy as np
from torch import Tensor
from tqdm import tqdm
from transformers import LlamaTokenizer, GenerationConfig, LlamaConfig
from LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm, LlamaAttention, LlamaMLP

import LLMPruner.torch_pruning as tp
from LLMPruner.pruner import hf_llama_pruner as llama_pruner
from LLMPruner.utils.logger import LoggerWithDepth
from LLMPruner.evaluator.ppl import PPLMetric
from LLMPruner.datasets.example_samples import get_examples
from LLMPruner.templates.prompts import prompts

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import csv
import pandas as pd


# from layerwrapper import BiasGPT


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# class MaskedLinear(nn.Linear):
#     def __init__(self, in_features, out_features, bias=True):
#         super(MaskedLinear, self).__init__(in_features, out_features, bias)
#         self.weight_mask = nn.Parameter(torch.ones(self.weight.size()), requires_grad=False)
#         self.bias_mask = nn.Parameter(torch.ones(self.bias.size()), requires_grad=False)
#
#     def forward(self, input):
#         masked_weight = self.weight * self.weight_mask
#         masked_bias = self.bias * self.bias_mask
#         output = F.linear(input, masked_weight, masked_bias)
#         # for i in masked_weight:
#         #     print(i)
#         return output
#
#     def apply_mask(self, idxs, prune_fn):
#         # self.weight_mask = nn.Parameter(torch.zeros(self.weight.size()), requires_grad=False)
#         pruned_params = 0
#         if prune_fn in ["linear_out"]:
#             for i in idxs:
#                 self.weight_mask[i, :] = 0
#                 pruned_params += len(self.weight_mask[i])
#         elif prune_fn in ["linear_in"]:
#             for i in idxs:
#                 self.weight_mask[:, i] = 0
#                 pruned_params += len(self.weight_mask)
#         return pruned_params


class TaylorImportance(tp.importance.Importance):
    def __init__(self, group_reduction="sum", normalizer=None, taylor=None):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.taylor = taylor

    def _reduce(self, group_imp):
        if self.group_reduction == "sum":
            group_imp = group_imp.sum(dim=0)
        elif self.group_reduction == "mean":
            group_imp = group_imp.mean(dim=0)
        elif self.group_reduction == "max":
            group_imp = group_imp.max(dim=0)[0]
        elif self.group_reduction == "prod":
            group_imp = torch.prod(group_imp, dim=0)
        elif self.group_reduction == 'first':
            group_imp = group_imp[0]
        elif self.group_reduction == 'second':
            group_imp = group_imp[1]
        elif self.group_reduction is None:
            group_imp = group_imp
        else:
            raise NotImplementedError
        return group_imp

    @torch.no_grad()
    def __call__(self, layer, prune_fn, idxs):

        group_imp = []

        idxs.sort()

        if prune_fn not in [
            "linear_out", "linear_in", "embedding_out", "rmsnorm_out"
        ]:
            return

        salience = layer.weight * layer.weight.grad

        if self.taylor in ['param_second']:
            salience = layer.weight * layer.weight.acc_grad * layer.weight
        elif self.taylor in ['param_mix']:
            salience = salience - 0.5 * layer.weight * layer.weight.acc_grad * layer.weight

        # Linear out_channels
        if prune_fn in ["linear_out"]:
            if self.taylor == 'vectorize':
                local_norm = salience.sum(1).abs()
            elif 'param' in self.taylor:
                local_norm = salience.abs().sum(1)
            else:
                raise NotImplementedError
            group_imp.append(local_norm)

        # Linear in_channels
        elif prune_fn in ["linear_in"]:
            if self.taylor == 'vectorize':
                local_norm = salience.sum(0).abs()
            elif 'param' in self.taylor:
                local_norm = salience.abs().sum(0)
            else:
                raise NotImplementedError
            local_norm = local_norm[idxs]
            group_imp.append(local_norm)

        # RMSNorm
        elif prune_fn == "rmsnorm_out":
            local_norm = salience.abs()
            group_imp.append(local_norm)

        # Embedding
        elif prune_fn == "embedding_out":
            if self.taylor == 'vectorize':
                local_norm = salience[:, idxs].sum(0).abs()
            elif 'param' in self.taylor:
                local_norm = salience[:, idxs].abs().sum(0)
            else:
                raise NotImplementedError
            group_imp.append(local_norm)

        if len(group_imp) == 0:
            return None

        min_imp_size = min([len(imp) for imp in group_imp])
        aligned_group_imp = []
        for imp in group_imp:
            if len(imp) > min_imp_size and len(imp) % min_imp_size == 0:
                imp = imp.view(len(imp) // min_imp_size, min_imp_size).sum(0)
                aligned_group_imp.append(imp)
            elif len(imp) == min_imp_size:
                aligned_group_imp.append(imp)
        group_imp = torch.stack(aligned_group_imp, dim=0)
        group_imp = self._reduce(group_imp)
        # if self.normalizer is not None:
        # group_imp = self.normalizer(group, group_imp)
        return group_imp


class RandomImportance(tp.importance.Importance):
    @torch.no_grad()
    def __call__(self, layer, prune_fn, idxs):
        return torch.rand(len(idxs))


def get_mask(imps, layer, prune_fn, target_sparsity, head_dim=1):
    if prune_fn in ["linear_out"]:
        current_channels = layer.out_features
    elif prune_fn in ["linear_in"]:
        current_channels = layer.in_features
    else:
        current_channels = layer.out_features
    n_pruned = current_channels - int(
        current_channels *
        (1 - target_sparsity)
    )
    if n_pruned <= 0:
        return

    if head_dim > 1:
        imps = imps.view(-1, head_dim).sum(1)

    imp_argsort = torch.argsort(imps)

    if head_dim > 1:
        # n_pruned//consecutive_groups
        pruning_groups = imp_argsort[:(n_pruned // head_dim)]
        group_size = head_dim
        pruning_idxs = torch.cat(
            [torch.tensor([j + group_size * i for j in range(group_size)])
             for i in pruning_groups], 0)
    else:
        pruning_idxs = imp_argsort[:n_pruned]
    # print(len(pruning_idxs))
    return pruning_idxs


def plot(figure, x, y):
    # plt.figure(figsize=figure)
    # for index in range(len(y)):
    #     plt.subplot((len(y) + 1) // 2, 2, index + 1)
    #     plt.title(y[index][0])
    #     plt.plot(x, y[index][1], label=y[index][0])
    # plt.show()
    return


#
# def prepare_calibration_input(model, dataloader, device):
#     """
#     Prepare inputs for model calibration.
#
#     Args:
#         model (nn.Module): The model to prepare inputs for.
#         dataloader (DataLoader): DataLoader object to fetch input data.
#         device (torch.device): Device on which the model is loaded.
#
#     Returns:
#         inps (torch.Tensor): Input tensor for calibration.
#         outs (torch.Tensor): Output tensor for calibration.
#         attention_mask (torch.Tensor): Attention mask tensor.
#         position_ids (torch.Tensor): Position IDs tensor.
#     """
#     use_cache = model.config.use_cache
#     model.config.use_cache = False
#     layers = model.model.layers
#
#     if "model.embed_tokens" in getattr(model, 'hf_device_map', {}):
#         device = model.hf_device_map["model.embed_tokens"]
#
#     dtype = next(iter(model.parameters())).dtype
#     inps = torch.zeros((2048, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
#     inps.requires_grad = False
#     cache = {'i': 0, 'attention_mask': None, "position_ids": None}
#
#     class Catcher(nn.Module):
#         def __init__(self, module):
#             super().__init__()
#             self.module = module
#
#         def forward(self, inp, **kwargs):
#             inps[cache['i']] = inp
#             cache['i'] += 1
#             cache['attention_mask'] = kwargs['attention_mask']
#             cache['position_ids'] = kwargs['position_ids']
#             raise ValueError
#
#     layers[0] = Catcher(layers[0])
#     for batch in dataloader:
#         try:
#             model(batch[0].to(device))
#         except ValueError:
#             pass
#     layers[0] = layers[0].module
#
#     outs = torch.zeros_like(inps)
#     attention_mask = cache['attention_mask']
#     position_ids = cache['position_ids']
#     model.config.use_cache = use_cache
#
#     return inps, outs, attention_mask, position_ids
#
#
# def find_layers(module, layers=[nn.Linear], name=''):
#     """
#     Recursively find the layers of a certain type in a module.
#
#     Args:
#         module (nn.Module): PyTorch module.
#         layers (list): List of layer types to find.
#         name (str): Name of the module.
#
#     Returns:
#         dict: Dictionary of layers of the given type(s) within the module.
#     """
#     if type(module) in layers:
#         return {name: module}
#     res = {}
#     for name1, child in module.named_children():
#         res.update(find_layers(
#             child, layers=layers, name=name + '.' + name1 if name != '' else name1
#         ))
#     return res
#
#
# def compress(layer, attn_mask, mlp_mask, attn_mean_inp, mlp_mean_inp, device, bias=True, unstr=False):
#     """
#     Compress a model layer by masking or pruning based on the given masks.
#
#     Args:
#         layer (nn.Module): The model layer to compress.
#         attn_mask (torch.Tensor): The mask to apply to the attention weights.
#         mlp_mask (torch.Tensor): The mask to apply to the MLP weights.
#         attn_mean_inp (torch.Tensor): The mean attention input.
#         mlp_mean_inp (torch.Tensor): The mean MLP input.
#         device (torch.device): Device on which the model is loaded.
#         bias (bool, optional): Whether to consider bias while compressing. Defaults to True.
#         unstr (bool, optional): If True, only mask without real pruning. Defaults to False.
#
#     Returns:
#         None: This function modifies the layer in-place and doesn't return anything.
#     """
#     if unstr:  # Only mask, do not really prune
#         # Attention Weight Masking
#         if attn_mask is not None:
#             retain_heads = torch.count_nonzero(attn_mask)
#             attn_mask = attn_mask.repeat_interleave(128)
#             # Apply the mask to the query, key and value projection weights
#             layer.self_attn.q_proj.weight.data *= attn_mask.unsqueeze(-1).to(device)
#             layer.self_attn.k_proj.weight.data *= attn_mask.unsqueeze(-1).to(device)
#             layer.self_attn.v_proj.weight.data *= attn_mask.unsqueeze(-1).to(device)
#
#             output_weight = layer.self_attn.o_proj.weight.data
#             if bias:
#                 # Add the additional bias to compensate for the loss
#                 output_bias = ((attn_mean_inp * ~attn_mask.to(device)) @ output_weight.T)
#
#             # Note: the weight data is masked, but the weight tensor shape remains unchanged
#             if bias:
#                 layer.self_attn.o_proj.bias.data = output_bias
#             layer.self_attn.o_proj.weight.data = output_weight
#
#         # MLP Weight Masking
#         if mlp_mask is not None:
#             # Apply the mask to the up and gate projection weights
#             layer.mlp.up_proj.weight.data *= mlp_mask.unsqueeze(-1).to(device)
#             layer.mlp.gate_proj.weight.data *= mlp_mask.unsqueeze(-1).to(device)
#
#             output_weight = layer.mlp.down_proj.weight.data
#             if bias:
#                 # Add the additional bias to compensate for the loss
#                 output_bias = ((mlp_mean_inp * ~mlp_mask.to(device)) @ output_weight.T)
#
#             # Note: the weight data is masked, but the weight tensor shape remains unchanged
#             if bias:
#                 layer.mlp.down_proj.bias.data = output_bias
#             layer.mlp.down_proj.weight.data = output_weight
#
#     else:
#         # Real Pruning
#         # Attention Weight Pruning
#         if attn_mask is not None:
#             retain_heads = torch.count_nonzero(attn_mask)
#             attn_mask = attn_mask.repeat_interleave(128)
#
#             # Prune the query, key and value projection weights
#             # We reduce the size of the weights based on the attention mask
#             layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[torch.where(attn_mask)[0]]
#             layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[torch.where(attn_mask)[0]]
#             layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[torch.where(attn_mask)[0]]
#
#             # Update output dimensions of q, k, v projections based on remaining heads
#             layer.self_attn.q_proj.out_features = attn_mask.sum().item()
#             layer.self_attn.k_proj.out_features = attn_mask.sum().item()
#             layer.self_attn.v_proj.out_features = attn_mask.sum().item()
#
#             output_weight = layer.self_attn.o_proj.weight.data
#
#             if bias:
#                 # Add the additional bias to compensate for the loss
#                 output_bias = ((attn_mean_inp * ~attn_mask.to(device)) @ output_weight.T)
#
#             # Prune the output projection weight
#             output_weight = layer.self_attn.o_proj.weight.data[:, torch.where(attn_mask)[0]]
#             # Update layer configurations for the new output shape after pruning
#             layer.self_attn.num_heads = retain_heads
#             layer.self_attn.hidden_size = retain_heads * 128
#
#             if bias:
#                 # Re-initialize the Linear layer with new shape and bias
#                 layer.self_attn.o_proj.in_features = attn_mask.sum().item()
#                 # layer.self_attn.o_proj = torch.nn.Linear(in_features=output_weight.shape[1], out_features=output_weight.shape[0], bias=True).to(device)
#                 layer.self_attn.o_proj.bias.data = output_bias
#             layer.self_attn.o_proj.in_features = attn_mask.sum().item()
#             # Assign the pruned weights
#             layer.self_attn.o_proj.weight.data = output_weight
#
#         # MLP Weight Pruning
#         if mlp_mask is not None:
#             # Prune the up and gate projection weights
#             layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[torch.where(mlp_mask)[0]]
#             layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[torch.where(mlp_mask)[0]]
#
#             # Update output dimensions of up and gate projections based on the mlp mask
#             layer.mlp.up_proj.out_features = mlp_mask.sum().item()
#             layer.mlp.gate_proj.out_features = mlp_mask.sum().item()
#
#             output_weight = layer.mlp.down_proj.weight.data
#             layer.mlp.intermediate_size = mlp_mask.sum().item()
#             if bias:
#                 # Add the additional bias to compensate for the loss
#                 output_bias = ((mlp_mean_inp * ~mlp_mask.to(device)) @ output_weight.T)
#
#             # Prune the down projection weight
#             output_weight = layer.mlp.down_proj.weight.data[:, torch.where(mlp_mask)[0]]
#
#             if bias:
#                 # Re-initialize the Linear layer with new shape and bias
#                 layer.mlp.down_proj.in_features = mlp_mask.sum().item()
#                 # layer.mlp.down_proj = torch.nn.Linear(in_features=output_weight.shape[1], out_features=output_weight.shape[0], bias=True).to(device)
#                 layer.mlp.down_proj.bias.data = output_bias
#             layer.mlp.down_proj.in_features = mlp_mask.sum().item()
#             # Assign the pruned weights
#             layer.mlp.down_proj.weight.data = output_weight
#
#     # Explicitly empty the CUDA cache to clean up some memory
#     torch.cuda.empty_cache()
#
#
# def prune_flap(args, model, tokenizer, device=torch.device("cuda:0")):
#     """
#     Our FLAP Pruning.
#
#     Args:
#         args (object): Command line arguments parsed via argparse.
#         model (nn.Module): PyTorch model to prune.
#         tokenizer (Tokenizer): Tokenizer associated with the model.
#         device (torch.device, optional): Device to move tensors to. Defaults to CUDA device 0.
#     """
#     use_cache = model.config.use_cache
#     model.config.use_cache = False
#
#     print("loading calibdation data")
#     from data import get_loaders
#     dataloader, _ = get_loaders("wikitext2", nsamples=args.num_examples, seed=42, seqlen=128,
#                                 tokenizer=tokenizer)
#     print("dataset loading complete")
#
#     with torch.no_grad():
#         inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
#     layers = model.model.layers
#
#     attn_metric_list, mlp_metric_list = [], []
#     attn_baseline_inp_list, mlp_baseline_inp_list = [], []
#     attn_mask, mlp_mask = [], []
#
#     # Split into sub-problems, separate statistics for each module
#     for i in tqdm(range(len(layers)), desc="Processing layers"):
#         layer = layers[i]
#         subset = {}
#         subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
#         subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})
#         wrapped_layers = {}
#         for name in subset:
#             wrapped_layers[name] = BiasGPT(subset[name], 'WIFV')
#
#         def add_batch(name):
#             def tmp(_, inp, out):
#                 wrapped_layers[name].add_batch(inp[0].data, out.data)
#
#             return tmp
#
#         if i not in range(args.block_attention_layer_start, args.block_attention_layer_end):
#             for j in range(args.num_examples):
#                 with torch.no_grad():
#                     outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
#             inps, outs = outs, inps
#             torch.cuda.empty_cache()
#         else:
#             handles = []
#             for name in wrapped_layers:
#                 handles.append(subset[name].register_forward_hook(add_batch(name)))
#             for j in range(args.num_examples):
#                 with torch.no_grad():
#                     outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
#             for h in handles:
#                 h.remove()
#             metrics = {
#                 'IFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp,
#                 'WIFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp * torch.sum(
#                     subset[name].weight.data.pow(2), dim=0),
#                 'WIFN': lambda wrapped_layers, subset, name: (torch.abs(subset[name].weight.data) * torch.sqrt(
#                     wrapped_layers[name].scaler_inp.reshape((1, -1)))).mean(axis=0),
#             }
#
#             for name in subset:
#                 if name == 'self_attn.o_proj':
#                     W_metric = metrics["WIFV"](wrapped_layers, subset, name) ** 2
#                     # if args.structure == "UL-UM":
#                     if args.global_pruning:
#                         attn_metric_list.append(W_metric.cpu())
#                     else:
#                         W_metric = W_metric.reshape(-1, 128).sum(dim=1)
#                         attn_metric_list.append(W_metric.cpu())
#                         thresh = torch.sort(W_metric.cuda())[0][
#                             int(args.pruning_ratio * layer.self_attn.num_heads)].cpu()
#                         W_mask = (W_metric >= thresh)
#                         attn_mask.append(W_mask)
#                     attn_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))
#                 else:
#                     W_metric = metrics["WIFV"](wrapped_layers, subset, name)
#                     if args.global_pruning:
#                         mlp_metric_list.append(W_metric.cpu())
#                     else:
#                         mlp_metric_list.append(W_metric.cpu())
#                         thresh = torch.sort(W_metric.cuda())[0][int(W_metric.numel() * args.pruning_ratio)].cpu()
#                         W_mask = (W_metric >= thresh)
#                         mlp_mask.append(W_mask)
#                     mlp_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))
#                 wrapped_layers[name].free()
#             inps, outs = outs, inps  # Use the original output as input to the next layer
#             torch.cuda.empty_cache()
#
#     standarlization = lambda x: (x - torch.mean(x, axis=1, keepdim=True)) / torch.std(x, axis=1, keepdim=True)
#
#     if args.global_pruning:
#         mlp_metric = torch.stack(mlp_metric_list)
#         mlp_metric = standarlization(mlp_metric)
#
#         attn_metric = torch.stack(attn_metric_list)
#         attn_metric = standarlization(attn_metric)
#         attn_metric = attn_metric.reshape(mlp_metric.shape[0], -1, 128).mean(dim=2)
#
#         # attn_metric_list=attn_metric.view(-1).tolist()
#
#         prune_metric = torch.cat([attn_metric.view(-1), mlp_metric.view(-1)])
#         sorted_prune, indices = torch.sort(prune_metric, descending=True)
#         compression_weight = torch.ones_like(indices)
#         compression_weight[indices < attn_metric.numel()] = 512.0 / 3
#         threshold = sorted_prune[torch.argmin(
#             torch.abs(torch.cumsum(compression_weight, 0) - torch.sum(compression_weight) * (1 - args.pruning_ratio)))]
#         attn_mask = (attn_metric > threshold)
#         mlp_mask = (mlp_metric > threshold)
#     else:
#         attn_mask = torch.stack(attn_mask)
#         mlp_mask = torch.stack(mlp_mask)
#     # pruning_ratio_mha = []
#     # pruning_ratio_mlp = []
#     # importance_mha = []
#     # importance_mlp = []
#     # plot((15, 10), [x for x in range(1024)],
#     #      [("Importance MHA", attn_metric_list)])
#     for idx in range(args.block_attention_layer_start, args.block_attention_layer_end):
#         # pruning_ratio_mha.append(torch.sum(attn_mask[idx - args.block_attention_layer_start] == False).item() /
#         #                          attn_mask[idx - args.block_attention_layer_start].shape[0])
#         # pruning_ratio_mlp.append(torch.sum(mlp_mask[idx - args.block_attention_layer_start] == False).item() /
#         #                          mlp_mask[idx - args.block_attention_layer_start].shape[0])
#         #
#         # importance_mha.append(
#         #     torch.linalg.vector_norm(attn_metric_list[idx - args.block_attention_layer_start]).tolist())
#         # importance_mlp.append(
#         #     torch.linalg.vector_norm(mlp_metric_list[idx - args.block_attention_layer_start]).tolist())
#
#         compress(model.model.layers[idx], attn_mask[idx - args.block_attention_layer_start], None,
#                  attn_baseline_inp_list[idx - args.block_attention_layer_start], None, device,
#                  bias=args.bias, unstr=False)
#         compress(model.model.layers[idx], None, mlp_mask[idx - args.block_attention_layer_start], None,
#                  mlp_baseline_inp_list[idx - args.block_attention_layer_start], device,
#                  bias=args.bias, unstr=False)
#     # plot((15, 10), [x for x in range(args.block_attention_layer_start, args.block_attention_layer_end)],
#     #      [("Importance MHA", importance_mha),
#     #       ("Importance MLP", importance_mlp),
#     #       ("Pruning ratio MHA", pruning_ratio_mha),
#     #       ("Pruning ratio MLP", pruning_ratio_mlp)])
#     print()

def apply_mask(layer, idxs, prune_fn):
    idxs.sort(reverse=True)
    before = layer.weight.numel()
    if prune_fn in ["linear_out"]:
        rows_mask = torch.ones(layer.weight.data.size(0), dtype=torch.bool)
        rows_mask[idxs] = False
        layer.weight.data = layer.weight.data[rows_mask]
        layer.out_features -= len(idxs)
    elif prune_fn in ["linear_in"]:
        cols_mask = torch.ones(layer.weight.data.size(1), dtype=torch.bool)
        cols_mask[idxs] = False
        layer.weight.data = layer.weight.data[:, cols_mask]
        layer.in_features -= len(idxs)
    return 1 - layer.weight.numel() / before


def main(args):
    logger = LoggerWithDepth(
        env_name="{}".format(args.save_ckpt_log_name),
        config=args.__dict__,
        root_dir='prune_log',
        setup_sublogger=True
    )
    set_random_seed(args.seed)
    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)

    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        low_cpu_mem_usage=True if args.torch_version >= 1.9 else False
    )
    # for i in range(32):
    #     model.model.layers[i].self_attn.o_proj.bias = torch.nn.Parameter(
    #         torch.zeros_like(model.model.layers[i].self_attn.o_proj.bias, device="cpu"))
    #     model.model.layers[i].mlp.down_proj.bias = torch.nn.Parameter(
    #         torch.zeros_like(model.model.layers[i].mlp.down_proj.bias, device="cpu"))
    #     torch.nn.init.zeros_(model.model.layers[i].self_attn.o_proj.bias)
    #     torch.nn.init.zeros_(model.model.layers[i].mlp.down_proj.bias)
    # model.seqlen = 128
    if args.device != "cpu":
        model.half()
    model.to(args.device)
    pruner_type = args.pruner_type.lower()
    assert pruner_type in ['random', 'l2', 'l1', 'taylor', 'variance']
    for param in model.parameters():
        # if param.requires_grad:
        #     print(param.numel())
        param.requires_grad_(True)
    before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    forward_prompts = torch.tensor([
        [1, 306, 4658, 278, 6593, 310, 2834, 338],
        [1, 3439, 17632, 1925, 29892, 278, 6368, 310],
        # [1, 319, 11473, 2643, 378, 629, 271, 18099],
        # [1, 4103, 9632, 4223, 304, 5176, 29901, 13],
    ]).to(
        args.device)

    # forward_prompts = get_examples('bookcorpus', tokenizer, 10, seq_len=64).to(args.device)
    wrapper_layers = {}
    if pruner_type == 'random':
        imp = RandomImportance()
    elif pruner_type == 'l1':
        imp = llama_pruner.MagnitudeImportance(p=1)
    elif pruner_type == 'l2':
        imp = llama_pruner.MagnitudeImportance(p=2)
    elif pruner_type == 'taylor':
        imp = TaylorImportance(group_reduction=args.grouping_strategy, taylor=args.taylor)
    elif pruner_type == 'variance':
        imp = RandomImportance()
    else:
        raise NotImplementedError

    logger.log("Use {} pruner...".format(pruner_type))
    logger.log("Pruning Attention Layer = {}".format(
        list(range(args.block_attention_layer_start, args.block_attention_layer_end))))
    logger.log("Pruning MLP Layer = {}".format(list(range(args.block_mlp_layer_start, args.block_mlp_layer_end))))
    logger.log("Start Pruning")

    model.eval()
    try:
        out = model(*forward_prompts)
    except:
        out = model(forward_prompts)

    for i in range(args.iterative_steps):
        # if pruner_type in ['variance']:
        #     prune_flap(args, model, tokenizer, args.device)
        #
        # else:
        if pruner_type in ['taylor']:
            example_prompts = get_examples('bookcorpus', tokenizer, 10, seq_len=64).to(args.device)
            logger.log("Start Backwarding in iterative steps = {}...".format(i))
            if args.taylor in ['param_mix', 'param_second']:
                for j in range(args.num_examples):
                    batch_input = example_prompts[j].unsqueeze(0)
                    loss = model(batch_input, labels=batch_input).loss
                    logger.log("Loss = {}".format(loss))
                    loss.backward()

                    for module_param in model.parameters():
                        if module_param.requires_grad:
                            module_param.grad = module_param.grad * module_param.grad / args.num_examples
                            if hasattr(module_param, 'acc_grad'):
                                module_param.acc_grad += module_param.grad
                            else:
                                module_param.acc_grad = copy.deepcopy(module_param.grad)
                    model.zero_grad()
                    del loss.grad
            loss = model(example_prompts, labels=example_prompts).loss
            logger.log("Loss = {}".format(loss))
            loss.backward()

        # pruner.step()
        if args.global_pruning:
            whole_imps_attn = torch.tensor([]).to(args.device)
            whole_imps_attn_scaled = torch.tensor([]).to(args.device)
            whole_imps_attn_scaled_layer = torch.tensor([]).to(args.device)

            whole_imps_mlp = torch.tensor([]).to(args.device)
            whole_imps_mlp_scaled = torch.tensor([]).to(args.device)
            whole_imps_mlp_scaled_layer = torch.tensor([]).to(args.device)

            pruning_ratio_mha = []
            pruning_ratio_mlp = []

            weight_norm_mha = []
            activation_norm_mha = []
            gradient_norm_mha = []
            importance_norm_mha = []
            importance_norm_mha_scale = []

            weight_norm_mlp = []
            activation_norm_mlp = []
            gradient_norm_mlp = []
            importance_norm_mlp = []
            importance_norm_mlp_scale = []

            layer_x_mha = [x for x in range(args.block_attention_layer_start, args.block_attention_layer_end)]
            layer_x_mlp = [x for x in range(args.block_mlp_layer_start, args.block_mlp_layer_end)]

            for z in range(args.block_attention_layer_start, args.block_attention_layer_end):
                layer = model.model.layers[z]
                weight_norm_mha.append(torch.linalg.matrix_norm(layer.self_attn.q_proj.weight).tolist())
                gradient_norm_mha.append(torch.linalg.matrix_norm(layer.self_attn.q_proj.weight.grad).tolist())
                imps = imp(layer.self_attn.q_proj, "linear_out", [1])
                importance_norm_mha.append(torch.linalg.vector_norm(imps).tolist())
                imps_scaled = imps.clone()
                mean = torch.mean(imps_scaled)
                stdev = torch.std(imps_scaled, unbiased=False)
                imps_scaled = (imps_scaled - mean) / stdev
                importance_norm_mha_scale.append(torch.linalg.vector_norm(imps_scaled).tolist())
                imps_scaled = imps_scaled.view(-1, layer.self_attn.head_dim).sum(1)
                whole_imps_attn_scaled_layer = torch.cat((whole_imps_attn_scaled_layer, imps_scaled), dim=0)
                imps = imps.view(-1, layer.self_attn.head_dim).sum(1)
                whole_imps_attn = torch.cat((whole_imps_attn, imps), dim=0)

            for z in range(args.block_mlp_layer_start, args.block_mlp_layer_end):
                layer = model.model.layers[z]
                weight_norm_mlp.append(torch.linalg.matrix_norm(layer.mlp.gate_proj.weight).tolist())
                gradient_norm_mlp.append(torch.linalg.matrix_norm(layer.mlp.gate_proj.weight.grad).tolist())
                imps = imp(layer.mlp.gate_proj, "linear_out", [1])
                importance_norm_mlp.append(torch.linalg.vector_norm(imps).tolist())
                imps_scaled = imps.clone()
                mean = torch.mean(imps_scaled)
                stdev = torch.std(imps_scaled, unbiased=False)
                imps_scaled = (imps_scaled - mean) / stdev
                importance_norm_mlp_scale.append(torch.linalg.vector_norm(imps_scaled).tolist())
                whole_imps_mlp_scaled_layer = torch.cat((whole_imps_mlp_scaled_layer, imps_scaled), dim=0)
                whole_imps_mlp = torch.cat((whole_imps_mlp, imps), dim=0)

            mean = torch.mean(whole_imps_attn)
            stdev = torch.std(whole_imps_attn, unbiased=False)
            whole_imps_attn_scaled = (whole_imps_attn - mean) / stdev

            plot((15, 10), [x for x in range(len(whole_imps_attn))],
                 [("No scale Importance MHA", whole_imps_attn.tolist()),
                  ("Scaled Importance global-wise", whole_imps_attn_scaled.tolist()),
                  ("Scaled Importance layer-wise", whole_imps_attn_scaled_layer.tolist())])

            mean = torch.mean(whole_imps_mlp)
            stdev = torch.std(whole_imps_mlp, unbiased=False)
            whole_imps_mlp_scaled = (whole_imps_mlp - mean) / stdev

            plot((15, 10), [x for x in range(len(whole_imps_mlp))],
                 [("No scale Importance MLP", whole_imps_mlp.tolist()),
                  ("Scaled Importance global-wise", whole_imps_mlp_scaled.tolist()),
                  ("Scaled Importance layer-wise", whole_imps_mlp_scaled_layer.tolist())])

            # imp_argsort = torch.argsort(whole_imps_attn_scaled_layer)
            imp_argsort = torch.argsort(whole_imps_attn)
            n_pruned = len(imp_argsort) - int(
                len(imp_argsort) *
                (1 - args.pruning_ratio)
            )
            pruning_groups = imp_argsort[:n_pruned]
            pruning_groups = pruning_groups.tolist()
            pruning_groups.sort()

            pruning_idxs_attn_record = []
            for z in range(args.block_attention_layer_start, args.block_attention_layer_end):
                layer = model.model.layers[z]
                pruning_idxs = torch.tensor([], dtype=torch.int8)
                for j in range(layer.self_attn.num_heads):
                    # i-> current layer index
                    # j-> current head index (inside current layer)
                    if (z - args.block_attention_layer_start) * layer.self_attn.num_heads + j in pruning_groups:
                        pruning_idxs = torch.cat(
                            (pruning_idxs,
                             torch.tensor(
                                 [j * layer.self_attn.head_dim + x for x in range(layer.self_attn.head_dim)])
                             ),
                            dim=0)
                pruning_ratio_mha.append(apply_mask(layer.self_attn.q_proj, pruning_idxs.tolist(), "linear_out"))
                apply_mask(layer.self_attn.k_proj, pruning_idxs.tolist(), "linear_out")
                apply_mask(layer.self_attn.v_proj, pruning_idxs.tolist(), "linear_out")
                apply_mask(layer.self_attn.o_proj, pruning_idxs.tolist(), "linear_in")
                pruning_idxs_attn_record.append(pruning_idxs.tolist())
            plot((18, 6), layer_x_mha,
                 [('mha', pruning_ratio_mha), ('gradient', gradient_norm_mha), ('weight', weight_norm_mha),
                  ('tylor imp', importance_norm_mha), ('scaled imp', importance_norm_mha_scale),
                  ])
            # for z in range(args.block_mlp_layer_start, args.block_mlp_layer_end):
            #     layer = model.model.layers[z]
            #     imps = imp(layer.mlp.gate_proj, "linear_out", [])
            #     pruning_idxs = get_mask(imps, layer.mlp.gate_proj, "linear_out",
            #                             args.pruning_ratio)
            #     apply_mask(layer.mlp.gate_proj, pruning_idxs.tolist(), "linear_out")
            #     apply_mask(layer.mlp.up_proj, pruning_idxs.tolist(), "linear_out")
            #     apply_mask(layer.mlp.down_proj, pruning_idxs.tolist(), "linear_in")
            imp_argsort = torch.argsort(whole_imps_mlp)
            n_pruned = len(imp_argsort) - int(
                len(imp_argsort) *
                (1 - args.pruning_ratio)
            )
            pruning_groups = imp_argsort[:n_pruned]
            pruning_groups = pruning_groups.tolist()
            pruning_groups.sort()
            pruning_idxs_mlp_record = []
            for z in range(args.block_mlp_layer_start, args.block_mlp_layer_end):
                layer = model.model.layers[z]
                pruning_idxs = torch.tensor([], dtype=torch.int8)
                for j in range(layer.mlp.gate_proj.out_features):
                    # z-> current layer index
                    # j-> current vector index (inside current layer)
                    if (
                            z - args.block_attention_layer_start) * layer.mlp.gate_proj.out_features + j in pruning_groups:
                        pruning_idxs = torch.cat(
                            (pruning_idxs,
                             torch.tensor([j])
                             ),
                            dim=0)
                pruning_ratio_mlp.append(apply_mask(layer.mlp.gate_proj, pruning_idxs.tolist(), "linear_out"))
                apply_mask(layer.mlp.up_proj, pruning_idxs.tolist(), "linear_out")
                apply_mask(layer.mlp.down_proj, pruning_idxs.tolist(), "linear_in")
                pruning_idxs_mlp_record.append(pruning_idxs.tolist())
            plot((18, 6), layer_x_mlp,
                 [('mlp', pruning_ratio_mlp), ('gradient', gradient_norm_mlp), ('weight', weight_norm_mlp),
                  ('tylor imp', importance_norm_mlp), ('scaled imp', importance_norm_mlp_scale),
                  ])

            # quantization_config = QuantoConfig(weights="int8")
            # model_q = LlamaForCausalLM.from_pretrained(args.base_model,
            #                                          low_cpu_mem_usage=True if args.torch_version >= 1.9 else False,
            #                                          quantization_config=quantization_config)
            # with torch.no_grad():
            #     for z in range(args.block_attention_layer_start, args.block_attention_layer_end):
            #         layer = model.model.layers[z]
            #         pruning_idxs = pruning_idxs_attn_record[z - args.block_attention_layer_start]
            #         apply_mask(layer.self_attn.q_proj, pruning_idxs, "linear_out")
            #         apply_mask(layer.self_attn.k_proj, pruning_idxs, "linear_out")
            #         apply_mask(layer.self_attn.v_proj, pruning_idxs, "linear_out")
            #         apply_mask(layer.self_attn.o_proj, pruning_idxs, "linear_in")
            #     for z in range(args.block_mlp_layer_start, args.block_mlp_layer_end):
            #         layer = model.model.layers[z]
            #         pruning_idxs = pruning_idxs_mlp_record[z - args.block_mlp_layer_start]
            #         apply_mask(layer.mlp.gate_proj, pruning_idxs, "linear_out")
            #         apply_mask(layer.mlp.up_proj, pruning_idxs, "linear_out")
            #         apply_mask(layer.mlp.down_proj, pruning_idxs, "linear_in")
            # print()
        else:

            for z in range(args.block_attention_layer_start, args.block_attention_layer_end):
                layer = model.model.layers[z]
                if args.mask_type_mha == 'q':
                    imps = imp(layer.self_attn.q_proj, "linear_out", [])
                    pruning_idxs = get_mask(imps, layer.self_attn.q_proj, "linear_out",
                                            args.pruning_ratio,
                                            layer.self_attn.head_dim if args.head_dim else 1)
                elif args.mask_type_mha == 'k':
                    imps = imp(layer.self_attn.k_proj, "linear_out", [])
                    pruning_idxs = get_mask(imps, layer.self_attn.k_proj, "linear_out",
                                            args.pruning_ratio,
                                            layer.self_attn.head_dim if args.head_dim else 1)
                elif args.mask_type_mha == 'v':
                    imps = imp(layer.self_attn.v_proj, "linear_out", [])
                    pruning_idxs = get_mask(imps, layer.self_attn.v_proj, "linear_out",
                                            args.pruning_ratio,
                                            layer.self_attn.head_dim if args.head_dim else 1)
                elif args.mask_type_mha == 'o':
                    imps = imp(layer.self_attn.o_proj, "linear_in", [])
                    pruning_idxs = get_mask(imps, layer.self_attn.o_proj, "linear_in",
                                            args.pruning_ratio,
                                            layer.self_attn.head_dim if args.head_dim else 1)
                else:
                    raise NotImplementedError()
                apply_mask(layer.self_attn.q_proj, pruning_idxs.tolist(), "linear_out")
                apply_mask(layer.self_attn.k_proj, pruning_idxs.tolist(), "linear_out")
                apply_mask(layer.self_attn.v_proj, pruning_idxs.tolist(), "linear_out")
                apply_mask(layer.self_attn.o_proj, pruning_idxs.tolist(), "linear_in")

            for z in range(args.block_mlp_layer_start, args.block_mlp_layer_end):
                layer = model.model.layers[z]
                if args.mask_type_mlp == 'gate_proj':
                    imps = imp(layer.mlp.gate_proj, "linear_out", [])
                    pruning_idxs = get_mask(imps, layer.mlp.gate_proj, "linear_out",
                                            args.pruning_ratio)
                elif args.mask_type_mlp == 'up_proj':
                    imps = imp(layer.mlp.up_proj, "linear_out", [])
                    pruning_idxs = get_mask(imps, layer.mlp.up_proj, "linear_out",
                                            args.pruning_ratio)
                elif args.mask_type_mlp == 'down_proj':
                    imps = imp(layer.mlp.down_proj, "linear_in", [])
                    pruning_idxs = get_mask(imps, layer.mlp.down_proj, "linear_in",
                                            args.pruning_ratio)
                else:
                    raise NotImplementedError()
                apply_mask(layer.mlp.gate_proj, pruning_idxs.tolist(), "linear_out")
                apply_mask(layer.mlp.up_proj, pruning_idxs.tolist(), "linear_out")
                apply_mask(layer.mlp.down_proj, pruning_idxs.tolist(), "linear_in")

    after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    for layer in model.model.layers:
        layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim
    logger.log("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters,
                                                                             after_pruning_parameters,
                                                                             100.0 * after_pruning_parameters / before_pruning_parameters))
    from torch.quantization import quantize_dynamic
    model = quantize_dynamic(
        model=model, qconfig_spec={nn.LSTM, nn.Linear}, dtype=torch.qint8, inplace=True
    )

    gc.collect()
    torch.cuda.empty_cache()

    if args.save_model:
        # model.half()
        torch.save({
            'model': model,
            'tokenizer': tokenizer,
        }, logger.best_checkpoint_path)

    # if args.eval_device != "cpu":
    #     model.half()
    model.to(args.eval_device)

    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if args.test_after_train:
        logger.log("\n==================Generation Results After Pruning================\n")

        model.eval()
        with torch.no_grad():
            for prompt in prompts:
                input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(args.eval_device)

                generation_output = model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    top_k=50,
                    max_length=args.max_seq_len,
                    top_p=args.top_p,
                    temperature=args.temperature,
                )

                result = tokenizer.decode(generation_output[0])
                logger.log(result)

        logger.log("\n==================Finish================\n")

    ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], args.max_seq_len, device=args.eval_device)
    logger.log("PPL after pruning: {}".format(ppl))
    logger.log("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated() / 1024 / 1024))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

    # argument for parsing
    parser.add_argument('--base_model', type=str, default="baffo32/decapoda-research-llama-7B-hf",
                        help='base model name')
    parser.add_argument('--save_ckpt_log_name', type=str, default="llama_prune",
                        help='the path for save the checkpoint and the log. The final path would be log/{your_name_here}_{pruner_type}_{pruning_ratio}')
    parser.add_argument('--pruning_ratio', type=float, default=0.5, help='pruning ratio')
    parser.add_argument('--pruner_type', type=str, default='taylor', help='pruner type')
    parser.add_argument('--mask_type_mha', type=str, default='q', help='use what layer as mask (in attention)')
    parser.add_argument('--mask_type_mlp', type=str, default='gate_proj', help='use what layer as mask (in MLP)')
    parser.add_argument('--head_dim', action='store_true', help='use head dimention when pruning MHA')
    # argument for generation
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='top p')
    parser.add_argument('--max_seq_len', type=int, default=128, help='max sequence length')

    # argument for layer-wise pruning/column-wise pruning
    parser.add_argument('--block_wise', action='store_true', help='block wise')
    parser.add_argument('--layer_wise', action='store_true', help='layer wise')
    parser.add_argument('--channel_wise', action='store_true', help='channel wise')
    parser.add_argument('--layer', type=int, default=12, help='remain the previous n layers')

    parser.add_argument('--block_attention_layer_start', type=int, help='start layer of block attention layers',
                        default=3)
    parser.add_argument('--block_attention_layer_end', type=int, help='end layer of block attention layers', default=31)
    parser.add_argument('--block_mlp_layer_start', type=int, help='start layer of block mlp layers', default=3)
    parser.add_argument('--block_mlp_layer_end', type=int, help='end layer of block mlp layers', default=31)

    parser.add_argument('--iterative_steps', type=int, default=1, help="Iteration step for pruning. Default=1")
    parser.add_argument('--grouping_strategy', type=str, default='sum', help='Reduce method for grouping')
    parser.add_argument('--global_pruning', action='store_true', help='whether global pruning')
    parser.add_argument('--taylor', type=str, default='param_first',
                        help='choose from [vectorize, param_second, param_first, param_mix]')
    parser.add_argument('--num_examples', type=int, default=10)
    parser.add_argument('--bias', action='store_true', help='bias compensation')
    # general argument
    parser.add_argument('--device', type=str, default="cpu", help='device')
    parser.add_argument('--test_before_train', action='store_true', help='whether test before train')
    parser.add_argument('--eval_device', type=str, default="cuda", help='eval device')
    parser.add_argument('--test_after_train', action='store_true', help='whether test after train')

    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--save_model', action='store_true', help='if save model')
    args = parser.parse_args()

    torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    args.torch_version = torch_version
    main(args)
