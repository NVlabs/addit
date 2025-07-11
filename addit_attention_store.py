# Copyright (C) 2025 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file
# located at the root directory.

import torch
from skimage import filters
import cv2
import torch.nn.functional as F
from skimage.filters import threshold_li, threshold_yen, threshold_multiotsu
import numpy as np
from visualization_utils import show_tensors
import matplotlib.pyplot as plt

def text_to_tokens(text, tokenizer):
    return [tokenizer.decode(x) for x in tokenizer(text, padding="longest", return_tensors="pt").input_ids[0]]

def flatten_list(l):
    return [item for sublist in l for item in sublist]

def gaussian_blur(heatmap, kernel_size=7, sigma=0):
    # Shape of heatmap: (H, W)
    heatmap = heatmap.cpu().numpy()
    heatmap = cv2.GaussianBlur(heatmap, (kernel_size, kernel_size), sigma)
    heatmap = torch.tensor(heatmap)
    
    return heatmap

def min_max_norm(x):
    return (x - x.min()) / (x.max() - x.min())

class AttentionStore:
    def __init__(self, prompts, tokenizer, 
                 subject_token=None, record_attention_steps=[],
                 is_cache_attn_ratio=False, attn_ratios_steps=[5]):
        
        self.text2image_store = {}
        self.image2text_store = {}
        self.count_per_layer = {}

        self.record_attention_steps = record_attention_steps
        self.record_attention_layers = ["transformer_blocks.13","transformer_blocks.14", "transformer_blocks.18", "single_transformer_blocks.23", "single_transformer_blocks.33"]

        self.attention_ratios = {}
        self._is_cache_attn_ratio = is_cache_attn_ratio
        self.attn_ratios_steps = attn_ratios_steps
        self.ratio_source = 'text'

        self.max_tokens_to_record = 10

        if isinstance(prompts, str):
            prompts = [prompts]
            batch_size = 1
        else:
            batch_size = len(prompts)

        tokens_per_prompt = []

        for prompt in prompts:
            tokens = text_to_tokens(prompt, tokenizer)
            tokens_per_prompt.append(tokens)

        self.tokens_to_record = []
        self.token_idxs_to_record = []

        if len(record_attention_steps) > 0:
            self.subject_tokens = flatten_list([text_to_tokens(x, tokenizer)[:-1] for x in [subject_token]])
            self.subject_tokens_idx = [tokens_per_prompt[1].index(x) for x in self.subject_tokens]
            self.add_token_idx = self.subject_tokens_idx[-1]

    def is_record_attention(self, layer_name, step_index):
        is_correct_layer = (self.record_attention_layers is None) or (layer_name in self.record_attention_layers)

        record_attention =  (step_index in self.record_attention_steps) and (is_correct_layer)

        return record_attention

    def store_attention(self, attention_probs, layer_name, batch_size, num_heads):
        text_len = 512
        timesteps = len(self.record_attention_steps)
        
        # Split batch and heads
        attention_probs = attention_probs.view(batch_size, num_heads, *attention_probs.shape[1:])

        # Mean over the heads
        attention_probs = attention_probs.mean(dim=1)

        # Attention: text -> image
        attention_probs_text2image = attention_probs[:, :text_len, text_len:]
        attention_probs_text2image = [attention_probs_text2image[0, self.subject_tokens_idx, :]]

        # Attention: image -> text
        attention_probs_image2text = attention_probs[:, text_len:, :text_len].transpose(1,2)
        attention_probs_image2text = [attention_probs_image2text[0, self.subject_tokens_idx, :]]

        if layer_name not in self.text2image_store:
            self.text2image_store[layer_name] = [x for x in attention_probs_text2image]
            self.image2text_store[layer_name] = [x for x in attention_probs_image2text]
        else:
            self.text2image_store[layer_name] = [self.text2image_store[layer_name][i] + x for i, x in enumerate(attention_probs_text2image)]
            self.image2text_store[layer_name] = [self.text2image_store[layer_name][i] + x for i, x in enumerate(attention_probs_image2text)]
    
    def is_cache_attn_ratio(self, step_index):
        return (self._is_cache_attn_ratio) and (step_index in self.attn_ratios_steps)
    
    def store_attention_ratios(self, attention_probs, step_index, layer_name):
        layer_prefix = layer_name.split(".")[0]
        
        if self.ratio_source == 'pixels':
            extended_attention_probs = attention_probs.mean(dim=0)[512:, :]
            extended_attention_probs_source = extended_attention_probs[:,:4096].sum(dim=1).view(64,64).float().cpu()
            extended_attention_probs_text = extended_attention_probs[:,4096:4096+512].sum(dim=1).view(64,64).float().cpu()
            extended_attention_probs_target = extended_attention_probs[:,4096+512:].sum(dim=1).view(64,64).float().cpu()
            token_attention = extended_attention_probs[:,4096+self.add_token_idx].view(64,64).float().cpu()

            stacked_attention_ratios = torch.cat([extended_attention_probs_source, extended_attention_probs_text, extended_attention_probs_target, token_attention], dim=1)
        elif self.ratio_source == 'text':
            extended_attention_probs = attention_probs.mean(dim=0)[:512, :]
            extended_attention_probs_source = extended_attention_probs[:,:4096].sum(dim=0).view(64,64).float().cpu()
            extended_attention_probs_target = extended_attention_probs[:,4096+512:].sum(dim=0).view(64,64).float().cpu()

            stacked_attention_ratios = torch.cat([extended_attention_probs_source, extended_attention_probs_target], dim=1)

        if step_index not in self.attention_ratios:
            self.attention_ratios[step_index] = {}

        if layer_prefix not in self.attention_ratios[step_index]:
            self.attention_ratios[step_index][layer_prefix] = []

        self.attention_ratios[step_index][layer_prefix].append(stacked_attention_ratios)

    def get_attention_ratios(self, step_indices=None, display_imgs=False):
        ratios = []

        if step_indices is None:
            step_indices = list(self.attention_ratios.keys())

        if len(step_indices) == 1:
            steps = f"Step: {step_indices[0]}"
        else:
            steps = f"Steps: [{step_indices[0]}-{step_indices[-1]}]"

        layer_prefixes = list(self.attention_ratios[step_indices[0]].keys())
        scores_per_layer = {}
        
        for layer_prefix in layer_prefixes:
            ratios = []

            for step_index in step_indices:
                if layer_prefix in self.attention_ratios[step_index]:
                    step_ratios = self.attention_ratios[step_index][layer_prefix]
                    step_ratios = torch.stack(step_ratios).mean(dim=0)
                    ratios.append(step_ratios)
            
            # Mean over the steps
            ratios = torch.stack(ratios).mean(dim=0)

            if self.ratio_source == 'pixels':
                source, text, target, token = torch.split(ratios, 64, dim=1)
                title = f"{steps}: Source={source.sum().item():.2f}, Text={text.sum().item():.2f}, Target={target.sum().item():.2f}, Token={token.sum().item():.2f}"
                ratios = min_max_norm(torch.cat([source, text, target], dim=1))
                token = min_max_norm(token)
                ratios = torch.cat([ratios, token], dim=1)
            elif self.ratio_source == 'text':
                source, target = torch.split(ratios, 64, dim=1)
                source_sum = source.sum().item()
                target_sum = target.sum().item()
                text_sum = 512 - (source_sum + target_sum)

                title = f"{steps}: Source={source_sum:.2f}, Target={target_sum:.2f}"
                ratios = min_max_norm(torch.cat([source, target], dim=1))
            
            if display_imgs:
                print(f"Layer: {layer_prefix}")
                show_tensors([ratios], [title])

            scores_per_layer[layer_prefix] = (source_sum, text_sum, target_sum)

        return scores_per_layer

    def plot_attention_ratios(self, step_indices=None):
        steps = list(self.attention_ratios.keys())
        score_per_layer = {
            'transformer_blocks': {},
            'single_transformer_blocks': {}
        }

        for i in steps:
            scores_per_layer = self.get_attention_ratios(step_indices=[i], display_imgs=False)

            for layer in self.attention_ratios[i]:
                source, text, target = scores_per_layer[layer]
                score_per_layer[layer][i] = (source, text, target)

        for layer_type in score_per_layer:
            x = list(score_per_layer[layer_type].keys())
            source_sums = [x[0] for x in score_per_layer[layer_type].values()]
            text_sums = [x[1] for x in score_per_layer[layer_type].values()]
            target_sums = [x[2] for x in score_per_layer[layer_type].values()]

            # Calculate the total sums for each stack (source + text + target)
            total_sums = [source_sums[j] + text_sums[j] + target_sums[j] for j in range(len(source_sums))]

            # Create stacked bar plots
            fig, ax = plt.subplots(figsize=(10, 6))
            indices = np.arange(len(x))

            # Plot source at the bottom
            ax.bar(indices, source_sums, label='Source', color='#6A2C70')

            # Plot text stacked on source
            ax.bar(indices, text_sums, label='Text', color='#B83B5E', bottom=source_sums)

            # Plot target stacked on text + source
            target_bottom = [source_sums[j] + text_sums[j] for j in range(len(source_sums))]
            ax.bar(indices, target_sums, label='Target', color='#F08A5D', bottom=target_bottom)

            # Annotate bars with percentage values
            for j, index in enumerate(indices):

                font_size = 12

                # Source percentage
                source_percentage = 100 * source_sums[j] / total_sums[j]
                ax.text(index, source_sums[j] / 2, f'{source_percentage:.1f}%', 
                        ha='center', va='center', rotation=90, color='white', 
                        fontsize=font_size, fontweight='bold')

                # Text percentage
                text_percentage = 100 * text_sums[j] / total_sums[j]
                ax.text(index, source_sums[j] + (text_sums[j] / 2), f'{text_percentage:.1f}%', 
                        ha='center', va='center', rotation=90, color='white', 
                        fontsize=font_size, fontweight='bold')

                # Target percentage
                target_percentage = 100 * target_sums[j] / total_sums[j]
                ax.text(index, source_sums[j] + text_sums[j] + (target_sums[j] / 2), f'{target_percentage:.1f}%', 
                        ha='center', va='center', rotation=90, color='white', 
                        fontsize=font_size, fontweight='bold')


            ax.set_xlabel('Step Index')
            ax.set_ylabel('Attention Ratio')
            ax.set_title(f'Attention Ratios for {layer_type}')
            ax.set_xticks(indices)
            ax.set_xticklabels(x)

            plt.legend()
            plt.show()

    def aggregate_attention(self, store, target_layers=None, resolution=None,
                            gaussian_kernel=3, thr_type='otsu', thr_number=0.5):
        if target_layers is None:
            store_vals = list(store.values())
        elif isinstance(target_layers, list):
            store_vals = [store[x] for x in target_layers]
        else:
            raise ValueError("target_layers must be a list of layer names or None.")

        # store vals = List[layers] of Tensor[batch_size, text_tokens, image_tokens]
        batch_size = len(store_vals[0])
        
        attention_maps = []
        attention_masks = []

        for i in range(batch_size):
            # Average over the layers
            agg_vals = torch.stack([x[i] for x in store_vals]).mean(dim=0)

            if resolution is None:
                size = int(agg_vals.shape[-1] ** 0.5)
                resolution = (size, size)
            
            agg_vals = agg_vals.view(agg_vals.shape[0], *resolution)

            if gaussian_kernel > 0:
                agg_vals = torch.stack([gaussian_blur(x.float(), kernel_size=gaussian_kernel) for x in agg_vals]).to(agg_vals.dtype)

            mask_vals = agg_vals.clone()

            for j in range(mask_vals.shape[0]):
                mask_vals[j] = (mask_vals[j] - mask_vals[j].min()) / (mask_vals[j].max() - mask_vals[j].min())
                np_vals = mask_vals[j].float().cpu().numpy()

                otsu_thr = filters.threshold_otsu(np_vals)
                li_thr = threshold_li(np_vals, initial_guess=otsu_thr)
                yen_thr = threshold_yen(np_vals)

                if thr_type == 'otsu':
                    thr = otsu_thr
                elif thr_type == 'yen':
                    thr = yen_thr
                elif thr_type == 'li':
                    thr = li_thr
                elif thr_type == 'number':
                    thr = thr_number
                elif thr_type == 'multiotsu':
                    thrs = threshold_multiotsu(np_vals, classes=3)

                    if thrs[1] > thrs[0] * 3.5:
                        thr = thrs[1]
                    else:
                        thr = thrs[0]

                    # Take the closest threshold to otsu_thr
                    # thr = thrs[np.argmin(np.abs(thrs - otsu_thr))]
                
                # alpha = 0.8
                # thr  = (alpha * thr + (1-alpha) * mask_vals[j].max())
                
                mask_vals[j] = (mask_vals[j] > thr).to(mask_vals[j].dtype)

            attention_maps.append(agg_vals)
            attention_masks.append(mask_vals)

        return attention_maps, attention_masks, self.tokens_to_record
