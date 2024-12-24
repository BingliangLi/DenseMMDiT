#!/usr/bin/env python
# coding: utf-8


import torch
import os
import numpy as np
import diffusers
import random
import pickle
import tqdm

from PIL import Image
from tqdm.auto import tqdm
from diffusers.pipelines.stable_diffusion_3.pipeline_dense_stable_diffusion_3 import DenseStableDiffusion3Pipeline
from diffusers import DDIMScheduler
from diffusers.models.attention_processor import AttnProcessor2_0

import math

import transformers
import torch.nn.functional as F

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import matplotlib.pyplot as plt

import torch
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

device= "cuda"

with open('./dataset/testset.pkl', 'rb') as f:
    dataset = pickle.load(f)
layout_img_root = './dataset/testset_layout/'
# with open('./dataset/valset.pkl', 'rb') as f:
#     dataset = pickle.load(f)
# layout_img_root = './dataset/valset_layout/'

output_root = './output/'
if not os.path.exists(output_root):
    os.makedirs(output_root)



pipe = DenseStableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    cache_dir='./models/diffusers/',
    text_encoder_3=None,
    tokenizer_3=None,
    torch_dtype=torch.bfloat16
    ).to(device)

class JointAttnProcessor_mod:
    """
    Attention processor for SD3-like self-attention projections.
    This processor handles both self-attention and cross-attention mechanisms.
    """

    def __init__(self):
        pass

    def __call__(
        self,
        attn,  # Attention module
        hidden_states: torch.FloatTensor,  # Input tensor [batch_size, seq_len, hidden_dim]
        encoder_hidden_states: torch.FloatTensor = None,  # Optional context tensor [batch_size, context_len, hidden_dim]
        attention_mask = None,  # Optional mask tensor [batch_size, seq_len]
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        # Store residual for skip connection
        residual = hidden_states  # [batch_size, seq_len, hidden_dim]
        batch_size = hidden_states.shape[0]

        # Project input into query, key, value spaces
        # Each projection: [batch_size, seq_len, hidden_dim]
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # Reshape for multi-head attention
        inner_dim = key.shape[-1]  # Total dimension across all heads
        head_dim = inner_dim // attn.heads  # Dimension per attention head

        # Reshape tensors to [batch_size, n_heads, seq_len, head_dim]
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Apply normalization if specified
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        

        # Handle cross-attention if encoder_hidden_states is provided
        if encoder_hidden_states is not None:
            # Project encoder states to query, key, value
            # [batch_size, context_len, hidden_dim] -> [batch_size, n_heads, context_len, head_dim]
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            # Reshape encoder projections
            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            # Apply normalization to encoder projections if specified
            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # Concatenate self and cross attention tensors
            # [batch_size, n_heads, seq_len + context_len, head_dim]
            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        # Compute scaled dot-product attention
        L, S = query.size(-2), key.size(-2)  # L: target sequence length, S: source sequence length
        scale_factor = 1 / math.sqrt(query.size(-1))  # Scaling factor for numerical stability
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attention_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attention_mask

        # Compute attention weights and apply them to values
        # [batch_size, n_heads, seq_len, seq_len]
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        
        ################################################# regulation for self attention and self attention
        sa_ = True if encoder_hidden_states is None else False
        global COUNT
        if COUNT/37 < STEPS * reg_part:
            treg = torch.pow(timesteps[COUNT//37]/1000, 5)
            if sa_:
                # self attention regulation
                min_value = attn_weight[int(attn_weight.size(0)/2):].min(-1)[0].unsqueeze(-1)
                max_value = attn_weight[int(attn_weight.size(0)/2):].max(-1)[0].unsqueeze(-1)  
                # print("attn.heads", attn.heads)
                mask = sreg_maps[attn_weight.size(2)].repeat(attn.heads,1,1)
                # print("mask", mask.shape)
                size_reg = reg_sizes[attn_weight.size(2)].repeat(attn.heads,1,1)
                # print("size_reg", size_reg.shape)
                
                # Apply positive and negative regulation for self-attention
                attn_weight[int(attn_weight.size(0)/2):] += (mask>0)*size_reg*sreg*treg*(max_value-attn_weight[int(attn_weight.size(0)/2):])
                attn_weight[int(attn_weight.size(0)/2):] -= ~(mask>0)*size_reg*sreg*treg*(attn_weight[int(attn_weight.size(0)/2):]-min_value)
            else:
                # cross attention regulation
                pass
            
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.0, train=False)
        # [batch_size, n_heads, seq_len, head_dim]
        hidden_states = attn_weight @ value
        # Reshape back to original dimensions
        # [batch_size, seq_len, hidden_dim]
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Handle cross-attention output
        if encoder_hidden_states is not None:
            # Split self-attention and cross-attention outputs
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # Final linear projection and dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        # Return appropriate output based on whether cross-attention was used
        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
        
joint_attn_processor = JointAttnProcessor_mod()

def mod_forward_sd3(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states = None,
    attention_mask = None,
    **cross_attention_kwargs,
) -> torch.Tensor:
    r"""
    The forward method of the `Attention` class.

    Args:
        hidden_states (`torch.Tensor`):
            The hidden states of the query.
        encoder_hidden_states (`torch.Tensor`, *optional*):
            The hidden states of the encoder.
        attention_mask (`torch.Tensor`, *optional*):
            The attention mask to use. If `None`, no mask is applied.
        **cross_attention_kwargs:
            Additional keyword arguments to pass along to the cross attention.

    Returns:
        `torch.Tensor`: The output of the attention layer.
    """
    # The `Attention` class can call different attention processors / attention functions
    # here we simply pass along all tensors to the selected processor class
    # For standard processors that are defined here, `**cross_attention_kwargs` is empty

    attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
    quiet_attn_parameters = {"ip_adapter_masks"}
    unused_kwargs = [
        k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters
    ]

    cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}
    if isinstance(self.processor, AttnProcessor2_0):
        pass
    else:
        self.processor = joint_attn_processor
    return self.processor(
        self,
        hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=attention_mask,
        **cross_attention_kwargs,
    )
for _module in pipe.transformer.modules():
    if _module.__class__.__name__ == "Attention":
        _module.__class__.__call__ = mod_forward_sd3


if __name__ == "__main__":
    STEPS=10
    pipe.scheduler.set_timesteps(STEPS)
    timesteps = pipe.scheduler.timesteps
    sp_sz = pipe.transformer.sample_size
    bsz = 1

    # loop through 0-249 with tqdm
    for idx in tqdm(range(250)):
        layout_img_path = layout_img_root+str(idx)+'.png'
        prompts = [dataset[idx]['textual_condition']] + dataset[idx]['segment_descriptions']

        ########### tokenizer 1 and text encoder 1
        text_input = pipe.tokenizer(prompts, padding="max_length", return_length=True, return_overflowing_tokens=False, 
                                    max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        cond_embeddings = pipe.text_encoder(text_input.input_ids.to(device))[0]

        uncond_input = pipe.tokenizer([""]*bsz, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                                    truncation=True, return_tensors="pt")
        uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]

        for i in range(1,len(prompts)):
            wlen = text_input['length'][i] - 2
            widx = text_input['input_ids'][i][1:1+wlen]
            for j in range(77):
                if (text_input['input_ids'][0][j:j+wlen] == widx).sum() == wlen:
                    break

        ############ tokenizer 2 and text encoder 2
        text_input_2 = pipe.tokenizer_2(prompts, padding="max_length", return_length=True, return_overflowing_tokens=False, 
                                    max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        cond_embeddings_2 = pipe.text_encoder_2(text_input_2.input_ids.to(device))[0]

        uncond_input_2 = pipe.tokenizer_2([""]*bsz, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                                    truncation=True, return_tensors="pt")
        uncond_embeddings_2 = pipe.text_encoder_2(uncond_input_2.input_ids.to(device))[0]

        for i in range(1,len(prompts)):
            wlen = text_input_2['length'][i] - 2
            widx = text_input_2['input_ids'][i][1:1+wlen]
            for j in range(77):
                if (text_input_2['input_ids'][0][j:j+wlen] == widx).sum() == wlen:
                    break
        ############
        layout_img_ = np.asarray(Image.open(layout_img_path).resize([sp_sz*8,sp_sz*8]))[:,:,:3]
        unique, counts = np.unique(np.reshape(layout_img_,(-1,3)), axis=0, return_counts=True)
        sorted_idx = np.argsort(-counts)

        layouts_ = []

        for i in range(len(prompts)-1):
            if (unique[sorted_idx[i]] == [0, 0, 0]).all() or (unique[sorted_idx[i]] == [255, 255, 255]).all():
                layouts_ = [((layout_img_ == unique[sorted_idx[i]]).sum(-1)==3).astype(np.uint8)] + layouts_
            else:
                layouts_.append(((layout_img_ == unique[sorted_idx[i]]).sum(-1)==3).astype(np.uint8))
                
        layouts = [torch.FloatTensor(l).unsqueeze(0).unsqueeze(0).cuda() for l in layouts_]
        layouts = F.interpolate(torch.cat(layouts),(sp_sz,sp_sz),mode='nearest')

        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds, creg_maps = pipe.encode_prompt(
            prompt=prompts,
            prompt_2=prompts,
            prompt_3=prompts,
            layouts=layouts,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            bsz=bsz,
        )

        ###########################
        ###### prep for sreg ###### 
        ###########################
        sreg_maps = {}
        reg_sizes = {}

        res = 64
        layouts_s = F.interpolate(layouts,(res, res),mode='nearest')
        layouts_s = (layouts_s.view(layouts_s.size(0),1,-1)*layouts_s.view(layouts_s.size(0),-1,1)).sum(0).unsqueeze(0).repeat(1, 1, 1)
        layouts_s = layouts_s.bool()
        reg_sizes[np.power(res, 2)] = 1-1.*layouts_s.sum(-1, keepdim=True)/(np.power(res, 2))
        sreg_maps[np.power(res, 2)] = layouts_s
            
            
        ###########################
        ###### prep for creg ######
        ###########################
        # pww_maps = torch.zeros(1, 77, sp_sz, sp_sz).to(device)
        # for i in range(1,len(prompts)):
        #     wlen = text_input['length'][i] - 2
        #     widx = text_input['input_ids'][i][1:1+wlen]
        #     for j in range(77):
        #         if (text_input['input_ids'][0][j:j+wlen] == widx).sum() == wlen:
        #             pww_maps[:,j:j+wlen,:,:] = layouts[i-1:i]
        #             cond_embeddings[0][j:j+wlen] = cond_embeddings[i][1:1+wlen]
        #             print(prompts[i], i, '-th segment is handled.')
        #             break

        # for i in range(1,len(prompts)):
        #     wlen = text_input_2['length'][i] - 2
        #     widx = text_input_2['input_ids'][i][1:1+wlen]
        #     for j in range(77):
        #         if (text_input_2['input_ids'][0][j:j+wlen] == widx).sum() == wlen:
        #             pww_maps[:,j:j+wlen,:,:] = layouts[i-1:i]
        #             cond_embeddings_2[0][j:j+wlen] = cond_embeddings_2[i][1:1+wlen]
        #             print(prompts[i], i, '-th segment is handled.')
        #             break
                    
        # creg_maps = {}
        # for r in range(4):
        #     res = int(sp_sz/np.power(2,r))
        #     layout_c = F.interpolate(pww_maps,(res,res),mode='nearest').view(1,77,-1).permute(0,2,1).repeat(bsz,1,1)
        #     creg_maps[np.power(res, 2)] = layout_c

            
        ###########################    
        #### prep for text_emb ####
        ###########################
        # text_cond = torch.cat([uncond_embeddings, cond_embeddings[:1].repeat(bsz,1,1)])

        # plot sreg map and creg map side by side, save to file, name as idx_reg_map.png to output_root
        sreg_map = sreg_maps[4096][0].cpu().numpy()
        creg_map = creg_maps[4096][0].cpu().numpy()
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(sreg_map, cmap='viridis')
        ax[1].imshow(creg_map, cmap='viridis')

        fig.savefig(f'{output_root}/{idx}_reg_map.png')

        reg_part = .2
        sreg = .2
        creg = 1.
        COUNT = 0

        with torch.no_grad():
            image = pipe(prompts, layouts=layouts, num_inference_steps=STEPS, guidance_scale=7).images[0]

        mask = layout_img_.astype(np.uint8)
        image_mask_overlap = mask+np.asarray(image)

        # save image, mask, and image_mask_overlap to output_root
        Image.fromarray(image_mask_overlap).save(f'{output_root}/{idx}_image_mask_overlap.png')
        Image.fromarray(mask).save(f'{output_root}/{idx}_mask.png')
        Image.fromarray(np.asarray(image)).save(f'{output_root}/{idx}.png')

        # save image, mask side by side to output_root
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image_mask_overlap)
        ax[1].imshow(mask)
        fig.savefig(f'{output_root}/{idx}_side_by_side.png')
        
        # clear memory
        del image, mask, image_mask_overlap
        torch.cuda.empty_cache()
