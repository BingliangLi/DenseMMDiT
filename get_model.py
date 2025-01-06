import torch
import os
import numpy as np
import diffusers
import random
import pickle

from PIL import Image
from tqdm.auto import tqdm
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_dense_stable_diffusion_3 import DenseStableDiffusion3Pipeline
from diffusers import DDIMScheduler

import transformers
from transformers import CLIPTextModel, CLIPTokenizer
import torch.nn.functional as F
from torchvision import transforms

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)
token = "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"
device= "mps"

with open('./dataset/testset.pkl', 'rb') as f:
    dataset = pickle.load(f)
layout_img_root = './dataset/testset_layout/'
# with open('./dataset/valset.pkl', 'rb') as f:
#     dataset = pickle.load(f)
# layout_img_root = './dataset/valset_layout/'

pipe = DenseStableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    cache_dir='./models/diffusers/',
    # text_encoder_3=None,
    # tokenizer_3=None,
    torch_dtype=torch.bfloat16
    ).to(device)
# pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing()