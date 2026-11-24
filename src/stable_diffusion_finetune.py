"""
Módulo de Fine-Tuning Eficiente (LoRA) para Stable Diffusion v1.5 utilizando Hugging Face diffusers.
Autor: Oscar - Programa Académico en Inteligencia (Noviembre de 2026)
Referencias: Rombach et al. (2022); Hu et al. (2022).
"""

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer


def setup_stable_diffusion_lora(model_id: str = "runwayml/stable-diffusion-v1-5", rank: int = 4, device: str = "cuda"):
    """Carga los componentes del modelo de difusión latente e inyecta adaptadores LoRA en la U-Net."""
    dtype = torch.float16 if device == "cuda" and torch.cuda.is_available() else torch.float32
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype).to(device)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype).to(device)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype).to(device)
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_cfg)
    return tokenizer, text_encoder, vae, unet, scheduler
