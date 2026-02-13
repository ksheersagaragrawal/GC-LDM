"""
genre conditional diffusion model

AudioLDM citation:
    https://github.com/haoheliu/AudioLDM
Core AudioLDM UNet class: audioldm/latent_diffusion/unet.py 
UNET2D citation:
    https://huggingface.co/docs/diffusers/main/api/models/unet2d-cond
    
We are heavily inspired by Audio LDM code and are using parts of their pipeline code such as using latent diffusion, cross-attention UNet and prediction of epsilon/noise but we differ in that text conditioning is replaced with genre. Also technically, we use HuggingFace's UNet2DConditionModel instead of AudioLDM's custom UNet but they are functionally accomplishing the same thing.

"""
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from models.gen_conditioning import GenreCondition

class GenreConditionedUNet(nn.Module):

    def __init__(self):
        super().__init__()
        # this entire block is inspired by AudioLDM's class UNetModel
        # they manually defined/build ResBlocks, Attention blocks, Downsampling, Upsampling and SpatialTransformer cross-attention 
        # we attempted to do this but due to issues, pivoted to HuggingFace's UNet2DConditionModel, which allows you to select
        # for all of those subparts

        self.unet = UNet2DConditionModel(
            # shape of VAE z
            sample_size=(256, 16),
            #latent channel counts set in encoder/decoder parts
            # in AudioLDM these are parameters in EncoderUNetModel() setup and used in line 577: conv_nd(dims, in_channels, model_channels, 3, padding=1)
            in_channels=8,
            out_channels=8,
            # each resolution level has this many ResBlocks
            # residual block countfor each never
            # in audioLDM these are parameters in UNetModel() setup and used in line 586 and 992:for _ in range(num_res_blocks)
            layers_per_block=2,
            # this sets the sizes for the channel variable
            # in line 471 and 869 they set channel_mult=(1, 2, 4, 8). What this does is act as a scaling factor of their model_channels var 
            # so its equivalent if we set 128, 256, 512, 1024
            block_out_channels=(128, 256, 512, 1024),
            # they inject attention with SpatialTransformer across the file in UNetModel
            # we will use an equivalent CrossAttnDownBlock2D
            # AudioLDM has 4 levels so we do four parts of downsampling
            down_block_types=("CrossAttnDownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D"),
            # for upsampling AudioLDM uses SpatialTransformer again so
            # we will use an equivalent CrossAttnUpBlock2D            
            # AudioLDM has 4 levels so we do four parts of upsampling
            up_block_types=("CrossAttnUpBlock2D","CrossAttnUpBlock2D", "CrossAttnUpBlock2D","CrossAttnUpBlock2D"),
            # AudioLDM uses context_dim = 512 so we will also use that
            cross_attention_dim=512,
            # AudioLDM reference does num_heads div by num_head_channels
            # so we do the same
            attention_head_dim=8
        )
        # our genre (where AudioLDM does their text work)
        self.genre_condition = GenreCondition()

    def forward(self, noisy_latents: torch.Tensor, timesteps: torch.Tensor, genre_ids: torch.Tensor,) -> torch.Tensor:
        # get tokens after they are projected
        genre_tokens = self.genre_condition(genre_ids)
        # context = conditioning tokens
        # step forward with latent, embedding, and genre tokens to condition
        out = self.unet(sample=noisy_latents, timestep=timesteps, encoder_hidden_states=genre_tokens)
        return out.sample
