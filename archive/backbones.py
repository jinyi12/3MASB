#!/usr/bin/env python3
"""
Backbone Architectures for C-CVEP
=================================

This module provides backbone neural network architectures for the C-CVEP framework.
The TimeConditionedMLP is designed to handle time-conditioned inputs for both
velocity and energy networks.

Key Features:
- Time embedding for conditioning
- Flexible hidden layer configuration
- Support for multi-dimensional data (e.g., GRF images)
- Compatible with both VelocityNetwork and EnergyNetwork
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import math


class TimeConditionedMLP(nn.Module):
    """
    A flexible time-conditioned MLP backbone for C-CVEP.
    
    This architecture embeds the time variable and concatenates it with the input data,
    then processes through a series of hidden layers with ReLU activations.
    """
    
    def __init__(self, data_dim: int, hidden_dims: List[int], time_embedding_dim: int = 32):
        super().__init__()
        self.data_dim = data_dim
        
        # Time embedding network
        self.time_embedder = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.ReLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )
        
        # Main MLP layers
        layers = []
        input_dim = data_dim + time_embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
            
        # Output layer (no activation)
        layers.append(nn.Linear(input_dim, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with time conditioning.
        
        Args:
            x: Input data tensor, shape (B, ...) where ... can be any spatial dimensions
            s: Time tensor, shape (B,) or (B, 1)
            
        Returns:
            Output tensor with same shape as x
        """
        # Ensure s is (B, 1) for embedding
        if s.dim() == 1:
            s = s.unsqueeze(-1)
        
        # Embed time
        s_emb = self.time_embedder(s)
        
        # Handle multi-dimensional data (e.g., GRF images) by flattening
        original_shape = x.shape
        if x.dim() > 2:
             x_flat = x.view(x.shape[0], -1)
        else:
             x_flat = x

        # Validate input dimension
        if x_flat.shape[1] != self.data_dim:
             raise ValueError(f"Input data dimension mismatch. Expected {self.data_dim}, got {x_flat.shape[1]}.")

        # Concatenate data and time embedding
        h = torch.cat([x_flat, s_emb], dim=1)
        
        # Forward through network
        out = self.net(h)
        
        # Reshape back to original spatial dimensions if necessary
        return out.view(original_shape)


class SimpleTimeConditionedMLP(nn.Module):
    """
    A simplified version of TimeConditionedMLP for lighter computational requirements.
    Uses direct time concatenation without embedding.
    """
    
    def __init__(self, data_dim: int, hidden_dims: List[int]):
        super().__init__()
        self.data_dim = data_dim
        
        # Main MLP layers with direct time concatenation
        layers = []
        input_dim = data_dim + 1  # +1 for time
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(input_dim, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with direct time concatenation.
        
        Args:
            x: Input data tensor, shape (B, ...) where ... can be any spatial dimensions
            s: Time tensor, shape (B,) or (B, 1)
            
        Returns:
            Output tensor with same shape as x
        """
        # Ensure s is (B, 1)
        if s.dim() == 1:
            s = s.unsqueeze(-1)
        
        # Handle multi-dimensional data by flattening
        original_shape = x.shape
        if x.dim() > 2:
             x_flat = x.view(x.shape[0], -1)
        else:
             x_flat = x

        # Validate input dimension
        if x_flat.shape[1] != self.data_dim:
             raise ValueError(f"Input data dimension mismatch. Expected {self.data_dim}, got {x_flat.shape[1]}.")

        # Concatenate data and time
        h = torch.cat([x_flat, s], dim=1)
        
        # Forward through network
        out = self.net(h)
        
        # Reshape back to original spatial dimensions
        return out.view(original_shape)


# =============================================================================
# U-Net Components for Spatial Backbones
# =============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for improved temporal conditioning."""
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0: 
            t = t.unsqueeze(0)
        device = t.device
        half_dim = self.embedding_dim // 2
        if half_dim < 1: 
            raise ValueError("embedding_dim must be >= 2")
        
        emb = math.log(10000) / max(1, (half_dim - 1))
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        
        t = t.view(-1, 1)
        emb = t * emb.unsqueeze(0)
        
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ConvBlock(nn.Module):
    """
    Convolutional block with time conditioning (FiLM) and circular padding.
    Designed for GRF data with periodic boundary conditions.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, num_groups=8):
        super().__init__()
        num_groups = min(num_groups, out_channels)
        if out_channels % num_groups != 0 and num_groups > 1:
             num_groups = 1

        # Use circular padding for GRF boundary conditions
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular')
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular')
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.activation = nn.SiLU()
        
        # FiLM conditioning
        if time_emb_dim > 0:
            self.film = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels * 2)
            )
        else:
            self.film = None

        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, t_emb=None):
        residual = self.residual_conv(x)
        h = self.activation(self.norm1(self.conv1(x)))
        
        # Apply FiLM
        if self.film is not None and t_emb is not None:
            film_params = self.film(t_emb)
            film_params = film_params.view(film_params.shape[0], film_params.shape[1], 1, 1)
            scale, shift = film_params.chunk(2, dim=1)
            h = h * (1 + scale) + shift
            
        h = self.activation(self.norm2(self.conv2(h)))
        return h + residual


class DownBlock(nn.Module):
    """Downsampling block for U-Net encoder."""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, time_emb_dim)
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x, t_emb):
        h = self.conv_block(x, t_emb)
        p = self.pool(h)
        return p, h


class UpBlock(nn.Module):
    """Upsampling block for U-Net decoder."""
    def __init__(self, in_channels, skip_channels, time_emb_dim):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_block = ConvBlock(in_channels + skip_channels, skip_channels, time_emb_dim)

    def forward(self, x, skip, t_emb):
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
             x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x, t_emb)


class TimeConditionedUNet(nn.Module):
    """
    A flexible time-conditioned U-Net backbone for C-CVEP spatial data.
    Designed to preserve spatial correlations in GRF data.
    """
    def __init__(self, 
                 data_dim: int, resolution: int, in_channels: int = 1, out_channels: int = 1,
                 hidden_dims: List[int] = [64, 128, 256], time_embedding_dim: int = 64):
        super().__init__()
        self.data_dim = data_dim
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_channels = out_channels

        if data_dim != in_channels * resolution * resolution:
             raise ValueError(f"data_dim {data_dim} inconsistent with channels {in_channels} and resolution {resolution}.")

        # Time embedding network
        self.time_embedder = nn.Sequential(
            SinusoidalTimeEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )
        
        # U-Net layers
        self.init_conv = ConvBlock(in_channels, hidden_dims[0], time_emb_dim=0)
        
        # Downsampling path
        self.downs = nn.ModuleList()
        in_ch = hidden_dims[0]
        current_res = resolution
        
        for i, out_ch in enumerate(hidden_dims[1:], 1):  # Skip first element since it's used for init_conv
            if current_res > 4:
                self.downs.append(DownBlock(in_ch, out_ch, time_embedding_dim))
                in_ch = out_ch
                current_res //= 2
            
        # Bottleneck
        self.bottleneck = ConvBlock(in_ch, in_ch, time_embedding_dim)
        
        # Upsampling path
        self.ups = nn.ModuleList()
        for i in range(len(self.downs) -1, -1, -1):
            # Skip connection has the output channels of the corresponding down block
            skip_ch = self.downs[i].conv_block.conv1.out_channels
            self.ups.append(UpBlock(in_ch, skip_ch, time_embedding_dim))
            in_ch = skip_ch
            
        # Final output convolution
        self.final_conv = nn.Conv2d(in_ch, out_channels, kernel_size=1)

    def _format_input(self, x: torch.Tensor) -> torch.Tensor:
        """Formats input tensor from (B, D) or (B, C, H, W) to (B, C, H, W)."""
        B = x.shape[0]
        if x.dim() == 2:
            return x.view(B, self.in_channels, self.resolution, self.resolution)
        elif x.dim() == 4:
            return x
        else:
            raise ValueError(f"Unsupported input dimensions: {x.dim()}. Expected 2 or 4.")

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x_spatial = self._format_input(x)
        original_shape = x.shape
        was_flattened = len(original_shape) == 2

        if s.dim() > 1: 
            s = s.view(-1)
        t_emb = self.time_embedder(s)
        
        h = self.init_conv(x_spatial)
        
        skips = []
        for down in self.downs:
            h, skip = down(h, t_emb)
            skips.append(skip)
            
        h = self.bottleneck(h, t_emb)
        
        skips = list(reversed(skips))
        for i, up in enumerate(self.ups):
            h = up(h, skips[i], t_emb)
            
        out = self.final_conv(h)
        
        if was_flattened:
            # If input was flattened, return flattened output with correct dimensions
            B = original_shape[0]
            output_dim = self.out_channels * self.resolution * self.resolution
            return out.view(B, output_dim)
        return out