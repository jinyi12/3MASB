"""
Data Generation Utilities
=========================

This module provides functions for generating synthetic data used in the
asymmetric bridge experiments, such as multiscale Gaussian Random Fields (GRF)
and spiral distributions.
"""

import torch
from torch import Tensor
import numpy as np
from tqdm import trange
from typing import Dict, Tuple

# ============================================================================
# Random Field Generation
# ============================================================================

def gaussian_blur_periodic(input_tensor: Tensor, kernel_size: int, sigma: float) -> Tensor:
    """Apply Gaussian blur with periodic boundary conditions."""
    if sigma <= 1e-9 or kernel_size <= 1:
        return input_tensor
    if kernel_size % 2 == 0:
        kernel_size += 1
    k = torch.arange(kernel_size, dtype=torch.float32, device=input_tensor.device)
    center = (kernel_size - 1) / 2
    gauss_1d = torch.exp(-0.5 * ((k - center) / sigma)**2)
    gauss_1d = gauss_1d / gauss_1d.sum()
    gauss_2d = torch.outer(gauss_1d, gauss_1d)
    if input_tensor.dim() != 4:
        raise ValueError("Input tensor must be [B, C, H, W]")
    C_in = input_tensor.shape[1]
    kernel = gauss_2d.expand(C_in, 1, kernel_size, kernel_size)
    padding = (kernel_size - 1) // 2
    padded_input = torch.nn.functional.pad(input_tensor, (padding, padding, padding, padding), mode='circular')
    output = torch.nn.functional.conv2d(padded_input, kernel, padding=0, groups=C_in)
    return output

class RandomFieldGenerator2D:
    """Generator for 2D Gaussian Random Fields with multiscale coarsening."""
    def __init__(self, nx=100, ny=100, lx=1.0, ly=1.0, device='cpu'):
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.device = device

    def generate_random_field(self, mean=10.0, std=2.0, correlation_length=0.2, covariance_type="exponential"):
        dx = self.lx / self.nx
        dy = self.ly / self.ny
        white_noise = np.random.normal(0, 1, (self.nx, self.ny))
        fourier_coefficients = np.fft.fft2(white_noise)
        kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ny, d=dy)
        Kx, Ky = np.meshgrid(kx, ky, indexing="ij")
        K = np.sqrt(Kx**2 + Ky**2)
        l = correlation_length
        if covariance_type == "exponential":
            denom = (1 + (l * K) ** 2)
            P = (2 * np.pi * l**2) / np.maximum(1e-9, denom ** (1.5))
        elif covariance_type == "gaussian":
            P = np.pi * l**2 * np.exp(-((l * K) ** 2) / 4)
        else:
            raise ValueError("Invalid covariance_type")
        P = np.nan_to_num(P)
        fourier_coefficients *= np.sqrt(P)
        field = np.fft.ifft2(fourier_coefficients).real
        field_std = np.std(field)
        if field_std > 1e-9:
            field = (field - np.mean(field)) / field_std * std + mean
        else:
            field = np.full_like(field, mean)
        return field

    def coarsen_field(self, field, H):
        if isinstance(field, np.ndarray):
            field = torch.from_numpy(field).to(self.device)
        if field.dim() == 3:
            field = field.unsqueeze(1)
            squeeze_channel = True
        elif field.dim() == 4:
            squeeze_channel = False
        else:
            raise ValueError("Unsupported field dimensions (must be 3 or 4)")
        
        pixel_size = self.lx / self.nx
        filter_sigma_phys = H / 6.0
        filter_sigma_pix = filter_sigma_phys / pixel_size
        
        if filter_sigma_pix < 1e-6:
            smooth = field
        else:
            kernel_size = int(6 * filter_sigma_pix)
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel_size = max(3, kernel_size)
            smooth = gaussian_blur_periodic(field, kernel_size=kernel_size, sigma=filter_sigma_pix)
        
        coarse = smooth
        if squeeze_channel:
            coarse = coarse.squeeze(1)
        return coarse

def generate_multiscale_grf_data(
    N_samples: int, T: float = 1.0, N_constraints: int = 5, resolution: int = 32,
    L_domain: float = 1.0, micro_corr_length: float = 0.1, H_max_factor: float = 0.5,
    mean_val: float = 10.0, std_val: float = 2.0, covariance_type: str = "exponential",
    device: str = 'cpu'
) -> Tuple[Dict[float, Tensor], int]:
    """Generate multiscale Gaussian Random Field data."""
    print(f"\n--- Generating Multiscale GRF Data (Resolution: {resolution}x{resolution}) ---")
    time_steps = torch.linspace(0, T, N_constraints)
    marginal_data = {}
    data_dim = resolution * resolution
    generator = RandomFieldGenerator2D(nx=resolution, ny=resolution, lx=L_domain, ly=L_domain, device=device)
    
    print("Generating base microscopic fields (t=0)...")
    micro_fields = []
    for _ in trange(N_samples):
        field = generator.generate_random_field(mean=mean_val, std=std_val, correlation_length=micro_corr_length, covariance_type=covariance_type)
        micro_fields.append(field)
    micro_fields_tensor = torch.tensor(np.array(micro_fields), dtype=torch.float32).to(device)
    
    print("Applying progressive coarsening filters...")
    H_max = L_domain * H_max_factor
    for t in time_steps:
        t_val = t.item()
        t_norm = t_val / T
        H_t = t_norm * H_max
        coarsened_fields = generator.coarsen_field(micro_fields_tensor, H=H_t)
        flattened_fields = coarsened_fields.reshape(N_samples, data_dim)
        marginal_data[t_val] = flattened_fields
        mean_std = torch.std(flattened_fields, dim=0).mean().item()
        print(f"  t={t_val:.2f}: H={H_t:.4f}, Mean Std Dev across field: {mean_std:.4f}")
        
    print("Multiscale data generation complete.")
    return marginal_data, data_dim

def generate_spiral_distributional_data(
    N_constraints: int = 5, 
    T: float = 1.0, 
    data_dim: int = 3,
    N_samples_per_marginal: int = 128,
    noise_std: float = 0.1
) -> Tuple[Dict[float, Tensor], Tensor]:
    """
    Generate synthetic distributional data around a spiral trajectory.
    """
    time_steps = torch.linspace(0, T, N_constraints)
    marginal_data = {}

    for t in time_steps:
        t_val = t.item()
        
        if data_dim == 3:
            # 3D spiral mean
            mu = torch.tensor([
                torch.sin(t * 2 * torch.pi / T),
                torch.cos(t * 2 * torch.pi / T), 
                t / T
            ])
        else:
            raise NotImplementedError("Only data_dim=3 supported for spiral data.")
        
        # Generate samples around the mean
        samples = mu + noise_std * torch.randn(N_samples_per_marginal, data_dim)
        marginal_data[t_val] = samples
    
    return marginal_data, time_steps
