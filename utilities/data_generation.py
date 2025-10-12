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
from typing import Dict, Tuple, Optional
# Imports required for KL decomposition
from scipy.spatial.distance import pdist, squareform
import scipy.stats as stats

# ============================================================================
# Utility Functions
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
    # Use 'circular' padding for periodic boundaries
    padded_input = torch.nn.functional.pad(input_tensor, (padding, padding, padding, padding), mode='circular')
    output = torch.nn.functional.conv2d(padded_input, kernel, padding=0, groups=C_in)
    return output

def transform_to_non_gaussian(
    gaussian_field: np.ndarray, mu_target: float, sigma_target: float, distribution: str = 'gamma'
) -> np.ndarray:
    """
    Transform a Gaussian random field to a non-Gaussian field using the Probability Integral Transform.
    (For future use).
    """
    # 1. Standardize the input field (ensure it is approximately standard normal)
    g_mean = np.mean(gaussian_field)
    g_std = np.std(gaussian_field)
    
    if g_std < 1e-9:
        standard_gaussian = np.zeros_like(gaussian_field)
    else:
        standard_gaussian = (gaussian_field - g_mean) / g_std

    # 2. Apply Standard Normal CDF (Z -> U)
    z_normcdf = stats.norm.cdf(standard_gaussian, 0, 1)
    # Clip for numerical stability near 0 and 1
    z_normcdf = np.clip(z_normcdf, 1e-9, 1 - 1e-9)

    # 3. Apply Inverse CDF (PPF) of the target distribution (U -> X)
    if distribution == 'gamma':
        # Parameters using Method of Moments
        shape = (mu_target / sigma_target)**2
        scale = (sigma_target**2) / mu_target
        non_gaussian_field = stats.gamma.ppf(z_normcdf, shape, scale=scale)
    elif distribution == 'lognormal':
        sigma_ln = np.sqrt(np.log(1 + (sigma_target/mu_target)**2))
        mu_ln = np.log(mu_target) - 0.5 * sigma_ln**2
        non_gaussian_field = stats.lognorm.ppf(z_normcdf, s=sigma_ln, scale=np.exp(mu_ln))
    else:
        raise ValueError(f"Unsupported distribution type: {distribution}")

    return non_gaussian_field

# ============================================================================
# Random Field Generation
# ============================================================================

class RandomFieldGenerator2D:
    """Generator for 2D Gaussian Random Fields with multiscale coarsening."""
    def __init__(self, nx=100, ny=100, lx=1.0, ly=1.0, device='cpu', generation_method='kl', kl_error_threshold=1e-3):
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.device = device
        self.generation_method = generation_method.lower()
        self.kl_error_threshold = kl_error_threshold

        if self.generation_method not in ['fft', 'kl']:
            raise ValueError("generation_method must be 'fft' or 'kl'")

        self.kl_cache = {}
        if self.generation_method == 'kl':
            self._initialize_kl_mesh()

    def _initialize_kl_mesh(self):
        # Create mesh grid coordinates for KL.
        # Use 'ij' indexing (Matrix-style: nx rows, ny columns) for consistency with FFT.
        x = np.linspace(0, self.lx, self.nx)
        y = np.linspace(0, self.ly, self.ny)
        X, Y = np.meshgrid(x, y, indexing='ij') 
        self.xy_coords = np.column_stack((X.flatten(), Y.flatten()))

    def _get_kl_transform_matrix(self, correlation_length, covariance_type):
        """Compute or retrieve the cached KL transformation matrix (Phi @ Sqrt(Lambda))."""
        cache_key = (correlation_length, covariance_type, self.kl_error_threshold)
        if cache_key in self.kl_cache:
            return self.kl_cache[cache_key]

        print(f"Computing KL decomposition for l={correlation_length}...")

        # 1. Compute covariance matrix (Variance=1 for standard GRF)
        distances = squareform(pdist(self.xy_coords, "euclidean"))
        l = correlation_length

        if covariance_type == "exponential":
            # C(r) = exp(-r/l)
            cov_matrix = np.exp(-distances / l)
        elif covariance_type == "gaussian":
            # C(r) = exp(-(r/l)^2 / 2)
            cov_matrix = np.exp(-((distances / l)**2) / 2.0)
        else:
            raise ValueError(f"Invalid covariance_type for KL method: {covariance_type}")

        # 2. Eigendecomposition (using stable eigh for symmetric matrices)
        eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)

        # Sort in descending order and ensure positivity
        idx = np.argsort(eig_vals)[::-1]
        eig_vals = np.maximum(0, eig_vals[idx])
        eig_vecs = eig_vecs[:, idx]

        # 3. Truncation based on explained variance
        total_variance = np.sum(eig_vals)
        if total_variance > 1e-9:
            error_func = 1 - (np.cumsum(eig_vals) / total_variance)
            # Find the first index where the error is below the threshold
            truncation_idx = np.where(error_func <= self.kl_error_threshold)[0]
            if truncation_idx.size > 0:
                n_truncate = truncation_idx[0] + 1
            else:
                n_truncate = len(eig_vals)
        else:
            n_truncate = 0

        print(f"KL decomposition truncated to {n_truncate} components.")

        if n_truncate == 0:
            kl_transform_matrix = np.empty((len(self.xy_coords), 0))
        else:
            # 4. Precompute Transformation Matrix (Phi @ Sqrt(Lambda))
            sqrt_eig_vals = np.sqrt(eig_vals[:n_truncate])
            # Efficient element-wise multiplication broadcasting
            kl_transform_matrix = eig_vecs[:, :n_truncate] * sqrt_eig_vals[np.newaxis, :]

        self.kl_cache[cache_key] = kl_transform_matrix
        return kl_transform_matrix

    def generate_random_field(self, mean=10.0, std=2.0, correlation_length=0.2, covariance_type="exponential"):
        """Generate a single realization of a Gaussian Random Field."""
        if self.generation_method == 'fft':
            return self._generate_fft(mean, std, correlation_length, covariance_type)
        elif self.generation_method == 'kl':
            return self._generate_kl(mean, std, correlation_length, covariance_type)

    def _generate_fft(self, mean, std, correlation_length, covariance_type):
        """Generate GRF using Fast Fourier Transform (Periodic)."""
        dx = self.lx / self.nx
        dy = self.ly / self.ny
        # Use (nx, ny) shape for 'ij' indexing consistency
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
            # Note: PSD definitions vary based on the spatial covariance definition used.
            # This matches the original data_generation.py definition.
            P = np.pi * l**2 * np.exp(-((l * K) ** 2) / 4)
        else:
            raise ValueError("Invalid covariance_type")
        
        P = np.nan_to_num(P)
        fourier_coefficients *= np.sqrt(P)
        field = np.fft.ifft2(fourier_coefficients).real
        
        # Normalize and scale (essential for FFT method)
        field_std = np.std(field)
        if field_std > 1e-9:
            field = (field - np.mean(field)) / field_std * std + mean
        else:
            field = np.full_like(field, mean)
        return field

    def _generate_kl(self, mean, std, correlation_length, covariance_type):
        """Generate GRF using the Karhunen-Loève expansion (Non-Periodic)."""
        # Get the precomputed transformation matrix
        kl_transform_matrix = self._get_kl_transform_matrix(correlation_length, covariance_type)
        n_kl = kl_transform_matrix.shape[1]

        if n_kl == 0:
             return np.full((self.nx, self.ny), mean)

        # 1. Generate standard normal random variables (xi)
        xi = np.random.normal(0, 1, n_kl)

        # 2. Compute KL expansion (G = A @ xi) -> Standard Gaussian Field
        field_flat = kl_transform_matrix @ xi

        # 3. Rescale and shift
        field_flat = field_flat * std + mean

        # 4. Reshape back to 2D grid (nx, ny) due to 'ij' indexing
        field = field_flat.reshape((self.nx, self.ny))
        return field

    def coarsen_field(self, field, H):
        """Apply coarsening (smoothing) filter to the field."""
        if isinstance(field, np.ndarray):
            field = torch.from_numpy(field).to(self.device)
        
        # Ensure tensor format is [B, C, H, W]
        original_dim = field.dim()
        if original_dim == 2:
             field = field.unsqueeze(0).unsqueeze(0) # [H, W] -> [1, 1, H, W]
        elif original_dim == 3:
            field = field.unsqueeze(1) # [B, H, W] -> [B, 1, H, W]
        elif original_dim != 4:
            raise ValueError(f"Unsupported field dimensions (must be 2, 3, or 4), got {original_dim}")

        # Assuming isotropic grid for filter calculation
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
            # Apply periodic blurring (consistent with homogenization framework)
            smooth = gaussian_blur_periodic(field, kernel_size=kernel_size, sigma=filter_sigma_pix)
        
        # Restore dimensions
        if original_dim == 2:
            coarse = smooth.squeeze(0).squeeze(0)
        elif original_dim == 3:
            coarse = smooth.squeeze(1)
        else:
            coarse = smooth
            
        return coarse

def generate_multiscale_grf_data(
    N_samples: int, T: float = 1.0, N_constraints: int = 5, resolution: int = 32,
    L_domain: float = 1.0, micro_corr_length: float = 0.1, H_max_factor: float = 0.5,
    mean_val: float = 10.0, std_val: float = 2.0, covariance_type: str = "exponential",
    device: str = 'cpu', generation_method: str = 'fft', kl_error_threshold: float = 1e-3
) -> Tuple[Dict[float, Tensor], int]:
    """Generate multiscale Gaussian Random Field data."""
    print(f"\n--- Generating Multiscale GRF Data (Resolution: {resolution}x{resolution}, Method: {generation_method.upper()}) ---")
    time_steps = torch.linspace(0, T, N_constraints)
    marginal_data = {}
    data_dim = resolution * resolution
    
    generator = RandomFieldGenerator2D(
        nx=resolution, ny=resolution, lx=L_domain, ly=L_domain, device=device,
        generation_method=generation_method, kl_error_threshold=kl_error_threshold
    )
    
    print("Generating base microscopic fields (t=0)...")
    
    # Pre-compute KL components if necessary (handled internally by the first call)
    if generation_method == 'kl':
        generator._get_kl_transform_matrix(micro_corr_length, covariance_type)

    micro_fields = []
    for _ in trange(N_samples):
        field = generator.generate_random_field(mean=mean_val, std=std_val, correlation_length=micro_corr_length, covariance_type=covariance_type)
        micro_fields.append(field)
    
    # Convert list of numpy arrays to a tensor [N_samples, nx, ny]
    micro_fields_tensor = torch.tensor(np.array(micro_fields), dtype=torch.float32).to(device)
    
    print("Applying progressive coarsening filters...")
    H_max = L_domain * H_max_factor
    for t in time_steps:
        t_val = t.item()
        t_norm = t_val / T if T > 0 else 0.0
        H_t = t_norm * H_max
        
        # Coarsen the entire batch
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
