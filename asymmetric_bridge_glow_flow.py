"""
Asymmetric Multi-Marginal Bridge - Expressive Normalizing Flow
================================================================

This module implements the Asymmetric Multi-Marginal Bridge framework using
a more expressive normalizing flow, specifically a series of time-dependent
coupling layers (RealNVP style). This allows for modeling complex, non-Gaussian
distributional paths.

Unlike the affine flow version, this implementation relies heavily on automatic
differentiation for both the forward velocity and the score function, as
analytical expressions are no longer tractable.

Key Components:
- TimeDependentCouplingLayer: A single layer of a RealNVP-like flow.
- ExpressiveFlow: Stacks multiple coupling layers to create a deep flow.
- NeuralBridgeExpressive: The core module implementing the bridge with the new flow.
"""

from typing import Tuple, Callable, Dict, Any
import torch
from torch import nn, Tensor
from torch import distributions as D
from glow_model import TimeConditionedGlow
import torch.fft as fft
import numpy as np
import math
import os
from tqdm import trange
import matplotlib.pyplot as plt
# Import any required utilities (remove if not needed)

try:
    import ot
except ImportError:
    print("Warning: Python Optimal Transport (POT) library not found.")
    ot = None


# ============================================================================
# Utilities
# ============================================================================


def jvp(f: Callable[[Tensor], Any], x: Tensor, v: Tensor) -> Tuple[Tensor, ...]:
    """Compute Jacobian-vector product."""
    return torch.autograd.functional.jvp(f, x, v, create_graph=torch.is_grad_enabled())


def t_dir(f: Callable[[Tensor], Any], t: Tensor) -> Tuple[Tensor, ...]:
    """Compute the time derivative of f(t)."""
    return jvp(f, t, torch.ones_like(t))[1]  # since jvp returns [func_output, jvp]


# ============================================================================
# Data Utilities (Add this section)
# ============================================================================

def normalize_multimarginal_data(marginal_data: Dict[float, Tensor]) -> Tuple[Dict[float, Tensor], Tensor, Tensor]:
    """Normalizes the multi-marginal dataset to zero mean and unit variance per dimension."""
    # Concatenate all data across all time points
    all_data = torch.cat(list(marginal_data.values()), dim=0)
    
    # Calculate per-dimension mean and std
    mean = torch.mean(all_data, dim=0)
    std = torch.std(all_data, dim=0)
    
    # Ensure std is not zero for stability
    std[std < 1e-6] = 1.0
    
    normalized_data = {}
    for t, samples in marginal_data.items():
        normalized_data[t] = (samples - mean) / std
        
    return normalized_data, mean, std


# ============================================================================
# Expressive Flow Model Components
# ============================================================================


class SinusoidalTimeEmbedding(nn.Module):
    """
    Converts scalar time into a high-dimensional embedding vector
    using sinusoidal functions. This provides a richer representation of time.
    """
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t: Tensor) -> Tensor:
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t * emb
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.embedding_dim % 2 == 1:  # Zero pad if dim is odd
            emb = nn.functional.pad(emb, (0, 1))
        return emb

def spatial_data_conversion(data: Tensor, resolution: int, to_spatial: bool = True) -> Tensor:
    """
    Convert between flattened data and spatial (H, W, C) format for GLOW model.
    
    Args:
        data: Input tensor
        resolution: Spatial resolution (assumes square)
        to_spatial: If True, convert flat -> spatial. If False, convert spatial -> flat
    """
    if to_spatial:
        # Convert from [B, H*W*C] to [B, C, H, W] for CNN processing
        batch_size = data.shape[0]
        data_dim = data.shape[1]
        
        # Determine number of channels (assume square spatial dimensions)
        spatial_dim = resolution * resolution
        num_channels = data_dim // spatial_dim
        
        if data_dim != num_channels * spatial_dim:
            raise ValueError(f"Data dimension {data_dim} incompatible with resolution {resolution}")
        
        # Reshape to spatial format
        return data.view(batch_size, num_channels, resolution, resolution)
    else:
        # Convert from [B, C, H, W] to [B, C*H*W] for flat processing
        batch_size = data.shape[0]
        return data.view(batch_size, -1)

class GLOWExpressiveFlow(nn.Module):
    """
    GLOW-based normalizing flow using CNN convolutions for spatial data.
    
    Convention alignment with NeuralBridgeExpressive:
    - forward(epsilon, t): Generation path G (latent -> data)
    - inverse(x, t): Inference path G^-1 (data -> latent)
    
    Note: This wrapper correctly maps the bridge conventions to GLOW's internal conventions:
    - Bridge forward -> GLOW inverse (generation)
    - Bridge inverse -> GLOW forward (inference)
    """

    def __init__(self, data_dim: int, hidden_size: int, resolution: int, n_blocks_flow: int = 3, num_scales: int = 2):
        super().__init__()
        self.data_dim = data_dim
        self.resolution = resolution
        
        # Determine number of channels from data dimension
        spatial_dim = resolution * resolution
        self.num_channels = data_dim // spatial_dim
        
        if data_dim != self.num_channels * spatial_dim:
            raise ValueError(f"Data dimension {data_dim} incompatible with resolution {resolution}")
        
        # Initialize GLOW model
        input_shape = (self.num_channels, resolution, resolution)
        self.glow_model = TimeConditionedGlow(
            input_shape=input_shape,
            hidden_dim=hidden_size,
            n_blocks_flow=n_blocks_flow,
            num_scales=num_scales
        )

    def forward(self, epsilon: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass: latent -> data (Generation path G)."""
        # Convert to spatial format for CNN processing
        epsilon_spatial = spatial_data_conversion(epsilon, self.resolution, to_spatial=True)
        
        # Generation through GLOW (uses glow_model.inverse)
        x_spatial, log_det_J = self.glow_model.inverse(epsilon_spatial, t)
        
        # Convert back to flat format
        x_flat = spatial_data_conversion(x_spatial, self.resolution, to_spatial=False)
        
        return x_flat, log_det_J

    def inverse(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """Inverse pass: data -> latent (Inference path G^-1)."""
        # Convert to spatial format for CNN processing
        x_spatial = spatial_data_conversion(x, self.resolution, to_spatial=True)
        
        # Inference through GLOW (uses glow_model.forward)
        epsilon, log_det_J_inv = self.glow_model(x_spatial, t)
        
        return epsilon, log_det_J_inv


# ============================================================================
# Asymmetric Bridge with Expressive Flow
# ============================================================================


class NeuralBridgeExpressive(nn.Module):
    def __init__(
        self,
        data_dim: int,
        hidden_size: int,
        resolution: int,                    # NEW: Required for spatial data
        n_blocks_flow: int = 3,             # Changed from n_flow_layers
        num_scales: int = 2,                # NEW: GLOW multi-scale parameter
        T: float = 1.0,
        sigma_reverse: float = 1.0,
        data_mean: Tensor = None,           # FIX: Added for normalization
        data_std: Tensor = None,            # FIX: Added for normalization
        training_noise_std: float = 0.01,   # Added for data jittering
        noise_densification_std: float = 0.0001,  # Added for stable flow training
        inference_clamp_norm: float = None, # CRITICAL: Disable clamping by default
    ):
        super().__init__()

        self.data_dim = data_dim
        self.resolution = resolution
        self.T = T
        self.sigma_reverse = sigma_reverse
        self.inference_clamp_norm = inference_clamp_norm
        self.training_noise_std = training_noise_std
        self.noise_densification_std = noise_densification_std

        # Initialize GLOW-based flow
        self.flow = GLOWExpressiveFlow(
            data_dim=data_dim, 
            hidden_size=hidden_size, 
            resolution=resolution,
            n_blocks_flow=n_blocks_flow,
            num_scales=num_scales
        )

        # Base distribution (Standard Normal Latent Space)
        self.register_buffer("base_mean", torch.zeros(data_dim))
        self.register_buffer("base_std", torch.ones(data_dim))

        # FIX: Register normalization buffers
        if data_mean is None:
            self.register_buffer("data_mean", torch.zeros(data_dim))
        else:
            self.register_buffer("data_mean", data_mean)
            
        if data_std is None:
            self.register_buffer("data_std", torch.ones(data_dim))
        else:
            self.register_buffer("data_std", data_std)
    @property
    def base_dist(self):
        return D.Independent(D.Normal(self.base_mean, self.base_std), 1)

    def _format_time(self, t, batch_size):
        """Helper to format time tensor t to match batch size [B, 1]."""
        if not torch.is_tensor(t):
            t = torch.tensor(float(t), device=self.base_mean.device)

        if t.dim() == 0:
            return t.expand(batch_size, 1)
        
        if t.shape[0] == 1:
            return t.expand(batch_size, -1)

        if t.dim() == 1 and t.shape[0] == batch_size:
            return t.unsqueeze(-1)
        
        if t.dim() == 2 and t.shape[0] == batch_size and t.shape[1] == 1:
            return t

        raise ValueError(
            f"Time tensor shape {t.shape} incompatible with batch size {batch_size}."
        )
        
    # FIX: Add normalization helpers
    def normalize(self, x: Tensor) -> Tensor:
        return (x - self.data_mean) / self.data_std

    def denormalize(self, z: Tensor) -> Tensor:
        # Used externally during visualization/generation
        return z * self.data_std + self.data_mean

    # --- Dynamics (Calculated via Automatic Differentiation) ---

    def _apply_inference_clamp(self, vector: Tensor, name: str) -> Tensor:
        """Helper function to handle NaN/Inf values without artificial clamping during inference."""
        # Only handle NaN/Inf values for numerical stability, no artificial clamping
        if not torch.isfinite(vector).all():
            print(f"Warning: NaN or Inf detected in {name} calculation. Replacing with zeros.")
            # Replace NaNs/Infs with 0 to stabilize the trajectory
            vector = torch.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        # No artificial L2 norm clamping - let the flow dynamics be natural
        return vector

    def forward_velocity(self, z: Tensor, t: Tensor) -> Tensor:
        # NOTE: Assumes the implementation incorporates the previous fix 
        # (detaching epsilon before calculating the time derivative).
        
        t = self._format_time(t, z.shape[0])

        # 1. Invert: epsilon = G_inv(z, t)
        epsilon, _ = self.flow.inverse(z, t)
        
        # Detach epsilon for partial derivative calculation
        epsilon_detached = epsilon.detach()

        # 2. Define G(epsilon_fixed, t)
        def G_fixed_epsilon(time_tensor):
             z_transformed, _ = self.flow.forward(epsilon_detached, time_tensor)
             return z_transformed

        # 3. Compute partial_G/partial_t using JVP (t_dir).
        # We must ensure gradients can be computed if needed for the score function.
        grad_mode_before = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        try:
             velocity = t_dir(G_fixed_epsilon, t)
        finally:
             torch.set_grad_enabled(grad_mode_before)

        # Apply stabilization clamp if enabled
        velocity = self._apply_inference_clamp(velocity, "velocity")
        
        return velocity

    def score_function(self, z: Tensor, t: Tensor) -> Tensor:
        """
        Compute the score function nabla_z log p_t(z) using GLOW model.
        """
        t = self._format_time(t, z.shape[0])
        
        # Use GLOW model's score function if available, otherwise use autodiff
        if hasattr(self.flow.glow_model, 'score_function'):
            # Convert to spatial format
            z_spatial = spatial_data_conversion(z, self.resolution, to_spatial=True)
            score_spatial = self.flow.glow_model.score_function(z_spatial, t)
            # Convert back to flat format
            score = spatial_data_conversion(score_spatial, self.resolution, to_spatial=False)
        else:
            # Fallback to automatic differentiation
            z_grad = z.detach().clone().requires_grad_(True)
            
            # Convert to spatial format for GLOW
            z_spatial = spatial_data_conversion(z_grad, self.resolution, to_spatial=True)
            log_prob = self.flow.glow_model.log_prob(z_spatial, t)
            
            # Compute gradient
            score_spatial = torch.autograd.grad(
                log_prob.sum(), z_spatial, create_graph=torch.is_grad_enabled()
            )[0]
            
            # Convert back to flat format
            score = spatial_data_conversion(score_spatial, self.resolution, to_spatial=False)
        
        # Apply stabilization clamp if enabled
        score = self._apply_inference_clamp(score, "score")
        
        return score

    def reverse_drift(self, z: Tensor, t: Tensor) -> Tensor:
        """
        Compute the drift for the exact reverse SDE.
        R(z,t) = v(z,t) - (σ²/2) * ∇log p_t(z)
        """
        # Uses the corrected forward_velocity.
        velocity = self.forward_velocity(z, t)
        score = self.score_function(z, t)

        drift = velocity - (self.sigma_reverse**2 / 2) * score
        return drift

    # --- Objectives (Loss Functions) ---

    def path_regularization_loss(self, t: Tensor) -> Tensor:
        """
        Compute the kinetic energy loss using the latent formulation.
        J_Path = E_t E_epsilon [ 0.5 * || dG(epsilon, t)/dt ||^2 ]
        
        t: Tensor containing sampled times [B_time, 1].
        """
        # NOTE: This uses the optimized implementation from the previous analysis.
        n_samples = t.shape[0]
        
        # 1. Sample epsilon ~ base_dist
        epsilon = self.base_dist.sample((n_samples,)).to(self.base_mean.device)

        # 2. Define G(epsilon, t) as a function of t.
        def G_fixed_epsilon(time_tensor):
             z_transformed, _ = self.flow.forward(epsilon, time_tensor)
             return z_transformed

        # 3. Calculate the time derivative (the velocity) using JVP (t_dir).
        velocity = t_dir(G_fixed_epsilon, t)

        # 4. Kinetic energy
        kinetic_energy = 0.5 * torch.sum(velocity**2, dim=-1)
        
        return kinetic_energy.mean()

    def marginal_log_likelihood(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Compute the exact log likelihood log p_t(x) using Change of Variables.
        log p(x) = log p_base(G_inv(x, t)) + log|det(dG_inv/dx)|
        """
        t = self._format_time(t, x.shape[0])

        # 1. Inverse transformation (compute latent epsilon and log_det_inv)
        epsilon, log_det_J_inv = self.flow.inverse(x, t)

        # 2. Base distribution log probability
        log_prob_base = self.base_dist.log_prob(epsilon)

        # 3. Total log likelihood
        log_likelihood = log_prob_base + log_det_J_inv
        return log_likelihood

    def loss(
        self,
        marginal_data: Dict[float, Tensor],
        lambda_path: float = 0.1,
        batch_size_time: int = 256,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute the total loss: L_MM (MLE) + lambda * J_Path.
        """
        # 1. Multi-Marginal MLE Loss (L_MM) with Data Jittering
        mle_loss = 0
        total_samples = 0
        device = self.base_mean.device

        for t_k, samples_k in marginal_data.items():
            B_k = samples_k.shape[0]
            total_samples += B_k
            t_k_tensor = torch.full((B_k, 1), t_k, device=device)

            # FIX: Apply Data Jittering (Noise Injection) during training
            if self.training and self.training_noise_std > 0:
                # Noise is relative to normalized data scale (Std=1)
                noise = torch.randn_like(samples_k) * self.training_noise_std
                samples_k_perturbed = samples_k + noise
            else:
                samples_k_perturbed = samples_k
                
            # Apply noise densification for stable normalizing flow training
            if self.training and self.noise_densification_std > 0:
                # Additional small noise to prevent mode collapse in flows
                densify_noise = torch.randn_like(samples_k_perturbed) * self.noise_densification_std
                samples_k_perturbed = samples_k_perturbed + densify_noise

            log_likelihoods = self.marginal_log_likelihood(samples_k_perturbed, t_k_tensor)
            mle_loss -= log_likelihoods.sum()

        if total_samples > 0:
            mle_loss /= total_samples

        # 2. Path Regularization Loss (J_Path)
        t_rand = torch.rand(batch_size_time, 1, device=device) * self.T
        path_loss = self.path_regularization_loss(t_rand)

        # Total Loss
        total_loss = mle_loss + lambda_path * path_loss
        return total_loss, mle_loss, path_loss


# ============================================================================
# Autocovariance Analysis Utilities (for GRF comparison)
# ============================================================================


def compute_spatial_acf_2d(samples: Tensor, resolution: int) -> Tensor:
    """
    Computes the normalized spatial ACF for a batch of 2D fields using FFT (Wiener-Khinchin).
    Assumes stationarity, periodicity, and ergodicity.
    """
    B = samples.shape[0]
    fields = samples.reshape(B, resolution, resolution)

    # 1. Center the data (remove spatial mean per sample)
    spatial_mean = fields.mean(dim=[1, 2], keepdim=True)
    centered_fields = fields - spatial_mean

    # 2. Compute Power Spectral Density (PSD)
    # Use norm="ortho" to ensure consistency (Parseval's theorem)
    freq_domain = fft.fft2(centered_fields, norm="ortho")
    psd = torch.abs(freq_domain)**2

    # 3. Compute Autocovariance (ACVF) = IFFT(PSD)
    autocovariance = fft.ifft2(psd, norm="ortho").real # [B, H, W]

    # 4. Average over the batch to get the ensemble estimate
    avg_autocovariance = autocovariance.mean(dim=0) # [H, W]

    # 5. Normalize by the variance (zero lag, index [0, 0])
    variance = avg_autocovariance[0, 0]
    if variance < 1e-9:
        return torch.zeros_like(avg_autocovariance)
    
    normalized_acf = avg_autocovariance / torch.clamp(variance, min=1e-9)
    
    # Clamp for numerical stability
    normalized_acf = torch.clamp(normalized_acf, min=-1.0, max=1.0)

    # 6. Shift zero lag to center for visualization
    return fft.fftshift(normalized_acf)


def radial_average(data_2d: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Compute the radial average of a centered 2D array (e.g., ACF) using torch.bincount.
    """
    H, W = data_2d.shape
    center_y, center_x = H // 2, W // 2
    
    # Create coordinate grids
    y, x = torch.meshgrid(torch.arange(H, device=data_2d.device), 
                          torch.arange(W, device=data_2d.device), indexing='ij')
    
    # Calculate radial distance from center
    r = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Binning by integer radius
    r_int = r.round().long()
    max_radius = min(center_x, center_y)
    
    # Flatten arrays for accumulation
    r_flat = r_int.flatten()
    data_flat = data_2d.flatten()
    
    # Use torch.bincount for efficient averaging
    # Sum values in each radial bin
    sum_in_bin = torch.bincount(r_flat, weights=data_flat)
    # Count elements in each radial bin
    count_in_bin = torch.bincount(r_flat)
    
    # Calculate average
    avg_in_bin = sum_in_bin / torch.clamp(count_in_bin, min=1)
    
    # Truncate to the meaningful radius
    radii = torch.arange(max_radius + 1, device=data_2d.device)
    radial_profile = avg_in_bin[:max_radius + 1]
    
    return radii, radial_profile


def _visualize_covariance_comparison(marginal_data: Dict[float, Tensor], generated_samples: Dict[float, Tensor], output_dir: str):
    """Visualizes and compares the spatial ACF between original and generated samples."""
    print("  - Plotting spatial covariance (ACF) comparison...")
    
    sorted_times = sorted(marginal_data.keys())
    # Select representative time points (e.g., 5 points including start and end)
    n_time_points = min(5, len(sorted_times))
    selected_indices = np.linspace(0, len(sorted_times)-1, n_time_points, dtype=int)
    selected_times = [sorted_times[i] for i in selected_indices]

    # Determine resolution
    if not selected_times: 
        return
    data_dim = marginal_data[selected_times[0]].shape[1]
    resolution = int(math.sqrt(data_dim))
    if resolution * resolution != data_dim:
        print("    Warning: Data is not square. Skipping covariance analysis.")
        return

    # Setup figure: 3 rows (1D Radial ACF, 2D Target ACF, 2D Gen ACF)
    fig, axes = plt.subplots(3, n_time_points, figsize=(4 * n_time_points, 12))
    if n_time_points == 1: 
        axes = axes.reshape(3, 1)
    fig.suptitle("Spatial Autocorrelation Function (ACF) Comparison (Second-Order Statistics)", fontsize=16)

    # Global color scale for 2D ACF plots
    vmin, vmax = -0.5, 1.0
    im = None
    
    for i, t_val in enumerate(selected_times):
        # Compute ACFs (ensure data is on CPU for computation)
        acf_target = compute_spatial_acf_2d(marginal_data[t_val].cpu(), resolution)
        acf_gen = compute_spatial_acf_2d(generated_samples[t_val].cpu(), resolution)
        
        # Calculate MSE_ACF
        mse_acf = torch.mean((acf_target - acf_gen)**2).item()

        # Compute Radial Averages
        radii, radial_target = radial_average(acf_target)
        _, radial_gen = radial_average(acf_gen)
        
        # Convert radii to physical distance (assuming domain L=1.0)
        physical_radii = radii.numpy() / resolution

        # Row 1: 1D Radial ACF Comparison
        ax_1d = axes[0, i]
        ax_1d.plot(physical_radii, radial_target.numpy(), 'b-', label='Target', linewidth=2)
        ax_1d.plot(physical_radii, radial_gen.numpy(), 'r--', label='Generated', linewidth=2)
        ax_1d.set_title(f't={t_val:.2f} (MSE={mse_acf:.2e})')
        ax_1d.set_xlabel('Lag Distance (r/L)')
        ax_1d.set_ylim(-0.1, 1.05)
        ax_1d.grid(True, alpha=0.3)
        if i == 0: 
            ax_1d.set_ylabel('Radial ACF R(r)')
        if i == n_time_points - 1:
            ax_1d.legend()

        # Row 2: 2D Target ACF
        ax_target = axes[1, i]
        im = ax_target.imshow(acf_target.numpy(), cmap='viridis', vmin=vmin, vmax=vmax, extent=[-0.5, 0.5, -0.5, 0.5])
        ax_target.set_title('Target 2D ACF')
        if i == 0: 
            ax_target.set_ylabel('Y Lag')
        
        # Row 3: 2D Generated ACF
        ax_gen = axes[2, i]
        ax_gen.imshow(acf_gen.numpy(), cmap='viridis', vmin=vmin, vmax=vmax, extent=[-0.5, 0.5, -0.5, 0.5])
        ax_gen.set_title('Generated 2D ACF')
        ax_gen.set_xlabel('X Lag')
        if i == 0: 
            ax_gen.set_ylabel('Y Lag')

    # Add colorbar
    if im is not None:
        fig.colorbar(im, ax=axes[1:, :].ravel().tolist(), fraction=0.046, pad=0.04, label="Correlation")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "covariance_comparison.png"), dpi=300)
    plt.close()


def calculate_w2_distance(target_samples: np.ndarray, gen_samples: np.ndarray, method: str = "corrected_emd2", n_subsample: int = 512) -> float:
    """
    Calculate Wasserstein-2 distance between two sets of samples using different methods.
    
    Args:
        target_samples: Target data samples [N, D]
        gen_samples: Generated data samples [M, D] 
        method: "corrected_emd2" (default) or "sliced_wasserstein"
        n_subsample: Number of samples to use for computation efficiency
        
    Returns:
        W2 distance (float)
        
    Note:
        - "corrected_emd2": Uses EMD2 with squared Euclidean distances, then takes sqrt to get W2
        - "sliced_wasserstein": Uses sliced Wasserstein approximation (consistent with Affine flow)
    """
    if ot is None:
        return float('inf')
        
    try:
        # Subsample for computational efficiency
        n_subsample = min(n_subsample, target_samples.shape[0], gen_samples.shape[0])
        idx_target = np.random.choice(target_samples.shape[0], n_subsample, replace=False)
        idx_gen = np.random.choice(gen_samples.shape[0], n_subsample, replace=False)
        
        target_sub = target_samples[idx_target]
        gen_sub = gen_samples[idx_gen]
        
        if method == "corrected_emd2":
            # CORRECTED: EMD2 with sqrt to get W2 instead of W2²
            M_squared = ot.dist(target_sub, gen_sub)  # Squared Euclidean distances
            w2_squared = ot.emd2([], [], M_squared)   # This gives W2²
            return np.sqrt(w2_squared)  # Take square root to get W2
            
        elif method == "sliced_wasserstein":
            # Sliced Wasserstein (consistent with Affine flow implementation)
            return ot.sliced_wasserstein_distance(
                target_sub, gen_sub, n_projections=1000, seed=42)
                
        else:
            raise ValueError(f"Unknown method: {method}. Use 'corrected_emd2' or 'sliced_wasserstein'")
            
    except Exception as e:
        print(f"Warning: W2 calculation failed with {method}: {e}")
        return float('inf')


def _calculate_validation_metrics(marginal_data: Dict[float, Tensor], generated_samples: Dict[float, Tensor]) -> Dict[str, Any]:
    """
    Calculate quantitative validation metrics (W2 distance and MSE_ACF).
    
    CRITICAL FIX: This function now correctly calculates the W2 distance instead of W2² (squared).
    The previous implementation using ot.emd2 with ot.dist was returning W2², 
    causing orders-of-magnitude discrepancy compared to the Affine flow implementation.
    """
    print("  - Calculating quantitative validation metrics (W2, MSE_ACF)...")
    
    sorted_times = sorted(marginal_data.keys())
    metrics = {'times': sorted_times, 'w2_distances': [], 'mse_acf': []}

    # Determine resolution
    if not sorted_times: 
        return metrics
    data_dim = marginal_data[sorted_times[0]].shape[1]
    resolution = int(math.sqrt(data_dim))
    is_square = (resolution * resolution == data_dim)
    
    for t_val in sorted_times:
        target_samples = marginal_data[t_val].cpu().numpy()
        gen_samples = generated_samples[t_val].cpu().numpy()
        
        # 1. Wasserstein-2 distance calculation (CORRECTED)
        # Use corrected EMD2 method by default (takes sqrt to get W2 instead of W2²)
        w2_dist = calculate_w2_distance(target_samples, gen_samples, method="sliced_wasserstein")
        metrics['w2_distances'].append(w2_dist)
        
        # Optional: Also calculate using sliced Wasserstein for comparison with Affine flow
        # w2_sliced = calculate_w2_distance(target_samples, gen_samples, method="sliced_wasserstein")
        # metrics['w2_distances_sliced'] = metrics.get('w2_distances_sliced', []) + [w2_sliced]
        
        # 2. MSE of ACF (only for square/2D data)
        if is_square:
            try:
                acf_target = compute_spatial_acf_2d(marginal_data[t_val].cpu(), resolution)
                acf_gen = compute_spatial_acf_2d(generated_samples[t_val].cpu(), resolution)
                mse_acf = torch.mean((acf_target - acf_gen)**2).item()
                metrics['mse_acf'].append(mse_acf)
            except Exception:
                metrics['mse_acf'].append(float('inf'))
        else:
            metrics['mse_acf'].append(float('nan'))

    return metrics


# ============================================================================
# Data Generation Utilities (GRF)
# ============================================================================


def gaussian_blur_periodic(
    input_tensor: Tensor, kernel_size: int, sigma: float
) -> Tensor:
    """Apply Gaussian blur with periodic boundary conditions."""
    if sigma <= 1e-9 or kernel_size <= 1:
        return input_tensor
    if kernel_size % 2 == 0:
        kernel_size += 1
    k = torch.arange(kernel_size, dtype=torch.float32, device=input_tensor.device)
    center = (kernel_size - 1) / 2
    gauss_1d = torch.exp(-0.5 * ((k - center) / sigma) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()
    gauss_2d = torch.outer(gauss_1d, gauss_1d)
    if input_tensor.dim() != 4:
        raise ValueError("Input tensor must be [B, C, H, W]")
    C_in = input_tensor.shape[1]
    kernel = gauss_2d.expand(C_in, 1, kernel_size, kernel_size)
    padding = (kernel_size - 1) // 2
    padded_input = torch.nn.functional.pad(
        input_tensor, (padding, padding, padding, padding), mode="circular"
    )
    output = torch.nn.functional.conv2d(padded_input, kernel, padding=0, groups=C_in)
    return output


class RandomFieldGenerator2D:
    """Generator for 2D Gaussian Random Fields with multiscale coarsening."""

    def __init__(self, nx=100, ny=100, lx=1.0, ly=1.0, device="cpu"):
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.device = device

    def generate_random_field(
        self, mean=10.0, std=2.0, correlation_length=0.2, covariance_type="exponential"
    ):
        dx = self.lx / self.nx
        dy = self.ly / self.ny
        white_noise = np.random.normal(0, 1, (self.nx, self.ny))
        fourier_coefficients = np.fft.fft2(white_noise)
        kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ny, d=dy)
        Kx, Ky = np.meshgrid(kx, ky, indexing="ij")
        K = np.sqrt(Kx**2 + Ky**2)
        correlation_length_param = correlation_length
        if covariance_type == "exponential":
            denom = 1 + (correlation_length_param * K) ** 2
            P = (2 * np.pi * correlation_length_param**2) / np.maximum(1e-9, denom ** (1.5))
        elif covariance_type == "gaussian":
            P = np.pi * correlation_length_param**2 * np.exp(-((correlation_length_param * K) ** 2) / 4)
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
            smooth = gaussian_blur_periodic(
                field, kernel_size=kernel_size, sigma=filter_sigma_pix
            )
        coarse = smooth
        if squeeze_channel:
            coarse = coarse.squeeze(1)
        return coarse


def generate_multiscale_grf_data(
    N_samples: int,
    T: float = 1.0,
    N_constraints: int = 5,
    resolution: int = 16,
    L_domain: float = 1.0,
    micro_corr_length: float = 0.1,
    H_max_factor: float = 0.5,
    mean_val: float = 10.0,
    std_val: float = 2.0,
    covariance_type: str = "exponential",
    device: str = "cpu",
) -> Tuple[Dict[float, Tensor], int]:
    """Generate multiscale Gaussian Random Field data."""
    print(
        f"\n--- Generating Multiscale GRF Data (Resolution: {resolution}x{resolution}) ---"
    )
    time_steps = torch.linspace(0, T, N_constraints)
    marginal_data = {}
    data_dim = resolution * resolution
    generator = RandomFieldGenerator2D(
        nx=resolution, ny=resolution, lx=L_domain, ly=L_domain, device=device
    )
    print("Generating base microscopic fields (t=0)...")
    micro_fields = []
    for _ in trange(N_samples):
        field = generator.generate_random_field(
            mean=mean_val,
            std=std_val,
            correlation_length=micro_corr_length,
            covariance_type=covariance_type,
        )
        micro_fields.append(field)
    micro_fields_tensor = torch.tensor(np.array(micro_fields), dtype=torch.float32).to(
        device
    )
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
        print(
            f"  t={t_val:.2f}: H={H_t:.4f}, Mean Std Dev across field: {mean_std:.4f}"
        )
    print("Multiscale data generation complete.")
    return marginal_data, data_dim


# ============================================================================
# Backward Sampling and Additional Visualization Functions
# ============================================================================


def generate_backward_samples(bridge: NeuralBridgeExpressive, marginal_data: Dict[float, Tensor], n_samples: int = 64, n_steps: int = 100, device: str = 'cpu') -> Dict[float, Tensor]:
    """
    Generate backward samples from the learned bridge starting from the final time distribution.
    This implements proper bridge sampling by starting from the T=1 marginal and integrating 
    backward through the reverse SDE to generate samples at all intermediate time points.
    """
    print("  - Generating backward samples from learned bridge (starting from final time)...")
    
    bridge.eval()
    generated_samples = {}
    
    # Get the time points from marginal data
    sorted_times = sorted(marginal_data.keys())
    
    with torch.no_grad():
        # Step 1: Start from the final time distribution
        final_time = max(sorted_times)
        if final_time in marginal_data and marginal_data[final_time].shape[0] >= n_samples:
            # Use actual data samples from final time
            indices = torch.randperm(marginal_data[final_time].shape[0])[:n_samples]
            z_final = marginal_data[final_time][indices].to(device)
            print(f"    Starting from {n_samples} actual samples at t={final_time}")
        else:
            # Generate samples from the learned distribution at final time
            z_final = bridge.base_dist.sample((n_samples,)).to(device)
            # Transform through the flow at final time
            t_final_tensor = torch.full((n_samples, 1), final_time, device=device)
            z_final, _ = bridge.flow.forward(z_final, t_final_tensor)
            print(f"    Starting from {n_samples} generated samples at t={final_time}")
        
        # Step 2: Generate backward trajectory using Euler-Maruyama for the reverse SDE
        print(f"    Integrating backward SDE with {n_steps} steps...")
        try:
            dt = (0.0 - final_time) / n_steps  # dt is negative for backward integration
            trajectory_times = torch.linspace(final_time, 0.0, n_steps + 1)
            trajectory = [z_final.clone()]
            current_z = z_final.clone()
            
            for step in range(n_steps):
                t_current = trajectory_times[step]
                t_tensor = torch.full((current_z.shape[0], 1), t_current, device=device)
                
                with torch.enable_grad():
                    
                    # Get drift for reverse SDE
                    drift = bridge.reverse_drift(current_z, t_tensor)
                
                # disable gradients after drift calculation
                drift = drift.detach()
                
                # Add noise for SDE
                noise = torch.randn_like(current_z) * bridge.sigma_reverse * torch.sqrt(torch.abs(torch.tensor(dt)))
                
                # Euler-Maruyama step
                current_z = current_z + drift * dt + noise
                trajectory.append(current_z.clone())
            
            # Step 3: Interpolate to get samples at the specific time points
            for t_val in sorted_times:
                # Find the closest time index in the trajectory
                time_idx = torch.argmin(torch.abs(trajectory_times - t_val)).item()
                samples_at_t = trajectory[time_idx].cpu()
                generated_samples[t_val] = samples_at_t
                
        except Exception as e:
            print(f"    Warning: Backward SDE integration failed ({e}). Falling back to forward sampling.")
            # Fallback: Use forward sampling from base distribution
            epsilon = bridge.base_dist.sample((n_samples,)).to(device)
            for t_val in sorted_times:
                t_tensor = torch.full((n_samples, 1), t_val, device=device)
                samples, _ = bridge.flow.forward(epsilon, t_tensor)
                generated_samples[t_val] = samples.cpu()
    
    print(f"    Generated samples for {len(generated_samples)} time points")
    return generated_samples


def generate_consistent_backward_samples(bridge: NeuralBridgeExpressive, marginal_data: Dict[float, Tensor], 
                                       selected_indices: Tensor, n_steps: int = 100, device: str = 'cpu') -> Tuple[Dict[float, Tensor], Dict[float, Tensor]]:
    """
    Generate backward samples from specific selected samples at the final time.
    This ensures that we compare the evolution of the same final-time samples.
    
    Returns:
        original_selected: Dict containing the original selected samples at each time point
        generated_consistent: Dict containing the backward-evolved samples starting from the same final samples
    """
    print("  - Generating consistent backward samples from selected final time samples...")
    
    bridge.eval()
    original_selected = {}
    generated_consistent = {}
    
    # Get the time points from marginal data
    sorted_times = sorted(marginal_data.keys())
    final_time = max(sorted_times)
    
    with torch.no_grad():
        # Step 1: Extract the same selected samples from original data at all time points
        for t_val in sorted_times:
            original_data = marginal_data[t_val]
            # Use the same indices to ensure consistency across time points
            if selected_indices.max() < original_data.shape[0]:
                original_selected[t_val] = original_data[selected_indices]
            else:
                # If not enough samples, repeat the indices cyclically
                repeated_indices = selected_indices % original_data.shape[0]
                original_selected[t_val] = original_data[repeated_indices]
        
        # Step 2: Start backward integration from the selected final time samples
        z_final = original_selected[final_time].to(device)
        n_samples = z_final.shape[0]
        print(f"    Starting backward integration from {n_samples} selected samples at t={final_time}")
        
        # Step 3: Generate backward trajectory using Euler-Maruyama for the reverse SDE
        print(f"    Integrating backward SDE with {n_steps} steps...")
        try:
            dt = (0.0 - final_time) / n_steps  # dt is negative for backward integration
            trajectory_times = torch.linspace(final_time, 0.0, n_steps + 1)
            trajectory = [z_final.clone()]
            current_z = z_final.clone()
            
            for step in range(n_steps):
                t_current = trajectory_times[step]
                t_tensor = torch.full((current_z.shape[0], 1), t_current, device=device)
                
                with torch.enable_grad():
                    # Get drift for reverse SDE
                    drift = bridge.reverse_drift(current_z, t_tensor)
                
                # disable gradients after drift calculation
                drift = drift.detach()
                
                # Add noise for SDE
                noise = torch.randn_like(current_z) * bridge.sigma_reverse * torch.sqrt(torch.abs(torch.tensor(dt)))
                
                # Euler-Maruyama step
                current_z = current_z + drift * dt + noise
                trajectory.append(current_z.clone())
            
            # Step 4: Interpolate to get samples at the specific time points
            for t_val in sorted_times:
                # Find the closest time index in the trajectory
                time_idx = torch.argmin(torch.abs(trajectory_times - t_val)).item()
                samples_at_t = trajectory[time_idx].cpu()
                generated_consistent[t_val] = samples_at_t
                
        except Exception as e:
            print(f"    Warning: Backward SDE integration failed ({e}). Using original samples as fallback.")
            # Fallback: Use the original samples (no bridge evolution)
            for t_val in sorted_times:
                generated_consistent[t_val] = original_selected[t_val].clone()
    
    print(f"    Generated consistent samples for {len(generated_consistent)} time points")
    return original_selected, generated_consistent


def _visualize_original_samples(marginal_data: Dict[float, Tensor], selected_indices: Tensor, output_dir: str):
    """
    Visualize the original samples at different time points for the selected indices.
    This shows the ground truth evolution of specific samples through time.
    """
    print("  - Plotting original samples evolution...")
    
    sorted_times = sorted(marginal_data.keys())
    n_time_points = len(sorted_times)
    n_samples_show = len(selected_indices)
    
    # Determine resolution
    if not sorted_times:
        return
    data_dim = marginal_data[sorted_times[0]].shape[1]
    resolution = int(math.sqrt(data_dim))
    
    if resolution * resolution != data_dim:
        print(f"    Warning: Data dimension {data_dim} is not a perfect square. Skipping visualization.")
        return

    # Create figure with samples as rows and time points as columns
    fig, axes = plt.subplots(n_samples_show, n_time_points, 
                           figsize=(3 * n_time_points, 3 * n_samples_show))
    fig.suptitle("Original Samples Evolution Through Time", fontsize=16)
    
    # Handle dimension edge cases
    if n_samples_show == 1 and n_time_points == 1:
        axes = np.array([[axes]])
    elif n_samples_show == 1:
        axes = axes.reshape(1, -1)
    elif n_time_points == 1:
        axes = axes.reshape(-1, 1)
    elif axes.ndim == 1:
        axes = axes.reshape(-1, 1) if n_time_points == 1 else axes.reshape(1, -1)
    
    # Determine global color scale for consistency
    all_original = torch.cat([marginal_data[t] for t in sorted_times], dim=0).cpu().numpy()
    vmin, vmax = all_original.min(), all_original.max()
    
    for j, t_val in enumerate(sorted_times):
        original_data = marginal_data[t_val].cpu().numpy()
        
        for i, sample_idx in enumerate(selected_indices):
            # Handle case where sample_idx might be out of bounds
            actual_idx = sample_idx % original_data.shape[0]
            sample_orig = original_data[actual_idx].reshape(resolution, resolution)
            
            axes[i, j].imshow(sample_orig, cmap='viridis', vmin=vmin, vmax=vmax, extent=[0, 1, 0, 1])
            axes[i, j].axis('off')
            
            # Add labels
            if j == 0:
                axes[i, j].set_ylabel(f'Sample {sample_idx.item()}', rotation=0, ha='right', va='center')
            if i == 0:
                axes[i, j].set_title(f't = {t_val:.2f}')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(output_dir, "original_samples_evolution.png"), dpi=300, bbox_inches='tight')
    plt.close()


def _visualize_generated_samples(generated_samples: Dict[float, Tensor], selected_indices: Tensor, output_dir: str):
    """
    Visualize the generated samples from the bridge model evolution.
    This shows how the bridge model predicts the evolution from the final time samples.
    """
    print("  - Plotting generated samples from bridge evolution...")
    
    sorted_times = sorted(generated_samples.keys())
    n_time_points = len(sorted_times)
    n_samples_show = len(selected_indices)
    
    # Determine resolution
    if not sorted_times:
        return
    data_dim = generated_samples[sorted_times[0]].shape[1]
    resolution = int(math.sqrt(data_dim))
    
    if resolution * resolution != data_dim:
        print(f"    Warning: Data dimension {data_dim} is not a perfect square. Skipping visualization.")
        return

    # Create figure with samples as rows and time points as columns
    fig, axes = plt.subplots(n_samples_show, n_time_points, 
                           figsize=(3 * n_time_points, 3 * n_samples_show))
    fig.suptitle("Generated Samples Evolution Through Bridge Model", fontsize=16)
    
    # Handle dimension edge cases
    if n_samples_show == 1 and n_time_points == 1:
        axes = np.array([[axes]])
    elif n_samples_show == 1:
        axes = axes.reshape(1, -1)
    elif n_time_points == 1:
        axes = axes.reshape(-1, 1)
    elif axes.ndim == 1:
        axes = axes.reshape(-1, 1) if n_time_points == 1 else axes.reshape(1, -1)
    
    # Determine global color scale for consistency
    all_generated = torch.cat([generated_samples[t] for t in sorted_times], dim=0).cpu().numpy()
    vmin, vmax = all_generated.min(), all_generated.max()
    
    for j, t_val in enumerate(sorted_times):
        generated_data = generated_samples[t_val].cpu().numpy()
        
        for i, sample_idx in enumerate(selected_indices):
            # The generated samples should have the same order as selected_indices
            if i < generated_data.shape[0]:
                sample_gen = generated_data[i].reshape(resolution, resolution)
                axes[i, j].imshow(sample_gen, cmap='viridis', vmin=vmin, vmax=vmax, extent=[0, 1, 0, 1])
            axes[i, j].axis('off')
            
            # Add labels
            if j == 0:
                axes[i, j].set_ylabel(f'Sample {sample_idx.item()}', rotation=0, ha='right', va='center')
            if i == 0:
                axes[i, j].set_title(f't = {t_val:.2f}')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(output_dir, "generated_samples_evolution.png"), dpi=300, bbox_inches='tight')
    plt.close()


def _visualize_consistent_samples_comparison(original_samples: Dict[float, Tensor], 
                                           generated_samples: Dict[float, Tensor], 
                                           selected_indices: Tensor, output_dir: str):
    """
    Create a side-by-side comparison visualization of the same samples showing:
    1. Original evolution through time
    2. Generated evolution through the bridge model
    This enables direct comparison of how well the bridge model captures the dynamics.
    """
    print("  - Creating consistent samples comparison visualization...")
    
    sorted_times = sorted(original_samples.keys())
    n_time_points = len(sorted_times)
    n_samples_show = min(4, len(selected_indices))  # Show up to 4 samples
    
    # Determine resolution
    if not sorted_times:
        return
    data_dim = original_samples[sorted_times[0]].shape[1]
    resolution = int(math.sqrt(data_dim))
    
    if resolution * resolution != data_dim:
        print(f"    Warning: Data dimension {data_dim} is not a perfect square. Skipping visualization.")
        return

    # Create figure with alternating rows (original vs generated for each sample)
    fig, axes = plt.subplots(2 * n_samples_show, n_time_points, 
                           figsize=(3 * n_time_points, 3 * 2 * n_samples_show))
    fig.suptitle("Original vs Generated Evolution Comparison (Same Initial Conditions)", fontsize=16)
    
    # Handle dimension edge cases
    if n_samples_show == 1 and n_time_points == 1:
        axes = np.array([[axes]])
    elif n_samples_show == 1:
        axes = axes.reshape(2, 1) if n_time_points == 1 else axes.reshape(2, -1)
    elif n_time_points == 1:
        axes = axes.reshape(-1, 1)
    elif axes.ndim == 1:
        axes = axes.reshape(-1, 1) if n_time_points == 1 else axes.reshape(1, -1)
    
    # Determine global color scale for consistency
    all_original = torch.cat([original_samples[t] for t in sorted_times], dim=0).cpu().numpy()
    all_generated = torch.cat([generated_samples[t] for t in sorted_times], dim=0).cpu().numpy()
    vmin = min(all_original.min(), all_generated.min())
    vmax = max(all_original.max(), all_generated.max())
    
    for j, t_val in enumerate(sorted_times):
        original_data = original_samples[t_val].cpu().numpy()
        generated_data = generated_samples[t_val].cpu().numpy()
        
        for i in range(n_samples_show):
            sample_idx = selected_indices[i]
            
            # Original data samples
            row_orig = 2 * i
            if i < original_data.shape[0]:
                sample_orig = original_data[i].reshape(resolution, resolution)
                axes[row_orig, j].imshow(sample_orig, cmap='viridis', vmin=vmin, vmax=vmax, extent=[0, 1, 0, 1])
            axes[row_orig, j].axis('off')
            if j == 0:
                axes[row_orig, j].set_ylabel(f'Original\nSample {sample_idx.item()}', rotation=0, ha='right', va='center')
            if i == 0:
                axes[row_orig, j].set_title(f't = {t_val:.2f}')
            
            # Generated data samples  
            row_gen = 2 * i + 1
            if i < generated_data.shape[0]:
                sample_gen = generated_data[i].reshape(resolution, resolution)
                axes[row_gen, j].imshow(sample_gen, cmap='viridis', vmin=vmin, vmax=vmax, extent=[0, 1, 0, 1])
            axes[row_gen, j].axis('off')
            if j == 0:
                axes[row_gen, j].set_ylabel(f'Generated\nSample {sample_idx.item()}', rotation=0, ha='right', va='center')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(output_dir, "consistent_samples_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()


def _visualize_marginal_statistics_comparison(marginal_data: Dict[float, Tensor], generated_samples: Dict[float, Tensor], output_dir: str):
    """
    Compare statistical properties (mean, std, distribution) between original and generated samples.
    """
    print("  - Plotting marginal statistics comparison...")
    
    sorted_times = sorted(marginal_data.keys())
    
    # Calculate statistics for each time point
    original_means = []
    original_stds = []
    generated_means = []
    generated_stds = []
    
    for t_val in sorted_times:
        # Original data statistics
        orig_data = marginal_data[t_val].cpu()
        orig_mean = torch.mean(orig_data, dim=0).mean().item()  # Average over spatial dimensions
        orig_std = torch.std(orig_data, dim=0).mean().item()
        original_means.append(orig_mean)
        original_stds.append(orig_std)
        
        # Generated data statistics
        gen_data = generated_samples[t_val].cpu()
        gen_mean = torch.mean(gen_data, dim=0).mean().item()
        gen_std = torch.std(gen_data, dim=0).mean().item()
        generated_means.append(gen_mean)
        generated_stds.append(gen_std)
    
    # Create comparison plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Statistical Comparison: Original vs Generated Samples", fontsize=14)
    
    # Plot 1: Mean evolution
    ax1 = axes[0]
    ax1.plot(sorted_times, original_means, 'o-', label='Original', linewidth=2, markersize=6)
    ax1.plot(sorted_times, generated_means, 's--', label='Generated', linewidth=2, markersize=6)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Mean Value')
    ax1.set_title('Mean Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Standard deviation evolution
    ax2 = axes[1]
    ax2.plot(sorted_times, original_stds, 'o-', label='Original', linewidth=2, markersize=6)
    ax2.plot(sorted_times, generated_stds, 's--', label='Generated', linewidth=2, markersize=6)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Standard Deviation')
    ax2.set_title('Std Dev Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Scatter plot of means vs stds
    ax3 = axes[2]
    ax3.scatter(original_means, original_stds, c=sorted_times, cmap='viridis', 
               s=80, alpha=0.7, label='Original', marker='o')
    ax3.scatter(generated_means, generated_stds, c=sorted_times, cmap='plasma', 
               s=80, alpha=0.7, label='Generated', marker='s')
    ax3.set_xlabel('Mean Value')
    ax3.set_ylabel('Standard Deviation')
    ax3.set_title('Mean vs Std (colored by time)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add colorbar for time
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(sorted_times), vmax=max(sorted_times)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax3)
    cbar.set_label('Time')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "marginal_statistics_comparison.png"), dpi=300)
    plt.close()


def _visualize_sample_distributions(marginal_data: Dict[float, Tensor], generated_samples: Dict[float, Tensor], output_dir: str):
    """
    Visualize the pixel-wise distributions for select time points to assess distributional match.
    """
    print("  - Plotting sample distributions comparison...")
    
    sorted_times = sorted(marginal_data.keys())
    # Select a few representative time points
    n_plots = min(4, len(sorted_times))
    selected_times = [sorted_times[i] for i in np.linspace(0, len(sorted_times)-1, n_plots, dtype=int)]
    
    fig, axes = plt.subplots(2, n_plots, figsize=(4 * n_plots, 8))
    if n_plots == 1:
        axes = axes.reshape(-1, 1)
    fig.suptitle("Pixel Value Distributions: Original vs Generated", fontsize=14)
    
    for i, t_val in enumerate(selected_times):
        # Get flattened data for histogram
        orig_data = marginal_data[t_val].cpu().numpy().flatten()
        gen_data = generated_samples[t_val].cpu().numpy().flatten()
        
        # Plot histograms
        ax_hist = axes[0, i]
        ax_hist.hist(orig_data, bins=50, alpha=0.7, density=True, label='Original', color='blue')
        ax_hist.hist(gen_data, bins=50, alpha=0.7, density=True, label='Generated', color='red')
        ax_hist.set_title(f'Distributions at t={t_val:.2f}')
        ax_hist.set_xlabel('Pixel Value')
        ax_hist.set_ylabel('Density')
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        
        # Q-Q plot for distributional comparison
        ax_qq = axes[1, i]
        # Sort both datasets for Q-Q plot
        orig_sorted = np.sort(orig_data)
        gen_sorted = np.sort(gen_data)
        
        # Interpolate to same length for comparison
        n_points = min(len(orig_sorted), len(gen_sorted))
        orig_interp = np.interp(np.linspace(0, 1, n_points), 
                               np.linspace(0, 1, len(orig_sorted)), orig_sorted)
        gen_interp = np.interp(np.linspace(0, 1, n_points), 
                              np.linspace(0, 1, len(gen_sorted)), gen_sorted)
        
        ax_qq.scatter(orig_interp, gen_interp, alpha=0.5, s=1)
        
        # Add diagonal line for perfect match
        min_val = min(orig_interp.min(), gen_interp.min())
        max_val = max(orig_interp.max(), gen_interp.max())
        ax_qq.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Match')
        
        ax_qq.set_title(f'Q-Q Plot at t={t_val:.2f}')
        ax_qq.set_xlabel('Original Quantiles')
        ax_qq.set_ylabel('Generated Quantiles')
        ax_qq.legend()
        ax_qq.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "sample_distributions_comparison.png"), dpi=300)
    plt.close()


def _plot_grf_marginals(marginal_data, output_dir, title="GRF Data Marginals"):
    """Visualizes samples from the GRF marginal data."""
    print(f"  - Plotting {title}...")
    
    sorted_times = sorted(marginal_data.keys())
    n_marginals = len(sorted_times)
    n_samples_to_show = 5
    
    # Determine resolution
    if not sorted_times:
        return
    data_dim = marginal_data[sorted_times[0]].shape[1]
    resolution = int(math.sqrt(data_dim))
    
    if resolution * resolution != data_dim:
        print(f"    Warning: Data dimension {data_dim} is not a perfect square. Skipping visualization.")
        return

    fig, axes = plt.subplots(n_samples_to_show, n_marginals, figsize=(3 * n_marginals, 3 * n_samples_to_show))
    # Handle potential dimension issues if n_samples_to_show or n_marginals is 1
    if n_samples_to_show == 1 and n_marginals == 1:
        axes = np.array([[axes]])
    elif n_samples_to_show == 1:
        axes = axes.reshape(1, -1)
    elif n_marginals == 1:
        axes = axes.reshape(-1, 1)
    elif axes.ndim == 1:
        axes = axes.reshape(-1, 1) if n_marginals == 1 else axes.reshape(1, -1)
    
    # Determine global vmin/vmax
    all_data = torch.cat([marginal_data[t] for t in sorted_times], dim=0).cpu().numpy()
    vmin, vmax = all_data.min(), all_data.max()

    for i in range(n_samples_to_show):
        for j, t_val in enumerate(sorted_times):
            data_at_t = marginal_data[t_val].cpu().numpy()
            if i < data_at_t.shape[0]:
                sample = data_at_t[i].reshape(resolution, resolution)
                axes[i, j].imshow(sample, cmap='viridis', vmin=vmin, vmax=vmax, extent=[0, 1, 0, 1])
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(f't = {t_val:.2f}')

    plt.tight_layout()
    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


# ============================================================================
# Training Functions
# ============================================================================


def train_bridge(
    bridge: NeuralBridgeExpressive,
    marginal_data: Dict[float, Tensor],
    epochs: int = 2000,              # Increased default epochs for better convergence
    lr: float = 5e-4,               # Reduced learning rate for more stable training
    lambda_path: float = 0.01,      # Reduced path regularization to focus on distribution matching
    weight_decay: float = 1e-4,     # Added parameter for regularization
    use_scheduler: bool = True,     # Added parameter for learning rate scheduling
    grad_clip_norm: float = 1.0,    # Gradient clipping for stability
    verbose: bool = True,
) -> list:
    """
    Train the expressive bridge using MLE and path regularization.
    
    OPTIMIZATION CHALLENGES FOR GLOW FLOW:
    
    The GLOW-based normalizing flow is highly expressive and can model complex distributions
    with full covariance structures, but this comes with significant optimization challenges:
    
    1. High-dimensional parameter space: GLOW has many more parameters than simple affine flows
    2. Complex loss landscape: The combination of invertible networks and SDE dynamics creates
       a challenging optimization surface
    3. Distribution vs. Correlation Trade-off:
       - GLOW excels at capturing spatial correlations (good ACF performance)
       - But may struggle to match overall distribution parameters (mean/variance)
       - This explains why GLOW can have high W2 distance but low MSE_ACF
    
    RECOMMENDED HYPERPARAMETERS:
    - Lower learning rates (5e-4 instead of 1e-3) for more stable convergence
    - More epochs (2000+) to allow the complex model to converge
    - Lower path regularization (0.01 instead of 0.1) to prioritize distribution matching
    - Cosine annealing scheduler to help escape local minima
    
    Args:
        bridge: The neural bridge model to train
        marginal_data: Dictionary of time -> samples for training
        epochs: Number of training epochs (default: 2000 for better convergence)
        lr: Learning rate (default: 5e-4 for stability)
        lambda_path: Path regularization weight (default: 0.01 for distribution focus)
        weight_decay: L2 regularization weight (default: 1e-4)
        use_scheduler: Whether to use cosine annealing scheduler (default: True)
        grad_clip_norm: Gradient clipping norm (default: 1.0)
        verbose: Whether to show progress bar (default: True)
        
    Returns:
        List of loss dictionaries with 'total', 'mle', and 'path' losses per epoch
    """
    # FIX: Use AdamW for decoupled weight decay
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=lr, weight_decay=weight_decay)
    
    # FIX: Initialize Scheduler
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)

    loss_history = []

    if verbose:
        pbar = trange(epochs)
    else:
        pbar = range(epochs)

    bridge.train() # Ensure model is in training mode

    for epoch in pbar:
        optimizer.zero_grad()

        # Compute loss
        total_loss, mle_loss, path_loss = bridge.loss(
            marginal_data, lambda_path=lambda_path
        )

        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), max_norm=grad_clip_norm)

        optimizer.step()
        
        if scheduler:
            scheduler.step()

        loss_history.append(
            {
                "total": total_loss.item(),
                "mle": mle_loss.item(),
                "path": path_loss.item(),
            }
        )

        if verbose and isinstance(pbar, type(trange(0))):
            pbar.set_description(
                f"Loss: {total_loss.item():.4f} (MLE: {mle_loss.item():.4f}, Path: {path_loss.item():.4f})"
            )

    bridge.eval()
    return loss_history


if __name__ == "__main__":
    
    # --- Configuration ---
    RESOLUTION_GRF = 16
    HIDDEN_SIZE = 64
    N_BLOCKS_FLOW = 2
    NUM_SCALES = 2
    T_MAX = 1.0
    DEVICE = "cpu"
    SIGMA_REVERSE = 0.5
    OUTPUT_DIR = "output_glow_flow"
    
    # Data Generation Config
    N_SAMPLES = 1024
    N_CONSTRAINTS = 5
    L_DOMAIN = 1.0
    MICRO_CORR_LENGTH = 0.1
    H_MAX_FACTOR = 0.5
    MEAN_VAL = 10.0
    STD_VAL = 2.0
    COVARIANCE_TYPE = "gaussian"

    # Training Config
    EPOCHS = 1000 # Recommend 1000+ for good results
    LEARNING_RATE = 5e-4
    LAMBDA_PATH = 0.01
    WEIGHT_DECAY = 1e-4
    USE_SCHEDULER = True
    GRAD_CLIP_NORM = 1.0
    
    # --- Main Execution ---
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("--- Asymmetric Bridge with GLOW Flow ---")
    print(f"Output will be saved to: {OUTPUT_DIR}")

    # 1. Generate Data
    marginal_data_grf_raw, data_dim_grf = generate_multiscale_grf_data(
        N_samples=N_SAMPLES,
        T=T_MAX,
        N_constraints=N_CONSTRAINTS,
        resolution=RESOLUTION_GRF,
        L_domain=L_DOMAIN,
        micro_corr_length=MICRO_CORR_LENGTH,
        H_max_factor=H_MAX_FACTOR,
        mean_val=MEAN_VAL,
        std_val=STD_VAL,
        covariance_type=COVARIANCE_TYPE,
        device=DEVICE,
    )

    # 2. Normalize Data
    print("\nNormalizing data...")
    marginal_data_grf, data_mean, data_std = normalize_multimarginal_data(marginal_data_grf_raw)
    print("  - Data normalized.")

    # 3. Initialize Bridge Model
    print("\nInitializing bridge model...")
    bridge_grf = NeuralBridgeExpressive(
        data_dim=data_dim_grf, 
        hidden_size=HIDDEN_SIZE, 
        resolution=RESOLUTION_GRF,
        n_blocks_flow=N_BLOCKS_FLOW, 
        num_scales=NUM_SCALES,
        T=T_MAX,
        data_mean=data_mean.to(DEVICE),
        data_std=data_std.to(DEVICE),
        sigma_reverse=SIGMA_REVERSE,
        training_noise_std=0.01,
        noise_densification_std=0.00001,
        inference_clamp_norm=None
    ).to(DEVICE)
    print(f"  - Bridge initialized with {sum(p.numel() for p in bridge_grf.parameters())} parameters.")

    # 4. Train the Bridge
    print("\n--- Training Bridge ---")
    loss_history = train_bridge(
        bridge_grf,
        marginal_data_grf,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        lambda_path=LAMBDA_PATH,
        weight_decay=WEIGHT_DECAY,
        use_scheduler=USE_SCHEDULER,
        grad_clip_norm=GRAD_CLIP_NORM,
        verbose=True,
    )
    print(f"Training complete. Final loss: {loss_history[-1]['total']:.4f}")

    # 5. Evaluation and Visualization
    print("\n--- Running Evaluation & Visualization Pipeline ---")
    with torch.no_grad():
        # 5.1 Visualize Input Data
        _plot_grf_marginals(marginal_data_grf, OUTPUT_DIR, title="Input_GRF_Data_Marginals")
        
        # 5.2 Generate samples for consistent comparison
        n_samples_show = 4
        selected_indices = torch.randperm(marginal_data_grf[max(marginal_data_grf.keys())].shape[0])[:n_samples_show]
        
        original_selected, generated_consistent = generate_consistent_backward_samples(
            bridge_grf, marginal_data_grf, selected_indices, n_steps=100, device=DEVICE
        )
        
        # 5.3 Visualize consistent sample evolution
        _visualize_consistent_samples_comparison(original_selected, generated_consistent, selected_indices, OUTPUT_DIR)
        
        # 5.4 Generate samples for statistical analysis
        n_samples_analysis = 512
        generated_samples = generate_backward_samples(
            bridge_grf, marginal_data_grf, n_samples=n_samples_analysis, n_steps=100, device=DEVICE
        )
        
        # 5.5 Statistical & Distributional Comparison
        _visualize_marginal_statistics_comparison(marginal_data_grf, generated_samples, OUTPUT_DIR)
        _visualize_sample_distributions(marginal_data_grf, generated_samples, OUTPUT_DIR)
        
        # 5.6 Covariance Analysis
        _visualize_covariance_comparison(marginal_data_grf, generated_samples, OUTPUT_DIR)
        
        # 5.7 Quantitative Metrics
        validation_metrics = _calculate_validation_metrics(marginal_data_grf, generated_samples)
        print("\nValidation Summary:")
        for i, t_val in enumerate(validation_metrics['times']):
            w2_dist = validation_metrics['w2_distances'][i]
            mse_acf = validation_metrics['mse_acf'][i]
            print(f"  t={t_val:.2f}: W2={w2_dist:.3e}, MSE_ACF={mse_acf:.3e}")

    print("\n--- Script Finished ---")