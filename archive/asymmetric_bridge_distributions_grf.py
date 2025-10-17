"""
Asymmetric Multi-Marginal Bridge Framework - Neural Implementation
========================================================================

This module implements the Asymmetric Multi-Marginal Bridge framework using
a Neural Network parameterization for the flow, enabling distributional 
interpolation. We utilize the Latent Space Reparameterization approach 
to ensure exact consistency between the forward ODE and the backward SDE.

This implementation uses an Affine Flow (Neural Gaussian Flow), adapting
architectural concepts and utilities from the SDE Matching literature.

Key Components:
- Utilities for automatic differentiation (jvp, t_dir)
- TimeDependentAffine: NN parameterization of the affine flow (mu(t), gamma(t))
- NeuralGaussianBridge: Core module implementing the framework
- Training and validation logic for distributional constraints
"""

from typing import Tuple, Callable, Optional, Dict, Any
import torch
from torch import nn, Tensor
from torch import distributions as D
import torch.fft as fft
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import transforms
from tqdm import trange
import numpy as np
import math
import traceback
try:
    import ot  # Optimal Transport for validation metrics
except ImportError:
    print("Warning: Python Optimal Transport (POT) library not found. Wasserstein distance validation will be unavailable.")
    ot = None
import os
from scipy.integrate import solve_ivp
from scipy.stats import gaussian_kde

# Import utilities for advanced visualization
from utilities.visualization import (
    visualize_bridge_results as util_visualize_bridge_results,
)
from utilities.simulation import generate_comparative_backward_samples


# ============================================================================
# Utilities (Adapted from sde_matching.py)
# ============================================================================

def jvp(f: Callable[[Tensor], Any], x: Tensor, v: Tensor) -> Tuple[Tensor, ...]:
    """Compute Jacobian-vector product. Used for time derivatives."""
    # Ensures gradients can flow back through the JVP calculation if needed
    return torch.autograd.functional.jvp(
        f, x, v,
        create_graph=torch.is_grad_enabled()
    )

def t_dir(f: Callable[[Tensor], Any], t: Tensor) -> Tuple[Tensor, ...]:
    """Compute the time derivative of f(t) by using jvp with v=1."""
    return jvp(f, t, torch.ones_like(t))

# ============================================================================
# Flow Model Components (Adapted from sde_matching.py PosteriorAffine)
# ============================================================================

class TimeDependentAffine(nn.Module):
    """
    Neural Network parameterizing an affine transformation over time.
    G(epsilon, t) = mu(t) + gamma(t) * epsilon.
    """
    def __init__(self, data_dim: int, hidden_size: int):
        super().__init__()
        # Input is time t (dim=1)
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * data_dim),
        )

    def get_coeffs(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        """Get the coefficients mu(t) and gamma(t)."""
        # Ensure t is [Batch, 1] or [1, 1]
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        mu, log_gamma = self.net(t).chunk(chunks=2, dim=1)
        gamma = torch.exp(log_gamma)
        # Clamp gamma for numerical stability in downstream calculations (division/log)
        gamma = torch.clamp(gamma, min=1e-6)
        return mu, gamma

    def forward(self, t: Tensor, return_t_dir: bool = False) -> Any:
        """
        Evaluate the coefficients and optionally their time derivatives.
        """
        if return_t_dir:
            # Compute time derivatives using automatic differentiation (t_dir/jvp)
            def f(t_in: Tensor) -> Tuple[Tensor, Tensor]:
                return self.get_coeffs(t_in)
            
            # t_dir returns ((mu, gamma), (dmu_dt, dgamma_dt))
            return t_dir(f, t)
        else:
            return self.get_coeffs(t)

# ============================================================================
# Asymmetric Bridge Implementation
# ============================================================================

class NeuralGaussianBridge(nn.Module):
    """
    Neural Gaussian Asymmetric Multi-Marginal Bridge.

    Implements the asymmetric bridge using an affine flow parameterized by a NN.
    """
    def __init__(
        self, 
        data_dim: int,
        hidden_size: int,
        T: float = 1.0,
        sigma_reverse: float = 1.0
    ):
        super().__init__()
        
        self.data_dim = data_dim
        self.T = T
        self.sigma_reverse = sigma_reverse
        
        # The flow model
        self.affine_flow = TimeDependentAffine(data_dim, hidden_size)

        # Base distribution (Standard Normal Latent Space)
        self.register_buffer('base_mean', torch.zeros(data_dim))
        self.register_buffer('base_std', torch.ones(data_dim))
    
    @property
    def base_dist(self):
        return D.Independent(D.Normal(self.base_mean, self.base_std), 1)
        
    def denormalize(self, z: Tensor) -> Tensor:
        """Denormalize data (identity for Gaussian bridge since no normalization is used)."""
        return z

    def get_params(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        """Get mu(t) and gamma(t)."""
        return self.affine_flow(t, return_t_dir=False)

    def get_params_and_derivs(self, t: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Get mu, gamma and their time derivatives (dmu_dt, dgamma_dt)."""
        (mu, gamma), (dmu_dt, dgamma_dt) = self.affine_flow(t, return_t_dir=True)
        return mu, gamma, dmu_dt, dgamma_dt
    
    # --- Dynamics (Forward ODE and Backward SDE) ---

    def forward_velocity(self, z: Tensor, t: Tensor) -> Tensor:
        """
        Compute the deterministic forward velocity field (PF-ODE).
        v(z,t) = dmu/dt + (dgamma/dt / gamma) * (z - mu)
        """
        # Ensure t is correctly shaped [B, 1]
        t = self._format_time(t, z.shape[0])

        mu, gamma, dmu_dt, dgamma_dt = self.get_params_and_derivs(t)
        
        # Gamma is clamped in TimeDependentAffine (min=1e-6) for stability.
        velocity = dmu_dt + (dgamma_dt / gamma) * (z - mu)
        return velocity
    
    def score_function(self, z: Tensor, t: Tensor) -> Tensor:
        """
        Compute the exact score function ∇_z log p_t(z).
        For Gaussian: ∇ log p(z) = -(z - mu) / gamma^2
        """
        t = self._format_time(t, z.shape[0])
        mu, gamma = self.get_params(t)
        
        score = -(z - mu) / (gamma**2)
        return score
    
    def reverse_drift(self, z: Tensor, t: Tensor) -> Tensor:
        """
        Compute the drift for the exact reverse SDE.
        Reverse drift: R(z,t) = v(z,t) - (σ²/2) * ∇log p_t(z)
        """
        velocity = self.forward_velocity(z, t)
        score = self.score_function(z, t)
        
        drift = velocity - (self.sigma_reverse**2 / 2) * score
        return drift
    
    def reverse_sde(self, z: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """Complete reverse SDE specification (drift and diffusion)."""
        drift = self.reverse_drift(z, t)
        diffusion = torch.ones_like(z) * self.sigma_reverse
        return drift, diffusion

    def _format_time(self, t, batch_size):
        """Helper to format time tensor t to match batch size [B, 1]."""
        if not torch.is_tensor(t):
            t = torch.tensor(float(t), device=self.base_mean.device)
        
        if t.dim() == 0:
            # Scalar time
            return t.expand(batch_size, 1)
        elif t.dim() == 1:
            if t.shape[0] == 1:
                return t.expand(batch_size, 1)
            elif t.shape[0] == batch_size:
                return t.unsqueeze(-1)
        elif t.dim() == 2:
            if t.shape[0] == batch_size and t.shape[1] == 1:
                return t
            elif t.shape[0] == 1 and t.shape[1] == 1:
                return t.expand(batch_size, 1)
        
        raise ValueError(f"Time tensor shape {t.shape} incompatible with batch size {batch_size}.")

    # --- Objectives (Loss Functions) ---

    def path_regularization_loss(self, t: Tensor) -> Tensor:
        """
        Compute the kinetic energy loss (Path Regularization J_Path).
        E[||v||²] = ||dmu/dt||² + sum((dgamma_i/dt)²)  (for diagonal gamma)
        """
        _, _, dmu_dt, dgamma_dt = self.get_params_and_derivs(t)
        
        mean_term = torch.sum(dmu_dt**2, dim=-1)
        var_term = torch.sum(dgamma_dt**2, dim=-1)
        
        kinetic_energy = (mean_term + var_term) / 2.0
        return kinetic_energy.mean()

    def marginal_log_likelihood(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Compute the exact log likelihood log p_t(x) using Change of Variables.
        For affine flow: log p(x) = log p_base( (x-mu)/gamma ) - sum(log(gamma))
        """
        mu, gamma = self.get_params(t)

        # 1. Inverse transformation (compute latent epsilon)
        epsilon = (x - mu) / gamma

        # 2. Base distribution log probability
        log_prob_base = self.base_dist.log_prob(epsilon)

        # 3. Log determinant of the Jacobian of the inverse
        log_det_J_inv = -torch.sum(torch.log(gamma), dim=-1)

        # 4. Total log likelihood
        log_likelihood = log_prob_base + log_det_J_inv
        return log_likelihood

    def loss(self, marginal_data: Dict[float, Tensor], lambda_path: float = 0.1, batch_size_time: int = 256) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute the total loss: L_MM (MLE) + lambda * J_Path.
        """
        
        # 1. Multi-Marginal MLE Loss (L_MM)
        mle_loss = 0
        total_samples = 0
        device = self.base_mean.device

        for t_k, samples_k in marginal_data.items():
            B_k = samples_k.shape[0]
            total_samples += B_k
            # Create time tensor [B_k, 1]
            t_k_tensor = torch.full((B_k, 1), t_k, device=device)
            
            log_likelihoods = self.marginal_log_likelihood(samples_k, t_k_tensor)
            mle_loss -= log_likelihoods.sum()

        if total_samples > 0:
            mle_loss /= total_samples

        # 2. Path Regularization Loss (J_Path)
        # Sample random times for Monte Carlo estimation
        t_rand = torch.rand(batch_size_time, 1, device=device) * self.T
        path_loss = self.path_regularization_loss(t_rand)

        # Total Loss
        total_loss = mle_loss + lambda_path * path_loss
        return total_loss, mle_loss, path_loss


# ============================================================================
# Simulation Utilities
# ============================================================================

def solve_gaussian_bridge_reverse_sde(
    bridge: NeuralGaussianBridge,
    z_start: Tensor,
    ts: float,
    tf: float,
    n_steps: int
) -> Tensor:
    """
    Specialized solver for the reverse SDE of the Gaussian bridge.
    Uses Exponential Integrator with Variance Matching for stability and accuracy.
    This is applicable because the NeuralGaussianBridge has affine dynamics.
    """
    if ts <= tf:
        raise ValueError("This solver is designed for backward integration (ts > tf).")

    dt = (tf - ts) / n_steps # dt is negative
    tt = torch.linspace(ts, tf, n_steps + 1)

    path = [z_start.clone()]
    current_z = z_start.clone()
    B = current_z.shape[0]
    sigma = bridge.sigma_reverse
    EPSILON = 1e-9 # Epsilon for numerical stability
    device = z_start.device
    
    for i in range(n_steps):
        t, t_next = tt[i], tt[i+1]

        # 1. Get parameters at t and t_next
        # Input time tensors must be [1, 1] for the NN forward pass with derivatives
        t_tensor = torch.tensor([[float(t)]], device=device, requires_grad=torch.is_grad_enabled())
        t_next_tensor = torch.tensor([[float(t_next)]], device=device, requires_grad=torch.is_grad_enabled())

        # Ensure NN evaluations happen in a graph-enabled context if needed
        with torch.set_grad_enabled(torch.is_grad_enabled()):
             # Get params and derivs at t
            mu_t_scalar, gamma_t_scalar, _, dgamma_dt_scalar = bridge.get_params_and_derivs(t_tensor)
            # Get params at t_next
            mu_next_scalar, gamma_next_scalar = bridge.get_params(t_next_tensor)

        # 2. Expand parameters to batch size
        mu_t = mu_t_scalar.expand(B, -1)
        gamma_t_raw = gamma_t_scalar.expand(B, -1)
        dgamma_dt = dgamma_dt_scalar.expand(B, -1)
        mu_next = mu_next_scalar.expand(B, -1)
        gamma_next = gamma_next_scalar.expand(B, -1)

        # 3. Calculate the drift coefficient C(t)
        # C(t) = (dgamma/dt / gamma) + (sigma^2 / (2*gamma^2))
        
        # Clamping handled by TimeDependentAffine, but reinforced here for safety
        gamma_t_clamped = torch.clamp(gamma_t_raw, min=EPSILON)
        gamma_sq_t_clamped = torch.clamp(gamma_t_raw**2, min=EPSILON**2)

        C_t = (dgamma_dt / gamma_t_clamped) + (sigma**2 / 2) / gamma_sq_t_clamped

        # 4. Calculate the amplification factor A = exp(C(t) dt)
        A_t = torch.exp(C_t * dt) # Stable as C_t*dt <= 0 for reverse time.

        # 5. Deterministic update of the deviation
        deviation_t = current_z - mu_t
        deterministic_update = A_t * deviation_t

        # 6. Calculate the required noise variance (Variance Matching)
        # Var(Noise) = gamma_{t+dt}^2 - A_t^2 * gamma_t^2
        variance = gamma_next**2 - (A_t**2) * gamma_t_raw**2
        
        # Clamp variance to be non-negative
        variance = torch.clamp(variance, min=0.0)
        std_dev = torch.sqrt(variance)

        # 7. Add noise and update
        noise = torch.randn_like(current_z) * std_dev
        current_z = mu_next + deterministic_update + noise
        
        if torch.isnan(current_z).any():
            print(f"Warning: NaN detected in simulation at t={t:.4f}. Stopping.")
            break

        path.append(current_z.clone())

    return torch.stack(path)


def solve_backward_sde_euler_maruyama(
    bridge: NeuralGaussianBridge,
    z_start: Tensor,
    ts: float,
    tf: float,
    n_steps: int
) -> Tensor:
    """
    Solve the backward SDE using Euler-Maruyama method.
    Alternative to the specialized Gaussian solver for comparison.
    
    Args:
        bridge: Trained NeuralGaussianBridge model
        z_start: Starting samples at time ts [B, D]
        ts: Start time (should be T, final time)
        tf: Final time (should be 0, initial time)
        n_steps: Number of integration steps
    
    Returns:
        Tensor of shape [n_steps+1, B, D] containing the trajectory
    """
    if ts <= tf:
        raise ValueError("Backward SDE requires ts > tf (reverse time integration)")
    
    dt = (tf - ts) / n_steps  # dt is negative for backward integration
    current_z = z_start.clone()
    path = [current_z.clone()]
    
    # Diffusion coefficient for the backward SDE
    sigma_backward = bridge.sigma_reverse
    
    for step in range(n_steps):
        t_current = ts + step * dt
        # Ensure t_current is a tensor on the correct device
        t_tensor = torch.tensor([[t_current]], device=current_z.device, dtype=torch.float32)
        
        # Use the bridge's reverse drift method
        drift_backward = bridge.reverse_drift(current_z, t_tensor)
        
        # Euler-Maruyama update
        noise = torch.randn_like(current_z) * math.sqrt(abs(dt)) * sigma_backward
        current_z = current_z + drift_backward * dt + noise
        
        path.append(current_z.clone())
    
    return torch.stack(path)


def generate_backward_samples(bridge: NeuralGaussianBridge, marginal_data: Dict[float, Tensor], n_samples: int = 64, n_steps: int = 100, device: str = 'cpu') -> Dict[float, Tensor]:
    """
    Generate backward samples from the learned bridge starting from the final time distribution.
    This implements proper bridge sampling by starting from the T=1 marginal and integrating 
    backward through the reverse SDE to generate samples at all intermediate time points.
    
    Args:
        bridge: Trained NeuralGaussianBridge model
        marginal_data: Dictionary of {time: samples} from original data
        n_samples: Number of samples to generate for backward trajectories
        n_steps: Number of SDE integration steps
        device: Device to run computations on
    
    Returns:
        Dictionary of {time: generated_samples} with same structure as marginal_data
    """
    print("  - Generating backward samples from learned bridge (starting from final time)...")
    
    bridge.eval()
    generated_samples = {}
    T = bridge.T
    
    # Get the time points from marginal data
    sorted_times = sorted(marginal_data.keys())
    
    with torch.no_grad():
        # Step 1: Start from the final time distribution
        # Use actual samples from the final time marginal data if available, 
        # otherwise sample from the learned distribution at T
        final_time = max(sorted_times)
        if final_time in marginal_data and marginal_data[final_time].shape[0] >= n_samples:
            # Use actual data samples from final time
            indices = torch.randperm(marginal_data[final_time].shape[0])[:n_samples]
            z_final = marginal_data[final_time][indices].to(device)
            print(f"    Starting from {n_samples} actual samples at t={final_time}")
        else:
            # Generate samples from the learned distribution at final time
            t_final_tensor = torch.tensor([[final_time]], device=device, dtype=torch.float32)
            mu_final, gamma_final = bridge.get_params(t_final_tensor)
            z_final = mu_final + gamma_final * torch.randn(n_samples, bridge.data_dim, device=device)
            print(f"    Starting from {n_samples} generated samples at t={final_time}")
        
        # Step 2: Solve backward SDE from final time to 0
        print(f"    Integrating backward SDE with {n_steps} steps...")
        try:
            # Use the Euler-Maruyama backward SDE solver
            backward_trajectory = solve_backward_sde_euler_maruyama(
                bridge, z_final, ts=final_time, tf=0.0, n_steps=n_steps
            )  # Shape: [n_steps+1, n_samples, data_dim]
            
            # Step 3: Interpolate to get samples at the specific time points
            trajectory_times = torch.linspace(final_time, 0.0, n_steps + 1)
            
            for t_val in sorted_times:
                # Find the closest time index in the trajectory
                time_idx = torch.argmin(torch.abs(trajectory_times - t_val)).item()
                samples_at_t = backward_trajectory[time_idx].cpu()
                generated_samples[t_val] = samples_at_t
                
        except Exception as e:
            print(f"    Warning: Backward SDE integration failed ({e}). Falling back to forward sampling.")
            # Fallback: Use forward sampling from base distribution
            epsilon = bridge.base_dist.sample((n_samples,))
            for t_val in sorted_times:
                t_tensor = torch.tensor([[t_val]], device=device, dtype=torch.float32)
                mu_t, gamma_t = bridge.get_params(t_tensor)
                samples = mu_t + gamma_t * epsilon.to(device)
                generated_samples[t_val] = samples.cpu()
    
    print(f"    Generated samples for {len(generated_samples)} time points")
    return generated_samples


def generate_consistent_backward_samples(bridge: NeuralGaussianBridge, marginal_data: Dict[float, Tensor], 
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
            # Use the Euler-Maruyama backward SDE solver
            backward_trajectory = solve_backward_sde_euler_maruyama(
                bridge, z_final, ts=final_time, tf=0.0, n_steps=n_steps
            )  # Shape: [n_steps+1, n_samples, data_dim]
            
            # Step 4: Interpolate to get samples at the specific time points
            trajectory_times = torch.linspace(final_time, 0.0, n_steps + 1)
            
            for t_val in sorted_times:
                # Find the closest time index in the trajectory
                time_idx = torch.argmin(torch.abs(trajectory_times - t_val)).item()
                samples_at_t = backward_trajectory[time_idx].cpu()
                generated_consistent[t_val] = samples_at_t
                
        except Exception as e:
            print(f"    Warning: Backward SDE integration failed ({e}). Using original samples as fallback.")
            # Fallback: Use the original samples (no bridge evolution)
            for t_val in sorted_times:
                generated_consistent[t_val] = original_selected[t_val].clone()
    
    print(f"    Generated consistent samples for {len(generated_consistent)} time points")
    return original_selected, generated_consistent


# ============================================================================
# 0. Random Field Generation Utilities (Multiscale Homogenization)
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
        self.nx = nx; self.ny = ny; self.lx = lx; self.ly = ly; self.device = device

    def generate_random_field(self, mean=10.0, std=2.0, correlation_length=0.2, covariance_type="exponential"):
        dx = self.lx / self.nx; dy = self.ly / self.ny
        white_noise = np.random.normal(0, 1, (self.nx, self.ny))
        fourier_coefficients = np.fft.fft2(white_noise)
        kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=dx); ky = 2 * np.pi * np.fft.fftfreq(self.ny, d=dy)
        Kx, Ky = np.meshgrid(kx, ky, indexing="ij"); K = np.sqrt(Kx**2 + Ky**2)
        l = correlation_length
        if covariance_type == "exponential":
            denom = (1 + (l * K) ** 2); P = (2 * np.pi * l**2) / np.maximum(1e-9, denom ** (1.5))
        elif covariance_type == "gaussian":
            P = np.pi * l**2 * np.exp(-((l * K) ** 2) / 4)
        else: raise ValueError("Invalid covariance_type")
        P = np.nan_to_num(P)
        fourier_coefficients *= np.sqrt(P); field = np.fft.ifft2(fourier_coefficients).real
        field_std = np.std(field)
        if field_std > 1e-9: field = (field - np.mean(field)) / field_std * std + mean
        else: field = np.full_like(field, mean)
        return field

    def coarsen_field(self, field, H):
        if isinstance(field, np.ndarray): field = torch.from_numpy(field).to(self.device)
        if field.dim() == 3: field = field.unsqueeze(1); squeeze_channel = True
        elif field.dim() == 4: squeeze_channel = False
        else: raise ValueError("Unsupported field dimensions (must be 3 or 4)")
        pixel_size = self.lx / self.nx; filter_sigma_phys = H / 6.0
        filter_sigma_pix = filter_sigma_phys / pixel_size
        if filter_sigma_pix < 1e-6: smooth = field
        else:
            kernel_size = int(6 * filter_sigma_pix)
            if kernel_size % 2 == 0: kernel_size += 1
            kernel_size = max(3, kernel_size)
            smooth = gaussian_blur_periodic(field, kernel_size=kernel_size, sigma=filter_sigma_pix)
        coarse = smooth
        if squeeze_channel: coarse = coarse.squeeze(1)
        return coarse

# ============================================================================
# Data Generation
# ============================================================================

def generate_multiscale_grf_data(
    N_samples: int, T: float = 1.0, N_constraints: int = 5, resolution: int = 32,
    L_domain: float = 1.0, micro_corr_length: float = 0.1, H_max_factor: float = 0.5,
    mean_val: float = 10.0, std_val: float = 2.0, covariance_type: str = "exponential",
    device: str = 'cpu'
) -> Tuple[Dict[float, Tensor], int]:
    """Generate multiscale Gaussian Random Field data."""
    print(f"\n--- Generating Multiscale GRF Data (Fixed Resolution: {resolution}x{resolution}) ---")
    time_steps = torch.linspace(0, T, N_constraints); marginal_data = {}; data_dim = resolution * resolution
    generator = RandomFieldGenerator2D(nx=resolution, ny=resolution, lx=L_domain, ly=L_domain, device=device)
    print("Generating base microscopic fields (t=0)..."); micro_fields = []
    for _ in trange(N_samples):
        field = generator.generate_random_field(mean=mean_val, std=std_val, correlation_length=micro_corr_length, covariance_type=covariance_type)
        micro_fields.append(field)
    micro_fields_tensor = torch.tensor(np.array(micro_fields), dtype=torch.float32).to(device)
    print("Applying progressive coarsening filters..."); H_max = L_domain * H_max_factor
    for t in time_steps:
        t_val = t.item(); t_norm = t_val / T; H_t = t_norm * H_max
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

# ============================================================================
# Training
# ============================================================================

def train_bridge(
    bridge: NeuralGaussianBridge, 
    marginal_data: Dict[float, Tensor],
    epochs: int = 2000, 
    lr: float = 1e-3,
    lambda_path: float = 0.1,
    verbose: bool = True
) -> list:
    """
    Train the Neural Gaussian bridge using MLE and path regularization.
    """
    optimizer = torch.optim.Adam(bridge.parameters(), lr=lr)
    loss_history = []
    
    if verbose:
        pbar = trange(epochs)
    else:
        pbar = range(epochs)
    
    for epoch in pbar:
        optimizer.zero_grad()
        
        # Compute loss
        total_loss, mle_loss, path_loss = bridge.loss(marginal_data, lambda_path=lambda_path)
        
        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), max_norm=1.0)

        optimizer.step()
        
        loss_history.append({
            'total': total_loss.item(),
            'mle': mle_loss.item(), 
            'path': path_loss.item()
        })
        
        if verbose and isinstance(pbar, type(trange(0))):
            pbar.set_description(
                f"Loss: {total_loss.item():.4f} (MLE: {mle_loss.item():.4f}, Path: {path_loss.item():.4f})"
            )
    
    return loss_history

# ============================================================================
# Validation and Visualization (Helper Functions)
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
    if not selected_times: return
    data_dim = marginal_data[selected_times[0]].shape[1]
    resolution = int(math.sqrt(data_dim))
    if resolution * resolution != data_dim:
        print("    Warning: Data is not square. Skipping covariance analysis.")
        return

    # Setup figure: 3 rows (1D Radial ACF, 2D Target ACF, 2D Gen ACF)
    fig, axes = plt.subplots(3, n_time_points, figsize=(4 * n_time_points, 12))
    if n_time_points == 1: axes = axes.reshape(3, 1)
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
        if i == 0: ax_target.set_ylabel('Y Lag')
        
        # Row 3: 2D Generated ACF
        ax_gen = axes[2, i]
        ax_gen.imshow(acf_gen.numpy(), cmap='viridis', vmin=vmin, vmax=vmax, extent=[-0.5, 0.5, -0.5, 0.5])
        ax_gen.set_title('Generated 2D ACF')
        ax_gen.set_xlabel('X Lag')
        if i == 0: ax_gen.set_ylabel('Y Lag')

    # Add colorbar
    if im is not None:
        fig.colorbar(im, ax=axes[1:, :].ravel().tolist(), fraction=0.046, pad=0.04, label="Correlation")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "covariance_comparison.png"), dpi=300)
    plt.close()

def _calculate_validation_metrics(marginal_data: Dict[float, Tensor], generated_samples: Dict[float, Tensor]) -> Dict[str, Any]:
    """Calculate quantitative validation metrics (W2 distance and MSE_ACF)."""
    print("  - Calculating quantitative validation metrics (W2, MSE_ACF)...")
    
    sorted_times = sorted(marginal_data.keys())
    metrics = {'times': sorted_times, 'w2_distances': [], 'mse_acf': []}

    # Determine resolution
    if not sorted_times: return metrics
    data_dim = marginal_data[sorted_times[0]].shape[1]
    resolution = int(math.sqrt(data_dim))
    is_square = (resolution * resolution == data_dim)
    
    for t_val in sorted_times:
        target_data_cpu = marginal_data[t_val].cpu()
        gen_data_cpu = generated_samples[t_val].cpu()
        
        # 1. Wasserstein Distance (Sliced)
        w2_dist = float('nan')
        if ot:
            try:
                # Ensure same number of samples
                min_samples = min(target_data_cpu.shape[0], gen_data_cpu.shape[0])
                # Use numpy arrays for POT library
                w2_dist = ot.sliced_wasserstein_distance(
                    target_data_cpu[:min_samples].numpy(), 
                    gen_data_cpu[:min_samples].numpy(), 
                    n_projections=1000, seed=42
                )
            except Exception as e:
                print(f"    Warning: W2 calculation failed at t={t_val:.2f}: {e}")
        metrics['w2_distances'].append(w2_dist)

        # 2. MSE ACF
        mse_acf = float('nan')
        if is_square:
            try:
                acf_target = compute_spatial_acf_2d(target_data_cpu, resolution)
                acf_gen = compute_spatial_acf_2d(gen_data_cpu, resolution)
                mse_acf = torch.mean((acf_target - acf_gen)**2).item()
            except Exception as e:
                 print(f"    Warning: MSE ACF calculation failed at t={t_val:.2f}: {e}")
        metrics['mse_acf'].append(mse_acf)
        
        print(f"    t={t_val:.2f}: W2={w2_dist:.4f}, MSE_ACF={mse_acf:.4e}")

    return metrics

def validate_asymmetric_consistency(
    bridge: NeuralGaussianBridge,
    T: float,
    n_particles: int = 512,
    n_steps: int = 100,
    n_validation_times: int = 5,
    device: str = 'cpu'
) -> dict:
    """
    Rigorous validation of asymmetric consistency.
    Compares reverse SDE simulation against analytical forward distribution.
    """
    bridge.eval()
    bridge = bridge.to(device)
    
    results = {
        'times': [],
        'mean_errors': [],
        'cov_errors': [], 
        'wasserstein_distances': []
    }
    
    with torch.no_grad():
        # 1. Initialize particles at t=T according to the learned distribution p_T(x).
        start_time = T
        # Input time tensor must be [1, 1] for the NN forward pass
        t_start_tensor = torch.tensor([[start_time]], device=device)
        mu_start, gamma_start = bridge.get_params(t_start_tensor)
        
        # Sample initial particles (z_start ~ N(mu_T, gamma_T^2))
        z_start = mu_start + gamma_start * torch.randn(n_particles, bridge.data_dim, device=device)
        
        print(f"\\nStarting reverse SDE simulation from t={start_time:.4f} to t=0.0")
        
        # 2. Solve reverse SDE from T to 0 using the stable solver
        try:
            path = solve_gaussian_bridge_reverse_sde(
                bridge=bridge, z_start=z_start, ts=start_time, tf=0.0, n_steps=n_steps
            )
            print(f"Successfully simulated full trajectory: {path.shape}")
            
        except Exception as e:
            print(f"Full SDE simulation failed: {e}")
            traceback.print_exc()
            return results
        
        # 3. Validation Setup
        val_times = torch.linspace(0.0, T, n_validation_times)
        time_points = torch.linspace(start_time, 0.0, n_steps + 1)
        
        # 4. For each validation time, find closest trajectory point and validate
        for t_val in val_times:
            t_val = t_val.item()
            
            closest_idx = torch.argmin(torch.abs(time_points - t_val)).item()
            actual_time = time_points[closest_idx].item()
            
            print(f"\\nValidating at t={t_val:.2f} (closest simulated: t={actual_time:.2f})")
            
            simulated_particles = path[closest_idx].cpu()
            
            # Get analytical ground truth (forward distribution)
            # Input time tensor must be [1, 1]
            t_tensor = torch.tensor([[t_val]], device=device)
            mu_gt, gamma_gt = bridge.get_params(t_tensor)
            mu_gt = mu_gt[0].cpu()
            gamma_gt = gamma_gt[0].cpu()
            cov_gt = torch.diag(gamma_gt**2)
            
            # Compute validation metrics
            simulated_mean = simulated_particles.mean(dim=0)
            mean_error = torch.norm(simulated_mean - mu_gt).item()

            try:
                simulated_cov = torch.cov(simulated_particles.T)
                cov_error = torch.norm(simulated_cov - cov_gt).item()
                
                w2_dist = float('nan')
                if ot:
                    # Generate GT samples for comparison
                    gt_dist = D.MultivariateNormal(mu_gt, cov_gt)
                    gt_samples = gt_dist.sample((n_particles,))

                    w2_dist = ot.sliced_wasserstein_distance(
                        X_s=simulated_particles.numpy(),
                        X_t=gt_samples.numpy(),
                        n_projections=500, seed=42
                    )
            except Exception as e:
                print(f"Error during metric calculation: {e}")
                cov_error = float('nan')
                w2_dist = float('nan')

            
            results['times'].append(actual_time)
            results['mean_errors'].append(mean_error)
            results['cov_errors'].append(cov_error)
            results['wasserstein_distances'].append(w2_dist)
            
            print(f"  Mean error: {mean_error:.6f}")
            print(f"  Cov error (Frobenius norm): {cov_error:.6f}")
            print(f"  W2 distance: {w2_dist:.6f}")

    return results


def plot_confidence_ellipse(ax, mu, cov, n_std=2.0, **kwargs):
    """
    Plots a confidence ellipse for a 2D Gaussian distribution.
    Projects a 3D covariance to 2D if necessary.
    """
    if mu.shape[0] > 2:
        mu_2d = mu[:2]
    else:
        mu_2d = mu

    if cov.shape[0] > 2:
        cov_2d = cov[:2, :2]
    else:
        cov_2d = cov

    if cov_2d.shape != (2, 2):
        raise ValueError("Covariance matrix must be 2x2 for ellipse plot.")

    lambda_, v = np.linalg.eigh(cov_2d)
    # Eigh can return negative eigenvalues for near-singular matrices. Clamp to 0.
    lambda_ = np.sqrt(np.maximum(lambda_, 0))

    angle = np.rad2deg(np.arctan2(*v[:, 0][::-1]))

    ell = Ellipse(xy=(mu_2d[0], mu_2d[1]),
                  width=lambda_[0]*n_std*2, height=lambda_[1]*n_std*2,
                  angle=angle, **kwargs)
    ell.set_facecolor('none')
    ax.add_patch(ell)


def _plot_marginal_distribution_comparison(
    bridge, T, n_viz_particles, n_sde_steps, output_dir, device
):
    print("  - Plotting marginal distribution comparisons...")

    # Define validation times
    validation_times = [T * f for f in [0.25, 0.5, 0.75]]

    # --- Forward ODE Simulation ---
    t0_tensor = torch.tensor([[0.0]], device=device, dtype=torch.float32)
    mu0, gamma0 = bridge.get_params(t0_tensor)
    z0 = mu0 + gamma0 * torch.randn(n_viz_particles, bridge.data_dim, device=device)

    def forward_ode_func(t, z_np):
        B = z_np.shape[0] // bridge.data_dim
        z = torch.from_numpy(z_np).float().to(device).reshape(B, bridge.data_dim)
        t_tensor = torch.tensor([[t]], device=device, dtype=torch.float32)
        with torch.no_grad():
            v = bridge.forward_velocity(z, t_tensor)
        return v.cpu().numpy().flatten()

    times_eval = torch.linspace(0, T, n_sde_steps + 1).numpy()
    z0_np = z0.cpu().numpy().flatten()
    sol = solve_ivp(
        fun=forward_ode_func, t_span=[0, T], y0=z0_np, method='RK45', t_eval=times_eval
    )
    # Reshape to [n_times, n_particles, data_dim]
    forward_path = torch.from_numpy(sol.y.T).float().reshape(len(times_eval), n_viz_particles, bridge.data_dim)

    # --- Reverse SDE Simulation ---
    tT_tensor = torch.tensor([[T]], device=device, dtype=torch.float32)
    muT, gammaT = bridge.get_params(tT_tensor)
    zT = muT + gammaT * torch.randn(n_viz_particles, bridge.data_dim, device=device)

    reverse_path = solve_gaussian_bridge_reverse_sde(
        bridge=bridge, z_start=zT, ts=T, tf=0.0, n_steps=n_sde_steps
    ).cpu() # Shape [n_steps+1, n_particles, dim]

    # --- Plotting ---
    fig, axes = plt.subplots(2, len(validation_times), figsize=(6 * len(validation_times), 11), sharex=True, sharey=True)
    fig.suptitle("Figure 3: Comparison of Marginal Distributions p_t(z)", fontsize=16)

    for i, t_val in enumerate(validation_times):
        ax_scatter = axes[0, i]
        ax_contour = axes[1, i]

        # Find closest time index
        forward_idx = np.argmin(np.abs(times_eval - t_val))
        reverse_times = np.linspace(T, 0, n_sde_steps + 1)
        reverse_idx = np.argmin(np.abs(reverse_times - t_val))

        # Get particles
        fwd_particles = forward_path[forward_idx].cpu().numpy()
        rev_particles = reverse_path[reverse_idx].cpu().numpy()

        # Get analytical distribution
        t_tensor = torch.tensor([[t_val]], device=device)
        mu_gt, gamma_gt = bridge.get_params(t_tensor)
        mu_gt = mu_gt[0].cpu().numpy()
        cov_gt = torch.diag(gamma_gt[0]**2).cpu().numpy()

        # --- Scatter Plot (Top Row) ---
        ax_scatter.scatter(fwd_particles[:, 0], fwd_particles[:, 1], alpha=0.3, label='Fwd ODE', color='blue', s=10)
        ax_scatter.scatter(rev_particles[:, 0], rev_particles[:, 1], alpha=0.3, label='Rev SDE', color='green', s=10)
        ax_scatter.scatter(mu_gt[0], mu_gt[1], marker='x', color='red', s=100, label='Learned Mean')
        plot_confidence_ellipse(ax_scatter, mu_gt, cov_gt, n_std=2.0, edgecolor='red', linewidth=2, linestyle='--')
        ax_scatter.set_title(f't = {t_val:.2f}')
        if i == 0:
            ax_scatter.set_ylabel('z₂ (Scatter)')
        ax_scatter.grid(True, linestyle='--')
        ax_scatter.legend()
        ax_scatter.set_aspect('equal', adjustable='box')

        # --- KDE Contour Plot (Bottom Row) ---
        all_particles = np.vstack([fwd_particles, rev_particles])
        xmin, xmax = all_particles[:, 0].min(), all_particles[:, 0].max()
        ymin, ymax = all_particles[:, 1].min(), all_particles[:, 1].max()
        x_range = xmax - xmin
        y_range = ymax - ymin
        grid_x, grid_y = np.mgrid[xmin - 0.1*x_range:xmax + 0.1*x_range:100j, ymin - 0.1*y_range:ymax + 0.1*y_range:100j]
        grid_pts = np.vstack([grid_x.ravel(), grid_y.ravel()])

        try:
            kde_fwd = gaussian_kde(fwd_particles[:, :2].T)
            kde_rev = gaussian_kde(rev_particles[:, :2].T)
            density_fwd = kde_fwd(grid_pts).reshape(grid_x.shape)
            density_rev = kde_rev(grid_pts).reshape(grid_y.shape)

            # Plot contours
            ax_contour.contour(grid_x, grid_y, density_fwd, colors='blue', linestyles='--', levels=5)
            ax_contour.contour(grid_x, grid_y, density_rev, colors='green', linestyles='-', levels=5)
            
            # Create custom legend entries
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], color='blue', ls='--', label='Fwd ODE'),
                               Line2D([0], [0], color='green', ls='-', label='Rev SDE')]
            ax_contour.legend(handles=legend_elements)

        except np.linalg.LinAlgError:
            ax_contour.text(0.5, 0.5, "KDE failed (singular matrix)", ha='center', va='center', transform=ax_contour.transAxes)

        ax_contour.set_title(f'KDE Contour Comparison')
        ax_contour.set_xlabel('z₁')
        if i == 0:
            ax_contour.set_ylabel('z₂ (KDE)')
        ax_contour.grid(True, linestyle=':')
        ax_contour.set_aspect('equal', adjustable='box')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "fig3_marginal_comparison.png"), dpi=300)
    plt.close()


def _plot_marginal_data_fit(bridge, marginal_data, T, output_dir, device):
    print("  - Plotting marginal data fit...")

    # --- Plotting ---
    # Determine the number of marginal constraints
    n_marginals = len(marginal_data)
    # Sort the times for consistent plotting order
    sorted_times = sorted(marginal_data.keys())

    fig, axes = plt.subplots(2, n_marginals, figsize=(6 * n_marginals, 11), sharex=True, sharey=True)
    if n_marginals == 1: # If only one subplot, axes is not a list
        axes = np.array([[axes[0]], [axes[1]]])

    fig.suptitle("Figure 4: Qualitative Fit to Marginal Data Constraints", fontsize=16)

    for i, t_k in enumerate(sorted_times):
        ax_scatter = axes[0, i]
        ax_contour = axes[1, i]
        samples_k = marginal_data[t_k].cpu().numpy()

        # Get analytical distribution at time t_k
        t_k_tensor = torch.tensor([[t_k]], device=device)
        mu_k, gamma_k = bridge.get_params(t_k_tensor)
        mu_k_np = mu_k[0].cpu().numpy()
        cov_k = torch.diag(gamma_k[0]**2).cpu().numpy()

        # --- Scatter Plot (Top Row) ---
        ax_scatter.scatter(samples_k[:, 0], samples_k[:, 1], alpha=0.5, label='Data Samples', s=15, color='gray')
        ax_scatter.scatter(mu_k_np[0], mu_k_np[1], marker='P', color='orange', s=150, label='Learned Mean', edgecolors='black')
        plot_confidence_ellipse(ax_scatter, mu_k_np, cov_k, n_std=2.0, edgecolor='orange', linewidth=2.5, linestyle='-')
        ax_scatter.set_title(f't = {t_k:.2f}')
        if i == 0:
            ax_scatter.set_ylabel('z₂ (Scatter)')
        ax_scatter.grid(True, linestyle='--')
        ax_scatter.legend()
        ax_scatter.set_aspect('equal', adjustable='box')

        # --- KDE Contour Plot (Bottom Row) ---
        # Generate samples from the learned Gaussian for KDE comparison
        learned_dist = D.MultivariateNormal(mu_k[0].cpu(), torch.from_numpy(cov_k).float())
        learned_samples = learned_dist.sample((samples_k.shape[0],)).numpy()

        all_particles = np.vstack([samples_k, learned_samples])
        xmin, xmax = all_particles[:, 0].min(), all_particles[:, 0].max()
        ymin, ymax = all_particles[:, 1].min(), all_particles[:, 1].max()
        x_range = xmax - xmin
        y_range = ymax - ymin
        grid_x, grid_y = np.mgrid[xmin - 0.1*x_range:xmax + 0.1*x_range:100j, ymin - 0.1*y_range:ymax + 0.1*y_range:100j]
        grid_pts = np.vstack([grid_x.ravel(), grid_y.ravel()])

        try:
            kde_data = gaussian_kde(samples_k[:, :2].T)
            kde_learned = gaussian_kde(learned_samples[:, :2].T)
            density_data = kde_data(grid_pts).reshape(grid_x.shape)
            density_learned = kde_learned(grid_pts).reshape(grid_y.shape)

            # Plot contours
            ax_contour.contour(grid_x, grid_y, density_data, colors='gray', linestyles='-', levels=5)
            ax_contour.contour(grid_x, grid_y, density_learned, colors='orange', linestyles='--', levels=5)

            # Create custom legend entries
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], color='gray', ls='-', label='Data'),
                               Line2D([0], [0], color='orange', ls='--', label='Learned')]
            ax_contour.legend(handles=legend_elements)

        except np.linalg.LinAlgError:
            ax_contour.text(0.5, 0.5, "KDE failed (singular matrix)", ha='center', va='center', transform=ax_contour.transAxes)

        ax_contour.set_title(f'KDE Contour Comparison')
        ax_contour.set_xlabel('z₁')
        if i == 0:
            ax_contour.set_ylabel('z₂ (KDE)')
        ax_contour.grid(True, linestyle=':')
        ax_contour.set_aspect('equal', adjustable='box')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "fig4_marginal_fit.png"), dpi=300)
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
        print(f"Warning: Data dimension {data_dim} is not a perfect square. Cannot visualize as 2D field.")
        return

    fig, axes = plt.subplots(n_samples_to_show, n_marginals, figsize=(3 * n_marginals, 3 * n_samples_to_show))
    # Handle potential dimension issues if n_samples_to_show or n_marginals is 1
    if n_samples_to_show == 1 or n_marginals == 1:
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        if n_samples_to_show > 1:
            axes = axes.reshape(-1, 1)
        elif n_marginals > 1:
            axes = axes.reshape(1, -1)
        elif n_samples_to_show == 1 and n_marginals == 1:
            axes = axes.reshape(1, 1)
    elif axes.ndim == 1:
        axes = np.array([axes])
    
    # Determine global vmin/vmax
    all_data = torch.cat([marginal_data[t] for t in sorted_times], dim=0).cpu().numpy()
    vmin, vmax = all_data.min(), all_data.max()

    for i in range(n_samples_to_show):
        for j, t_k in enumerate(sorted_times):
            ax = axes[i, j]
            # Check if sample index exists
            if i < marginal_data[t_k].shape[0]:
                 sample = marginal_data[t_k][i].cpu().numpy().reshape(resolution, resolution)
                 ax.imshow(sample, cmap='viridis', vmin=vmin, vmax=vmax, extent=[0, 1, 0, 1])
            ax.axis('off')
            if i == 0:
                ax.set_title(f"t = {t_k:.2f}")

    plt.tight_layout()
    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

def _plot_learned_grf_evolution(bridge, T, output_dir, device):
    """Visualizes the evolution of the learned GRF distribution (mean and std dev fields)."""
    print("  - Plotting learned GRF evolution (Mean and Std)...")

    data_dim = bridge.data_dim
    resolution = int(math.sqrt(data_dim))
    if resolution * resolution != data_dim:
        print(f"Warning: Data dimension {data_dim} is not a perfect square. Cannot visualize as 2D field.")
        return

    n_steps = 5
    time_steps = torch.linspace(0, T, n_steps)
    
    # Get learned parameters (mu and gamma)
    t_tensor = time_steps.unsqueeze(-1).to(device)
    with torch.no_grad():
        mu_t, gamma_t = bridge.get_params(t_tensor)
    
    mu_t = mu_t.cpu().numpy().reshape(n_steps, resolution, resolution)
    gamma_t = gamma_t.cpu().numpy().reshape(n_steps, resolution, resolution)
    
    fig, axes = plt.subplots(2, n_steps, figsize=(3 * n_steps, 6))
    fig.suptitle("Learned GRF Evolution: Mean μ(t) and Std Dev γ(t)", fontsize=16)

    vmin_mu, vmax_mu = mu_t.min(), mu_t.max()
    vmin_gamma, vmax_gamma = gamma_t.min(), gamma_t.max()

    for i in range(n_steps):
        t_val = time_steps[i].item()
        
        # Plot Mean μ(t)
        ax_mu = axes[0, i]
        im_mu = ax_mu.imshow(mu_t[i], cmap='viridis', vmin=vmin_mu, vmax=vmax_mu, extent=[0, 1, 0, 1])
        ax_mu.set_title(f"μ(t={t_val:.2f})")
        ax_mu.axis('off')
        
        # Plot Std Dev γ(t)
        ax_gamma = axes[1, i]
        im_gamma = ax_gamma.imshow(gamma_t[i], cmap='plasma', vmin=vmin_gamma, vmax=vmax_gamma, extent=[0, 1, 0, 1])
        ax_gamma.set_title(f"γ(t={t_val:.2f})")
        ax_gamma.axis('off')

    # Add colorbars
    fig.colorbar(im_mu, ax=axes[0, :], fraction=0.046, pad=0.04, label="Mean")
    fig.colorbar(im_gamma, ax=axes[1, :], fraction=0.046, pad=0.04, label="Std Dev")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "learned_grf_evolution.png"), dpi=300)
    plt.close()


def _visualize_backward_samples_comparison(marginal_data: Dict[float, Tensor], generated_samples: Dict[float, Tensor], output_dir: str):
    """
    Visualize comparison between original marginal data and backward generated samples.
    Shows multiple samples side by side for qualitative assessment.
    """
    print("  - Plotting backward samples vs marginal data comparison...")
    
    sorted_times = sorted(marginal_data.keys())
    n_time_points = len(sorted_times)
    n_samples_show = 4  # Number of samples to show per time point
    
    # Determine resolution
    if not sorted_times:
        return
    data_dim = marginal_data[sorted_times[0]].shape[1]
    resolution = int(math.sqrt(data_dim))
    
    if resolution * resolution != data_dim:
        print(f"    Warning: Data dimension {data_dim} is not a perfect square. Skipping visualization.")
        return

    # Create figure with 2 rows (original vs generated) and multiple columns
    fig, axes = plt.subplots(2 * n_samples_show, n_time_points, 
                           figsize=(3 * n_time_points, 3 * 2 * n_samples_show))
    fig.suptitle("Backward Generated Samples vs Original Marginal Data", fontsize=16)
    
    # Handle dimension edge cases
    if axes.ndim == 1:
        axes = axes.reshape(-1, 1) if n_time_points == 1 else axes.reshape(1, -1)
    
    # Determine global color scale for consistency
    all_original = torch.cat([marginal_data[t] for t in sorted_times], dim=0).cpu().numpy()
    all_generated = torch.cat([generated_samples[t] for t in sorted_times], dim=0).cpu().numpy()
    vmin = min(all_original.min(), all_generated.min())
    vmax = max(all_original.max(), all_generated.max())
    
    for j, t_val in enumerate(sorted_times):
        original_data = marginal_data[t_val].cpu().numpy()
        generated_data = generated_samples[t_val].cpu().numpy()
        
        for i in range(n_samples_show):
            # Original data samples
            row_orig = 2 * i
            if i < original_data.shape[0]:
                sample_orig = original_data[i].reshape(resolution, resolution)
                axes[row_orig, j].imshow(sample_orig, cmap='viridis', vmin=vmin, vmax=vmax, extent=[0, 1, 0, 1])
            axes[row_orig, j].axis('off')
            if j == 0:
                axes[row_orig, j].set_ylabel(f'Original {i+1}', rotation=0, ha='right', va='center')
            if i == 0:
                axes[row_orig, j].set_title(f't = {t_val:.2f}')
            
            # Generated data samples  
            row_gen = 2 * i + 1
            if i < generated_data.shape[0]:
                sample_gen = generated_data[i].reshape(resolution, resolution)
                axes[row_gen, j].imshow(sample_gen, cmap='viridis', vmin=vmin, vmax=vmax, extent=[0, 1, 0, 1])
            axes[row_gen, j].axis('off')
            if j == 0:
                axes[row_gen, j].set_ylabel(f'Generated {i+1}', rotation=0, ha='right', va='center')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(output_dir, "backward_samples_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()


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


def _plot_learned_grf_evolution(bridge, T, output_dir, device):
    """Visualizes the evolution of the learned GRF distribution (mean and std dev fields)."""
    print("  - Plotting learned GRF evolution (Mean and Std)...")

    data_dim = bridge.data_dim
    resolution = int(math.sqrt(data_dim))
    if resolution * resolution != data_dim:
        print(f"Warning: Data dimension {data_dim} is not a perfect square. Cannot visualize as 2D field.")
        return

    n_steps = 5
    time_steps = torch.linspace(0, T, n_steps)
    
    # Get learned parameters (mu and gamma)
    t_tensor = time_steps.unsqueeze(-1).to(device)
    with torch.no_grad():
        mu_t, gamma_t = bridge.get_params(t_tensor)
    
    mu_t = mu_t.cpu().numpy().reshape(n_steps, resolution, resolution)
    gamma_t = gamma_t.cpu().numpy().reshape(n_steps, resolution, resolution)
    
    fig, axes = plt.subplots(2, n_steps, figsize=(3 * n_steps, 6))
    fig.suptitle("Learned GRF Evolution: Mean μ(t) and Std Dev γ(t)", fontsize=16)

    vmin_mu, vmax_mu = mu_t.min(), mu_t.max()
    vmin_gamma, vmax_gamma = gamma_t.min(), gamma_t.max()

    for i in range(n_steps):
        t_val = time_steps[i].item()
        
        # Plot Mean μ(t)
        ax_mu = axes[0, i]
        im_mu = ax_mu.imshow(mu_t[i], cmap='viridis', vmin=vmin_mu, vmax=vmax_mu, extent=[0, 1, 0, 1])
        ax_mu.set_title(f"μ(t={t_val:.2f})")
        ax_mu.axis('off')
        
        # Plot Std Dev γ(t)
        ax_gamma = axes[1, i]
        im_gamma = ax_gamma.imshow(gamma_t[i], cmap='plasma', vmin=vmin_gamma, vmax=vmax_gamma, extent=[0, 1, 0, 1])
        ax_gamma.set_title(f"γ(t={t_val:.2f})")
        ax_gamma.axis('off')

    # Add colorbars
    fig.colorbar(im_mu, ax=axes[0, :], fraction=0.046, pad=0.04, label="Mean")
    fig.colorbar(im_gamma, ax=axes[1, :], fraction=0.046, pad=0.04, label="Std Dev")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "learned_grf_evolution.png"), dpi=300)
    plt.close()

def _plot_data_and_trajectory(bridge, marginal_data, T, output_dir, device):
    """Plot data and mean trajectory for spiral data."""
    print("  - Plotting data and mean trajectory...")
    
    # Evaluate the learned trajectory
    # Input time tensor [N_times, 1]
    fine_times = torch.linspace(0, T, 200, device=device).unsqueeze(-1)
    mu_traj, gamma_traj = bridge.get_params(fine_times)
    mu_traj = mu_traj.cpu()
    gamma_traj = gamma_traj.cpu()

    # fig = plt.figure(figsize=(18, 6))
    fig = plt.figure(figsize=(7, 4))
    fig.suptitle("Figure 1: Learned Distributional Flow", fontsize=12)

    # 3D Trajectory and Data
    # ax1 = fig.add_subplot(131, projection='3d')
    ax1 = fig.add_subplot(111, projection='3d')

    # Plot data samples
    for t_k, samples_k in marginal_data.items():
        ax1.scatter(samples_k[:, 0], samples_k[:, 1], samples_k[:, 2], alpha=0.1, label=f'Data t={t_k:.1f}')
    
    # Plot learned mean trajectory
    ax1.plot(mu_traj[:, 0], mu_traj[:, 1], mu_traj[:, 2], 'r-', linewidth=2.5, label='Learned Mean μ(t)')
    
    ax1.set_title('A) Data Marginals and Mean Trajectory', fontsize=12)
    ax1.set_xlabel('z₁')
    ax1.set_ylabel('z₂')
    ax1.set_zlabel('z₃')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "fig1_flow_parameters.png"), dpi=300)
    plt.close()

def _plot_data_and_trajectory_grf(bridge, marginal_data, T, output_dir, device):
    """Plot data and trajectory visualization specifically for GRF data."""
    print("  - Plotting GRF data and learned parameters evolution...")
    
    # Evaluate the learned trajectory
    fine_times = torch.linspace(0, T, 100, device=device).unsqueeze(-1)
    with torch.no_grad():
        mu_traj, gamma_traj = bridge.get_params(fine_times)
    mu_traj = mu_traj.cpu()
    gamma_traj = gamma_traj.cpu()

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle("Figure 1: GRF Bridge Parameters Evolution", fontsize=14)

    # Mean field evolution over time (showing average)
    ax1 = fig.add_subplot(131)
    mean_values = torch.mean(mu_traj, dim=1)
    ax1.plot(fine_times.cpu().squeeze(), mean_values, 'b-', linewidth=2.5, label='Mean Field')
    ax1.set_title('A) Mean Field Evolution')
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Average μ(t)')
    ax1.grid(True, linestyle='--')
    ax1.legend()

    # Standard deviation evolution over time
    ax2 = fig.add_subplot(132)
    std_values = torch.mean(gamma_traj, dim=1)
    ax2.plot(fine_times.cpu().squeeze(), std_values, 'r-', linewidth=2.5, label='Std Dev Field')
    ax2.set_title('B) Standard Deviation Evolution')
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Average γ(t)')
    ax2.grid(True, linestyle='--')
    ax2.legend()
    ax2.set_ylim(bottom=0)

    # Path regularization (kinetic energy) over time
    ax3 = fig.add_subplot(133)
    # We estimate the kinetic energy at the fine_times
    _, _, dmu_dt, dgamma_dt = bridge.get_params_and_derivs(fine_times)
    dmu_dt = dmu_dt.cpu()
    dgamma_dt = dgamma_dt.cpu()

    mean_term = torch.sum(dmu_dt**2, dim=-1)
    var_term = torch.sum(dgamma_dt**2, dim=-1)
    kinetic_energy = (mean_term + var_term) / 2.0

    ax3.plot(fine_times.cpu().squeeze(), kinetic_energy, 'purple', linewidth=2.5)
    ax3.set_title('C) Path Kinetic Energy ½E[||v||²]')
    ax3.set_xlabel('Time t')
    ax3.set_ylabel('Kinetic Energy')
    ax3.grid(True, linestyle='--')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "fig1_grf_parameters.png"), dpi=300)
    plt.close()

def _plot_asymmetric_dynamics(bridge, T, n_viz_particles, n_sde_steps, output_dir, device):
    print("  - Plotting asymmetric dynamics...")
    
    # Input time tensor [N_times, 1]
    fine_times = torch.linspace(0, T, 200, device=device).unsqueeze(-1)
    mu_traj, _ = bridge.get_params(fine_times)
    mu_traj = mu_traj.cpu()

    # --- Forward ODE Simulation ---
    # Start from the learned initial distribution p_0(x)
    # Input time tensor [1, 1]
    t0_tensor = torch.tensor([[0.0]], device=device, dtype=torch.float32)
    mu0, gamma0 = bridge.get_params(t0_tensor)
    z0 = mu0 + gamma0 * torch.randn(n_viz_particles, bridge.data_dim, device=device)

    def forward_ode_func(t, z_np):
        """Wrapper for bridge.forward_velocity for solve_ivp (handles batching)."""
        B = z_np.shape[0] // bridge.data_dim
        z = torch.from_numpy(z_np).float().to(device).reshape(B, bridge.data_dim)
        # Time input t is scalar float
        t_tensor = torch.tensor([[t]], device=device, dtype=torch.float32)
        with torch.no_grad():
            # forward_velocity handles the time formatting internally
            v = bridge.forward_velocity(z, t_tensor)
        return v.cpu().numpy().flatten()

    times_eval = torch.linspace(0, T, 100).numpy()

    # Solve ODE for the batch of particles simultaneously
    z0_np = z0.cpu().numpy().flatten()
    sol = solve_ivp(
        fun=forward_ode_func,
        t_span=[0, T],
        y0=z0_np,
        method='RK45', # Adaptive solver for deterministic ODE
        t_eval=times_eval
    )
    # sol.y is shape (data_dim*B, n_times)
    forward_trajectories = torch.from_numpy(sol.y.T).float() # (n_times, data_dim*B)
    forward_trajectories = forward_trajectories.reshape(-1, n_viz_particles, bridge.data_dim).permute(1, 0, 2)


    # --- Reverse SDE Simulation ---
    # Start from the learned final distribution p_T(x)
    # Input time tensor [1, 1]
    tT_tensor = torch.tensor([[T]], device=device, dtype=torch.float32)
    muT, gammaT = bridge.get_params(tT_tensor)
    zT = muT + gammaT * torch.randn(n_viz_particles, bridge.data_dim, device=device)

    # Use the stable specialized solver for the reverse SDE
    reverse_trajectories = solve_gaussian_bridge_reverse_sde(
        bridge=bridge, z_start=zT, ts=T, tf=0.0, n_steps=n_sde_steps
    ).cpu()
    reverse_trajectories = reverse_trajectories.permute(1, 0, 2)

    # --- Plotting ---
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("Figure 2: Asymmetric Dynamics", fontsize=16)

    # Forward ODE plot
    ax_fwd = fig.add_subplot(121, projection='3d')
    for i in range(n_viz_particles):
        ax_fwd.plot(forward_trajectories[i, :, 0], forward_trajectories[i, :, 1], forward_trajectories[i, :, 2], alpha=0.5, linewidth=1)
    ax_fwd.plot(mu_traj[:, 0], mu_traj[:, 1], mu_traj[:, 2], 'r--', linewidth=2.5, label='Mean Trajectory')
    ax_fwd.set_title('A) Forward Deterministic Trajectories (ODE)')
    ax_fwd.set_xlabel('z₁')
    ax_fwd.set_ylabel('z₂')
    ax_fwd.set_zlabel('z₃')
    ax_fwd.legend()

    # Reverse SDE plot
    ax_rev = fig.add_subplot(122, projection='3d')
    for i in range(n_viz_particles):
        ax_rev.plot(reverse_trajectories[i, :, 0], reverse_trajectories[i, :, 1], reverse_trajectories[i, :, 2], alpha=0.5, linewidth=1)
    ax_rev.plot(mu_traj[:, 0], mu_traj[:, 1], mu_traj[:, 2], 'r--', linewidth=2.5, label='Mean Trajectory')
    ax_rev.set_title(f'B) Reverse Stochastic Trajectories (SDE, σ={bridge.sigma_reverse:.2f})')
    ax_rev.set_xlabel('z₁')
    ax_rev.set_ylabel('z₂')
    ax_rev.set_zlabel('z₃')
    ax_rev.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "fig2_asymmetric_dynamics.png"), dpi=300)
    plt.close()


def visualize_bridge_results(bridge: NeuralGaussianBridge, marginal_data: Dict[float, Tensor], T: float, n_viz_particles: int = 50, n_sde_steps: int = 100, output_dir: str = "output", is_grf: bool = False):
    """
    Generate and save a comprehensive set of visualizations for the trained bridge.
    Implements specific pipelines for GRF and non-GRF data.
    """
    
    print(f"\n--- Starting Visualization Pipeline (Output: {output_dir}) ---")
    bridge.eval()
    device = bridge.base_mean.device

    if not is_grf:
        # --- Non-GRF (e.g., Spiral) Visualization Pipeline ---
        print("Using visualization pipeline for non-GRF data.")
        
        # 1. Data and Trajectory
        _plot_data_and_trajectory(bridge, marginal_data, T, output_dir, device)
        
        # 2. Asymmetric Dynamics (Forward ODE vs Reverse SDE)
        # Use fewer particles for trajectory visualization
        _plot_asymmetric_dynamics(bridge, T, n_viz_particles=min(n_viz_particles, 50), n_sde_steps=n_sde_steps, output_dir=output_dir, device=device)
        
        # 3. Marginal Distribution Comparison (Consistency Check)
        _plot_marginal_distribution_comparison(bridge, T, n_viz_particles=n_viz_particles, n_sde_steps=n_sde_steps, output_dir=output_dir, device=device)

        # 4. Marginal Data Fit (Qualitative)
        _plot_marginal_data_fit(bridge, marginal_data, T, output_dir, device)

        print("Visualization pipeline complete.")
        return

    # --- GRF Specific Visualization Pipeline ---
    print("Using specialized visualization pipeline for GRF data.")

    with torch.no_grad():
        # 1. Visualize Input Data
        _plot_grf_marginals(marginal_data, output_dir, title="Input GRF Data Marginals")
        
        # 2. Visualize Learned Parameters and Evolution
        _plot_data_and_trajectory_grf(bridge, marginal_data, T, output_dir, device)
        _plot_learned_grf_evolution(bridge, T, output_dir, device)
        
        # 3. Select specific samples for consistent comparison
        n_samples_show = 6  # Number of samples to track through time
        selected_indices = torch.randperm(marginal_data[max(marginal_data.keys())].shape[0])[:n_samples_show]
        
        # 4. Generate consistent backward samples starting from the same final time samples
        print("\n4. Generating consistent backward samples for detailed comparison...")
        original_selected, generated_consistent = generate_consistent_backward_samples(
            bridge, marginal_data, selected_indices, n_steps=n_sde_steps, device=device
        )
        
        # 5. Visualize original samples evolution (separate figure)
        _visualize_original_samples(original_selected, selected_indices, output_dir)
        
        # 6. Visualize generated samples evolution (separate figure)
        _visualize_generated_samples(generated_consistent, selected_indices, output_dir)
        
        # 7. Create side-by-side comparison (combined figure)
        _visualize_consistent_samples_comparison(original_selected, generated_consistent, selected_indices, output_dir)
        
        # 8. Generate additional samples for statistical analysis
        n_samples_analysis = max(n_viz_particles, 256) 
        generated_samples = generate_backward_samples(
            bridge, marginal_data, n_samples=n_samples_analysis, n_steps=n_sde_steps, device=device
        )
        
        # 9. Statistical Comparison (First-order statistics: Mean, Std Dev)
        _visualize_marginal_statistics_comparison(marginal_data, generated_samples, output_dir)
        
        # 10. Covariance Analysis (Second-order statistics: ACF)
        _visualize_covariance_comparison(marginal_data, generated_samples, output_dir)
        
        # 11. Sample Distributions Comparison
        _visualize_sample_distributions(marginal_data, generated_samples, output_dir)

        # Note: Quantitative metrics (W2, MSE_ACF) are calculated in the main validation step (Step 5).

    print("--- GRF Visualization Pipeline Complete ---")


# ============================================================================
# Main Execution
# ============================================================================

def main(use_grf: bool = False):
    """Main execution function for the Neural Gaussian Bridge implementation."""
    # Configuration
    # Use CPU as GPU is typically not available/needed for this scale
    DEVICE = "cpu" 
    torch.manual_seed(42)
    np.random.seed(42)
    
    if use_grf:
        OUTPUT_DIR = "output_neural_bridge_grf"
        print("="*80)
        print("NEURAL ASYMMETRIC MULTI-MARGINAL BRIDGE (GRF DATA)")
        print(f"Device: {DEVICE}")
        print("="*80)
    else:
        OUTPUT_DIR = "output_neural_bridge"
        print("="*80)
        print("NEURAL ASYMMETRIC MULTI-MARGINAL BRIDGE (GAUSSIAN FLOW)")
        print(f"Device: {DEVICE}")
        print("="*80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Hyperparameters
    T_MAX = 1.0
    LEARNING_RATE = 1e-3
    EPOCHS = 500
    LAMBDA_PATH = 0.001 # Weight for path regularization
    SIGMA_REVERSE = 0.5 # Diffusion for the reverse SDE
    
    if use_grf:
        # GRF-specific parameters
        RESOLUTION = 16  # Start with smaller resolution for testing
        DATA_DIM = RESOLUTION * RESOLUTION
        N_CONSTRAINTS = 5
        N_SAMPLES = 128  # Number of GRF samples per marginal
        HIDDEN_SIZE = 256  # Larger network for high-dimensional GRF data
        
        # GRF generation parameters
        L_DOMAIN = 1.0
        MICRO_CORR_LENGTH = 0.1
        H_MAX_FACTOR = 0.5
        MEAN_VAL = 10.0
        STD_VAL = 2.0
        COVARIANCE_TYPE = "gaussian"
        
        print(f"\n1. Generating multiscale GRF data (Resolution: {RESOLUTION}x{RESOLUTION})...")
        marginal_data, data_dim = generate_multiscale_grf_data(
            N_samples=N_SAMPLES, T=T_MAX, N_constraints=N_CONSTRAINTS, resolution=RESOLUTION,
            L_domain=L_DOMAIN, micro_corr_length=MICRO_CORR_LENGTH, H_max_factor=H_MAX_FACTOR,
            mean_val=MEAN_VAL, std_val=STD_VAL, covariance_type=COVARIANCE_TYPE, device=DEVICE
        )
        
        # Move data to device
        marginal_data = {t: samples.to(DEVICE) for t, samples in marginal_data.items()}
        
        print(f"Generated {len(marginal_data)} marginal distributions with {data_dim} dimensions.")
        
    else:
        # Spiral data parameters
        DATA_DIM = 3
        N_CONSTRAINTS = 5
        N_SAMPLES_PER_MARGINAL = 512
        DATA_NOISE_STD = 0.1
        HIDDEN_SIZE = 128
        
        print("\n1. Generating synthetic distributional spiral data...")
        marginal_data, time_steps = generate_spiral_distributional_data(
            N_constraints=N_CONSTRAINTS, 
            T=T_MAX, 
            data_dim=DATA_DIM,
            N_samples_per_marginal=N_SAMPLES_PER_MARGINAL,
            noise_std=DATA_NOISE_STD
        )
        
        # Move data to device
        marginal_data = {t: samples.to(DEVICE) for t, samples in marginal_data.items()}
        data_dim = DATA_DIM
        
        print(f"Generated {len(marginal_data)} marginal distributions.")
    
    # Step 2: Initialize bridge
    print("\n2. Initializing Neural Gaussian Bridge...")
    bridge = NeuralGaussianBridge(
        data_dim=data_dim,
        hidden_size=HIDDEN_SIZE,
        T=T_MAX,
        sigma_reverse=SIGMA_REVERSE
    ).to(DEVICE)
    
    print(f"Bridge initialized with {sum(p.numel() for p in bridge.parameters())} parameters")
    
    # Step 3: Train the bridge
    print("\n3. Training the bridge (MLE + Path Regularization)...")
    loss_history = train_bridge(
        bridge, 
        marginal_data, 
        epochs=EPOCHS, 
        lr=LEARNING_RATE, 
        lambda_path=LAMBDA_PATH,
        verbose=True
    )
    
    print(f"Training complete. Final Loss: {loss_history[-1]['total']:.4f}")
    
    # Step 4: Visualize results
    print("\n4. Visualizing results...")
    visualize_bridge_results(bridge, marginal_data, T_MAX, n_viz_particles=1000, n_sde_steps=100, output_dir=OUTPUT_DIR, is_grf=use_grf)
    
    # Step 5: Validation
    print("\n5. Performing rigorous validation...")
    if not use_grf:
        # ... (Existing validation logic for spiral data using validate_asymmetric_consistency) ...
        validation_results = validate_asymmetric_consistency(
            bridge=bridge,
            T=T_MAX,
            n_particles=1024,
            n_steps=100,
            n_validation_times=5,
            device=DEVICE
        )
        
        # Print validation summary
        print("\nVALIDATION SUMMARY:")
        print("-" * 40)
        
        # Calculate means, handle potential NaNs
        if validation_results and validation_results['mean_errors']:
            mean_errors = np.nanmean(validation_results['mean_errors'])
            cov_errors = np.nanmean(validation_results['cov_errors'])
            w2_distances = np.nanmean(validation_results['wasserstein_distances'])

            print(f"Mean of mean errors: {mean_errors:.4f}")
            print(f"Mean of covariance errors: {cov_errors:.4f}")
            print(f"Mean Wasserstein-2 distance: {w2_distances:.4f}")
            
            # Validation criteria
            if mean_errors < 0.05 and cov_errors < 0.1:
                 print("\nDistributional interpolation and asymmetric consistency validated")
            else:
                print("\nValidation metrics exceeded thresholds.")

        else:
            print("Validation failed to run or produced no results.")
    else:
        # GRF Validation (Quantitative Metrics: W2 and MSE_ACF)
        print("Performing quantitative validation for GRF data.")
        
        # Generate backward samples specifically for validation (ensure sufficient samples)
        N_SAMPLES_VALIDATION = 512
        # Ensure we do not request more samples than available in the data
        max_data_samples = max(s.shape[0] for s in marginal_data.values())
        N_SAMPLES_VALIDATION = min(N_SAMPLES_VALIDATION, max_data_samples)

        generated_samples_val = generate_backward_samples(
            bridge, marginal_data, n_samples=N_SAMPLES_VALIDATION, n_steps=100, device=DEVICE
        )
        
        # Calculate metrics
        validation_metrics = _calculate_validation_metrics(marginal_data, generated_samples_val)
        
        # Print Summary
        print("\nGRF VALIDATION SUMMARY:")
        print("-" * 40)
        
        if validation_metrics and validation_metrics['w2_distances']:
            avg_w2 = np.nanmean(validation_metrics['w2_distances'])
            avg_mse_acf = np.nanmean(validation_metrics['mse_acf'])

            print(f"Average Sliced Wasserstein-2 distance: {avg_w2:.6f}")
            print(f"Average MSE of Autocorrelation Function (ACF): {avg_mse_acf:.6e}")
            
            # Define success criteria (example thresholds, may need tuning)
            W2_THRESHOLD = 0.5
            MSE_ACF_THRESHOLD = 1e-3
            
            if avg_w2 < W2_THRESHOLD and (np.isnan(avg_mse_acf) or avg_mse_acf < MSE_ACF_THRESHOLD):
                print("\nValidation SUCCESS: Distributional interpolation and covariance structure preserved.")
            else:
                print(f"\nValidation WARNING: Metrics exceeded thresholds (W2<{W2_THRESHOLD}, MSE_ACF<{MSE_ACF_THRESHOLD}).")
        else:
            print("Validation failed to run or produced no results.")

    print("\n" + "="*80)
    if use_grf:
        print("GRF BRIDGE IMPLEMENTATION RUN COMPLETE")
    else:
        print("SPIRAL BRIDGE IMPLEMENTATION RUN COMPLETE")
    print("="*80)

def run_grf_example():
    """Run the Neural Gaussian Bridge with GRF data."""
    main(use_grf=True)

def run_spiral_example():
    """Run the Neural Gaussian Bridge with spiral data."""
    main(use_grf=False)


if __name__ == "__main__":
    # Check if running in an interactive environment (like Jupyter/Colab)
    try:
        from IPython import get_ipython
        if get_ipython():
            print("Interactive mode detected. Components loaded.")
            print("To run GRF example: run_grf_example()")
            print("To run spiral example: run_spiral_example()")
            # You can optionally run main() here if desired in interactive mode
            # run_grf_example()  # Uncomment to run GRF example by default
            # run_spiral_example()  # Uncomment to run spiral example by default
        else:
            # Default: run GRF example when executed as script
            run_grf_example()
    except ImportError:
        # Not interactive or IPython not installed, run main()
        # We check if torch is available before running main, as it requires it.
        try:
            import torch
            # Default: run GRF example when executed as script
            run_grf_example()
        except ImportError:
            print("PyTorch not found. Cannot run the main execution.")

