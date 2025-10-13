#!/usr/bin/env python3
"""
Asymmetric Multi-Marginal Bridge - Decoupled GLOW Flow Training Script
=========================================================================

Decoupled approach: Separate density estimation (StaticGlow) from dynamics (InvertibleNeuralFlow).
This resolves the latent consistency conflict by:
  1. G_phi (StaticGlow): Models initial density p_0 only
  2. T_theta (InvertibleNeuralFlow): Models transport map x_t = T(x_0, t)

Usage:
    python train_glow_flow_decoupled.py [--grf | --spiral] [--epochs EPOCHS] [--lr LR]

Examples:
    python train_glow_flow_decoupled.py --grf --epochs 2000
    python train_glow_flow_decoupled.py --spiral --epochs 1000 --lr 5e-4
"""

import argparse
import os
import json
import math
from pathlib import Path
import torch
import numpy as np
from typing import Dict, Tuple
from torch import nn, Tensor

# Import decoupled components
from glow_model import StaticGlow, InvertibleNeuralFlow
from utilities.common import t_dir
from utilities.data_generation import (
    generate_multiscale_grf_data,
    generate_spiral_distributional_data,
)
from utilities.visualization import visualize_bridge_results
from utilities.validation import (
    calculate_validation_metrics,
)


# ============================================================================
# Data Utilities
# ============================================================================


def normalize_multimarginal_data(
    marginal_data: Dict[float, Tensor],
) -> Tuple[Dict[float, Tensor], Tensor, Tensor]:
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


def spatial_data_conversion(
    data: Tensor, resolution: int, to_spatial: bool = True
) -> Tensor:
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
            raise ValueError(
                f"Data dimension {data_dim} incompatible with resolution {resolution}"
            )

        # Reshape to spatial format
        return data.view(batch_size, num_channels, resolution, resolution)
    else:
        # Convert from [B, C, H, W] to [B, C*H*W] for flat processing
        batch_size = data.shape[0]
        return data.view(batch_size, -1)


# ============================================================================
# Decoupled Bridge Architecture
# ============================================================================

class DecoupledBridge(nn.Module):
    """
    Decoupled bridge using separate models for density and dynamics.
    
    Components:
      - G_phi (StaticGlow): Time-independent density model for p_0
      - T_theta (InvertibleNeuralFlow): Transport map x_t = T(x_0, t)
    
    This architecture resolves the latent consistency conflict by:
      1. Training G_phi only on x_0 data (single Gaussianization)
      2. Training T_theta on paired trajectories (x_0, x_t) via supervised regression
      3. Computing p_t via pushforward: p_t(x) = p_0(T^{-1}(x,t)) |det J_{T^{-1}}|
    """
    
    def __init__(
        self,
        data_dim: int,
        hidden_size: int,
        resolution: int,
        n_blocks_flow: int = 2,
        num_scales: int = 2,
        T: float = 1.0,
        sigma_reverse: float = 0.5,
        data_mean: Tensor = None,
        data_std: Tensor = None,
        training_noise_std: float = 0.0001,
        score_clamp_norm: float = 100,
    ):
        super().__init__()
        
        self.data_dim = data_dim
        self.resolution = resolution
        self.T = T
        self.sigma_reverse = sigma_reverse
        self.training_noise_std = training_noise_std
        self.score_clamp_norm = score_clamp_norm

        # Determine spatial structure
        spatial_dim = resolution * resolution
        self.num_channels = data_dim // spatial_dim
        
        if data_dim != self.num_channels * spatial_dim:
            raise ValueError(f"Data dimension {data_dim} incompatible with resolution {resolution}")
        
        self.input_shape = (self.num_channels, resolution, resolution)
        
        # Component 1: Density Model (G_phi) - StaticGlow
        print("Initializing StaticGlow for density estimation (p_0)")
        self.density_model = StaticGlow(
            input_shape=self.input_shape,
            hidden_dim=hidden_size,
            n_blocks_flow=n_blocks_flow,
            num_scales=num_scales,
        )
        
        # Component 2: Dynamics Model (T_theta) - InvertibleNeuralFlow
        print("Initializing InvertibleNeuralFlow for dynamics (transport map)")
        self.dynamics_model = InvertibleNeuralFlow(
            input_shape=self.input_shape,
            hidden_dim=hidden_size,
            n_blocks_flow=n_blocks_flow,
            num_scales=num_scales,
        )
        
        # Normalization buffers
        if data_mean is None:
            self.register_buffer("data_mean", torch.zeros(data_dim))
        else:
            self.register_buffer("data_mean", data_mean)
        
        if data_std is None:
            self.register_buffer("data_std", torch.ones(data_dim))
        else:
            self.register_buffer("data_std", data_std)
    
    def normalize(self, x: Tensor) -> Tensor:
        """Normalize data to zero mean, unit variance."""
        return (x - self.data_mean) / self.data_std
    
    def denormalize(self, z: Tensor) -> Tensor:
        """Denormalize data back to original scale."""
        return z * self.data_std + self.data_mean
    
    def _format_time(self, t, batch_size):
        """Helper to format time tensor t to match batch size [B, 1]."""
        if not torch.is_tensor(t):
            t = torch.tensor(float(t), device=self.data_mean.device)
        
        if t.dim() == 0:
            t = t.view(1, 1).expand(batch_size, 1)
        
        if t.shape[0] == 1:
            t = t.expand(batch_size, 1)
        
        if t.dim() == 1 and t.shape[0] == batch_size:
            t = t.unsqueeze(1)
        
        if t.dim() == 2 and t.shape[0] == batch_size and t.shape[1] == 1:
            return t
        
        raise ValueError(f"Time tensor shape {t.shape} incompatible with batch size {batch_size}.")
    
    def _apply_noise_if_training(self, x: Tensor) -> Tensor:
        """Apply training noise (dequantization) if in training mode."""
        if self.training and self.training_noise_std > 0:
            noise = torch.randn_like(x) * self.training_noise_std
            return x + noise
        return x
    
    # ========================================================================
    # Core Methods for Decoupled Architecture
    # ========================================================================
    
    def sample_initial(self, n_samples: int, device: str = "cpu") -> Tensor:
        """
        Sample from initial distribution p_0 using density model G_phi.
        
        Args:
            n_samples: Number of samples to generate
            device: Device to generate samples on
        
        Returns:
            x_0: Samples from p_0 in flat format [B, D]
        """
        # Sample from base Gaussian
        epsilon = torch.randn(n_samples, self.data_dim, device=device)
        epsilon_spatial = spatial_data_conversion(epsilon, self.resolution, to_spatial=True)
        
        # Generate via G_phi
        x_0_spatial, _ = self.density_model.inverse(epsilon_spatial)  # Returns tuple (x, log_det)
        x_0_flat = spatial_data_conversion(x_0_spatial, self.resolution, to_spatial=False)
        
        return x_0_flat
    
    def transport(self, x_0: Tensor, t: Tensor) -> Tensor:
        """
        Apply transport map: x_t = T_theta(x_0, t).
        
        Args:
            x_0: Initial state [B, D]
            t: Time [B, 1] or scalar
        
        Returns:
            x_t: Transported state [B, D]
        """
        t = self._format_time(t, x_0.shape[0])
        
        # Convert to spatial
        x_0_spatial = spatial_data_conversion(x_0, self.resolution, to_spatial=True)
        
        # Apply transport map
        x_t_spatial, _ = self.dynamics_model.forward(x_0_spatial, t)
        
        # Convert back to flat
        x_t_flat = spatial_data_conversion(x_t_spatial, self.resolution, to_spatial=False)
        
        return x_t_flat
    
    def inverse_transport(self, x_t: Tensor, t: Tensor) -> Tensor:
        """
        Apply inverse transport map: x_0 = T_theta^{-1}(x_t, t).
        
        Args:
            x_t: State at time t [B, D]
            t: Time [B, 1] or scalar
        
        Returns:
            x_0: Initial state [B, D]
        """
        t = self._format_time(t, x_t.shape[0])
        
        # Convert to spatial
        x_t_spatial = spatial_data_conversion(x_t, self.resolution, to_spatial=True)
        
        # Apply inverse transport
        x_0_spatial, _ = self.dynamics_model.inverse(x_t_spatial, t)
        
        # Convert back to flat
        x_0_flat = spatial_data_conversion(x_0_spatial, self.resolution, to_spatial=False)
        
        return x_0_flat
    
    def log_prob_initial(self, x_0: Tensor) -> Tensor:
        """
        Compute log probability log p_0(x_0) using density model G_phi.
        
        Args:
            x_0: Initial state [B, D]
        
        Returns:
            log_prob: Log probability [B]
        """
        # Convert to spatial
        x_0_spatial = spatial_data_conversion(x_0, self.resolution, to_spatial=True)
        
        # Compute log prob via G_phi
        log_prob = self.density_model.log_prob(x_0_spatial)
        
        return log_prob
    
    def log_prob_pushforward(self, x_t: Tensor, t: Tensor) -> Tensor:
        """
        Compute log p_t(x_t) via pushforward formula:
        log p_t(x_t) = log p_0(T^{-1}(x_t, t)) + log |det J_{T^{-1}}(x_t, t)|
        
        Args:
            x_t: State at time t [B, D]
            t: Time [B, 1] or scalar
        
        Returns:
            log_prob: Log probability [B]
        """
        t = self._format_time(t, x_t.shape[0])
        
        # Convert to spatial
        x_t_spatial = spatial_data_conversion(x_t, self.resolution, to_spatial=True)
        
        # Inverse transport with log det jacobian
        x_0_spatial, log_det_J_inv = self.dynamics_model.inverse(x_t_spatial, t)
        
        # Compute log p_0(x_0)
        log_p_0 = self.density_model.log_prob(x_0_spatial)
        
        # Pushforward formula
        log_p_t = log_p_0 + log_det_J_inv
        
        return log_p_t
    
    def forward_velocity(self, x_t: Tensor, t: Tensor) -> Tensor:
        """
        Compute forward velocity field v(x_t, t) = ∂T/∂t|_{x_0=T^{-1}(x_t,t)}.
        
        Uses automatic differentiation to compute time derivative.
        
        Args:
            x_t: State at time t [B, D]
            t: Time [B, 1] or scalar
        
        Returns:
            velocity: Forward velocity [B, D]
        """
        t = self._format_time(t, x_t.shape[0])
        
        # Get x_0 via inverse transport
        x_0 = self.inverse_transport(x_t, t)
        x_0_detached = x_0.detach()
        
        # Define T(x_0, t) for fixed x_0
        def transport_fixed_x0(time_tensor):
            return self.transport(x_0_detached, time_tensor)
        
        # Compute ∂T/∂t using JVP
        grad_mode_before = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        try:
            velocity_result = t_dir(transport_fixed_x0, t, create_graph=torch.is_grad_enabled() and self.training)
            velocity_flat = velocity_result[1] if isinstance(velocity_result, tuple) else velocity_result
        finally:
            torch.set_grad_enabled(grad_mode_before)
        
        return velocity_flat
    
    def score_function(self, x_t: Tensor, t: Tensor) -> Tensor:
        """
        Compute score function ∇_x log p_t(x_t) via autodiff.
        REVISED: Includes robust score clamping for SDE stability.
        """
        t = self._format_time(t, x_t.shape[0])
        
        # Require gradients for x_t
        x_t_grad = x_t.detach().clone().requires_grad_(True)

        grad_mode_before = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        try:
            # Compute log_p_t using the corrected pushforward density
            log_p_t = self.log_prob_pushforward(x_t_grad, t)
            score = torch.autograd.grad(
                log_p_t.sum(), x_t_grad, create_graph=False, retain_graph=False
            )[0]
        finally:
            torch.set_grad_enabled(grad_mode_before)

        # --- FIX: Robust Score Clamping ---
        # This prevents score explosion during reverse SDE simulation.
        score = score.detach()
        
        # 1. Handle NaN/Inf
        if torch.isnan(score).any() or torch.isinf(score).any():
            print("Warning: NaN/Inf detected in score function. Replacing with zeros.")
            score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. L2 Norm Clamping
        if self.score_clamp_norm is not None and self.score_clamp_norm > 0:
            max_norm = self.score_clamp_norm
            
            score_norm = torch.norm(score.view(score.shape[0], -1), p=2, dim=1)
            
            # Create mask and view shapes for broadcasting
            norm_view_shape = (-1,) + tuple([1] * (score.dim() - 1))
            mask = (score_norm > max_norm).view(norm_view_shape)
            
            # Clamp the score
            clamped_score = score * (max_norm / (score_norm.view(norm_view_shape) + 1e-6))
            
            # Apply clamping only where needed
            score = torch.where(mask, clamped_score, score)

        return score
    
    def reverse_drift(self, x_t: Tensor, t: Tensor) -> Tensor:
        """
        Compute reverse SDE drift:
        R(x_t, t) = v(x_t, t) - (σ²/2) ∇_x log p_t(x_t)
        
        Args:
            x_t: State at time t [B, D]
            t: Time [B, 1] or scalar
        
        Returns:
            drift: Reverse drift [B, D]
        """
        velocity = self.forward_velocity(x_t, t)
        score = self.score_function(x_t, t)
        
        drift = velocity - (self.sigma_reverse**2 / 2) * score
        
        return drift
    
    # ========================================================================
    # Compatibility Methods for Visualization
    # ========================================================================
    
    @property
    def base_dist(self):
        """Base distribution compatibility for visualization."""
        return self.density_model.base_dist
    
    @property
    def flow(self):
        """Flow compatibility for visualization - return dynamics model."""
        return self.dynamics_model
    
    def get_params(self, t: Tensor):
        """Get distribution parameters for visualization compatibility.
        For decoupled bridge, we return simple Gaussian parameters.
        """
        batch_size = t.shape[0] if t.dim() > 0 else 1
        device = t.device if torch.is_tensor(t) else self.data_mean.device
        
        # Return simple Gaussian parameters centered at data mean
        mu = self.data_mean.unsqueeze(0).expand(batch_size, -1).to(device)
        gamma = torch.ones_like(mu)  # Unit variance
        
        return mu, gamma


# ============================================================================
# Training Functions for Decoupled Architecture
# ============================================================================

def train_density_model(
    bridge: DecoupledBridge,
    initial_data: Tensor,
    epochs: int = 1000,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    use_scheduler: bool = True,
    grad_clip_norm: float = 1.0,
    verbose: bool = True,
) -> list:
    """
    Train density model G_phi on initial distribution p_0 only.
    
    Args:
        bridge: DecoupledBridge instance
        initial_data: Initial state samples x_0 [N, D]
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay
        use_scheduler: Use cosine annealing scheduler
        grad_clip_norm: Gradient clipping norm
        verbose: Print progress
    
    Returns:
        loss_history: Training loss history
    """
    from tqdm import trange
    
    # Optimizer for density model only
    optimizer = torch.optim.AdamW(
        bridge.density_model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay
    )
    
    # Scheduler
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    loss_history = []
    
    if verbose:
        pbar = trange(epochs)
    else:
        pbar = range(epochs)
    
    bridge.density_model.train()
    
    for epoch in pbar:
        optimizer.zero_grad()
        
        # Apply training noise (dequantization)
        x_0_noisy = bridge._apply_noise_if_training(initial_data)
        
        # Negative log likelihood
        log_prob = bridge.log_prob_initial(x_0_noisy)
        loss = -log_prob.mean()
        
        loss.backward()
        
        # Gradient clipping
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(bridge.density_model.parameters(), grad_clip_norm)
        
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        loss_history.append(loss.item())
        
        if verbose and isinstance(pbar, type(trange(0))):
            pbar.set_postfix({"density_loss": f"{loss.item():.6f}"})
    
    bridge.density_model.eval()
    return loss_history


def train_dynamics_model(
    bridge: DecoupledBridge,
    marginal_data: Dict[float, Tensor],
    epochs: int = 1000,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_path: float = 0.01, # FIX: Added path regularization weight
    use_scheduler: bool = True,
    grad_clip_norm: float = 1.0,
    batch_size: int = 128,  # Batch size for memory management
    verbose: bool = True,
) -> list:
    """
    Train dynamics model T_theta on paired trajectories.
    
    REVISED: Includes Path Regularization (Kinetic Energy) to stabilize derivatives.
    Loss: L = L_map + lambda_path * L_path
    L_map = E[ sum_k ||T(x_0, t_k) - x_{t_k}||^2 ]
    L_path = E[ sum_k ||∂T/∂t||^2 ]
    """
    from tqdm import trange
    
    # Optimizer for dynamics model only
    optimizer = torch.optim.AdamW(
        bridge.dynamics_model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Scheduler
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    loss_history = []
    
    if verbose:
        pbar = trange(epochs)
    else:
        pbar = range(epochs)
    
    bridge.dynamics_model.train()
    
    # Get initial data
    x_0 = marginal_data[0.0]
    times = sorted([t for t in marginal_data.keys() if t > 0])
    n_samples = x_0.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for epoch in pbar:
        epoch_loss = 0.0
        epoch_regression = 0.0
        epoch_path_reg = 0.0
        
        # Process in batches with gradient accumulation
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            optimizer.zero_grad()
            
            total_loss = 0.0
            regression_loss = 0.0
            path_reg_loss = 0.0
            
            # Get batch data
            x_0_batch = x_0[start_idx:end_idx]
            
            # --- Supervised Regression Loss (L_map) and Path Regularization (L_path) ---
            for t_val in times:
                x_target_batch = marginal_data[t_val][start_idx:end_idx]
                t_tensor = torch.tensor([t_val], device=x_0_batch.device, dtype=x_0_batch.dtype)
                
                if lambda_path > 0:
                    # Calculate T(x0, t) and dT/dt simultaneously using t_dir (JVP).
                    
                    # Define T(t) for fixed x_0
                    def transport_fixed_x0(time_tensor):
                        return bridge.transport(x_0_batch, time_tensor)

                    # Format time tensor correctly for the bridge helper
                    formatted_t = bridge._format_time(t_tensor, x_0_batch.shape[0])

                    # Compute prediction and velocity
                    # t_dir returns (f(t), f'(t))
                    x_pred, velocity = t_dir(transport_fixed_x0, formatted_t, create_graph=True)

                    # Path regularization loss (Kinetic energy)
                    path_loss_t = 0.5 * torch.mean(velocity**2)
                    path_reg_loss += path_loss_t
                    
                else:
                     # Standard MSE only
                    x_pred = bridge.transport(x_0_batch, t_tensor)
                
                # MSE loss
                loss_t = torch.mean((x_pred - x_target_batch) ** 2)
                regression_loss += loss_t
            
            # Average over time points
            regression_loss = regression_loss / len(times)
            total_loss += regression_loss
            
            if lambda_path > 0:
                path_reg_loss = path_reg_loss / len(times)
                total_loss += lambda_path * path_reg_loss

            # --- Training Step ---
            total_loss.backward()
            
            # Gradient clipping
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(bridge.dynamics_model.parameters(), grad_clip_norm)
            
            optimizer.step()
            
            # Accumulate losses
            epoch_loss += total_loss.item()
            epoch_regression += regression_loss.item()
            if lambda_path > 0:
                epoch_path_reg += path_reg_loss.item()
            
            # Memory cleanup
            del x_pred, total_loss, regression_loss
            if lambda_path > 0:
                del velocity, path_loss_t, path_reg_loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Average over batches
        epoch_loss /= n_batches
        epoch_regression /= n_batches
        epoch_path_reg /= n_batches
        
        if scheduler:
            scheduler.step()
        
        loss_history.append(epoch_loss)
        
        if verbose and isinstance(pbar, type(trange(0))):
            # Update postfix with detailed loss components
            pbar.set_postfix({
                "Map": f"{epoch_regression:.4f}",
                "Path": f"{epoch_path_reg:.4f}" if lambda_path > 0 else "0.0000",
                "Total": f"{epoch_loss:.4f}"
            })
    
    bridge.dynamics_model.eval()
    return loss_history



def train_decoupled_bridge(
    bridge: DecoupledBridge,
    marginal_data: Dict[float, Tensor],
    density_epochs: int = 1000,
    dynamics_epochs: int = 1000,
    density_lr: float = 1e-3,
    dynamics_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    use_scheduler: bool = True,
    grad_clip_norm: float = 1.0,
    training_batch_size: int = 128,  # Batch size for dynamics training
    verbose: bool = True,
) -> Dict[str, list]:
    """
    Train both components of the decoupled bridge sequentially.
    
    Step 1: Train G_phi (density model) on x_0
    Step 2: Train T_theta (dynamics model) on trajectories
    
    Args:
        bridge: DecoupledBridge instance
        marginal_data: Dictionary {time: samples}, trajectory-aligned
        density_epochs: Epochs for density model
        dynamics_epochs: Epochs for dynamics model
        density_lr: Learning rate for density model
        dynamics_lr: Learning rate for dynamics model
        weight_decay: Weight decay
        use_scheduler: Use cosine annealing scheduler
        grad_clip_norm: Gradient clipping norm
        verbose: Print progress
    
    Returns:
        history: Dictionary with both loss histories
    """
    print("\n" + "="*80)
    print("DECOUPLED TRAINING: Step 1 - Density Model (G_phi)")
    print("="*80)
    
    # Step 1: Train density model on x_0
    density_history = train_density_model(
        bridge=bridge,
        initial_data=marginal_data[0.0],
        epochs=density_epochs,
        lr=density_lr,
        weight_decay=weight_decay,
        use_scheduler=use_scheduler,
        grad_clip_norm=grad_clip_norm,
        verbose=verbose,
    )
    
    print(f"\nDensity model training completed. Final loss: {density_history[-1]:.6f}")
    
    print("\n" + "="*80)
    print("DECOUPLED TRAINING: Step 2 - Dynamics Model (T_theta)")
    print("="*80)
    
    # Step 2: Train dynamics model on trajectories
    dynamics_history = train_dynamics_model(
        bridge=bridge,
        marginal_data=marginal_data,
        epochs=dynamics_epochs,
        lr=dynamics_lr,
        weight_decay=weight_decay,
        lambda_path=0,  # Path regularization weight
        use_scheduler=use_scheduler,
        grad_clip_norm=grad_clip_norm,
        batch_size=training_batch_size,
        verbose=verbose,
    )
    
    print(f"\nDynamics model training completed. Final loss: {dynamics_history[-1]:.6f}")
    
    return {
        "density": density_history,
        "dynamics": dynamics_history,
    }


# ============================================================================
# Experiment Setup and Execution
# ============================================================================

def setup_experiment(args) -> Dict:
    """Setup experiment configuration based on arguments."""
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Flat configuration
    config = {
        # Core experiment settings
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "data_type": args.data_type,
        "output_dir": args.output_dir if getattr(args, 'output_dir', None) else f"output_glow_flow_decoupled_{args.data_type}",
        
        # Training parameters - Decoupled
        "density_epochs": getattr(args, 'density_epochs', args.epochs // 2),
        "dynamics_epochs": getattr(args, 'dynamics_epochs', args.epochs // 2),
        "density_lr": getattr(args, 'density_lr', args.lr),
        "dynamics_lr": getattr(args, 'dynamics_lr', args.lr),
        "weight_decay": getattr(args, 'weight_decay', 1e-4),
        "lambda_path": getattr(args, 'lambda_path', 0.01),
        "use_scheduler": getattr(args, 'use_scheduler', True),
        "grad_clip_norm": getattr(args, 'grad_clip_norm', 1.0),
        
        # Model architecture
        "hidden_size": getattr(args, 'hidden_size', 64),
        "n_blocks_flow": getattr(args, 'n_blocks_flow', 2),
        "num_scales": getattr(args, 'num_scales', 2),
        
        # Data generation
        "n_samples": getattr(args, 'n_samples', 1024),
        "n_constraints": getattr(args, 'n_constraints', 5),
        "resolution": getattr(args, 'resolution', 16 if args.data_type == 'grf' else 8),
        
        # GRF-specific parameters
        "micro_corr_length": getattr(args, 'micro_corr_length', 0.1),
        "l_domain": getattr(args, 'l_domain', 1.0),
        "h_max_factor": getattr(args, 'h_max_factor', 0.5),
        "mean_val": getattr(args, 'mean_val', 10.0),
        "std_val": getattr(args, 'std_val', 2.0),
        "covariance_type": getattr(args, 'covariance_type', 'gaussian'),
        
        # Spiral-specific parameters
        "n_samples_per_marginal": getattr(args, 'n_samples_per_marginal', 512),
        "noise_std": getattr(args, 'noise_std', 0.1),
        
        # Training stability
        "training_noise_std": getattr(args, 'training_noise_std', 0.0001),
        
        # Toggles
        "save_metrics": getattr(args, 'save_metrics', False),
        "enable_covariance_analysis": getattr(args, 'enable_covariance_analysis', False),
        
        # Bridge dynamics
        "T": getattr(args, 'T', 1.0),
        "sigma_reverse": getattr(args, 'sigma_reverse', 0.5),
        
        # Evaluation parameters
        "n_sde_steps": getattr(args, 'n_sde_steps', 100),
        "n_viz_particles": getattr(args, 'n_viz_particles', 256),
        "n_validation_trajectories": getattr(args, 'n_validation_trajectories', 512),
        "validation_batch_size": getattr(args, 'validation_batch_size', None),
        "training_batch_size": getattr(args, 'training_batch_size', 64),  # Conservative for 64x64 resolution
    }
    
    # Compute derived parameters
    config["data_dim"] = config["resolution"] ** 2 if config["data_type"] == "grf" else 64
    if config["validation_batch_size"] is None:
        config["validation_batch_size"] = min(128, config["n_validation_trajectories"])
    
    # Scale training batch size based on resolution (memory usage scales with resolution^2)
    if config["training_batch_size"] is None or config["training_batch_size"] == 64:
        if config["resolution"] >= 64:
            config["training_batch_size"] = 32  # Very conservative for 64x64
        elif config["resolution"] >= 32:
            config["training_batch_size"] = 64
        else:
            config["training_batch_size"] = 128
    
    return config


def generate_data(config):
    """Generate marginal data based on configuration."""
    
    print("\n--- Data Generation ---")
    
    if config["data_type"] == "grf":
        print(f"Generating multiscale GRF data (Resolution: {config['resolution']}x{config['resolution']})...")
        marginal_data, data_dim = generate_multiscale_grf_data(
            N_samples=config["n_samples"],
            T=config["T"],
            N_constraints=config["n_constraints"],
            resolution=config["resolution"],
            L_domain=config["l_domain"],
            micro_corr_length=config["micro_corr_length"],
            H_max_factor=config["h_max_factor"],
            mean_val=config["mean_val"],
            std_val=config["std_val"],
            covariance_type=config["covariance_type"],
            device=config["device"],
        )
    else:
        print("Generating spiral distributional data...")
        marginal_data, data_dim = generate_spiral_distributional_data(
            N_samples=config["n_samples_per_marginal"],
            T=config["T"],
            N_constraints=config["n_constraints"],
            noise_std=config["noise_std"],
            device=config["device"],
        )
        
        # Pad spiral data for spatial structure
        # (Implementation from original if needed)
    
    # Move data to device
    marginal_data = {
        t: samples.to(config["device"]) for t, samples in marginal_data.items()
    }
    
    print(f"Generated {len(marginal_data)} marginal distributions with {data_dim} dimensions.")
    print("Note: Data is generated with trajectory alignment preserved across time points")
    
    return marginal_data, data_dim


def create_model(config, data_dim, data_mean, data_std):
    """Create and initialize the decoupled bridge model."""
    
    print("\n--- Model Initialization ---")
    
    bridge = DecoupledBridge(
        data_dim=data_dim,
        hidden_size=config["hidden_size"],
        resolution=config["resolution"],
        n_blocks_flow=config["n_blocks_flow"],
        num_scales=config["num_scales"],
        T=config["T"],
        sigma_reverse=config["sigma_reverse"],
        data_mean=data_mean.to(config["device"]),
        data_std=data_std.to(config["device"]),
        training_noise_std=config["training_noise_std"],
    ).to(config["device"])
    
    n_params_density = sum(p.numel() for p in bridge.density_model.parameters())
    n_params_dynamics = sum(p.numel() for p in bridge.dynamics_model.parameters())
    n_params_total = n_params_density + n_params_dynamics
    
    print(f"Density model (G_phi): {n_params_density:,} parameters")
    print(f"Dynamics model (T_theta): {n_params_dynamics:,} parameters")
    print(f"Total parameters: {n_params_total:,}")
    
    return bridge


def train_model(bridge, marginal_data, config):
    """Train the decoupled bridge model."""
    
    print("\n--- Training Decoupled Bridge ---")
    print(f"Density epochs: {config['density_epochs']}, Dynamics epochs: {config['dynamics_epochs']}")
    print(f"Density LR: {config['density_lr']}, Dynamics LR: {config['dynamics_lr']}")
    
    history = train_decoupled_bridge(
        bridge=bridge,
        marginal_data=marginal_data,
        density_epochs=config["density_epochs"],
        dynamics_epochs=config["dynamics_epochs"],
        density_lr=config["density_lr"],
        dynamics_lr=config["dynamics_lr"],
        weight_decay=config["weight_decay"],
        use_scheduler=config["use_scheduler"],
        grad_clip_norm=config["grad_clip_norm"],
        training_batch_size=config["training_batch_size"],
        verbose=True,
    )
    
    print("\n" + "="*80)
    print("Training completed!")
    print(f"Final density loss: {history['density'][-1]:.6f}")
    print(f"Final dynamics loss: {history['dynamics'][-1]:.6f}")
    print("="*80)

    bridge.eval()
    
    return history


def validate_model(bridge, marginal_data, config):
    """Validate the trained decoupled bridge model."""
    
    print("\n--- Validation ---")
    
    if config["data_type"] == "grf":
        # Generate backward samples using reverse SDE
        print("Generating comparative backward samples...")
        
        # Get final distribution samples
        x_T = marginal_data[config["T"]]
        
        # Sample backward trajectories
        backward_samples = {}
        times = sorted(marginal_data.keys())
        
        # Reverse SDE integration (batched to control memory)
        total_samples = x_T.shape[0]
        batch_size = config.get("validation_batch_size", None)
        if batch_size is None:
            batch_size = min(128, total_samples)
        batch_size = max(1, min(batch_size, total_samples))

        dt = -config["T"] / config["n_sde_steps"]
        sqrt_dt = math.sqrt(abs(dt))

        # Accumulate per-time batches then concatenate
        backward_accumulator = {t_val: [] for t_val in times}

        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            z = x_T[start:end].clone()

            for step in range(config["n_sde_steps"] + 1):
                t_current = config["T"] + step * dt

                # Store samples at constraint times
                for t_constraint in times:
                    if abs(t_current - t_constraint) < abs(dt) / 2:
                        backward_accumulator[t_constraint].append(z.clone().cpu())  # Move to CPU

                if step < config["n_sde_steps"]:
                    # Reverse SDE step
                    t_tensor = torch.tensor([t_current], device=z.device, dtype=z.dtype)
                    drift = bridge.reverse_drift(z, t_tensor)
                    noise = torch.randn_like(z)
                    diffusion = bridge.sigma_reverse * noise * sqrt_dt
                    z = z + drift * dt + diffusion
                    
                    # Memory cleanup every 10 steps
                    if step % 10 == 0:
                        del drift, noise, diffusion
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        backward_samples = {}
        feature_shape = x_T.shape[1:]
        for t_val, chunks in backward_accumulator.items():
            if chunks:
                backward_samples[t_val] = torch.cat(chunks, dim=0).to(x_T.device)
            else:
                backward_samples[t_val] = torch.empty(
                    0, *feature_shape, device=x_T.device, dtype=x_T.dtype
                )
        
        # Validation metrics
        validation_metrics = calculate_validation_metrics(
            marginal_data=marginal_data,
            generated_samples=backward_samples,
        )
        
        # Save metrics if requested
        if config["save_metrics"]:
            output_dir = Path(config["output_dir"])
            output_dir.mkdir(exist_ok=True)
            metrics_file = output_dir / "validation_metrics.json"
            
            with open(metrics_file, "w") as f:
                json.dump(validation_metrics, f, indent=2)
            
            print(f"Validation metrics saved to {metrics_file}")
        
        # Print summary
        print("\nValidation Results:")
        for t_val in times:
            w2 = validation_metrics.get("w2_distances", [None])[times.index(t_val)]
            if w2 is not None:
                print(f"  t={t_val:.2f}: W2 distance = {w2:.6f}")
        
        # Return both metrics and samples for reuse in visualization
        return validation_metrics, backward_samples
    
    # For non-GRF data, return None for both
    return None, None


def visualize_results(bridge, marginal_data, config, validation_samples=None):
    """Generate visualizations of the results.
    
    Args:
        validation_samples: Optional dict of pre-generated backward samples from validation.
                          If provided, these will be reused for statistical visualizations,
                          avoiding duplicate SDE integration.
    """
    
    print("\n--- Visualization ---")
    if validation_samples is not None:
        print("Reusing validation samples for visualization (saves ~50% computation)")
    print(f"Generating visualizations in '{config['output_dir']}'...")
    
    try:
        visualize_bridge_results(
            bridge=bridge,
            marginal_data=marginal_data,
            T=config["T"],
            output_dir=config["output_dir"],
            is_grf=(config["data_type"] == "grf"),  # Pass is_grf parameter
            n_viz_particles=config["n_viz_particles"],
            n_sde_steps=config["n_sde_steps"],
            enable_covariance_analysis=config["enable_covariance_analysis"],
            validation_samples=validation_samples,  # Pass samples for reuse
        )
        print("✓ Visualizations generated successfully")
    except Exception as e:
        print(f"Warning: Visualization failed: {e}")


def main():
    """Main training script."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train Decoupled Asymmetric Multi-Marginal Bridge with GLOW Flow"
    )
    parser.add_argument("--grf", action="store_true", help="Use GRF data")
    parser.add_argument("--spiral", action="store_true", help="Use spiral data")
    parser.add_argument("--epochs", type=int, default=2000, help="Total training epochs (split between density and dynamics)")
    parser.add_argument("--density_epochs", type=int, default=None, help="Epochs for density model (default: epochs//2)")
    parser.add_argument("--dynamics_epochs", type=int, default=None, help="Epochs for dynamics model (default: epochs//2)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--density_lr", type=float, default=None, help="Density model LR (default: lr)")
    parser.add_argument("--dynamics_lr", type=float, default=None, help="Dynamics model LR (default: lr)")
    
    # Training parameters
    parser.add_argument("--n_trajectories", type=int, help="Number of training trajectories")
    parser.add_argument("--n_validation_trajectories", type=int, help="Number of validation trajectories")
    parser.add_argument("--lambda_path", type=float, help="Path regularization weight")
    
    # Model architecture
    parser.add_argument("--n_samples", type=int, help="Number of training samples")
    parser.add_argument("--experiment_name", type=str, help="Experiment name for saving models")
    parser.add_argument("--micro_corr_length", type=float, help="GRF correlation length")
    parser.add_argument("--covariance_type", type=str, help="GRF kernel type")
    parser.add_argument("--resolution", type=int, help="Spatial resolution")
    parser.add_argument("--l_domain", type=float, help="GRF domain size")
    parser.add_argument("--h_max_factor", type=float, help="GRF maximum factor")
    parser.add_argument("--mean_val", type=float, help="GRF mean value")
    parser.add_argument("--std_val", type=float, help="GRF standard deviation")
    parser.add_argument("--hidden_size", type=int, help="Model hidden size")
    parser.add_argument("--n_blocks_flow", type=int, help="Number of GLOW blocks")
    parser.add_argument("--weight_decay", type=float, help="Weight decay")
    parser.add_argument("--grad_clip_norm", type=float, help="Gradient clipping norm")
    parser.add_argument("--training_noise_std", type=float, help="Training noise std (dequantization)")
    parser.add_argument("--T", type=float, help="Time horizon")
    parser.add_argument("--sigma_reverse", type=float, help="Reverse SDE noise")
    
    # Output and evaluation
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--save_metrics", action="store_true", help="Save validation metrics")
    parser.add_argument("--enable_covariance_analysis", action="store_true", help="Enable covariance analysis")
    parser.add_argument("--n_viz_particles", type=int, help="Number of visualization particles")
    parser.add_argument("--n_sde_steps", type=int, help="SDE integration steps")
    parser.add_argument("--validation_batch_size", type=int, help="Reverse SDE validation batch size (controls memory usage)")
    parser.add_argument("--training_batch_size", type=int, help="Training batch size for dynamics model (controls memory usage)")
    
    args = parser.parse_args()
    
    # Determine data type
    if args.grf:
        args.data_type = "grf"
    else:
        args.data_type = "spiral"
    
    # Setup experiment
    config = setup_experiment(args)
    os.makedirs(config["output_dir"], exist_ok=True)
    
    print("=" * 80)
    print("DECOUPLED ASYMMETRIC MULTI-MARGINAL BRIDGE - GLOW FLOW")
    print("=" * 80)
    print(f"Data Type: {config['data_type'].upper()}")
    print(f"Device: {config['device']}")
    print(f"Output: {config['output_dir']}")
    print(f"Resolution: {config['resolution']}x{config['resolution']}")
    print(f"GLOW Blocks: {config['n_blocks_flow']}, Scales: {config['num_scales']}")
    print("Architecture: Decoupled (G_phi + T_theta)")
    print("=" * 80)
    
    try:
        # Generate data
        marginal_data, data_dim = generate_data(config)
        
        # Normalize data
        normalized_data, data_mean, data_std = normalize_multimarginal_data(marginal_data)
        
        # Create model
        bridge = create_model(config, data_dim, data_mean, data_std)
        
        # Train model
        history = train_model(bridge, normalized_data, config)
        
        # Save training history
        history_file = Path(config["output_dir"]) / "training_history.json"
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)
        print(f"\nTraining history saved to {history_file}")
        
        # Validate model (returns metrics and samples)
        validation_metrics, validation_samples = validate_model(bridge, normalized_data, config)
        
        # Visualize results (reuse validation samples to avoid duplicate computation)
        visualize_results(bridge, normalized_data, config, validation_samples=validation_samples)
        
        print("\n" + "=" * 80)
        print("✓ Decoupled training completed successfully!")
        print(f"Results saved to: {config['output_dir']}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
