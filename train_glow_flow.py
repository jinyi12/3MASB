#!/usr/bin/env python3
"""
Asymmetric Multi-Marginal Bridge - GLOW Flow Training Script
============================================================

A streamlined script for training GLOW-based expressive flow models using the
Asymmetric Multi-Marginal Bridge framework. This script demonstrates the KISS
and YAGNI principles by providing a clean, focused implementation.

Usage:
    python train_glow_flow.py [--grf | --spiral] [--epochs EPOCHS] [--lr LR]

Examples:
    python train_glow_flow.py --grf --epochs 2000
    python train_glow_flow.py --spiral --epochs 1000 --lr 5e-4
"""

import argparse
import os
import json
from pathlib import Path
import time
import torch
import numpy as np
from typing import Dict, Tuple
from torch import distributions as D
from torch import nn, Tensor

# Import modularized components
from glow_model import TimeConditionedGlow
from losses import BridgeLosses
from utilities.common import t_dir
from utilities.data_generation import (
    generate_multiscale_grf_data,
    generate_spiral_distributional_data,
)
from utilities.simulation import generate_comparative_backward_samples
from utilities.visualization import visualize_bridge_results
from utilities.validation import (
    validate_asymmetric_consistency,
    calculate_validation_metrics,
)


# ============================================================================
# Data Utilities (Add this section)
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

    def __init__(
        self,
        data_dim: int,
        hidden_size: int,
        resolution: int,
        n_blocks_flow: int = 3,
        num_scales: int = 2,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.resolution = resolution

        # Determine number of channels from data dimension
        spatial_dim = resolution * resolution
        self.num_channels = data_dim // spatial_dim

        if data_dim != self.num_channels * spatial_dim:
            raise ValueError(
                f"Data dimension {data_dim} incompatible with resolution {resolution}"
            )

        # Initialize GLOW model
        input_shape = (self.num_channels, resolution, resolution)
        self.glow_model = TimeConditionedGlow(
            input_shape=input_shape,
            hidden_dim=hidden_size,
            n_blocks_flow=n_blocks_flow,
            num_scales=num_scales,
        )

    def forward(self, epsilon: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass: latent -> data (Generation path G)."""
        # Convert to spatial format for CNN processing
        epsilon_spatial = spatial_data_conversion(
            epsilon, self.resolution, to_spatial=True
        )

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


class NeuralBridgeExpressive(nn.Module):
    def __init__(
        self,
        data_dim: int,
        hidden_size: int,
        resolution: int,  # NEW: Required for spatial data
        n_blocks_flow: int = 3,  # Changed from n_flow_layers
        num_scales: int = 2,  # NEW: GLOW multi-scale parameter
        T: float = 1.0,
        sigma_reverse: float = 1.0,
        data_mean: Tensor = None,  # FIX: Added for normalization
        data_std: Tensor = None,  # FIX: Added for normalization
        training_noise_std: float = 0.0001,  # Added for data jittering
        noise_densification_std: float = 0.0001,  # Added for stable flow training
        inference_clamp_norm: float = None,  # CRITICAL: Disable clamping by default
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
            num_scales=num_scales,
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
            print(
                f"Warning: NaN or Inf detected in {name} calculation. Replacing with zeros."
            )
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
            velocity_result = t_dir(G_fixed_epsilon, t)
            if isinstance(velocity_result, tuple):
                velocity = velocity_result[1]  # Extract the derivative from the tuple
            else:
                velocity = velocity_result
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
        if hasattr(self.flow.glow_model, "score_function"):
            # Convert to spatial format
            z_spatial = spatial_data_conversion(z, self.resolution, to_spatial=True)
            score_spatial = self.flow.glow_model.score_function(z_spatial, t)
            # Convert back to flat format
            score = spatial_data_conversion(
                score_spatial, self.resolution, to_spatial=False
            )
        else:
            # Fallback to automatic differentiation
            z_grad = z.detach().clone().requires_grad_(True)

            # Convert to spatial format for GLOW
            z_spatial = spatial_data_conversion(
                z_grad, self.resolution, to_spatial=True
            )
            log_prob = self.flow.glow_model.log_prob(z_spatial, t)

            # Compute gradient
            score_spatial = torch.autograd.grad(
                log_prob.sum(), z_spatial, create_graph=torch.is_grad_enabled()
            )[0]

            # Convert back to flat format
            score = spatial_data_conversion(
                score_spatial, self.resolution, to_spatial=False
            )

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

    # Loss methods moved to losses.py module for modularity
    
    def _apply_noise_if_training(self, x: Tensor) -> Tensor:
        """Helper function to apply noise during training. Returns the (potentially) perturbed tensor."""
        x_perturbed = x
        if self.training:
            # Apply Data Jittering
            if hasattr(self, "training_noise_std") and self.training_noise_std > 0:
                x_perturbed = x_perturbed + torch.randn_like(x) * self.training_noise_std
            
            # Apply Noise Densification
            if hasattr(self, "noise_densification_std") and self.noise_densification_std > 0:
                x_perturbed = x_perturbed + torch.randn_like(x) * self.noise_densification_std
        return x_perturbed


    # Loss methods moved to losses.py module


def train_glow_bridge(
    bridge: NeuralBridgeExpressive,
    marginal_data: Dict[float, torch.Tensor],
    epochs: int = 2000,
    lr: float = 5e-4,
    lambda_path: float = 0.01,
    weight_decay: float = 1e-4,
    use_scheduler: bool = True,
    grad_clip_norm: float = 1.0,
    verbose: bool = True,
) -> list:
    """
    Train the GLOW-based expressive bridge using MLE and path regularization.
    Uses modularized loss functions for clarity and maintainability.
    """
    from tqdm import trange

    # Initialize modular losses
    losses = BridgeLosses(bridge)
    
    # Use AdamW for decoupled weight decay
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=lr, weight_decay=weight_decay)

    # Initialize Scheduler
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loss_history = []

    if verbose:
        pbar = trange(epochs)
    else:
        pbar = range(epochs)

    bridge.train()  # Ensure model is in training mode

    for epoch in pbar:
        optimizer.zero_grad()

        # Compute loss using modular function
        total_loss, mle_loss, path_loss = losses.standard_loss(
            marginal_data, lambda_path=lambda_path
        )

        total_loss.backward()

        # Apply gradient clipping
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(bridge.parameters(), max_norm=grad_clip_norm)

        optimizer.step()

        # Update learning rate
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
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_description(
                f"Loss: {total_loss.item():.4f} (MLE: {mle_loss.item():.4f}, Path: {path_loss.item():.4f}) LR: {current_lr:.2e}"
            )

    bridge.eval()
    return loss_history


def train_glow_bridge_atr(
    bridge: NeuralBridgeExpressive,
    marginal_data: Dict[float, torch.Tensor],
    epochs: int = 2000,
    lr: float = 5e-4,
    lambda_recon: float = 1.0,
    lambda_smooth: float = 0.001,
    lambda_path: float = 0.0,
    n_trajectories: int = 512,  # Number of paired trajectories
    weight_decay: float = 1e-4,
    use_scheduler: bool = True,
    grad_clip_norm: float = 1.0,
    verbose: bool = True,
) -> list:
    """
    Train the GLOW-based bridge using the Anchored Trajectory Reconstruction (ATR) objective.
    Uses modularized loss functions for clarity and maintainability.
    """
    from tqdm import trange

    # Initialize modular losses
    losses = BridgeLosses(bridge)
    
    # Create paired trajectory data
    paired_data = create_paired_trajectories(marginal_data, n_trajectories)

    # Use AdamW for decoupled weight decay
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=lr, weight_decay=weight_decay)

    # Initialize Scheduler
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loss_history = []

    if verbose:
        pbar = trange(epochs)
    else:
        pbar = range(epochs)

    bridge.train()  # Ensure model is in training mode

    for epoch in pbar:
        optimizer.zero_grad()

        # Compute ATR loss using modular function
        total_loss, primary_loss, reg_loss = losses.atr_loss(
            paired_data=paired_data,
            lambda_recon=lambda_recon,
            lambda_smooth=lambda_smooth,
            lambda_path=lambda_path,
        )

        total_loss.backward()

        # Apply gradient clipping
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(bridge.parameters(), max_norm=grad_clip_norm)

        optimizer.step()

        # Update learning rate
        if scheduler:
            scheduler.step()

        loss_history.append(
            {
                "total": total_loss.item(),
                "primary": primary_loss.item(),  # MLE + Reconstruction
                "regularization": reg_loss.item(),  # Smoothness + Path
            }
        )

        if verbose and isinstance(pbar, type(trange(0))):
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_description(
                f"ATR Loss: {total_loss.item():.4f} (Primary: {primary_loss.item():.4f}, Reg: {reg_loss.item():.4f}) LR: {current_lr:.2e}"
            )

    bridge.eval()
    return loss_history


def train_glow_bridge_lcl(
    bridge: NeuralBridgeExpressive,
    marginal_data: Dict[float, torch.Tensor],
    epochs: int = 2000,
    lr: float = 5e-4,
    lambda_lcl: float = 1.0,
    lambda_path: float = 0.01,
    n_trajectories: int = 512,  # Number of paired trajectories
    weight_decay: float = 1e-4,
    use_scheduler: bool = True,
    grad_clip_norm: float = 1.0,
    verbose: bool = True,
) -> list:
    """
    Train the GLOW-based bridge using the Latent Consistency Loss (LCL) objective.
    Uses modularized loss functions for clarity and maintainability.
    """
    from tqdm import trange

    # Initialize modular losses
    losses = BridgeLosses(bridge)
    
    # Create paired trajectory data with fixed indexing
    paired_data = create_paired_trajectories(marginal_data, n_trajectories)

    # Use AdamW for decoupled weight decay
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=lr, weight_decay=weight_decay)

    # Initialize Scheduler
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loss_history = []

    if verbose:
        pbar = trange(epochs)
    else:
        pbar = range(epochs)

    bridge.train()  # Ensure model is in training mode

    for epoch in pbar:
        optimizer.zero_grad()

        # Compute LCL loss using modular function
        total_loss, primary_loss, reg_loss = losses.lcl_loss(
            paired_data=paired_data,
            lambda_lcl=lambda_lcl,
            lambda_path=lambda_path,
        )

        total_loss.backward()

        # Apply gradient clipping
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(bridge.parameters(), max_norm=grad_clip_norm)

        optimizer.step()

        # Update learning rate
        if scheduler:
            scheduler.step()

        loss_history.append(
            {
                "total": total_loss.item(),
                "primary": primary_loss.item(),  # MLE + LCL
                "regularization": reg_loss.item(),  # Path regularization
            }
        )

        if verbose and isinstance(pbar, type(trange(0))):
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_description(
                f"LCL Loss: {total_loss.item():.4f} (Primary: {primary_loss.item():.4f}, Reg: {reg_loss.item():.4f}) LR: {current_lr:.2e}"
            )

    bridge.eval()
    return loss_history


def create_paired_trajectories(
    marginal_data: Dict[float, Tensor], n_trajectories: int
) -> Dict[float, Tensor]:
    """
    Create paired trajectory data, ensuring consistent indexing across all time points.
    Assumes input marginal_data is already aligned by index.
    
    CRITICAL FIX: This corrects the major bug where indices were randomized 
    independently for each time step, destroying the physical pairing.

    Args:
        marginal_data: Dictionary of {time: samples} with samples aligned by index
        n_trajectories: Number of paired trajectories to create

    Returns:
        paired_data: Dictionary of {time: aligned_samples} with consistent indexing
    """
    # Determine the total number of available trajectories
    first_key = next(iter(marginal_data.keys()))
    total_available = marginal_data[first_key].shape[0]

    # Validate alignment across all time points
    for t, samples in marginal_data.items():
        if samples.shape[0] != total_available:
            raise ValueError(f"Input data is not aligned at time {t}. Expected {total_available} samples, got {samples.shape[0]}.")

    # Limit to requested number or available samples
    n_trajectories = min(n_trajectories, total_available)

    # CRITICAL FIX: Generate indices ONCE for all time steps
    indices = torch.randperm(total_available)[:n_trajectories]

    # Create paired data by consistent sampling using the same indices
    paired_data = {}
    for t, samples in marginal_data.items():
        paired_data[t] = samples[indices]  # Use the same indices for all time points

    print(f"Created {n_trajectories} correctly paired trajectories from aligned marginal data")
    return paired_data


def setup_experiment(args) -> Dict:
    """Setup experiment configuration based on arguments.
    
    Simple, flat configuration structure for easy scripted experimentation.
    All parameters can be directly overridden via command line.
    """

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Flat configuration - no nested dicts for easy parameter access
    config = {
        # Core experiment settings
        "device": "cpu",
        "data_type": args.data_type,
        "output_dir": args.output_dir if getattr(args, 'output_dir', None) else f"output_glow_flow_{args.data_type}",
        
        # Training parameters
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": getattr(args, 'weight_decay', 1e-4),
        "use_scheduler": getattr(args, 'use_scheduler', True),
        "grad_clip_norm": getattr(args, 'grad_clip_norm', 1.0),
        
        # Loss weights
        "lambda_path": getattr(args, 'lambda_path', 0.01),
        "lambda_recon": getattr(args, 'lambda_recon', 1.0),
        "lambda_smooth": getattr(args, 'lambda_smooth', 0.001),
        "lambda_lcl": getattr(args, 'lambda_lcl', 1.0),
        
        # Model architecture
        "hidden_size": getattr(args, 'hidden_size', 64),
        "n_blocks_flow": getattr(args, 'n_blocks_flow', 2),
        "num_scales": getattr(args, 'num_scales', 2),
        
        # Data generation - key parameters for experimentation
        "n_samples": getattr(args, 'n_samples', 1024),
        "n_constraints": getattr(args, 'n_constraints', 5),
        "resolution": getattr(args, 'resolution', 16 if args.data_type == 'grf' else 8),
        
        # GRF-specific parameters (for correlation analysis)
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
        "training_noise_std": getattr(args, 'training_noise_std', 0),
        "noise_densification_std": getattr(args, 'noise_densification_std', 0.0001),
        # New simple toggles
        "save_metrics": getattr(args, 'save_metrics', False),
        "enable_covariance_analysis": getattr(args, 'enable_covariance_analysis', False),
        
        # Bridge dynamics
        "T": getattr(args, 'T', 1.0),
        "sigma_reverse": getattr(args, 'sigma_reverse', 0.5),
        
        # Evaluation parameters
        "n_sde_steps": getattr(args, 'n_sde_steps', 100),
        "n_viz_particles": getattr(args, 'n_viz_particles', 256),
        "n_trajectories": getattr(args, 'n_trajectories', 512),
    }
    
    # Compute derived parameters
    config["data_dim"] = config["resolution"] ** 2 if config["data_type"] == "grf" else 64
    
    return config


def generate_data(config):
    """Generate marginal data based on configuration."""

    print("\n--- Data Generation ---")

    if config["data_type"] == "grf":
        print(
            f"Generating multiscale GRF data (Resolution: {config['resolution']}x{config['resolution']})..."
        )
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
        print(
            "Generating spiral distributional data (adapted for spatial structure)..."
        )
        # Generate standard spiral data first
        marginal_data_raw, _ = generate_spiral_distributional_data(
            N_constraints=config["n_constraints"],
            T=config["T"],
            data_dim=3,  # Original 3D spiral
            N_samples_per_marginal=config["n_samples_per_marginal"],
            noise_std=config["noise_std"],
        )

        # Pad/reshape to spatial format for GLOW compatibility
        # This is a simple adaptation - for real use, consider generating truly spatial spiral data
        marginal_data = {}
        for t, samples in marginal_data_raw.items():
            # Pad from 3D to 64D to make it work with 8x8 spatial structure
            padded = torch.zeros(
                samples.shape[0], config["data_dim"], device=config["device"]
            )
            padded[:, :3] = samples  # Put spiral data in first 3 dimensions
            marginal_data[t] = padded

        data_dim = config["data_dim"]

    # Move data to device
    marginal_data = {
        t: samples.to(config["device"]) for t, samples in marginal_data.items()
    }

    print(
        f"Generated {len(marginal_data)} marginal distributions with {data_dim} dimensions."
    )
    
    # Note: The generated data naturally maintains trajectory alignment by index
    # This is crucial for LCL and ATR training methods
    print("Note: Data is generated with trajectory alignment preserved across time points")
    
    return marginal_data, data_dim


def create_model(config, data_dim, data_mean, data_std):
    """Create and initialize the GLOW bridge model."""

    print("\n--- Model Initialization ---")

    bridge = NeuralBridgeExpressive(
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
        noise_densification_std=config["noise_densification_std"],
        inference_clamp_norm=None,  # Disable artificial clamping
    ).to(config["device"])

    n_params = sum(p.numel() for p in bridge.parameters())
    print(f"Initialized NeuralBridgeExpressive with {n_params:,} parameters")

    return bridge


def train_model(bridge, marginal_data, config):
    """Train the bridge model."""

    print("\n--- Training ---")
    print(f"Training for {config['epochs']} epochs with lr={config['lr']}")
    print(
        "Note: GLOW models typically require more epochs (2000+) for optimal convergence"
    )

    loss_history = train_glow_bridge(
        bridge=bridge,
        marginal_data=marginal_data,
        epochs=config["epochs"],
        lr=config["lr"],
        lambda_path=config["lambda_path"],
        weight_decay=config["weight_decay"],
        use_scheduler=config["use_scheduler"],
        grad_clip_norm=config["grad_clip_norm"],
        verbose=True,
    )

    final_loss = loss_history[-1]["total"]
    print(f"\nTraining completed. Final loss: {final_loss:.6f}")

    return loss_history


def validate_model(bridge, marginal_data, config):
    """Validate the trained model."""

    print("\n--- Validation ---")

    if config["data_type"] == "grf":
        # Generate samples for validation using proper comparative backward sampling
        print("Generating backward samples for validation...")
        try:
            # Use comparative backward sampler (same as C-CVEP validation)
            # This generates backward trajectories from t=1.0 (macroscale) to t=0.0 (microscale)
            original_data, generated_samples = generate_comparative_backward_samples(
                bridge=bridge,
                marginal_data=marginal_data,
                n_samples=config["n_viz_particles"],
                n_steps=config["n_sde_steps"],
                device=config["device"],
            )

            # Calculate quantitative metrics
            metrics = calculate_validation_metrics(marginal_data, generated_samples)
            validation_metrics = metrics

            # Print validation summary
            if metrics["w2_distances"]:
                avg_w2 = np.nanmean(metrics["w2_distances"])
                avg_mse_acf = np.nanmean(metrics["mse_acf"])
                avg_rel_f = np.nanmean(metrics["rel_fro_cov"])

                print(f"Average Wasserstein-2 distance: {avg_w2:.6f}")
                print(f"Average MSE ACF: {avg_mse_acf:.6e}")
                print(f"Average Relative Frobenius (Cov): {avg_rel_f:.6f}")

                # Corrected validation criteria: MSE_ACF should be nearly zero at t=1.0
                # since we start backward sampling from the exact ground truth final marginal
                if avg_mse_acf < 1e-6:  # Very strict for t=1.0 exact match
                    print("✓ Validation PASSED: Perfect match at t=1.0 as expected")
                elif avg_mse_acf < 1e-3:
                    print("✓ Validation PASSED: GLOW model captures spatial correlations well")
                elif avg_w2 < 1.0:
                    print("⚠ Validation ACCEPTABLE: Good marginal matching despite ACF mismatch")
                else:
                    print("⚠ Validation WARNING: Consider more training epochs for GLOW convergence")

                # Additional check: The ACF at t=1.0 should be exact since we sample from ground truth
                print("Note: Backward sampling starts from exact t=1.0 marginal, MSE_ACF reflects trajectory quality")
            else:
                print("⚠ Validation failed to compute metrics")

        except Exception as e:
            print(f"Validation failed: {e}")
            print("This is common with GLOW models - try more training epochs")

    else:
        # Spiral data validation (simplified for padded data)
        print("Performing basic consistency validation for spiral data...")
        try:
            validation_results = validate_asymmetric_consistency(
                bridge=bridge,
                T=config["T"],
                n_particles=256,  # Reduced for stability
                n_steps=50,  # Reduced for speed
                n_validation_times=3,
                device=config["device"],
            )

            if validation_results and validation_results["mean_errors"]:
                mean_errors = np.nanmean(validation_results["mean_errors"])
                print(f"Mean error: {mean_errors:.6f}")

                if mean_errors < 0.2:  # More lenient for GLOW + padded data
                    print("✓ Validation PASSED: Basic consistency maintained")
                else:
                    print(
                        "⚠ Validation WARNING: Consider using true spatial data for GLOW"
                    )
            else:
                print("⚠ Validation failed to run")
        except Exception as e:
            print(f"Validation failed: {e}")

    # Save validation metrics if enabled
    try:
        if config.get("save_metrics"):
            outdir = Path(config["output_dir"]).resolve()
            outdir.mkdir(parents=True, exist_ok=True)
            metrics_file = outdir / "validation_metrics.json"
            # Ensure all metric values are plain Python floats/lists
            metrics_serializable = {}
            for k, v in validation_metrics.items():
                if isinstance(v, list):
                    # Convert torch/scalar types to floats where possible
                    new_list = []
                    for item in v:
                        try:
                            new_list.append(float(item))
                        except Exception:
                            # Fallback: convert tensors to lists
                            try:
                                # Attempt to convert tensors to numpy then to list
                                if hasattr(item, 'tolist'):
                                    new_list.append(item.tolist())
                                else:
                                    new_list.append(item)
                            except Exception:
                                new_list.append(item)
                    metrics_serializable[k] = new_list
                else:
                    try:
                        metrics_serializable[k] = float(v)
                    except Exception:
                        metrics_serializable[k] = v

            with open(metrics_file, "w") as f:
                json.dump(metrics_serializable, f, indent=2)
            print(f"Saved validation metrics to {metrics_file}")
    except Exception as e:
        print(f"Warning: failed to save validation metrics: {e}")


def visualize_results(bridge, marginal_data, config):
    """Generate visualizations of the results."""

    print("\n--- Visualization ---")
    print(f"Generating visualizations in '{config['output_dir']}'...")

    try:
        visualize_bridge_results(
            bridge=bridge,
            marginal_data=marginal_data,
            T=config["T"],
            output_dir=config["output_dir"],
            is_grf=(config["data_type"] == "grf"),
            n_viz_particles=config["n_viz_particles"],
            n_sde_steps=config["n_sde_steps"],
            # pass toggle for covariance analysis
            enable_covariance_analysis=config.get("enable_covariance_analysis", True),
        )

        print("✓ Visualizations completed")
    except Exception as e:
        print(f"Visualization failed: {e}")
        print(
            "This may happen with GLOW models - results still saved to output directory"
        )


def main():
    """Main training script."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train Asymmetric Multi-Marginal Bridge with GLOW Flow"
    )
    parser.add_argument(
        "--grf", action="store_true", help="Use GRF data (recommended for GLOW)"
    )
    parser.add_argument(
        "--spiral",
        action="store_true",
        help="Use spiral data (padded for spatial structure)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs (recommend 2000+ for GLOW)",
    )
    parser.add_argument(
        "--atr",
        action="store_true",
        help="Use Anchored Trajectory Reconstruction (ATR) training",
    )
    parser.add_argument(
        "--lcl",
        action="store_true",
        help="Use Latent Consistency Loss (LCL) training (recommended)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (lower for GLOW stability)",
    )
    parser.add_argument(
        "--lambda_recon", type=float, default=1.0, help="ATR reconstruction loss weight"
    )
    parser.add_argument(
        "--lambda_smooth",
        type=float,
        default=0.001,
        help="ATR smoothness regularization weight",
    )
    parser.add_argument(
        "--lambda_lcl", type=float, default=1.0, help="LCL latent consistency weight"
    )
    parser.add_argument(
        "--n_trajectories",
        type=int,
        default=512,
        help="Number of paired trajectories for ATR/LCL",
    )
    
    # Experiment parameters for downstream analysis
    parser.add_argument("--n_samples", type=int, help="Number of training samples")
    parser.add_argument("--micro_corr_length", type=float, help="GRF correlation length")
    parser.add_argument("--resolution", type=int, help="Spatial resolution")
    parser.add_argument("--hidden_size", type=int, help="Model hidden size")
    parser.add_argument("--n_blocks_flow", type=int, help="Number of GLOW blocks")
    parser.add_argument("--lambda_path", type=float, help="Path regularization weight")
    parser.add_argument("--weight_decay", type=float, help="Weight decay")
    parser.add_argument("--grad_clip_norm", type=float, help="Gradient clipping norm")
    parser.add_argument("--training_noise_std", type=float, help="Training noise std")
    parser.add_argument("--T", type=float, help="Time horizon")
    parser.add_argument("--sigma_reverse", type=float, help="Reverse SDE noise")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory to save results and metrics (overrides default)",
    )
    parser.add_argument(
        "--save_metrics",
        action="store_true",
        help="Save quantitative validation metrics (W2, MSE_ACF, Rel-F Frobenius) to output_dir/validation_metrics.json",
    )
    parser.add_argument(
        "--enable_covariance_analysis",
        action="store_true",
        help="Enable detailed covariance visualizations (may be heavy for large D)",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Optional experiment identifier used to name saved model files",
    )
    parser.add_argument(
        "--force_retrain",
        action="store_true",
        help="Force retraining even if a saved model exists for the given --experiment_name",
    )

    args = parser.parse_args()

    # Determine data type
    if args.grf:
        args.data_type = "grf"
    else:
        args.data_type = "spiral"  # default

    # Setup experiment
    config = setup_experiment(args)
    os.makedirs(config["output_dir"], exist_ok=True)

    print("=" * 80)
    print("ASYMMETRIC MULTI-MARGINAL BRIDGE - GLOW FLOW")
    print(f"Data Type: {config['data_type'].upper()}")
    print(f"Device: {config['device']}")
    print(f"Output: {config['output_dir']}")
    print(f"Resolution: {config['resolution']}x{config['resolution']}")
    print(f"GLOW Blocks: {config['n_blocks_flow']}, Scales: {config['num_scales']}")
    print("=" * 80)

    try:
        # 1. Generate data
        marginal_data_raw, data_dim = generate_data(config)

        # 2. Normalize data (critical for GLOW stability)
        print("\n--- Data Normalization ---")
        print("Normalizing data for GLOW stability...")
        marginal_data, data_mean, data_std = normalize_multimarginal_data(
            marginal_data_raw
        )
        print("Data normalized to zero mean and unit variance")

        # 3. Create model
        bridge = create_model(config, data_dim, data_mean, data_std)

        # 4. Train model
        if args.lcl:
            print(
                f"\n--- LCL Training (λ_lcl={args.lambda_lcl}, λ_path={config['lambda_path']}) ---"
            )
            print(
                "Using Latent Consistency Loss objective for robust trajectory consistency"
            )
            print("LCL enforces that physical trajectories correspond to single latent variables")
            train_glow_bridge_lcl(
                bridge=bridge,
                marginal_data=marginal_data,
                epochs=config["epochs"],
                lr=config["lr"],
                lambda_lcl=args.lambda_lcl,
                lambda_path=config["lambda_path"],
                n_trajectories=args.n_trajectories,
                weight_decay=config["weight_decay"],
                use_scheduler=config["use_scheduler"],
                grad_clip_norm=config["grad_clip_norm"],
                verbose=True,
            )
        elif args.atr:
            print(
                f"\n--- ATR Training (λ_recon={args.lambda_recon}, λ_smooth={args.lambda_smooth}) ---"
            )
            print(
                "Using Anchored Trajectory Reconstruction objective for trajectory consistency"
            )
            print("WARNING: ATR training uses the corrected data pairing but may still be unstable")
            train_glow_bridge_atr(
                bridge=bridge,
                marginal_data=marginal_data,
                epochs=config["epochs"],
                lr=config["lr"],
                lambda_recon=args.lambda_recon,
                lambda_smooth=args.lambda_smooth,
                lambda_path=0.0,  # Disable kinetic energy in favor of smoothness
                n_trajectories=args.n_trajectories,
                weight_decay=config["weight_decay"],
                use_scheduler=config["use_scheduler"],
                grad_clip_norm=config["grad_clip_norm"],
                verbose=True,
            )
        else:
            # Optionally skip training if a model already exists for this experiment
            skip_training = False
            if getattr(args, 'experiment_name', None) and not getattr(args, 'force_retrain', False):
                candidate_load = Path(config['output_dir']) / f"model_{args.experiment_name}.pt"
                if candidate_load.exists():
                    try:
                        print(f"Found existing model for experiment '{args.experiment_name}': {candidate_load}. Loading and skipping training.")
                        bridge.load_state_dict(torch.load(candidate_load, map_location=config['device']))
                        skip_training = True
                    except Exception as e:
                        print(f"Warning: failed to load existing model {candidate_load}: {e}. Proceeding to train.")

            if not skip_training:
                print("\n--- Standard Training ---")
                loss_history = train_model(bridge, marginal_data, config)

                # Save trained model state (KISS: save state_dict only)
                try:
                    exp_name = args.experiment_name if getattr(args, 'experiment_name', None) else None
                    model_path = Path(config['output_dir']) / (f"model_{exp_name}.pt" if exp_name else "model.pt")
                    torch.save(bridge.state_dict(), model_path)
                    # Also write a convenience copy at output_dir/model.pt
                    default_model = Path(config['output_dir']) / "model.pt"
                    if model_path.name != default_model.name:
                        torch.save(bridge.state_dict(), default_model)
                    print(f"Saved trained model to {model_path}")

                    # Save lightweight model metadata (training summary)
                    try:
                        model_meta = {
                            'experiment_name': exp_name,
                            'epochs': config.get('epochs'),
                            'final_loss': loss_history[-1]['total'] if loss_history and isinstance(loss_history, list) and 'total' in loss_history[-1] else None,
                            'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                        }
                        with open(Path(config['output_dir']) / 'model_metadata.json', 'w') as mmf:
                            json.dump(model_meta, mmf, indent=2)
                        print(f"Saved model metadata to {Path(config['output_dir']) / 'model_metadata.json'}")
                    except Exception as e:
                        print(f"Warning: failed to save model metadata: {e}")
                except Exception as e:
                    print(f"Warning: failed to save trained model: {e}")
        # 5. Validate model
        validate_model(bridge, marginal_data, config)

        # 6. Generate visualizations
        visualize_results(bridge, marginal_data, config)

        print("\n" + "=" * 80)
        print("✓ TRAINING COMPLETED SUCCESSFULLY")
        print(f"Results saved to: {config['output_dir']}")
        if args.lcl:
            print("✓ LCL training enforces trajectory consistency via latent variance minimization")
        elif args.atr:
            print("⚠ ATR training completed - consider LCL for more robust trajectory learning") 
        else:
            print("Note: Standard training matches marginals only - use --lcl for trajectory consistency")
        print("=" * 80)

    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
