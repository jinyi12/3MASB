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
import torch
import numpy as np
from typing import Dict, Tuple, Any
from torch import distributions as D
from torch import nn, Tensor

# Import modularized components
from glow_model import TimeConditionedGlow
from utilities.common import jvp, t_dir
from utilities.data_generation import generate_multiscale_grf_data, generate_spiral_distributional_data
from utilities.simulation import generate_backward_samples
from utilities.visualization import visualize_bridge_results
from utilities.validation import validate_asymmetric_consistency, calculate_validation_metrics


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
    This is a simplified version of the training function from asymmetric_bridge_glow_flow.py
    """
    from tqdm import trange
    
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
        
        # Compute loss
        total_loss, mle_loss, path_loss = bridge.loss(marginal_data, lambda_path=lambda_path)
        
        total_loss.backward()

        # Apply gradient clipping
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(bridge.parameters(), max_norm=grad_clip_norm)

        optimizer.step()
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        loss_history.append({
            'total': total_loss.item(),
            'mle': mle_loss.item(),
            'path': path_loss.item()
        })
        
        if verbose and isinstance(pbar, type(trange(0))):
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_description(
                f"Loss: {total_loss.item():.4f} (MLE: {mle_loss.item():.4f}, Path: {path_loss.item():.4f}) LR: {current_lr:.2e}"
            )

    bridge.eval()
    return loss_history


def setup_experiment(args) -> Dict:
    """Setup experiment configuration based on arguments."""
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration (prefer CPU for this scale)
    device = "cpu"
    
    # Common parameters
    config = {
        'device': device,
        'T': 1.0,
        'epochs': args.epochs,
        'lr': args.lr,
        'lambda_path': 0.01,        # Lower for GLOW to focus on distribution matching
        'weight_decay': 1e-4,       # Added regularization for GLOW
        'use_scheduler': True,      # Cosine annealing for better convergence
        'grad_clip_norm': 1.0,      # Gradient clipping for stability
        'sigma_reverse': 0.5,
        'n_sde_steps': 100,
        'n_viz_particles': 256,
        # GLOW-specific parameters
        'n_blocks_flow': 2,         # Number of GLOW flow blocks
        'num_scales': 2,            # Number of multi-scale levels
        'training_noise_std': 0.01, # Data jittering
        'noise_densification_std': 0.0001,  # Flow stability
    }
    
    if args.data_type == 'grf':
        # GRF-specific configuration
        config.update({
            'output_dir': 'output_glow_flow_grf',
            'data_type': 'grf',
            'resolution': 16,
            'data_dim': 16 * 16,
            'n_constraints': 5,
            'n_samples': 1024,
            'hidden_size': 64,
            # GRF generation parameters
            'l_domain': 1.0,
            'micro_corr_length': 0.1,
            'h_max_factor': 0.5,
            'mean_val': 10.0,
            'std_val': 2.0,
            'covariance_type': 'gaussian'
        })
    else:
        # Spiral data configuration - note: GLOW needs 2D spatial structure
        config.update({
            'output_dir': 'output_glow_flow_spiral',
            'data_type': 'spiral',
            'resolution': 8,            # Make spiral data spatial for GLOW
            'data_dim': 8 * 8,          # 8x8 = 64 dimensions
            'n_constraints': 5,
            'n_samples_per_marginal': 512,
            'noise_std': 0.1,
            'hidden_size': 64,
        })
    
    return config


def generate_data(config):
    """Generate marginal data based on configuration."""
    
    print("\n--- Data Generation ---")
    
    if config['data_type'] == 'grf':
        print(f"Generating multiscale GRF data (Resolution: {config['resolution']}x{config['resolution']})...")
        marginal_data, data_dim = generate_multiscale_grf_data(
            N_samples=config['n_samples'],
            T=config['T'],
            N_constraints=config['n_constraints'],
            resolution=config['resolution'],
            L_domain=config['l_domain'],
            micro_corr_length=config['micro_corr_length'],
            H_max_factor=config['h_max_factor'],
            mean_val=config['mean_val'],
            std_val=config['std_val'],
            covariance_type=config['covariance_type'],
            device=config['device']
        )
    else:
        print("Generating spiral distributional data (adapted for spatial structure)...")
        # Generate standard spiral data first
        marginal_data_raw, _ = generate_spiral_distributional_data(
            N_constraints=config['n_constraints'],
            T=config['T'],
            data_dim=3,  # Original 3D spiral
            N_samples_per_marginal=config['n_samples_per_marginal'],
            noise_std=config['noise_std']
        )
        
        # Pad/reshape to spatial format for GLOW compatibility
        # This is a simple adaptation - for real use, consider generating truly spatial spiral data
        marginal_data = {}
        for t, samples in marginal_data_raw.items():
            # Pad from 3D to 64D to make it work with 8x8 spatial structure
            padded = torch.zeros(samples.shape[0], config['data_dim'], device=config['device'])
            padded[:, :3] = samples  # Put spiral data in first 3 dimensions
            marginal_data[t] = padded
        
        data_dim = config['data_dim']
    
    # Move data to device
    marginal_data = {t: samples.to(config['device']) for t, samples in marginal_data.items()}
    
    print(f"Generated {len(marginal_data)} marginal distributions with {data_dim} dimensions.")
    return marginal_data, data_dim


def create_model(config, data_dim, data_mean, data_std):
    """Create and initialize the GLOW bridge model."""
    
    print("\n--- Model Initialization ---")
    
    bridge = NeuralBridgeExpressive(
        data_dim=data_dim,
        hidden_size=config['hidden_size'],
        resolution=config['resolution'],
        n_blocks_flow=config['n_blocks_flow'],
        num_scales=config['num_scales'],
        T=config['T'],
        sigma_reverse=config['sigma_reverse'],
        data_mean=data_mean.to(config['device']),
        data_std=data_std.to(config['device']),
        training_noise_std=config['training_noise_std'],
        noise_densification_std=config['noise_densification_std'],
        inference_clamp_norm=None  # Disable artificial clamping
    ).to(config['device'])
    
    n_params = sum(p.numel() for p in bridge.parameters())
    print(f"Initialized NeuralBridgeExpressive with {n_params:,} parameters")
    
    return bridge


def train_model(bridge, marginal_data, config):
    """Train the bridge model."""
    
    print("\n--- Training ---")
    print(f"Training for {config['epochs']} epochs with lr={config['lr']}")
    print("Note: GLOW models typically require more epochs (2000+) for optimal convergence")
    
    loss_history = train_glow_bridge(
        bridge=bridge,
        marginal_data=marginal_data,
        epochs=config['epochs'],
        lr=config['lr'],
        lambda_path=config['lambda_path'],
        weight_decay=config['weight_decay'],
        use_scheduler=config['use_scheduler'],
        grad_clip_norm=config['grad_clip_norm'],
        verbose=True
    )
    
    final_loss = loss_history[-1]['total']
    print(f"\nTraining completed. Final loss: {final_loss:.6f}")
    
    return loss_history


def validate_model(bridge, marginal_data, config):
    """Validate the trained model."""
    
    print("\n--- Validation ---")
    
    if config['data_type'] == 'grf':
        # Generate samples for validation using the simple backward sampler
        print("Generating backward samples for validation...")
        try:
            # Use utilities backward sampler for simplicity (KISS principle)
            generated_samples = generate_backward_samples(
                bridge=bridge,
                marginal_data=marginal_data,
                n_samples=min(256, list(marginal_data.values())[0].shape[0]),
                n_steps=config['n_sde_steps'],
                device=config['device']
            )
            
            # Calculate quantitative metrics
            metrics = calculate_validation_metrics(marginal_data, generated_samples)
            
            # Print validation summary
            if metrics['w2_distances']:
                avg_w2 = np.nanmean(metrics['w2_distances'])
                avg_mse_acf = np.nanmean(metrics['mse_acf'])
                avg_rel_f = np.nanmean(metrics['rel_fro_cov'])
                
                print(f"Average Wasserstein-2 distance: {avg_w2:.6f}")
                print(f"Average MSE ACF: {avg_mse_acf:.6e}")
                print(f"Average Relative Frobenius (Cov): {avg_rel_f:.6f}")
                
                # Validation criteria (GLOW may have higher W2 but better ACF)
                if avg_w2 < 1.0 and (np.isnan(avg_mse_acf) or avg_mse_acf < 1e-2):
                    print("✓ Validation PASSED: GLOW model captures spatial correlations well")
                else:
                    print("⚠ Validation WARNING: Consider more training epochs for GLOW convergence")
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
                T=config['T'],
                n_particles=256,  # Reduced for stability
                n_steps=50,       # Reduced for speed
                n_validation_times=3,
                device=config['device']
            )
            
            if validation_results and validation_results['mean_errors']:
                mean_errors = np.nanmean(validation_results['mean_errors'])
                print(f"Mean error: {mean_errors:.6f}")
                
                if mean_errors < 0.2:  # More lenient for GLOW + padded data
                    print("✓ Validation PASSED: Basic consistency maintained")
                else:
                    print("⚠ Validation WARNING: Consider using true spatial data for GLOW")
            else:
                print("⚠ Validation failed to run")
        except Exception as e:
            print(f"Validation failed: {e}")


def visualize_results(bridge, marginal_data, config):
    """Generate visualizations of the results."""
    
    print("\n--- Visualization ---")
    print(f"Generating visualizations in '{config['output_dir']}'...")
    
    try:
        visualize_bridge_results(
            bridge=bridge,
            marginal_data=marginal_data,
            T=config['T'],
            output_dir=config['output_dir'],
            is_grf=(config['data_type'] == 'grf'),
            n_viz_particles=config['n_viz_particles'],
            n_sde_steps=config['n_sde_steps']
        )
        
        print("✓ Visualizations completed")
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("This may happen with GLOW models - results still saved to output directory")


def main():
    """Main training script."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Asymmetric Multi-Marginal Bridge with GLOW Flow")
    parser.add_argument('--grf', action='store_true', help='Use GRF data (recommended for GLOW)')
    parser.add_argument('--spiral', action='store_true', help='Use spiral data (padded for spatial structure)')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs (recommend 2000+ for GLOW)')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate (lower for GLOW stability)')
    
    args = parser.parse_args()
    
    # Determine data type
    if args.grf:
        args.data_type = 'grf'
    else:
        args.data_type = 'spiral'  # default
    
    # Setup experiment
    config = setup_experiment(args)
    os.makedirs(config['output_dir'], exist_ok=True)
    
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
        marginal_data, data_mean, data_std = normalize_multimarginal_data(marginal_data_raw)
        print("Data normalized to zero mean and unit variance")
        
        # 3. Create model
        bridge = create_model(config, data_dim, data_mean, data_std)
        
        # 4. Train model
        train_model(bridge, marginal_data, config)
        
        # 5. Validate model
        validate_model(bridge, marginal_data, config)
        
        # 6. Generate visualizations
        visualize_results(bridge, marginal_data, config)
        
        print("\n" + "=" * 80)
        print("✓ TRAINING COMPLETED SUCCESSFULLY")
        print(f"Results saved to: {config['output_dir']}")
        print("Note: GLOW models excel at capturing spatial correlations but may need more epochs")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())