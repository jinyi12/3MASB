#!/usr/bin/env python3
"""
Loss Functions for Asymmetric Multi-Marginal Bridge
================================================

Unified loss computation system following KISS and YAGNI principles.
Eliminates code duplication through dynamic loss composition.

Key Design:
- Single `compute_loss()` method handles all loss types
- Loss components computed via internal methods
- Configuration-based composition (no code duplication)
- Supports: Standard MLE, LCL (mean latent), Initial Condition
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from torch import Tensor
from utilities.common import t_dir


class BridgeLosses:
    """
    Unified loss computation for bridge training.
    Eliminates code duplication by composing losses dynamically.
    """
    
    def __init__(self, bridge_model):
        """Initialize with reference to the bridge model."""
        self.bridge = bridge_model
    
    def marginal_log_likelihood(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Compute exact log likelihood log p_t(x) using Change of Variables.
        log p(x) = log p_base(G_inv(x, t)) + log|det(dG_inv/dx)|
        """
        t = self.bridge._format_time(t, x.shape[0])
        
        # Inverse transformation
        epsilon, log_det_J_inv = self.bridge.flow.inverse(x, t)
        
        # Base distribution log probability
        log_prob_base = self.bridge.base_dist.log_prob(epsilon)
        
        # Total log likelihood
        log_likelihood = log_prob_base + log_det_J_inv
        return log_likelihood
    
    def _compute_path_regularization(self, batch_size: int = 256) -> Tensor:
        """
        Kinetic energy loss: J_Path = E_t E_epsilon [ 0.5 * || dG(epsilon, t)/dt ||^2 ]
        """
        device = self.bridge.base_mean.device
        t = torch.rand(batch_size, 1, device=device) * self.bridge.T
        
        # Sample epsilon from base distribution
        epsilon = self.bridge.base_dist.sample((batch_size,)).to(device)
        
        # Define G(epsilon, t) as function of t
        def G_fixed_epsilon(time_tensor):
            z_transformed, _ = self.bridge.flow.forward(epsilon, time_tensor)
            return z_transformed
        
        # Calculate velocity using JVP
        velocity_result = t_dir(G_fixed_epsilon, t)
        if isinstance(velocity_result, tuple):
            velocity = velocity_result[1]
        else:
            velocity = velocity_result
        
        # Kinetic energy
        kinetic_energy = 0.5 * torch.sum(velocity**2, dim=-1)
        return kinetic_energy.mean()
    
    def _compute_mle_loss(self, marginal_data: Dict[float, Tensor]) -> Tuple[Tensor, int]:
        """
        Compute MLE loss across all marginals.
        Returns (mle_loss, total_samples)
        """
        mle_loss = 0
        total_samples = 0
        device = self.bridge.base_mean.device
        
        for t_k, samples_k in marginal_data.items():
            B_k = samples_k.shape[0]
            total_samples += B_k
            t_k_tensor = torch.full((B_k, 1), t_k, device=device)
            
            # Apply noise if training
            samples_k_perturbed = self.bridge._apply_noise_if_training(samples_k)
            
            # Compute log likelihood
            log_likelihoods = self.marginal_log_likelihood(samples_k_perturbed, t_k_tensor)
            mle_loss -= log_likelihoods.sum()
        
        if total_samples > 0:
            mle_loss /= total_samples
        
        return mle_loss, total_samples
    
    def _compute_latent_consistency(self, paired_data: Dict[float, Tensor], mode: str = 'mean') -> Tuple[Tensor, Tensor, int]:
        """
        Compute latent consistency loss for paired trajectories.
        
        Args:
            paired_data: Dictionary of time -> samples
            mode: 'mean' for LCL (mean latent), 'initial' for initial condition loss
        
        Returns:
            (mle_loss, consistency_loss, total_samples)
        """
        device = self.bridge.base_mean.device
        times = sorted(paired_data.keys())
        
        mle_loss = 0
        total_samples = 0
        inferred_latents = []
        
        # Calculate MLE and infer latents at all time points
        for t_k in times:
            x_k = paired_data[t_k]
            B_k = x_k.shape[0]
            total_samples += B_k
            t_k_tensor = torch.full((B_k, 1), t_k, device=device)
            
            # Apply noise handling
            x_k_perturbed = self.bridge._apply_noise_if_training(x_k)
            
            # Inverse transformation
            epsilon_k, log_det_J_inv = self.bridge.flow.inverse(x_k_perturbed, t_k_tensor)
            inferred_latents.append(epsilon_k)
            
            # MLE loss calculation
            log_prob_base = self.bridge.base_dist.log_prob(epsilon_k)
            log_likelihood = log_prob_base + log_det_J_inv
            mle_loss -= log_likelihood.sum()
        
        if total_samples > 0:
            mle_loss /= total_samples
        
        # Latent consistency loss (different modes)
        latents_stack = torch.stack(inferred_latents, dim=0)
        
        if mode == 'mean':
            # LCL: minimize variance around mean latent
            target_latent = torch.mean(latents_stack, dim=0)
        elif mode == 'initial':
            # Initial condition: enforce G^{-1}(x_k, t_k) = epsilon_0
            target_latent = latents_stack[0]
        else:
            raise ValueError(f"Unknown consistency mode: {mode}")
        
        consistency_loss = F.mse_loss(
            latents_stack,
            target_latent.detach().expand_as(latents_stack),
            reduction="mean"
        )
        
        return mle_loss, consistency_loss, total_samples
    
    def compute_loss(
        self,
        marginal_data: Dict[float, Tensor] = None,
        paired_data: Dict[float, Tensor] = None,
        loss_config: Dict = None,
    ) -> Dict[str, Tensor]:
        """
        Unified loss computation with dynamic composition.
        
        Args:
            marginal_data: Dictionary of time -> samples (for standard MLE)
            paired_data: Dictionary of time -> samples (for trajectory-based losses)
            loss_config: Configuration dict with keys:
                - lambda_path: Path regularization weight (default: 0.01)
                - lambda_consistency: Latent consistency weight (default: 1.0)
                - consistency_mode: 'mean' (LCL) or 'initial' (default: None)
                - batch_size_time: Batch size for path regularization (default: 256)
        
        Returns:
            Dictionary with 'total', 'mle', 'consistency', 'path' losses
        """
        if loss_config is None:
            loss_config = {}
        
        # Extract configuration
        lambda_path = loss_config.get('lambda_path', 0.01)
        lambda_consistency = loss_config.get('lambda_consistency', 1.0)
        consistency_mode = loss_config.get('consistency_mode', None)
        batch_size_time = loss_config.get('batch_size_time', 256)
        
        losses = {}
        
        # Compute MLE loss
        if paired_data is not None and consistency_mode is not None:
            # Trajectory-based training
            mle_loss, consistency_loss, _ = self._compute_latent_consistency(
                paired_data, mode=consistency_mode
            )
            losses['mle'] = mle_loss
            losses['consistency'] = consistency_loss
        elif marginal_data is not None:
            # Standard MLE training
            mle_loss, _ = self._compute_mle_loss(marginal_data)
            losses['mle'] = mle_loss
            losses['consistency'] = torch.tensor(0.0, device=mle_loss.device)
        else:
            raise ValueError("Must provide either marginal_data or paired_data")
        
        # Compute path regularization
        if lambda_path > 0:
            losses['path'] = self._compute_path_regularization(batch_size_time)
        else:
            losses['path'] = torch.tensor(0.0, device=losses['mle'].device)
        
        # Compose total loss
        losses['total'] = (
            losses['mle'] + 
            lambda_consistency * losses['consistency'] + 
            lambda_path * losses['path']
        )
        
        # Primary loss (without regularization)
        losses['primary'] = losses['mle'] + lambda_consistency * losses['consistency']
        
        return losses