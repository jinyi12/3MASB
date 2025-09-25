#!/usr/bin/env python3
"""
Loss Functions for Asymmetric Multi-Marginal Bridge
================================================

Modularized loss functions following KISS and YAGNI principles.
All losses are extracted from the main model for clarity and reusability.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from torch import Tensor, distributions as D
from utilities.common import t_dir


class BridgeLosses:
    """
    Collection of loss functions for bridge training.
    
    Simple, focused implementation following KISS principle.
    Only implements what's actually needed (YAGNI).
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
    
    def path_regularization_loss(self, t: Tensor) -> Tensor:
        """
        Kinetic energy loss: J_Path = E_t E_epsilon [ 0.5 * || dG(epsilon, t)/dt ||^2 ]
        """
        n_samples = t.shape[0]
        
        # Sample epsilon from base distribution
        epsilon = self.bridge.base_dist.sample((n_samples,)).to(self.bridge.base_mean.device)
        
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
    
    def acceleration_regularization_loss(self, t: Tensor) -> Tensor:
        """
        Smoothness loss: J_Smooth = E_t E_epsilon [ || d^2G(epsilon, t)/dt^2 ||^2 ]
        """
        n_samples = t.shape[0]
        device = self.bridge.base_mean.device
        
        # Sample epsilon from base distribution
        epsilon = self.bridge.base_dist.sample((n_samples,)).to(device)
        
        # Define G(epsilon, t) as function of t
        def G_fixed_epsilon(time_tensor):
            z_transformed, _ = self.bridge.flow.forward(epsilon, time_tensor)
            return z_transformed
        
        # Define velocity computation
        v_t = torch.ones_like(t)
        
        try:
            from torch.autograd.functional import jvp
        except ImportError:
            from utilities.common import jvp
        
        try:
            def compute_velocity(time_tensor):
                v_t_inner = torch.ones_like(time_tensor)
                _, vel = jvp(G_fixed_epsilon, time_tensor, v_t_inner, create_graph=True)
                return vel
            
            # Calculate acceleration using nested JVP
            _, acceleration = jvp(compute_velocity, t, v_t, create_graph=True)
            
        except RuntimeError as e:
            print(f"Warning: Error in acceleration calculation: {e}. Returning 0.")
            return torch.tensor(0.0, device=device)
        
        # Smoothness energy
        smoothness_energy = torch.sum(acceleration**2, dim=-1)
        return smoothness_energy.mean()
    
    def standard_loss(
        self, 
        marginal_data: Dict[float, Tensor], 
        lambda_path: float = 0.1,
        batch_size_time: int = 256
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Standard MM-MLE + Path regularization loss.
        """
        # MLE loss across all marginals
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
        
        # Path regularization
        t_rand = torch.rand(batch_size_time, 1, device=device) * self.bridge.T
        path_loss = self.path_regularization_loss(t_rand)
        
        # Total loss
        total_loss = mle_loss + lambda_path * path_loss
        return total_loss, mle_loss, path_loss
    
    def atr_loss(
        self,
        paired_data: Dict[float, Tensor],
        lambda_recon: float = 1.0,
        lambda_smooth: float = 0.001,
        lambda_path: float = 0.0,
        batch_size_time: int = 64,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Anchored Trajectory Reconstruction loss.
        """
        device = self.bridge.base_mean.device
        if not paired_data:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        
        times = sorted(paired_data.keys())
        t0_anchor = times[0]
        
        N = paired_data[t0_anchor].shape[0]
        if N == 0:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        
        x0 = paired_data[t0_anchor]
        t0_tensor = torch.full((N, 1), t0_anchor, device=device)
        
        # Anchor latent inference (clean data for reconstruction)
        epsilon0_clean, log_det_J_inv_clean = self.bridge.flow.inverse(x0, t0_tensor)
        
        # MLE loss (with noise)
        x0_perturbed = self.bridge._apply_noise_if_training(x0)
        is_noisy = self.bridge.training and (x0_perturbed is not x0)
        
        if is_noisy:
            epsilon0_noisy, log_det_J_inv_noisy = self.bridge.flow.inverse(x0_perturbed, t0_tensor)
        else:
            epsilon0_noisy, log_det_J_inv_noisy = epsilon0_clean, log_det_J_inv_clean
        
        try:
            log_prob_base_t0 = self.bridge.base_dist.log_prob(epsilon0_noisy)
        except ValueError:
            log_prob_base_t0 = D.Independent(
                D.Normal(torch.zeros_like(epsilon0_noisy), torch.ones_like(epsilon0_noisy)), 1
            ).log_prob(epsilon0_noisy)
        
        log_likelihood_t0 = log_prob_base_t0 + log_det_J_inv_noisy
        mle_loss_t0 = -log_likelihood_t0.mean()
        
        # Reconstruction loss
        recon_loss = 0
        K_recon = 0
        
        for t_k in times[1:]:
            x_k_target = paired_data[t_k]
            K_recon += 1
            t_k_tensor = torch.full((N, 1), t_k, device=device)
            
            # Forward using clean latents
            x_k_pred, _ = self.bridge.flow.forward(epsilon0_clean, t_k_tensor)
            mse = F.mse_loss(x_k_pred, x_k_target, reduction="mean")
            recon_loss += mse
        
        if K_recon > 0:
            recon_loss /= K_recon
        
        primary_loss = mle_loss_t0 + lambda_recon * recon_loss
        
        # Regularization losses
        reg_loss = torch.tensor(0.0, device=device)
        if batch_size_time > 0:
            t_rand = torch.rand(batch_size_time, 1, device=device) * self.bridge.T
            
            if lambda_smooth > 0:
                smooth_loss = self.acceleration_regularization_loss(t_rand)
                reg_loss += lambda_smooth * smooth_loss
            
            if lambda_path > 0:
                path_loss_val = self.path_regularization_loss(t_rand)
                reg_loss += lambda_path * path_loss_val
        
        total_loss = primary_loss + reg_loss
        return total_loss, primary_loss, reg_loss
    
    def lcl_loss(
        self,
        paired_data: Dict[float, Tensor],
        lambda_lcl: float = 1.0,
        lambda_path: float = 0.01,
        batch_size_time: int = 256,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Latent Consistency Loss objective.
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
        
        # Latent consistency loss
        latents_stack = torch.stack(inferred_latents, dim=0)
        mean_latent = torch.mean(latents_stack, dim=0)
        
        lcl_loss = F.mse_loss(
            latents_stack,
            mean_latent.detach().expand_as(latents_stack),
            reduction="mean"
        )
        
        # Path regularization
        t_rand = torch.rand(batch_size_time, 1, device=device) * self.bridge.T
        path_loss = self.path_regularization_loss(t_rand)
        
        # Total loss
        total_loss = mle_loss + lambda_lcl * lcl_loss + lambda_path * path_loss
        primary_loss = mle_loss + lambda_lcl * lcl_loss
        reg_loss = lambda_path * path_loss
        
        return total_loss, primary_loss, reg_loss