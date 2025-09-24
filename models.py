"""
Neural Models for Asymmetric Multi-Marginal Bridge Framework
=============================================================

This module contains the core neural network models for implementing the
Asymmetric Multi-Marginal Bridge framework using affine flows.

Key Components:
- TimeDependentAffine: Neural network parameterizing affine transformations
- NeuralGaussianBridge: Main bridge model implementing the framework
"""

from typing import Tuple, Callable, Dict, Any
import torch
from torch import nn, Tensor
from torch import distributions as D

# ============================================================================
# Utility Functions for Automatic Differentiation
# ============================================================================

def jvp(f: Callable[[Tensor], Any], x: Tensor, v: Tensor) -> Tuple[Tensor, ...]:
    """Compute Jacobian-vector product. Used for time derivatives."""
    return torch.autograd.functional.jvp(
        f, x, v,
        create_graph=torch.is_grad_enabled()
    )

def t_dir(f: Callable[[Tensor], Any], t: Tensor) -> Tuple[Tensor, ...]:
    """Compute the time derivative of f(t) by using jvp with v=1."""
    return jvp(f, t, torch.ones_like(t))

# ============================================================================
# Flow Model Components
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
        # Clamp gamma for numerical stability
        gamma = torch.clamp(gamma, min=1e-6)
        return mu, gamma

    def forward(self, t: Tensor, return_t_dir: bool = False) -> Any:
        """
        Evaluate the coefficients and optionally their time derivatives.
        """
        if return_t_dir:
            # Compute time derivatives using automatic differentiation
            def f(t_in: Tensor) -> Tuple[Tensor, Tensor]:
                return self.get_coeffs(t_in)
            
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
            t = torch.tensor(float(t), device=self.base_mean.device, dtype=torch.float32)
        
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