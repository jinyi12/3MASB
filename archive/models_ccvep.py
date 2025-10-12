"""
Neural Models for Asymmetric Multi-Marginal Bridge Framework
=============================================================

This module contains both the original affine flow model and the new C-CVEP model
for implementing the Asymmetric Multi-Marginal Bridge framework.

Models:
- NeuralGaussianBridge: Original affine flow implementation (kept for compatibility)
- CCVEPBridge: New C-CVEP implementation using velocity-energy parameterization
"""

from typing import Tuple, Callable, Dict, Any
import torch
from torch import nn, Tensor
from torch import distributions as D
import torch.autograd as autograd

# Import C-CVEP components
from ccvep_core import VelocityNetwork, EnergyNetwork, InterpolatorNetwork, RBFMetricHandler

# ============================================================================
# Utility Functions for Automatic Differentiation (Original)
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
# Original Affine Flow Model (Kept for Compatibility)
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


# ============================================================================
# C-CVEP Bridge Model (New Implementation)
# ============================================================================

class CCVEPBridge(nn.Module):
    """
    Coupled Consistent Velocity-Energy Parameterization (C-CVEP) Bridge.
    
    This model implements the C-CVEP framework combining Metric Flow Matching (MFM)
    and Dual Score Matching (DSM) for simulation-free training with two-stage optimization.
    Updated to support optional RBF metric for enhanced spatial modeling.
    """
    def __init__(
        self,
        velocity_backbone: nn.Module,
        energy_backbone: nn.Module,
        interpolator_backbone: nn.Module,
        metric_handler: RBFMetricHandler = None,
        T: float = 1.0,
        sigma_reverse: float = 1.0,
        data_mean: Tensor = None,
        data_std: Tensor = None
    ):
        super().__init__()
        self.T = T
        self.sigma_reverse = sigma_reverse
        
        # Initialize Networks using externally provided backbones
        self.v_theta = VelocityNetwork(velocity_backbone)
        # EnergyNetwork backbone represents S_phi
        self.U_phi = EnergyNetwork(energy_backbone)
        # InterpolatorNetwork for MFM trajectories
        self.phi_eta = InterpolatorNetwork(interpolator_backbone)
        
        # RBF Metric Handler (G_rbf) - optional
        self.G_rbf = metric_handler
        
        # Add a base distribution for compatibility with existing utilities
        # This is used for sampling initial conditions in visualizations
        data_dim = velocity_backbone.data_dim
        self.register_buffer('base_mean', torch.zeros(data_dim))
        self.register_buffer('base_std', torch.ones(data_dim))
        
        # Data normalization parameters (following GLOW Flow pattern)
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
        """Base distribution for compatibility with existing utilities."""
        return D.Independent(D.Normal(self.base_mean, self.base_std), 1)

    def _format_time(self, t, batch_size):
        """Helper to robustly format time tensor t to (B,) for network input."""
        device = next(self.parameters()).device
        if not torch.is_tensor(t):
            t = torch.tensor(float(t), device=device)
        
        if t.dim() == 0:
            # Scalar time
            return t.expand(batch_size)
        elif t.dim() == 1:
            if t.shape[0] == 1:
                return t.expand(batch_size)
            elif t.shape[0] == batch_size:
                return t
            else:
                raise ValueError(f"1D time tensor length {t.shape[0]} incompatible with batch size {batch_size}.")
        elif t.dim() == 2:
            if t.shape == (1, 1):
                return t.squeeze().expand(batch_size)
            elif t.shape == (batch_size, 1):
                return t.squeeze(-1)
            else:
                raise ValueError(f"2D time tensor shape {t.shape} incompatible with batch size {batch_size}.")
        else:
            raise ValueError(f"Time tensor with {t.dim()} dimensions not supported.")

    # --- Dynamics (Forward ODE and Backward SDE) Interface ---

    def forward_velocity(self, x: Tensor, t: Tensor) -> Tensor:
        """Compute the deterministic forward velocity field v_theta(x, t)."""
        t_formatted = self._format_time(t, x.shape[0])
        return self.v_theta(x, t_formatted)
    
    def score_function(self, x: Tensor, t: Tensor) -> Tensor:
        """Compute the estimated score function: -nabla_x U_phi(x, t)."""
        t_formatted = self._format_time(t, x.shape[0])
        
        # Compute spatial score efficiently during inference/simulation
        # Use detach().requires_grad_(True) to avoid cloning memory if possible
        x_inp = x.detach().requires_grad_(True)
        
        with torch.enable_grad():
            energy = self.U_phi(x_inp, t_formatted)
            spatial_score = autograd.grad(
                outputs=energy,
                inputs=x_inp,
                grad_outputs=torch.ones_like(energy),
                create_graph=False  # No need for higher-order gradients during inference
            )[0]
        
        # Score = - nabla_x U
        return -spatial_score
    
    def reverse_drift(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Compute the drift for the reverse SDE.
        Maintains the definition used in the original affine flow implementation:
        R(x,t) = v(x,t) - (sigma^2 / 2) * Score(x,t)
        """
        velocity = self.forward_velocity(x, t)
        score = self.score_function(x, t)
        
        drift = velocity - (self.sigma_reverse**2 / 2.0) * score
        return drift
    
    def reverse_sde(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """Complete reverse SDE specification."""
        drift = self.reverse_drift(x, t)
        diffusion = torch.ones_like(x) * self.sigma_reverse
        return drift, diffusion

    def normalize(self, x: Tensor) -> Tensor:
        """Normalize data to zero mean and unit variance."""
        return (x - self.data_mean) / self.data_std
    
    def denormalize(self, z: Tensor) -> Tensor:
        """Denormalize data back to original scale."""
        return z * self.data_std + self.data_mean