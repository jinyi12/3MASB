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
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import transforms
from tqdm import trange
import numpy as np
import traceback
try:
    import ot  # Optimal Transport for validation metrics
except ImportError:
    print("Warning: Python Optimal Transport (POT) library not found. Wasserstein distance validation will be unavailable.")
    ot = None
import os
from scipy.integrate import solve_ivp

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

# ============================================================================
# Data Generation
# ============================================================================

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
    fig, axes = plt.subplots(1, len(validation_times), figsize=(6 * len(validation_times), 6), sharex=True, sharey=True)
    fig.suptitle("Figure 3: Comparison of Marginal Distributions p_t(z)", fontsize=16)

    for i, t_val in enumerate(validation_times):
        ax = axes[i]

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
        cov_gt = torch.diag(gamma_gt[0].cpu().numpy()**2)

        # Plot (projected on z1-z2 plane)
        ax.scatter(fwd_particles[:, 0], fwd_particles[:, 1], alpha=0.3, label='Fwd ODE', color='blue', s=10)
        ax.scatter(rev_particles[:, 0], rev_particles[:, 1], alpha=0.3, label='Rev SDE', color='green', s=10)

        # Plot analytical mean and covariance ellipse
        ax.scatter(mu_gt[0], mu_gt[1], marker='x', color='red', s=100, label='Learned Mean')
        plot_confidence_ellipse(ax, mu_gt, cov_gt, n_std=2.0, edgecolor='red', linewidth=2, linestyle='--')

        ax.set_title(f't = {t_val:.2f}')
        ax.set_xlabel('z₁')
        if i == 0:
            ax.set_ylabel('z₂')
        ax.grid(True, linestyle='--')
        ax.legend()
        ax.set_aspect('equal', adjustable='box')

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

    fig, axes = plt.subplots(1, n_marginals, figsize=(6 * n_marginals, 6), sharex=True, sharey=True)
    if n_marginals == 1: # If only one subplot, axes is not a list
        axes = [axes]

    fig.suptitle("Figure 4: Qualitative Fit to Marginal Data Constraints", fontsize=16)

    for i, t_k in enumerate(sorted_times):
        ax = axes[i]
        samples_k = marginal_data[t_k].cpu().numpy()

        # Get analytical distribution at time t_k
        t_k_tensor = torch.tensor([[t_k]], device=device)
        mu_k, gamma_k = bridge.get_params(t_k_tensor)
        mu_k = mu_k[0].cpu().numpy()
        cov_k = torch.diag(gamma_k[0].cpu().numpy()**2)

        # Plot the data samples (projected on z1-z2 plane)
        ax.scatter(samples_k[:, 0], samples_k[:, 1], alpha=0.5, label='Data Samples', s=15, color='gray')

        # Plot the learned distribution's mean and covariance ellipse
        ax.scatter(mu_k[0], mu_k[1], marker='P', color='orange', s=150, label='Learned Mean', edgecolors='black')
        plot_confidence_ellipse(ax, mu_k, cov_k, n_std=2.0, edgecolor='orange', linewidth=2.5, linestyle='-')

        ax.set_title(f't = {t_k:.2f}')
        ax.set_xlabel('z₁')
        if i == 0:
            ax.set_ylabel('z₂')
        ax.grid(True, linestyle='--')
        ax.legend()
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "fig4_marginal_fit.png"), dpi=300)
    plt.close()


def _plot_data_and_trajectory(bridge, marginal_data, T, output_dir, device):
    print("  - Plotting data and mean trajectory...")
    
    # Evaluate the learned trajectory
    # Input time tensor [N_times, 1]
    fine_times = torch.linspace(0, T, 200, device=device).unsqueeze(-1)
    mu_traj, gamma_traj = bridge.get_params(fine_times)
    mu_traj = mu_traj.cpu()
    gamma_traj = gamma_traj.cpu()

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle("Figure 1: Learned Distributional Flow", fontsize=16)

    # 3D Trajectory and Data
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Plot data samples
    for t_k, samples_k in marginal_data.items():
        ax1.scatter(samples_k[:, 0], samples_k[:, 1], samples_k[:, 2], alpha=0.1, label=f'Data t={t_k:.1f}')
    
    # Plot learned mean trajectory
    ax1.plot(mu_traj[:, 0], mu_traj[:, 1], mu_traj[:, 2], 'r-', linewidth=2.5, label='Learned Mean μ(t)')
    
    ax1.set_title('A) Data Marginals and Mean Trajectory')
    ax1.set_xlabel('z₁')
    ax1.set_ylabel('z₂')
    ax1.set_zlabel('z₃')

    # Learned Standard Deviation evolution
    ax2 = fig.add_subplot(132)
    # Plotting each dimension of gamma
    for i in range(bridge.data_dim):
        ax2.plot(fine_times.cpu().squeeze(), gamma_traj[:, i], linewidth=2.5, label=f'γ_{i+1}(t)')
    
    ax2.set_title('B) Learned Standard Deviation γ(t)')
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('γ(t)')
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
    plt.savefig(os.path.join(output_dir, "fig1_flow_parameters.png"), dpi=300)
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


def visualize_bridge_results(bridge: NeuralGaussianBridge, marginal_data: Dict[float, Tensor], T: float, n_viz_particles: int = 50, n_sde_steps: int = 100, output_dir: str = "output"):
    """
    Generate and save a comprehensive set of visualizations for the trained bridge.
    """
    bridge.eval()
    device = bridge.base_mean.device

    print("\\n--- Generating and Saving Visualizations ---")
    
    # Move data to CPU for plotting
    marginal_data_cpu = {t: samples.cpu() for t, samples in marginal_data.items()}

    with torch.no_grad():
        _plot_data_and_trajectory(bridge, marginal_data_cpu, T, output_dir, device)
        _plot_asymmetric_dynamics(bridge, T, n_viz_particles, n_sde_steps, output_dir, device)
        _plot_marginal_distribution_comparison(bridge, T, n_viz_particles, n_sde_steps, output_dir, device)
        _plot_marginal_data_fit(bridge, marginal_data, T, output_dir, device)


    print(f"Visualizations saved to '{output_dir}' directory.")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function for the Neural Gaussian Bridge implementation."""
    # Configuration
    # Use CPU as GPU is typically not available/needed for this scale
    DEVICE = "cpu" 
    torch.manual_seed(42)
    OUTPUT_DIR = "output_neural_bridge"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Hyperparameters
    DATA_DIM = 3
    T_MAX = 1.0
    N_CONSTRAINTS = 5
    N_SAMPLES_PER_MARGINAL = 256
    DATA_NOISE_STD = 0.1
    HIDDEN_SIZE = 128
    LEARNING_RATE = 1e-3
    EPOCHS = 4000
    LAMBDA_PATH = 0.05 # Weight for path regularization
    SIGMA_REVERSE = 0.5 # Diffusion for the reverse SDE

    print("="*80)
    print("NEURAL ASYMMETRIC MULTI-MARGINAL BRIDGE (GAUSSIAN FLOW)")
    print(f"Device: {DEVICE}")
    print("="*80)
    
    # Step 1: Generate synthetic distributional data
    print("\\n1. Generating synthetic distributional spiral data...")
    marginal_data, time_steps = generate_spiral_distributional_data(
        N_constraints=N_CONSTRAINTS, 
        T=T_MAX, 
        data_dim=DATA_DIM,
        N_samples_per_marginal=N_SAMPLES_PER_MARGINAL,
        noise_std=DATA_NOISE_STD
    )
    
    # Move data to device
    marginal_data = {t: samples.to(DEVICE) for t, samples in marginal_data.items()}
    
    print(f"Generated {len(marginal_data)} marginal distributions.")
    
    # Step 2: Initialize bridge
    print("\\n2. Initializing Neural Gaussian Bridge...")
    bridge = NeuralGaussianBridge(
        data_dim=DATA_DIM,
        hidden_size=HIDDEN_SIZE,
        T=T_MAX,
        sigma_reverse=SIGMA_REVERSE
    ).to(DEVICE)
    
    print(f"Bridge initialized with {sum(p.numel() for p in bridge.parameters())} parameters")
    
    # Step 3: Train the bridge
    print("\\n3. Training the bridge (MLE + Path Regularization)...")
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
    print("\\n4. Visualizing results...")
    visualize_bridge_results(bridge, marginal_data, T_MAX, n_viz_particles=50, n_sde_steps=100, output_dir=OUTPUT_DIR)
    
    # Step 5: Rigorous validation
    print("\\n5. Performing rigorous asymmetric consistency validation...")
    
    validation_results = validate_asymmetric_consistency(
        bridge=bridge,
        T=T_MAX,
        n_particles=1024,
        n_steps=100,
        n_validation_times=5,
        device=DEVICE
    )
    
    # Print validation summary
    print("\\nVALIDATION SUMMARY:")
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
             print("\\n✅ SUCCESS: Distributional interpolation and asymmetric consistency validated!")
        else:
            print("\\n⚠️  Validation metrics exceeded thresholds.")

    else:
        print("Validation failed to run or produced no results.")


    print("\\n" + "="*80)
    print("IMPLEMENTATION AND VALIDATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    # Check if running in an interactive environment (like Jupyter/Colab)
    try:
        from IPython import get_ipython
        if get_ipython():
            print("Interactive mode detected. Components loaded.")
            # You can optionally run main() here if desired in interactive mode
            # main()
        else:
            main()
    except ImportError:
        # Not interactive or IPython not installed, run main()
        # We check if torch is available before running main, as it requires it.
        try:
            import torch
            main()
        except ImportError:
            print("PyTorch not found. Cannot run the main execution.")

