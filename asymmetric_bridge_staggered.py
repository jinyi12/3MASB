"""
Asymmetric Multi-Marginal Bridge Framework - Unified Staggered Implementation
==============================================================================

This module provides a unified implementation of the Asymmetric Multi-Marginal Bridge
framework using the RS-IPFO (Rectified Schrödinger-Iterative Proportional Fitting
Operator) training scheme. It integrates data generation, state cost functions, and
enhanced visualization capabilities in a single, self-contained script.

Key Features:
- RS-IPFO 4-step staggered training scheme
- Enhanced data generation for various scenarios (simple, homogenization)
- Advanced visualization with confidence ellipses and flow analysis
- Modular state cost functions
- Clean, maintainable code structure
"""

import torch
from torch import nn, Tensor
from torch import distributions as D
import copy
from typing import Tuple, Callable, Optional, Dict, Any
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os
import traceback
try:
    import pandas as pd
except ImportError:
    print("Warning: Pandas not found. Loss history plotting may be limited.")
    pd = None

# ============================================================================
# 1. Utilities (Adapted from asymmetric_bridge_distributions.py)
# ============================================================================

def jvp(f: Callable[[Tensor], Any], x: Tensor, v: Tensor) -> Tuple[Any, Any]:
    """Compute Jacobian-vector product. Returns (f(x), JVP)."""
    # torch.autograd.functional.jvp returns (output, jvp_result)
    return torch.autograd.functional.jvp(
        f, x, v,
        create_graph=torch.is_grad_enabled()
    )

def t_dir(f: Callable[[Tensor], Any], t: Tensor) -> Tuple[Any, Any]:
    """
    Compute the time derivative of f(t) by using jvp with v=1.
    Returns (f(t), df/dt).
    """
    return jvp(f, t, torch.ones_like(t))

# ============================================================================
# 2. Enhanced Data Generation Functions
# ============================================================================

def generate_simple_gaussian_data(T: float = 1.0, data_dim: int = 2, N_samples: int = 512):
    """Generate simple data for testing: Gaussian moving from (-2,..) to (2,...)."""
    marginal_data = {}
    
    mu_start = torch.full((data_dim,), -2.0)
    mu_end = torch.full((data_dim,), 2.0)
    std = 0.2
    
    dist_start = D.Independent(D.Normal(mu_start, std), 1)
    dist_end = D.Independent(D.Normal(mu_end, std), 1)

    marginal_data[0.0] = dist_start.sample((N_samples,))
    marginal_data[T] = dist_end.sample((N_samples,))
    
    return marginal_data

def generate_coarsening_gaussian_data(
    N_constraints: int = 5, 
    T: float = 1.0, 
    data_dim: int = 2,
    N_samples_per_marginal: int = 256,
    std_start: float = 0.1, # Microscale (sharp)
    std_end: float = 1.0    # Macroscale (coarse)
) -> Tuple[Dict[float, Tensor], Tensor]:
    """
    Generate synthetic distributional data representing a homogenization process.
    Interpolates Gaussian distributions from microscale to macroscale.
    """
    time_steps = torch.linspace(0, T, N_constraints)
    marginal_data = {}

    # Assume stationary mean at the origin for pure coarsening
    mu = torch.zeros(data_dim)

    for t in time_steps:
        t_val = t.item()
        t_norm = t_val / T
        
        # Interpolate Standard Deviation (Coarsening)
        # Linear interpolation of std dev for data generation
        std_t = std_start * (1 - t_norm) + std_end * t_norm
        
        # Generate samples
        cov_t = torch.eye(data_dim) * (std_t**2)
        dist_t = D.MultivariateNormal(mu, cov_t)
        
        samples = dist_t.sample((N_samples_per_marginal,))
        marginal_data[t_val] = samples
    
    return marginal_data, time_steps

# ============================================================================
# 3. MMD Loss Implementation
# ============================================================================

def gaussian_kernel(x: Tensor, y: Tensor, sigma: float = 1.0) -> Tensor:
    """Computes the Gaussian (RBF) kernel."""
    dist_sq = torch.cdist(x, y, p=2)**2
    return torch.exp(-dist_sq / (2 * sigma**2))

def compute_mmd_U(X: Tensor, Y: Tensor, kernel: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
    """Computes the unbiased MMD² U-statistic."""
    N = X.shape[0]
    M = Y.shape[0]
    
    if N < 2 or M < 2:
        return torch.tensor(0.0, device=X.device)

    K_XX = kernel(X, X)
    K_YY = kernel(Y, Y)
    K_XY = kernel(X, Y)
    
    # U-statistic calculation (excluding diagonals)
    term_XX = (torch.sum(K_XX) - torch.trace(K_XX)) / (N * (N - 1))
    term_YY = (torch.sum(K_YY) - torch.trace(K_YY)) / (M * (M - 1))
    term_XY = torch.mean(K_XY)
             
    mmd_sq = term_XX + term_YY - 2.0 * term_XY
    # Clamp to ensure non-negativity
    return torch.clamp(mmd_sq, min=0.0)

# ============================================================================
# 4. Model Architectures (Flow G_theta and Volatility g_phi)
# ============================================================================

class TimeDependentAffine(nn.Module):
    """
    Neural Network parameterizing the affine flow G_theta.
    G(epsilon, t) = mu(t) + gamma(t) * epsilon.
    """
    def __init__(self, data_dim: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * data_dim),
        )

    def get_coeffs(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        """Get the coefficients mu(t) and gamma(t)."""
        # Ensure t is [B, 1]
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        mu, log_gamma = self.net(t).chunk(chunks=2, dim=1)
        gamma = torch.exp(log_gamma)
        # Clamp gamma for numerical stability
        gamma = torch.clamp(gamma, min=1e-6)
        return mu, gamma

    def forward(self, t: Tensor, return_t_dir: bool = False) -> Any:
        """Evaluate the coefficients and optionally their time derivatives."""
        if return_t_dir:
            def f(t_in: Tensor) -> Tuple[Tensor, Tensor]:
                return self.get_coeffs(t_in)
            # t_dir returns ((mu, gamma), (dmu_dt, dgamma_dt))
            return t_dir(f, t)
        else:
            return self.get_coeffs(t)

class TimeDependentVolatility(nn.Module):
    """
    A simple time-dependent volatility model g_phi(t) (State-independent).
    """
    def __init__(self, data_dim: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, data_dim),
            nn.Softplus() # Ensure volatility is positive
        )

    def forward(self, t: Tensor) -> Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        # Add small epsilon for stability
        return self.net(t) + 1e-5

# ============================================================================
# 5. RS-IPFO Bridge Implementation
# ============================================================================

class RSIPFOBridge(nn.Module):
    """
    Implements the RS-IPFO framework using an affine Gaussian flow (G_theta)
    and a learnable volatility (g_phi).
    """
    def __init__(
        self, 
        data_dim: int,
        hidden_size: int,
        T: float = 1.0,
        V_t: Optional[Callable[[Tensor, Tensor], Tensor]] = None
    ):
        super().__init__()
        
        self.data_dim = data_dim
        self.T = T
        self.V_t = V_t # State cost function V_t(x)
        
        # The flow model (theta parameters)
        self.affine_flow = TimeDependentAffine(data_dim, hidden_size)
        
        # The volatility model (phi parameters)
        self.volatility_model = TimeDependentVolatility(data_dim, hidden_size//2)

        # Base distribution (Standard Normal Latent Space)
        self.register_buffer('base_mean', torch.zeros(data_dim))
        self.register_buffer('base_std', torch.ones(data_dim))
    
    @property
    def base_dist(self):
        return D.Independent(D.Normal(self.base_mean, self.base_std), 1)

    # --- Helper Functions ---

    def get_params(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        return self.affine_flow(t, return_t_dir=False)

    def get_params_and_derivs(self, t: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        (mu, gamma), (dmu_dt, dgamma_dt) = self.affine_flow(t, return_t_dir=True)
        return mu, gamma, dmu_dt, dgamma_dt
    
    def G_transform(self, epsilon: Tensor, t: Tensor) -> Tensor:
        """The forward map G_theta(epsilon, t)."""
        # Assumes t is [B_time, 1] and epsilon is [B_latent, D].
        mu, gamma = self.get_params(t)
        
        B_time = mu.shape[0]
        B_latent = epsilon.shape[0]

        if B_time == B_latent:
             # Standard case: one time per latent code
             return mu + gamma * epsilon
        elif B_time == 1:
             # Case: single time t_k, multiple latent codes (e.g., MMD calculation)
             return mu + gamma * epsilon
        else:
             # Case: multiple times, multiple latent codes (e.g., MC estimation)
             # Requires careful broadcasting, often handled outside this function.
             raise NotImplementedError(f"Broadcasting mismatch: Time batch {B_time}, Latent batch {B_latent}")

    # --- Dynamics and Score (Step 3) ---

    def score_function_exact(self, z: Tensor, t: Tensor) -> Tensor:
        """
        Step 3: Exact score function ∇_z log p_t(z) via CNF inversion.
        For Gaussian: ∇ log p(z) = -(z - mu) / gamma^2
        """
        # Assumes t and z have matching batch dimensions.
        mu, gamma = self.get_params(t)
        score = -(z - mu) / (gamma**2)
        return score

    # --- Loss Components ---

    def kinetic_energy(self, t: Tensor) -> Tensor:
        """
        J_KE (Analytical): E[1/2 ||v||²] = 1/2 * (||dmu/dt||² + sum((dgamma_i/dt)²))
        """
        _, _, dmu_dt, dgamma_dt = self.get_params_and_derivs(t)
        
        mean_term = torch.sum(dmu_dt**2, dim=-1)
        var_term = torch.sum(dgamma_dt**2, dim=-1)
        
        ke = (mean_term + var_term) / 2.0
        return ke

    def state_cost(self, t: Tensor, n_mc_samples: int = 128) -> Tensor:
        """
        J_V (Monte Carlo): E[V_t(x)]
        """
        if self.V_t is None:
            return torch.zeros(t.shape[0], device=t.device)

        # Efficient MC estimation over the latent space for a batch of times
        mu, gamma = self.get_params(t)
        B_time = mu.shape[0]

        # Expand for MC samples: [B_time, 1, D]
        mu = mu.unsqueeze(1)
        gamma = gamma.unsqueeze(1)

        # Sample epsilon: [B_time, n_mc_samples, D]
        epsilon = torch.randn(B_time, n_mc_samples, self.data_dim, device=t.device)
        x_samples = mu + gamma * epsilon

        # Time tensor expansion: [B_time, n_mc_samples, 1]
        t_expanded = t.unsqueeze(1).expand(-1, n_mc_samples, -1)
        
        costs = self.V_t(x_samples, t_expanded) # Shape [B_time, n_mc_samples]

        # Average over MC samples, result shape [B_time]
        return costs.mean(dim=1)

    def loss_Action(self, t: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        A(theta): Total Action (J_KE + J_V). Returns means over the time batch.
        """
        ke = self.kinetic_energy(t)
        sc = self.state_cost(t)
        action = ke + sc
        return action.mean(), ke.mean(), sc.mean()

    def loss_MMD(self, marginal_data: Dict[float, Tensor], n_samples_model: int, mmd_sigma: float = 1.0) -> Tensor:
        """
        L_MM: Multi-Marginal MMD Loss.
        """
        mmd_loss = 0.0
        device = self.base_mean.device
        
        def kernel(x, y):
            return gaussian_kernel(x, y, sigma=mmd_sigma)

        # Sample latent variables once
        epsilon = self.base_dist.sample((n_samples_model,))

        for t_k, samples_data in marginal_data.items():
            # Generate samples from the model p_t_k
            # Time tensor [1, 1] for parameter retrieval at specific t_k
            t_k_tensor = torch.tensor([[t_k]], device=device)
            
            samples_model = self.G_transform(epsilon, t_k_tensor)
            
            # Compute MMD
            mmd_sq = compute_mmd_U(samples_data, samples_model, kernel)
            mmd_loss += mmd_sq

        return mmd_loss / len(marginal_data)

    def loss_DLA(self, t: Tensor, target_model: 'RSIPFOBridge') -> Tensor:
        """
        L_DLA (Analytical): E[ ||mu_theta - mu_target||² + sum( (gamma_theta - gamma_target)² ) ]
        Efficient analytical calculation for Affine Flows.
        """
        # Ensure target model gradients are detached
        target_model.eval()
        
        mu_theta, gamma_theta = self.get_params(t)
        
        with torch.no_grad():
            mu_target, gamma_target = target_model.get_params(t)
        
        mean_term = torch.sum((mu_theta - mu_target)**2, dim=-1)
        var_term = torch.sum((gamma_theta - gamma_target)**2, dim=-1)
        
        dla_loss = (mean_term + var_term)
        return dla_loss.mean()

    def loss_Backward_Energy(self, t: Tensor, n_mc_samples: int = 128) -> Tensor:
        """
        J_Bwd (Step 4 Objective): Minimize backward kinetic energy.
        J_Bwd = E[ 1/2 || v_theta - 1/2 (div(D_phi) + D_phi * score) ||^2 ]
        """
        # MC estimation over p_t^theta
        mu, gamma, dmu_dt, dgamma_dt = self.get_params_and_derivs(t)
        B_time = mu.shape[0]

        # 1. Sample X_t (using multiple MC samples for robustness)
        # Expand params: [B_time, 1, D]
        mu_e = mu.unsqueeze(1)
        gamma_e = gamma.unsqueeze(1)
        dmu_dt_e = dmu_dt.unsqueeze(1)
        dgamma_dt_e = dgamma_dt.unsqueeze(1)

        # Sample epsilon: [B_time, n_mc_samples, D]
        epsilon = torch.randn(B_time, n_mc_samples, self.data_dim, device=t.device)
        X_t = mu_e + gamma_e * epsilon
        
        # 2. Compute Forward velocity v_theta(X_t, t)
        # v(z,t) = dmu/dt + (dgamma/dt / gamma) * (z - mu)
        v_theta = dmu_dt_e + (dgamma_dt_e / gamma_e) * (X_t - mu_e)

        # 3. Compute Score (Step 3: Exact Inversion)
        # score = -(z - mu) / gamma^2
        score = -(X_t - mu_e) / (gamma_e**2)
        
        # 4. Compute Volatility g_phi(t) and Diffusion D_phi(t)
        g_phi = self.volatility_model(t)
        D_phi = g_phi**2 # Diagonal diffusion tensor
        
        # Expand D_phi: [B_time, 1, D]
        D_phi_e = D_phi.unsqueeze(1)
        
        # 5. Divergence div(D_phi)
        # Since D_phi is state-independent in this implementation, div(D_phi) = 0.

        # 6. Backward drift R
        # R = v_theta - 1/2 * D_phi * score
        R = v_theta - 0.5 * D_phi_e * score
        
        # 7. Kinetic energy
        J_Bwd = 0.5 * torch.mean(torch.sum(R**2, dim=-1))
        return J_Bwd

# ============================================================================
# State Cost Functions (Optional Advanced Features)
# ============================================================================

def diffusive_consistency_state_cost(D: float = 0.5):
    """
    Create a state cost function that enforces diffusive consistency for homogenization.
    This is an example of how state costs can be integrated into the RS-IPFO framework.
    
    Args:
        D: Effective diffusivity parameter
    
    Returns:
        A callable state cost function V_t(x, t)
    """
    def V_t(x: Tensor, t: Tensor) -> Tensor:
        """
        Diffusive consistency state cost.
        Penalizes deviations from expected diffusive behavior.
        """
        # For this example, we use a simple quadratic potential
        # In practice, this would encode physical constraints
        batch_shape = x.shape[:-1]  # All dimensions except the last (data dimension)
        
        # Simple quadratic cost scaled by time and diffusivity
        cost = 0.5 * D * torch.sum(x**2, dim=-1) * t.squeeze(-1)
        
        return cost.reshape(batch_shape)
    
    return V_t

def create_custom_state_cost(cost_type: str = "none", **kwargs):
    """
    Factory function to create different types of state costs.
    
    Args:
        cost_type: Type of state cost ("none", "diffusive", "quadratic")
        **kwargs: Additional parameters for the specific cost type
    
    Returns:
        A state cost function or None
    """
    if cost_type == "none" or cost_type is None:
        return None
    elif cost_type == "diffusive":
        D = kwargs.get('D', 0.5)
        return diffusive_consistency_state_cost(D)
    elif cost_type == "quadratic":
        strength = kwargs.get('strength', 1.0)
        def V_t(x: Tensor, t: Tensor) -> Tensor:
            batch_shape = x.shape[:-1]
            cost = 0.5 * strength * torch.sum(x**2, dim=-1)
            return cost.reshape(batch_shape)
        return V_t
    else:
        raise ValueError(f"Unknown cost type: {cost_type}")

# ============================================================================
# 7. RS-IPFO Training Loop (Staggered Scheme)
# ============================================================================

def train_rs_ipfo(
    bridge: RSIPFOBridge, 
    marginal_data: Dict[float, Tensor],
    config: Dict[str, Any]
) -> list:
    """
    Train the bridge using the 4-step RS-IPFO staggered scheme.
    """
    # Configuration extraction
    iterations = config.get('iterations', 10)
    sub_steps = config.get('sub_steps', 200)
    lr_theta = config.get('lr_theta', 1e-3)
    lr_phi = config.get('lr_phi', 1e-3)
    lambda_MM = config.get('lambda_MM', 1.0)
    lambda_Match = config.get('lambda_Match', 50.0)
    batch_size_time = config.get('batch_size_time', 128)
    n_samples_mmd = config.get('n_samples_mmd', 256)

    # Separate optimizers for theta (flow) and phi (volatility)
    optimizer_theta = torch.optim.Adam(bridge.affine_flow.parameters(), lr=lr_theta)
    optimizer_phi = torch.optim.Adam(bridge.volatility_model.parameters(), lr=lr_phi)
    
    history = []
    device = bridge.base_mean.device
    T = bridge.T

    print("Starting RS-IPFO Training...")

    for iteration in range(iterations):
        print(f"\n--- Iteration {iteration + 1}/{iterations} ---")
        
        # ====================================================================
        # Step 1: GSB Path Optimization (theta -> theta_target)
        # Objective: J_GSB = lambda_MM * L_MM + A(theta)
        # ====================================================================
        bridge.train()
        pbar = trange(sub_steps, desc="Step 1 (J_GSB)")
        for step in pbar:
            optimizer_theta.zero_grad()
            
            # 1. L_MM
            loss_mmd = bridge.loss_MMD(marginal_data, n_samples_model=n_samples_mmd)
            
            # 2. A(theta)
            t_rand = torch.rand(batch_size_time, 1, device=device) * T
            loss_action, loss_ke, loss_sc = bridge.loss_Action(t_rand)
            
            # Total J_GSB
            loss_step1 = lambda_MM * loss_mmd + loss_action
            
            loss_step1.backward()
            torch.nn.utils.clip_grad_norm_(bridge.affine_flow.parameters(), max_norm=1.0)
            optimizer_theta.step()
            
            pbar.set_description(f"S1 J_GSB: {loss_step1.item():.4f} (MMD: {loss_mmd.item():.3f}, KE: {loss_ke.item():.3f})")

        # Store the result as the target model (theta_target).
        target_bridge = copy.deepcopy(bridge)
        J_GSB_target = loss_step1.item()

        # ====================================================================
        # Step 2: Field Refinement (theta_target -> theta_i+1)
        # Objective: J_KE(theta) + lambda_Match * L_DLA(theta; theta_target)
        # ====================================================================
        bridge.train()
        pbar = trange(sub_steps, desc="Step 2 (Refinement)")
        for step in pbar:
            optimizer_theta.zero_grad()
            
            t_rand = torch.rand(batch_size_time, 1, device=device) * T
            
            # 1. J_KE
            loss_ke = bridge.kinetic_energy(t_rand).mean()
            
            # 2. L_DLA
            loss_dla = bridge.loss_DLA(t_rand, target_bridge)
            
            # Total Step 2 Loss
            loss_step2 = loss_ke + lambda_Match * loss_dla
            
            loss_step2.backward()
            torch.nn.utils.clip_grad_norm_(bridge.affine_flow.parameters(), max_norm=1.0)
            optimizer_theta.step()
            
            pbar.set_description(f"S2 Loss: {loss_step2.item():.4f} (KE: {loss_ke.item():.3f}, DLA: {loss_dla.item():.3f})")

        # ====================================================================
        # Step 3: Score Approximation
        # ====================================================================
        # Using exact CNF inversion (implemented in bridge.score_function_exact).

        # ====================================================================
        # Step 4: Volatility Update (Optimize phi)
        # Objective: J_Bwd(theta_i+1, phi)
        # ====================================================================
        bridge.train()
        # Freeze theta parameters during phi optimization
        for param in bridge.affine_flow.parameters():
            param.requires_grad = False

        pbar = trange(sub_steps, desc="Step 4 (J_Bwd)")
        for step in pbar:
            optimizer_phi.zero_grad()

            t_rand = torch.rand(batch_size_time, 1, device=device) * T
            loss_bwd = bridge.loss_Backward_Energy(t_rand)

            loss_bwd.backward()
            torch.nn.utils.clip_grad_norm_(bridge.volatility_model.parameters(), max_norm=1.0)
            optimizer_phi.step()

            pbar.set_description(f"S4 J_Bwd: {loss_bwd.item():.4f}")

        # Unfreeze theta parameters
        for param in bridge.affine_flow.parameters():
            param.requires_grad = True

        # --- Evaluation and Logging ---
        with torch.no_grad():
            bridge.eval()
            # Re-evaluate J_GSB(theta_i+1) accurately
            t_eval = torch.rand(1024, 1, device=device) * T
            loss_mmd_final = bridge.loss_MMD(marginal_data, n_samples_model=n_samples_mmd)
            loss_action_final, _, _ = bridge.loss_Action(t_eval)
            J_GSB_refined = (lambda_MM * loss_mmd_final + loss_action_final).item()
            J_Bwd_final = bridge.loss_Backward_Energy(t_eval).item()

        # Check convergence guarantee
        warning = ""
        if J_GSB_refined > J_GSB_target + 1e-4:
             warning = f"(Warn: J_GSB increased by {J_GSB_refined - J_GSB_target:.4f})"

        print(f"Iteration Summary: J_GSB Target: {J_GSB_target:.4f} -> Refined: {J_GSB_refined:.4f} {warning}. J_Bwd: {J_Bwd_final:.4f}")
        
        history.append({
            'iteration': iteration,
            'J_GSB_target': J_GSB_target,
            'J_GSB_refined': J_GSB_refined,
            'J_Bwd': J_Bwd_final
        })
        
    return history

# ============================================================================
# 8. Enhanced Visualization Functions
# ============================================================================

def plot_confidence_ellipse(ax, mu, cov, n_std=2.0, **kwargs):
    """Plots a confidence ellipse for a 2D Gaussian distribution."""
    if mu.shape[0] != 2 or cov.shape != (2, 2):
        return 

    try:
        lambda_, v = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return
        
    lambda_ = np.sqrt(np.maximum(lambda_, 1e-9))

    if np.isnan(lambda_).any():
        return

    angle = np.rad2deg(np.arctan2(*v[:, 0][::-1]))

    ell = Ellipse(xy=(mu[0], mu[1]),
                  width=lambda_[0]*n_std*2, height=lambda_[1]*n_std*2,
                  angle=angle, **kwargs)
    ell.set_facecolor('none')
    ax.add_patch(ell)

def visualize_enhanced_results(bridge, history, marginal_data, T, output_dir="output_rsipfo"):
    """
    Enhanced visualization combining RS-IPFO convergence plots with detailed analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    bridge.eval()
    device = bridge.base_mean.device
    
    print("\n--- Generating Enhanced Visualizations ---")
    
    # 1. Convergence Analysis
    _plot_convergence_analysis(history, output_dir)
    
    # 2. Flow Analysis (if data_dim == 2)
    if bridge.data_dim == 2:
        _plot_flow_analysis_rsipfo(bridge, T, output_dir, device)
        _plot_marginal_fit_2d_rsipfo(bridge, marginal_data, T, output_dir, device)
    
    print(f"Enhanced visualizations saved to '{output_dir}' directory.")

def _plot_convergence_analysis(history, output_dir):
    """Plot RS-IPFO convergence analysis."""
    print("  - Plotting convergence analysis...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("RS-IPFO Training Convergence Analysis", fontsize=14)
    
    if history:
        iters = [h['iteration'] for h in history]
        target_loss = [h['J_GSB_target'] for h in history]
        refined_loss = [h['J_GSB_refined'] for h in history]
        j_bwd = [h['J_Bwd'] for h in history]
        
        # A) GSB Objective Evolution
        ax1 = axes[0]
        ax1.plot(iters, target_loss, marker='o', linestyle='--', linewidth=2, 
                label='J_GSB(Target) - Step 1', color='blue')
        ax1.plot(iters, refined_loss, marker='x', linestyle='-', linewidth=2, 
                label='J_GSB(Refined) - Step 2', color='red')
        ax1.set_title("A) GSB Objective (Monotonic Guarantee)")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("GSB Objective")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # B) Backward Energy Evolution
        ax2 = axes[1]
        ax2.plot(iters, j_bwd, marker='s', linewidth=2, color='purple', 
                label='J_Bwd (Backward Energy)')
        ax2.set_title("B) Backward Energy Convergence (Step 4)")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("J_Bwd")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "No training history available", 
                   ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "convergence.png"), dpi=300)
    plt.close()

def _plot_flow_analysis_rsipfo(bridge, T, output_dir, device):
    """Plot detailed flow analysis for RS-IPFO."""
    print("  - Plotting flow trajectory analysis...")
    
    # Evaluate the learned trajectory
    fine_times = torch.linspace(0, T, 100, device=device).unsqueeze(-1)
    fine_times.requires_grad = True 
    
    with torch.enable_grad():
        mu_traj, gamma_traj, dmu_dt, dgamma_dt = bridge.get_params_and_derivs(fine_times)

    # Detach for plotting
    time_points = fine_times.detach().cpu().squeeze()
    mu_traj = mu_traj.detach().cpu()
    gamma_traj = gamma_traj.detach().cpu()
    dmu_dt = dmu_dt.detach().cpu()
    dgamma_dt = dgamma_dt.detach().cpu()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("RS-IPFO Flow Analysis", fontsize=14)

    # A) Mean trajectory
    ax1 = axes[0]
    for i in range(bridge.data_dim):
        ax1.plot(time_points, mu_traj[:, i], linewidth=2.5, label=f'μ_{i+1}(t)')
    ax1.set_title('A) Mean Trajectory μ(t)')
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('μ(t)')
    ax1.grid(True, linestyle='--')
    ax1.legend()

    # B) Standard deviation evolution
    ax2 = axes[1]
    for i in range(bridge.data_dim):
        ax2.plot(time_points, gamma_traj[:, i], linewidth=2.5, label=f'γ_{i+1}(t)')
    ax2.set_title('B) Standard Deviation γ(t)')
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('γ(t)')
    ax2.grid(True, linestyle='--')
    ax2.legend()
    ax2.set_ylim(bottom=0)

    # C) Kinetic energy evolution
    ax3 = axes[2]
    kinetic_energy = (torch.sum(dmu_dt**2, dim=-1) + torch.sum(dgamma_dt**2, dim=-1)) / 2.0
    ax3.plot(time_points, kinetic_energy, 'purple', linewidth=2.5, label='Kinetic Energy')
    ax3.set_title('C) Kinetic Energy Evolution')
    ax3.set_xlabel('Time t')
    ax3.set_ylabel('Kinetic Energy')
    ax3.grid(True, linestyle='--')
    ax3.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "flow_analysis.png"), dpi=300)
    plt.close()

def _plot_marginal_fit_2d_rsipfo(bridge, marginal_data, T, output_dir, device):
    """Plot marginal distribution fit for 2D case."""
    print("  - Plotting marginal data fit (2D)...")

    if bridge.data_dim != 2:
        return

    n_marginals = len(marginal_data)
    sorted_times = sorted(marginal_data.keys())

    fig_width = min(20, 4 * n_marginals)
    fig, axes = plt.subplots(1, n_marginals, figsize=(fig_width, 5), sharex=True, sharey=True)
    if n_marginals == 1: 
        axes = np.array([axes])

    fig.suptitle("Learned Distribution vs Data Constraints", fontsize=14)

    for i, t_k in enumerate(sorted_times):
        ax = axes[i]
        samples_k = marginal_data[t_k].cpu().numpy()

        # Get learned distribution at time t_k
        t_k_tensor = torch.tensor([[t_k]], device=device, requires_grad=False)
        with torch.no_grad():
            mu_k, gamma_k = bridge.get_params(t_k_tensor)
        mu_k_np = mu_k[0].cpu().numpy()
        cov_k = torch.diag(gamma_k[0]**2).cpu().numpy()

        # Plot Data
        ax.scatter(samples_k[:, 0], samples_k[:, 1], alpha=0.3, s=10, color='gray', label='Data')
        
        # Plot learned distribution (Ellipses)
        plot_confidence_ellipse(ax, mu_k_np, cov_k, n_std=1.0, 
                              edgecolor='orange', linewidth=1.5, linestyle='-')
        plot_confidence_ellipse(ax, mu_k_np, cov_k, n_std=2.0, 
                              edgecolor='orange', linewidth=2.5, linestyle='--')
        
        ax.set_title(f't = {t_k:.2f}')
        ax.set_xlabel('x₁')
        if i == 0:
            ax.set_ylabel('x₂')
            # Create custom legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='gray', marker='o', linestyle='None', 
                       markersize=5, label='Data'),
                Line2D([0], [0], color='orange', ls='-', lw=1.5, label='Learned (1-std)'),
                Line2D([0], [0], color='orange', ls='--', lw=2.5, label='Learned (2-std)')
            ]
            ax.legend(handles=legend_elements, loc='upper right')

        ax.grid(True, linestyle=':')
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "marginal_fit.png"), dpi=300)
    plt.close()

# Legacy visualization function (kept for backward compatibility)
def visualize_rsipfo_results(bridge, history, marginal_data, output_dir="output_rsipfo"):
    """Legacy visualization function - redirects to enhanced version."""
    T = bridge.T if hasattr(bridge, 'T') else 1.0
    visualize_enhanced_results(bridge, history, marginal_data, T, output_dir)

# ============================================================================
# 9. Example Usage and Applications
# ============================================================================

def run_rs_ipfo_example():
    """
    Run an example using the RS-IPFO bridge with simple Gaussian data.
    """
    # Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    
    # Hyperparameters
    DATA_DIM = 2
    T_MAX = 1.0
    N_SAMPLES = 512
    HIDDEN_SIZE = 128
    
    # RS-IPFO Configuration
    config = {
        'iterations': 15,       # Number of outer iterations (i)
        'sub_steps': 300,       # Number of inner optimization steps
        'lr_theta': 1e-3,       # Learning rate for flow (theta)
        'lr_phi': 1e-3,         # Learning rate for volatility (phi)
        'lambda_MM': 10.0,      # Weight for MMD constraints
        'lambda_Match': 100.0,  # Weight for DLA matching (must be strong)
        'batch_size_time': 128,
        'n_samples_mmd': N_SAMPLES
    }

    print("="*80)
    print(f"RS-IPFO Implementation (4-Step Staggered Scheme) | Device: {DEVICE}")
    print("="*80)
    
    # 1. Generate Data
    marginal_data = generate_simple_gaussian_data(T=T_MAX, data_dim=DATA_DIM, N_samples=N_SAMPLES)
    marginal_data = {t: s.to(DEVICE) for t, s in marginal_data.items()}
    
    # 2. Initialize Bridge (V_t=0 for this example)
    bridge = RSIPFOBridge(
        data_dim=DATA_DIM, hidden_size=HIDDEN_SIZE, T=T_MAX, V_t=None
    ).to(DEVICE)
    
    # 3. Train with RS-IPFO
    history = train_rs_ipfo(bridge, marginal_data, config)
    
    print("\nTraining complete.")
    
    # 4. Visualize
    visualize_enhanced_results(bridge, history, marginal_data, T_MAX)
    print("Visualizations saved to 'output_rsipfo' directory.")

def run_homogenization_example():
    """
    Run an example using the coarsening Gaussian data for homogenization modeling.
    """
    # Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)
    OUTPUT_DIR = "output_homogenization_rsipfo"
    
    # Experiment Hyperparameters
    DATA_DIM = 2
    T_MAX = 1.0
    N_CONSTRAINTS = 5
    N_SAMPLES_PER_MARGINAL = 1024
    
    # Homogenization Data Parameters
    STD_MICRO = 0.05  # Sharp microscale distribution
    STD_MACRO = 2.0   # Coarse macroscale distribution
    
    # Model Hyperparameters
    HIDDEN_SIZE = 128
    
    # RS-IPFO Configuration for homogenization
    config = {
        'iterations': 10,       
        'sub_steps': 500,       
        'lr_theta': 5e-4,       
        'lr_phi': 5e-4,         
        'lambda_MM': 100.0,      # Higher weight for marginal constraints
        'lambda_Match': 50.0,  # Strong matching weight
        'batch_size_time': 256,
        'n_samples_mmd': N_SAMPLES_PER_MARGINAL
    }

    print("="*80)
    print("RS-IPFO FOR HOMOGENIZATION MODELING")
    print(f"Device: {DEVICE} | Dim: {DATA_DIM}")
    print(f"Coarsening: {STD_MICRO:.3f} → {STD_MACRO:.2f}")
    print("="*80)
    
    # Step 1: Generate synthetic homogenization data
    print("\n1. Generating coarsening Gaussian data (Micro to Macro)...")
    marginal_data, time_steps = generate_coarsening_gaussian_data(
        N_constraints=N_CONSTRAINTS, T=T_MAX, data_dim=DATA_DIM,
        N_samples_per_marginal=N_SAMPLES_PER_MARGINAL,
        std_start=STD_MICRO, std_end=STD_MACRO
    )
    
    # Move data to device
    marginal_data = {t: samples.to(DEVICE) for t, samples in marginal_data.items()}
    
    # Step 2: Initialize bridge
    print("\n2. Initializing RS-IPFO Bridge...")
    bridge = RSIPFOBridge(
        data_dim=DATA_DIM,
        hidden_size=HIDDEN_SIZE,
        T=T_MAX,
        V_t=None  # No additional state cost for this example
    ).to(DEVICE)
    
    # Step 3: Train the bridge
    print("\n3. Training with RS-IPFO staggered scheme...")
    try:
        history = train_rs_ipfo(bridge, marginal_data, config)
        
        if history:
            final_gsb = history[-1]['J_GSB_refined']
            final_bwd = history[-1]['J_Bwd']
            print(f"Training complete. Final J_GSB: {final_gsb:.4f}, J_Bwd: {final_bwd:.4f}")
        else:
            print("Training stopped prematurely.")
            return

        # Step 4: Visualize results
        print("\n4. Visualizing results...")
        visualize_enhanced_results(
            bridge, history, marginal_data, T_MAX, output_dir=OUTPUT_DIR
        )
        
    except Exception as e:
        print(f"An error occurred during training/visualization: {e}")
        traceback.print_exc()

    print("\n" + "="*80)
    print("HOMOGENIZATION EXPERIMENT COMPLETE")
    print("="*80)

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """
    Main execution function with multiple example options.
    """
    print("="*80)
    print("ASYMMETRIC BRIDGE STAGGERED IMPLEMENTATION")
    print("Choose an example to run:")
    print("1. Simple RS-IPFO example (basic Gaussian data)")
    print("2. Homogenization example (coarsening process)")
    print("="*80)
    
    try:
        # For automated execution, default to homogenization example
        choice = "2"  # Can be modified or made interactive
        
        if choice == "1":
            run_rs_ipfo_example()
        elif choice == "2":
            run_homogenization_example()
        else:
            print("Invalid choice. Running homogenization example by default.")
            run_homogenization_example()
            
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # Set default dtype and run main
    torch.set_default_dtype(torch.float32)
    main()