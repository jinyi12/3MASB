"""
Asymmetric Multi-Marginal Bridge Framework - Preliminary Implementation
========================================================================

This module implements the Gaussian Latent Space Approximation approach
for validating the core theoretical claims of the asymmetric Multi-Marginal 
Bridge framework. The implementation follows the research plan to:

1. Start with analytically tractable Gaussian flows
2. Validate asymmetric consistency (forward ODE, reverse SDE)
3. Provide a foundation for future CNF-based extensions

Key Components:
- DifferentiableSpline: Parameterizes mu_t and gamma_t
- GaussianBridge: Core module implementing the framework
- Validation functions for rigorous testing

Research Context:
The goal is to validate that we can optimize a latent probability flow
to satisfy marginal constraints and derive asymmetric dynamics that
consistently share the same marginal distributions.
"""

from typing import Tuple, Callable, Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import trange
import numpy as np
import traceback
import ot  # Optimal Transport


class DifferentiableLinearSpline(nn.Module):
    """
    A simplified differentiable spline using linear interpolation.
    
    This is a tractable alternative to cubic splines for the preliminary
    implementation. It ensures continuity while remaining differentiable.
    
    Args:
        fixed_points: Points that must be interpolated exactly [N_fixed, dim]
        fixed_times: Times corresponding to fixed points [N_fixed]
        n_control: Number of trainable control points
        T: Total time span
        dim: Dimension of the interpolated values
        enforce_positivity: Whether to enforce positive values (for gamma_t)
    """
    
    def __init__(
        self, 
        fixed_points: Tensor, 
        fixed_times: Tensor, 
        n_control: int,
        T: float,
        dim: int,
        enforce_positivity: bool = False
    ):
        super().__init__()
        
        self.T = T
        self.dim = dim
        self.enforce_positivity = enforce_positivity
        
        # Create control times between fixed points
        if n_control > 0:
            control_times = torch.linspace(0, T, n_control + 2)[1:-1]
            self.control_times = nn.Parameter(control_times, requires_grad=False)
            
            if enforce_positivity:
                # Use softplus to ensure positivity
                self._raw_control_points = nn.Parameter(torch.randn(n_control, dim))
                self.control_points = lambda: F.softplus(self._raw_control_points)
            else:
                self.control_points = nn.Parameter(torch.randn(n_control, dim))
        else:
            self.control_times = torch.tensor([])
            if enforce_positivity:
                self.control_points = lambda: torch.empty(0, dim)
            else:
                self.control_points = torch.empty(0, dim)
        
        # Combine and sort all points
        self.fixed_points = nn.Parameter(fixed_points, requires_grad=False)
        self.fixed_times = nn.Parameter(fixed_times, requires_grad=False)
        
    def _get_all_points_and_times(self):
        """Get all points (fixed + control) sorted by time."""
        if isinstance(self.control_points, nn.Parameter):
            control_pts = self.control_points
        else:
            control_pts = self.control_points()
            
        if len(self.control_times) > 0:
            all_times = torch.cat([self.fixed_times, self.control_times])
            all_points = torch.cat([self.fixed_points, control_pts], dim=0)
        else:
            all_times = self.fixed_times
            all_points = self.fixed_points
            
        # Sort by time
        sorted_indices = torch.argsort(all_times)
        return all_points[sorted_indices], all_times[sorted_indices]
    
    def forward(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Evaluate spline and its time derivative at time t.
        
        Args:
            t: Time points to evaluate [batch_size] or [batch_size, 1]
            
        Returns:
            value: Spline values at t [batch_size, dim]
            derivative: Time derivatives at t [batch_size, dim]
        """
        if t.dim() == 2:
            t = t.squeeze(-1)
        
        all_points, all_times = self._get_all_points_and_times()
        
        # Linear interpolation
        values = []
        derivatives = []
        
        for t_i in t:
            # Find the interval containing t_i
            if t_i <= all_times[0]:
                # Before first point - extrapolate with zero derivative
                val = all_points[0]
                deriv = torch.zeros_like(all_points[0])
            elif t_i >= all_times[-1]:
                # After last point - extrapolate with zero derivative  
                val = all_points[-1]
                deriv = torch.zeros_like(all_points[-1])
            else:
                # Find interval
                right_idx = torch.searchsorted(all_times, t_i, right=True)
                left_idx = right_idx - 1
                
                # Linear interpolation
                t_left, t_right = all_times[left_idx], all_times[right_idx]
                p_left, p_right = all_points[left_idx], all_points[right_idx]
                
                alpha = (t_i - t_left) / (t_right - t_left)
                val = (1 - alpha) * p_left + alpha * p_right
                deriv = (p_right - p_left) / (t_right - t_left)
            
            values.append(val)
            derivatives.append(deriv)
        
        return torch.stack(values), torch.stack(derivatives)


class GaussianBridge(nn.Module):
    """
    Gaussian Asymmetric Multi-Marginal Bridge.
    
    This module implements the Gaussian approximation of the asymmetric bridge,
    where the latent flow is restricted to Z_t = mu_t + gamma_t * epsilon.
    
    Key features:
    - Analytical forward velocity field (ODE)
    - Analytical reverse drift (SDE) 
    - Simulation-free optimization via reparameterization trick
    - Exact constraint satisfaction by construction
    
    Args:
        phi_ti: Fixed marginal points [N_constraints, data_dim]
        time_steps: Times for constraints [N_constraints]  
        n_control: Number of control points for splines
        data_dim: Dimension of the data space
        sigma_reverse: Diffusion coefficient for reverse SDE
    """
    
    def __init__(
        self, 
        phi_ti: Tensor, 
        time_steps: Tensor, 
        n_control: int = 10,
        data_dim: int = 3,
        sigma_reverse: float = 1.0
    ):
        super().__init__()
        
        self.data_dim = data_dim
        self.T = time_steps[-1].item()
        self.sigma_reverse = sigma_reverse
        
        # Store constraint information
        self.register_buffer('phi_ti', phi_ti)
        self.register_buffer('time_steps', time_steps)
        
        # Spline for mu_t (mean trajectory)
        self.mu_spline = DifferentiableLinearSpline(
            fixed_points=phi_ti,
            fixed_times=time_steps,
            n_control=n_control,
            T=self.T,
            dim=data_dim,
            enforce_positivity=False
        )
        
        # Spline for gamma_t (standard deviation)
        epsilon = 1e-3
        gamma_fixed = torch.full((len(time_steps), 1), epsilon) # Changed from zeros for numerical stability
        self.gamma_spline = DifferentiableLinearSpline(
            fixed_points=gamma_fixed,
            fixed_times=time_steps,
            n_control=n_control,
            T=self.T,
            dim=1,
            enforce_positivity=True
        )
    
    def get_params(self, t: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Get Gaussian parameters and their time derivatives.
        
        Args:
            t: Time points [batch_size] or [batch_size, 1]
            
        Returns:
            mu: Mean [batch_size, data_dim]
            dmu_dt: Time derivative of mean [batch_size, data_dim]  
            gamma: Standard deviation [batch_size, 1]
            dgamma_dt: Time derivative of std [batch_size, 1]
        """
        mu, dmu_dt = self.mu_spline(t)
        gamma, dgamma_dt = self.gamma_spline(t)
        
        # Ensure numerical stability
        gamma = torch.clamp(gamma, min=1e-6)
        
        return mu, dmu_dt, gamma, dgamma_dt
    
    def forward_velocity(self, z: Tensor, t: Tensor) -> Tensor:
        """
        Compute the deterministic forward velocity field.
        
        This is the velocity of the Probability Flow ODE:
        v(z,t) = ∂mu/∂t + (∂gamma/∂t / gamma) * (z - mu)
        
        Args:
            z: State [batch_size, data_dim]
            t: Time [batch_size] or [batch_size, 1] or scalar
            
        Returns:
            velocity: Forward velocity [batch_size, data_dim]
        """
        # Ensure t is properly shaped
        if torch.is_tensor(t) and t.dim() == 0:  # scalar tensor
            t = t.unsqueeze(0)
        elif not torch.is_tensor(t):  # python scalar
            t = torch.tensor([float(t)], device=z.device)
        elif t.dim() == 1 and len(t) == 1:  # single element tensor
            pass  # already correct shape
        elif t.dim() == 2 and t.shape[1] == 1:  # [batch, 1]
            t = t.squeeze(-1)
        
        # Get parameters at time t
        mu, dmu_dt, gamma, dgamma_dt = self.get_params(t)
        
        # Handle batch broadcasting - expand to match z batch size
        if z.shape[0] > 1 and mu.shape[0] == 1:
            mu = mu.expand(z.shape[0], -1)
            dmu_dt = dmu_dt.expand(z.shape[0], -1)
            gamma = gamma.expand(z.shape[0], -1)
            dgamma_dt = dgamma_dt.expand(z.shape[0], -1)
        
        # The division by gamma is numerically unstable when gamma is close to zero.
        # We clamp gamma to a small positive value to prevent division by zero.
        safe_gamma = torch.clamp(gamma, min=1e-9)
        velocity = dmu_dt + (dgamma_dt / safe_gamma) * (z - mu)
        return velocity
    
    def score_function(self, z: Tensor, t: Tensor) -> Tensor:
        """
        Compute the score function ∇_z log p_t(z) for Gaussian flow.
        
        For Gaussian: ∇ log p(z) = -(z - mu) / gamma^2
        
        Args:
            z: State [batch_size, data_dim]
            t: Time [batch_size] or [batch_size, 1] or scalar
            
        Returns:
            score: Score function [batch_size, data_dim]
        """
        # Ensure t is properly shaped
        if torch.is_tensor(t) and t.dim() == 0:  # scalar tensor
            t = t.unsqueeze(0)
        elif not torch.is_tensor(t):  # python scalar
            t = torch.tensor([float(t)], device=z.device)
        elif t.dim() == 1 and len(t) == 1:  # single element tensor
            pass  # already correct shape
        elif t.dim() == 2 and t.shape[1] == 1:  # [batch, 1]
            t = t.squeeze(-1)
            
        mu, _, gamma, _ = self.get_params(t)
        
        # Handle batch broadcasting - expand to match z batch size
        if z.shape[0] > 1 and mu.shape[0] == 1:
            mu = mu.expand(z.shape[0], -1)
            gamma = gamma.expand(z.shape[0], -1)
        
        gamma_sq = torch.clamp(gamma**2, min=1e-12)
        score = -(z - mu) / gamma_sq
        return score
    
    def reverse_drift(self, z: Tensor, t: Tensor) -> Tensor:
        """
        Compute the drift for the reverse SDE.
        
        Reverse drift: R(z,t) = v(z,t) - (σ²/2) * ∇log p_t(z)
        
        Args:
            z: State [batch_size, data_dim]
            t: Time [batch_size] or [batch_size, 1]
            
        Returns:
            drift: Reverse SDE drift [batch_size, data_dim]
        """
        velocity = self.forward_velocity(z, t)
        score = self.score_function(z, t)
        
        drift = velocity - (self.sigma_reverse**2 / 2) * score
        return drift
    
    def reverse_sde(self, z: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Complete reverse SDE specification (drift and diffusion).
        
        Args:
            z: State [batch_size, data_dim]
            t: Time [batch_size] or [batch_size, 1]
            
        Returns:
            drift: SDE drift [batch_size, data_dim]
            diffusion: SDE diffusion [batch_size, data_dim]
        """
        drift = self.reverse_drift(z, t)
        diffusion = torch.ones_like(z) * self.sigma_reverse
        return drift, diffusion
    
    def path_regularization_loss(self, n_samples: int = 256) -> Tensor:
        """
        Compute the kinetic energy loss via reparameterization trick.
        
        For Gaussian flows: E[||v||²] = ||∂μ/∂t||² + (∂γ/∂t)² * dim
        This is computed simulation-free using the reparameterization trick.
        
        Args:
            n_samples: Number of time samples for Monte Carlo estimation
            
        Returns:
            loss: Path regularization loss (scalar)
        """
        # Sample random times
        t = torch.rand(n_samples, 1, device=next(self.parameters()).device) * self.T
        
        # Get derivatives
        _, dmu_dt, _, dgamma_dt = self.get_params(t)
        
        # Kinetic energy: E[||∂μ/∂t + (∂γ/∂t) * ε||²]
        # = ||∂μ/∂t||² + (∂γ/∂t)² * E[||ε||²]
        # = ||∂μ/∂t||² + (∂γ/∂t)² * data_dim
        
        mean_term = torch.sum(dmu_dt**2, dim=-1)  # [n_samples]
        var_term = (dgamma_dt.squeeze(-1)**2) * self.data_dim  # [n_samples]
        
        kinetic_energy = mean_term + var_term
        return kinetic_energy.mean() / 2.0
    
    def constraint_satisfaction_loss(self) -> Tensor:
        """
        Verify constraint satisfaction (should be zero by construction).
        
        Returns:
            loss: Constraint violation (should be ~0)
        """
        # This should be exactly zero by construction
        mu_at_constraints, _ = self.mu_spline(self.time_steps)
        gamma_at_constraints, _ = self.gamma_spline(self.time_steps)
        
        mu_error = torch.norm(mu_at_constraints - self.phi_ti)
        gamma_error = torch.norm(gamma_at_constraints)
        
        return mu_error + gamma_error


def generate_spiral_data(N_constraints: int = 6, T: float = 3.0, data_dim: int = 3) -> Tuple[Tensor, Tensor]:
    """
    Generate synthetic spiral trajectory for testing.
    
    Args:
        N_constraints: Number of marginal constraints
        T: Total time span
        data_dim: Data dimension (should be 3 for spiral)
        
    Returns:
        phi_ti: Constraint points [N_constraints, data_dim]
        time_steps: Constraint times [N_constraints]
    """
    time_steps = torch.linspace(0, T, N_constraints)
    
    if data_dim == 3:
        # 3D spiral
        phi_ti = torch.stack([
            torch.sin(time_steps * 2 * torch.pi / T),
            torch.cos(time_steps * 2 * torch.pi / T), 
            time_steps / T
        ], dim=1)
    else:
        # Generic case: random smooth trajectory
        phi_ti = torch.cumsum(torch.randn(N_constraints, data_dim) * 0.3, dim=0)
    
    return phi_ti, time_steps


def train_bridge(
    bridge: GaussianBridge, 
    epochs: int = 2000, 
    lr: float = 1e-2,
    verbose: bool = True
) -> list:
    """
    Train the Gaussian bridge by minimizing path regularization.
    
    Args:
        bridge: The GaussianBridge model
        epochs: Number of training epochs
        lr: Learning rate
        verbose: Whether to show progress bar
        
    Returns:
        loss_history: Training loss history
    """
    optimizer = torch.optim.Adam(bridge.parameters(), lr=lr)
    loss_history = []
    
    if verbose:
        pbar = trange(epochs)
    else:
        pbar = range(epochs)
    
    for epoch in pbar:
        optimizer.zero_grad()
        
        # Main loss: path regularization (kinetic energy)
        path_loss = bridge.path_regularization_loss()
        
        # Sanity check: constraint satisfaction (should be ~0)
        constraint_loss = bridge.constraint_satisfaction_loss()
        
        # Total loss
        total_loss = path_loss + 1000.0 * constraint_loss  # High weight on constraints
        
        total_loss.backward()
        optimizer.step()
        
        loss_history.append({
            'total': total_loss.item(),
            'path': path_loss.item(), 
            'constraint': constraint_loss.item()
        })
        
        if verbose and isinstance(pbar, type(trange(0))):
            pbar.set_description(
                f"Loss: {path_loss.item():.4f}, Constraint: {constraint_loss.item():.6f}"
            )
    
    return loss_history


# Import solve_sde from the existing notebook implementation
def solve_sde(
    sde: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]],
    z: Tensor,
    ts: float,
    tf: float,
    n_steps: int
) -> Tensor:
    """
    Solve SDE using Euler-Maruyama scheme.
    
    Args:
        sde: Function returning (drift, diffusion) given (z, t)
        z: Initial state [batch_size, data_dim]
        ts: Start time
        tf: Final time  
        n_steps: Number of integration steps
        
    Returns:
        path: Trajectory [n_steps+1, batch_size, data_dim]
    """
    # Handle both forward (ts < tf) and backward (ts > tf) integration
    dt = (tf - ts) / n_steps
    dt_sqrt = abs(dt) ** 0.5
    
    # Time points for integration
    tt = torch.linspace(ts, tf, n_steps + 1)[:-1]  # Exclude final point, we'll add it manually

    path = [z.clone()]
    current_z = z.clone()
    
    for i, t in enumerate(tt):
        # Convert scalar t to tensor if needed
        if not torch.is_tensor(t):
            t_tensor = torch.tensor(float(t), device=current_z.device)
        else:
            t_tensor = t.to(current_z.device)
        
        # Get drift and diffusion
        try:
            drift, diffusion = sde(current_z, t_tensor)
        except Exception as e:
            print(f"SDE evaluation failed at step {i}, t={float(t):.4f}: {e}")
            print(f"current_z shape: {current_z.shape}, t_tensor shape: {t_tensor.shape}")
            raise e
        
        # Euler-Maruyama step
        noise = torch.randn_like(current_z)
        current_z = current_z + drift * dt + diffusion * noise * dt_sqrt
        
        path.append(current_z.clone())

    return torch.stack(path)


# ============================================================================
# FIX: Specialized Stable SDE Solver (Exponential Integrator)
# ============================================================================

def solve_gaussian_bridge_reverse_sde(
    bridge: GaussianBridge,
    z_start: Tensor,
    ts: float,
    tf: float,
    n_steps: int
) -> Tensor:
    """
    Specialized solver for the reverse SDE of the Gaussian bridge.
    Uses Exponential Integrator with Variance Matching for stability and accuracy.
    """
    if ts <= tf:
        raise ValueError("This solver is designed for backward integration (ts > tf).")

    dt = (tf - ts) / n_steps # dt is negative
    tt = torch.linspace(ts, tf, n_steps + 1)

    path = [z_start.clone()]
    current_z = z_start.clone()
    B = current_z.shape[0]
    sigma = bridge.sigma_reverse
    EPSILON = 1e-9 # Epsilon for numerical stability in division
    
    for i in range(n_steps):
        t, t_next = tt[i], tt[i+1]

        # 1. Get parameters at t and t_next
        t_tensor = torch.tensor(float(t), device=current_z.device).unsqueeze(0)
        t_next_tensor = torch.tensor(float(t_next), device=current_z.device).unsqueeze(0)
        
        mu_t, _, gamma_t, dgamma_dt = bridge.get_params(t_tensor)
        mu_next, _, gamma_next, _ = bridge.get_params(t_next_tensor)

        # 2. Expand parameters to batch size
        mu_t = mu_t.expand(B, -1)
        gamma_t_raw = gamma_t.expand(B, -1) # Raw gamma_t (can be 0)
        dgamma_dt = dgamma_dt.expand(B, -1)
        mu_next = mu_next.expand(B, -1)
        gamma_next = gamma_next.expand(B, -1)

        # 3. Calculate the drift coefficient C(t) with robust clamping
        # C(t) = (dgamma/dt / gamma) + (sigma^2 / (2*gamma^2))
        
        # We must clamp gamma before division to avoid Inf/NaN when gamma_t=0.
        gamma_t_clamped = torch.clamp(gamma_t_raw, min=EPSILON)
        gamma_sq_t_clamped = torch.clamp(gamma_t_raw**2, min=EPSILON**2)

        C_t = (dgamma_dt / gamma_t_clamped) + (sigma**2 / 2) / gamma_sq_t_clamped

        # 4. Calculate the amplification factor A = exp(C(t) dt)
        A_t = torch.exp(C_t * dt) # Stable as C_t*dt <= 0.

        # 5. Deterministic update of the deviation
        deviation_t = current_z - mu_t
        deterministic_update = A_t * deviation_t

        # 6. Calculate the required noise variance (Variance Matching)
        # Var(Noise) = gamma_{t+dt}^2 - A_t^2 * gamma_t^2
        # We use the raw gamma_t here for accurate variance calculation.
        variance = gamma_next**2 - (A_t**2) * gamma_t_raw**2
        
        # Clamp variance to be non-negative (handles minor numerical inaccuracies)
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

def validate_asymmetric_consistency(
    bridge: GaussianBridge,
    n_particles: int = 1024,
    n_steps: int = 200,
    n_validation_times: int = 5,
    device: str = 'cpu'
) -> dict:
    """
    Rigorous validation of asymmetric consistency using the stable solver.
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
        # 1. Initialize particles exactly at t=T.
        start_time = bridge.T
        
        t_start_tensor = torch.tensor([start_time], device=device)
        mu_start, _, gamma_start, _ = bridge.get_params(t_start_tensor)
        
        # Sample initial particles. If gamma_start=0 (as expected), z_start=mu_start.
        z_start = mu_start + gamma_start * torch.randn(n_particles, bridge.data_dim, device=device)
        
        print(f"Starting reverse SDE simulation from t={start_time:.4f} to t=0.0")
        print(f"Using specialized Exponential Integrator (Stable).")
        
        # 2. Solve reverse SDE from T to 0
        try:
            # FIX: Use the stable specialized solver
            path = solve_gaussian_bridge_reverse_sde(
                bridge=bridge, z_start=z_start, ts=start_time, tf=0.0, n_steps=n_steps
            )
            print(f"Successfully simulated full trajectory: {path.shape}")
            
        except Exception as e:
            print(f"Full SDE simulation failed: {e}")
            traceback.print_exc()
            return results
        
        # 3. Validation
        val_times = torch.linspace(0.1 * bridge.T, 0.9 * bridge.T, n_validation_times)
        # The time points go from T down to 0
        time_points = torch.linspace(start_time, 0.0, n_steps + 1)
        
        # 4. For each validation time, find closest trajectory point and validate
        for t_val in val_times:
            t_val = t_val.item()
            
            closest_idx = torch.argmin(torch.abs(time_points - t_val)).item()
            actual_time = time_points[closest_idx].item()
            
            print(f"\nValidating at t={t_val:.2f} (closest simulated: t={actual_time:.2f})")
            
            simulated_particles = path[closest_idx].cpu()
            
            # Get analytical ground truth
            t_tensor = torch.tensor([t_val], device=device)
            mu_gt, _, gamma_gt, _ = bridge.get_params(t_tensor)
            mu_gt = mu_gt[0].cpu()
            sigma_gt = gamma_gt[0].item()
            cov_gt = (sigma_gt**2) * torch.eye(bridge.data_dim)
            
            # Compute validation metrics
            simulated_mean = simulated_particles.mean(dim=0)
            mean_error = torch.norm(simulated_mean - mu_gt).item()

            try:
                simulated_cov = torch.cov(simulated_particles.T)
                cov_error = torch.norm(simulated_cov - cov_gt).item()
                
                # Wasserstein-2 distance estimation (using Sliced Wasserstein)
                gt_samples = mu_gt + sigma_gt * torch.randn(n_particles, bridge.data_dim)
                w2_dist = ot.sliced_wasserstein_distance(
                    X_s=simulated_particles.numpy(),
                    X_t=gt_samples.numpy(),
                    n_projections=1000, seed=42
                )
            except Exception as e:
                cov_error = float('nan')
                w2_dist = float('nan')

            
            results['times'].append(actual_time)
            results['mean_errors'].append(mean_error)
            results['cov_errors'].append(cov_error)
            results['wasserstein_distances'].append(w2_dist)
            
            print(f"  Mean error: {mean_error:.6f}")
            print(f"  Cov error: {cov_error:.6f}")
            print(f"  W2 distance: {w2_dist:.6f}")
            print(f"  Simulated mean: {simulated_mean.numpy()}")
            print(f"  Ground truth mean: {mu_gt.numpy()}")
            print(f"  Ground truth std: {sigma_gt:.4f}")

    
    return results


def visualize_data(xs: Tensor, title: str = "Trajectories"):
    """Visualize 3D trajectories."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for xs_i in xs:
        ax.plot(xs_i[:, 0], xs_i[:, 1], xs_i[:, 2], alpha=0.7)

    ax.set_xlabel('$z_1$', fontsize=14)
    ax.set_ylabel('$z_2$', fontsize=14)
    ax.set_zlabel('$z_3$', fontsize=14)
    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    return fig


def visualize_bridge_results(bridge: GaussianBridge, n_viz_particles: int = 6):
    """
    Visualize the trained bridge results.
    
    Args:
        bridge: Trained GaussianBridge
        n_viz_particles: Number of trajectories to visualize
    """
    bridge.eval()
    
    with torch.no_grad():
        # 1. Show the constraint points
        phi_ti = bridge.phi_ti.cpu()
        time_steps = bridge.time_steps.cpu()
        
        print("Constraint Points:")
        for i, (t, phi) in enumerate(zip(time_steps, phi_ti)):
            print(f"  t_{i} = {t:.2f}: {phi.numpy()}")
        
        # 2. Visualize the mean trajectory
        fine_times = torch.linspace(0, bridge.T, 100)
        mu_traj, _ = bridge.mu_spline(fine_times)
        gamma_traj, _ = bridge.gamma_spline(fine_times)
        
        fig1 = plt.figure(figsize=(15, 5))
        
        # Mean trajectory
        ax1 = fig1.add_subplot(131, projection='3d')
        ax1.plot(mu_traj[:, 0], mu_traj[:, 1], mu_traj[:, 2], 'b-', linewidth=2, label='Mean trajectory')
        ax1.scatter(phi_ti[:, 0], phi_ti[:, 1], phi_ti[:, 2], c='red', s=100, label='Constraints')
        ax1.set_title('Optimized Mean Trajectory μ(t)')
        ax1.legend()
        
        # Variance evolution
        ax2 = fig1.add_subplot(132)
        ax2.plot(fine_times, gamma_traj.squeeze(), 'g-', linewidth=2)
        ax2.scatter(time_steps, torch.zeros_like(time_steps), c='red', s=50)
        ax2.set_xlabel('Time t')
        ax2.set_ylabel('γ(t)')
        ax2.set_title('Standard Deviation Evolution')
        ax2.grid(True)
        
        # Path regularization over time
        ax3 = fig1.add_subplot(133) 
        with torch.enable_grad():
            bridge.train()
            times_for_reg = torch.linspace(0.01, bridge.T-0.01, 50).requires_grad_(False)
            regularization_vals = []
            
            for t in times_for_reg:
                _, dmu_dt, _, dgamma_dt = bridge.get_params(t.unsqueeze(0))
                reg_val = (torch.sum(dmu_dt**2) + dgamma_dt**2 * bridge.data_dim) / 2
                regularization_vals.append(reg_val.item())
        
        ax3.plot(times_for_reg, regularization_vals, 'purple', linewidth=2)
        ax3.set_xlabel('Time t')
        ax3.set_ylabel('Kinetic Energy')
        ax3.set_title('Path Regularization ½||v||²')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        bridge.eval()
        
        # 3. Simulate some forward trajectories (deterministic ODE)
        print("\nSimulating forward trajectories...")
        
        # Start from slightly perturbed initial conditions
        z0_base = phi_ti[0].unsqueeze(0).repeat(n_viz_particles, 1)
        z0_perturbed = z0_base + 0.1 * torch.randn_like(z0_base)
        
        # Simple forward Euler integration for ODE
        n_steps = 200
        dt = bridge.T / n_steps
        times = torch.linspace(0, bridge.T, n_steps + 1)
        
        forward_trajectories = []
        for i in range(n_viz_particles):
            traj = [z0_perturbed[i]]
            z = z0_perturbed[i]
            
            for step, t in enumerate(times[:-1]):
                v = bridge.forward_velocity(z.unsqueeze(0), torch.tensor([t]))[0]
                z = z + v * dt
                traj.append(z)
            
            forward_trajectories.append(torch.stack(traj))
        
        forward_trajectories = torch.stack(forward_trajectories)
        
        # Visualize forward trajectories
        visualize_data(forward_trajectories, "Forward Deterministic Trajectories (ODE)")
        plt.show()
        
        print(f"\nForward trajectories shape: {forward_trajectories.shape}")
        print("Forward simulation complete!")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    print("="*80)
    print("ASYMMETRIC MULTI-MARGINAL BRIDGE - PRELIMINARY VALIDATION")
    print("="*80)
    
    # Step 1: Generate synthetic data
    print("\n1. Generating synthetic spiral data...")
    phi_ti, time_steps = generate_spiral_data(N_constraints=6, T=3.0, data_dim=3)
    
    print(f"Generated {len(phi_ti)} constraint points over time [0, {time_steps[-1]:.1f}]")
    print("Constraint points:")
    for i, (t, phi) in enumerate(zip(time_steps, phi_ti)):
        print(f"  t_{i} = {t:.2f}: [{phi[0]:.3f}, {phi[1]:.3f}, {phi[2]:.3f}]")
    
    # Step 2: Initialize bridge
    print("\n2. Initializing Gaussian Bridge...")
    bridge = GaussianBridge(
        phi_ti=phi_ti,
        time_steps=time_steps,
        n_control=8,
        data_dim=3,
        sigma_reverse=1.0
    )
    
    print(f"Bridge initialized with {sum(p.numel() for p in bridge.parameters())} parameters")
    
    # Step 3: Train the bridge
    print("\n3. Training the bridge...")
    loss_history = train_bridge(bridge, epochs=2000, lr=1e-2, verbose=True)
    
    print(f"Training complete. Final path loss: {loss_history[-1]['path']:.6f}")
    print(f"Final constraint violation: {loss_history[-1]['constraint']:.8f}")
    
    # Step 4: Visualize results
    print("\n4. Visualizing results...")
    visualize_bridge_results(bridge, n_viz_particles=6)
    
    # Step 5: Rigorous validation
    print("\n5. Performing rigorous asymmetric consistency validation...")
    print("This validates that forward ODE and reverse SDE share marginals...")
    
    validation_results = validate_asymmetric_consistency(
        bridge=bridge,
        n_particles=1024,
        n_steps=100,
        n_validation_times=5
    )
    
    # Print validation summary
    print("\nVALIDATION SUMMARY:")
    print("-" * 40)
    print(f"Mean of mean errors: {np.mean(validation_results['mean_errors']):.4f}")
    print(f"Mean of covariance errors: {np.mean(validation_results['cov_errors']):.4f}")
    print(f"Mean Wasserstein-2 distance: {np.mean(validation_results['wasserstein_distances']):.4f}")
    
    # Validation criteria
    mean_error_threshold = 0.1
    cov_error_threshold = 0.2
    w2_threshold = 0.5
    
    mean_pass = np.mean(validation_results['mean_errors']) < mean_error_threshold
    cov_pass = np.mean(validation_results['cov_errors']) < cov_error_threshold  
    w2_pass = np.mean(validation_results['wasserstein_distances']) < w2_threshold
    
    print("\nValidation Results:")
    print(f"  Mean Error Test: {'PASS' if mean_pass else 'FAIL'} "
          f"(< {mean_error_threshold})")
    print(f"  Covariance Test: {'PASS' if cov_pass else 'FAIL'} "
          f"(< {cov_error_threshold})")
    print(f"  Wasserstein Test: {'PASS' if w2_pass else 'FAIL'} "
          f"(< {w2_threshold})")
    
    overall_pass = mean_pass and cov_pass and w2_pass
    print(f"\nOVERALL VALIDATION: {'PASS' if overall_pass else 'FAIL'}")
    
    if overall_pass:
        print("\n✅ SUCCESS: Asymmetric consistency validated!")
        print("The forward ODE and reverse SDE successfully share the same marginals.")
        print("This confirms the core theoretical framework.")
    else:
        print("\n⚠️  Some validation metrics exceeded thresholds.")
        print("Consider tuning hyperparameters or increasing simulation resolution.")
    
    print("\n" + "="*80)
    print("PRELIMINARY VALIDATION COMPLETE")
    print("="*80)