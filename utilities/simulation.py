"""
Simulation Utilities for Asymmetric Bridges
===========================================

This module provides functions for simulating the forward and backward
dynamics of the asymmetric bridge models. It includes different SDE solvers
and methods for generating trajectories.
"""

import torch
from torch import Tensor
import math
import traceback
from typing import Dict, Tuple, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from ..asymmetric_bridge_distributions_grf import NeuralGaussianBridge
    from ..asymmetric_bridge_expressive_flow import NeuralBridgeExpressive


# ============================================================================
# SDE Solvers
# ============================================================================

def solve_gaussian_bridge_reverse_sde(
    bridge, # Should be NeuralGaussianBridge
    z_start: Tensor,
    ts: float,
    tf: float,
    n_steps: int
) -> Tensor:
    """
    Specialized solver for the reverse SDE of the Gaussian bridge.
    Uses an exponential integrator with variance matching for stability.
    """
    if ts <= tf:
        raise ValueError("This solver is for backward integration (ts > tf).")

    dt = (tf - ts) / n_steps # dt is negative
    tt = torch.linspace(ts, tf, n_steps + 1)

    path = [z_start.clone()]
    current_z = z_start.clone()
    B = current_z.shape[0]
    sigma = bridge.sigma_reverse
    EPSILON = 1e-9
    device = z_start.device
    
    for i in range(n_steps):
        t, t_next = tt[i], tt[i+1]

        t_tensor = torch.tensor([[float(t)]], device=device, requires_grad=torch.is_grad_enabled())
        t_next_tensor = torch.tensor([[float(t_next)]], device=device, requires_grad=torch.is_grad_enabled())

        with torch.set_grad_enabled(torch.is_grad_enabled()):
            mu_t_scalar, gamma_t_scalar, _, dgamma_dt_scalar = bridge.get_params_and_derivs(t_tensor)
            mu_next_scalar, gamma_next_scalar = bridge.get_params(t_next_tensor)

        mu_t = mu_t_scalar.expand(B, -1)
        gamma_t_raw = gamma_t_scalar.expand(B, -1)
        dgamma_dt = dgamma_dt_scalar.expand(B, -1)
        mu_next = mu_next_scalar.expand(B, -1)
        gamma_next = gamma_next_scalar.expand(B, -1)

        gamma_t_clamped = torch.clamp(gamma_t_raw, min=EPSILON)
        gamma_sq_t_clamped = torch.clamp(gamma_t_raw**2, min=EPSILON**2)

        C_t = (dgamma_dt / gamma_t_clamped) + (sigma**2 / 2) / gamma_sq_t_clamped
        A_t = torch.exp(C_t * dt)

        deviation_t = current_z - mu_t
        deterministic_update = A_t * deviation_t

        variance = gamma_next**2 - (A_t**2) * gamma_t_raw**2
        variance = torch.clamp(variance, min=0.0)
        std_dev = torch.sqrt(variance)

        noise = torch.randn_like(current_z) * std_dev
        current_z = mu_next + deterministic_update + noise
        
        if torch.isnan(current_z).any():
            print(f"Warning: NaN detected in simulation at t={t:.4f}. Stopping.")
            break

        path.append(current_z.clone())

    return torch.stack(path)


def solve_backward_sde_euler_maruyama(
    bridge, # Can be any bridge model
    z_start: Tensor,
    ts: float,
    tf: float,
    n_steps: int
) -> Tensor:
    """
    Generic solver for the backward SDE using the Euler-Maruyama method.
    """
    if ts <= tf:
        raise ValueError("Backward SDE requires ts > tf.")
    
    dt = (tf - ts) / n_steps
    current_z = z_start.clone()
    path = [current_z.clone()]
    
    sigma_backward = bridge.sigma_reverse
    
    for step in range(n_steps):
        t_current = ts + step * dt
        t_tensor = torch.tensor([[t_current]], device=current_z.device, dtype=torch.float32)
        
        # Enable gradients for this step even if we're in a no_grad context
        with torch.enable_grad():
            # Ensure current_z requires grad for score function computation
            current_z = current_z.detach().requires_grad_(True)
            
            drift_backward = bridge.reverse_drift(current_z, t_tensor)
        
        # Update position (detach to break gradients for next iteration)
        noise = torch.randn_like(current_z) * math.sqrt(abs(dt)) * sigma_backward
        current_z = current_z.detach() + drift_backward.detach() * dt + noise
        
        path.append(current_z.clone())
    
    return torch.stack(path)

# ============================================================================
# Sample Generation
# ============================================================================

def generate_backward_samples(
    bridge, 
    marginal_data: Dict[float, Tensor], 
    n_samples: int = 64, 
    n_steps: int = 100, 
    device: str = 'cpu',
    solver: Callable = solve_backward_sde_euler_maruyama,
    z_final_ground_truth: Tensor = None
) -> Dict[float, Tensor]:
    """
    Generate backward samples from the learned bridge, starting from the final time.
    This function is generic and accepts a solver function.
    
    Args:
        bridge: The trained bridge model
        marginal_data: Dictionary of time -> samples for marginal constraints
        n_samples: Number of samples to generate (ignored if z_final_ground_truth is provided)
        n_steps: Number of SDE integration steps
        device: Device to run computations on
        solver: SDE solver function to use
        z_final_ground_truth: Optional tensor of ground truth samples at final time.
                            If provided, these will be used as starting points instead
                            of sampling from the learned distribution.
    
    Returns:
        Dictionary mapping times to generated samples
    """
    if z_final_ground_truth is not None:
        print("  - Generating backward samples from ground truth final data...")
    else:
        print("  - Generating backward samples from learned bridge...")
    
    bridge.eval()
    generated_samples = {}
    T = bridge.T
    
    sorted_times = sorted(marginal_data.keys())
    
    with torch.no_grad():
        final_time = max(sorted_times)
        
        # Start from either ground truth or learned distribution at the final time
        t_final_tensor = torch.tensor([[final_time]], device=device, dtype=torch.float32)
        
        if z_final_ground_truth is not None:
            # Use provided ground truth samples
            z_final = z_final_ground_truth.to(device)
            actual_n_samples = z_final.shape[0]
            print(f"    Starting from {actual_n_samples} ground truth samples at t={final_time}")
        else:
            # Generate initial samples z_T ~ p_T from learned distribution
            if hasattr(bridge, 'get_params'): # Gaussian Bridge
                 mu_final, gamma_final = bridge.get_params(t_final_tensor)
                 z_final = mu_final + gamma_final * torch.randn(n_samples, bridge.data_dim, device=device)
            else: # Expressive Flow Bridge
                epsilon = bridge.base_dist.sample((n_samples,)).to(device)
                z_final, _ = bridge.flow.forward(epsilon, t_final_tensor.expand(n_samples, -1))
            actual_n_samples = n_samples
            print(f"    Starting from {actual_n_samples} generated samples at t={final_time}")

        
        # Solve backward SDE from final time to 0
        print(f"    Integrating backward SDE with {n_steps} steps using {solver.__name__}...")
        try:
            backward_trajectory = solver(
                bridge, z_final, ts=final_time, tf=0.0, n_steps=n_steps
            )
            
            trajectory_times = torch.linspace(final_time, 0.0, n_steps + 1)
            
            for t_val in sorted_times:
                time_idx = torch.argmin(torch.abs(trajectory_times - t_val)).item()
                samples_at_t = backward_trajectory[time_idx].cpu()
                generated_samples[t_val] = samples_at_t
                
        except Exception as e:
            print(f"    Warning: Backward SDE integration failed ({e}).")
            traceback.print_exc()
            # Fallback to forward sampling
            if z_final_ground_truth is None:
                epsilon = bridge.base_dist.sample((actual_n_samples,))
                for t_val in sorted_times:
                    t_tensor = torch.tensor([[t_val]], device=device, dtype=torch.float32)
                    if hasattr(bridge, 'get_params'):
                        mu_t, gamma_t = bridge.get_params(t_tensor)
                        samples = mu_t + gamma_t * epsilon.to(device)
                    else:
                        samples, _ = bridge.flow.forward(epsilon, t_tensor.expand(actual_n_samples, -1))
                    generated_samples[t_val] = samples.cpu()
            else:
                print("    Cannot perform fallback with ground truth data. Returning empty samples.")
                for t_val in sorted_times:
                    generated_samples[t_val] = torch.zeros_like(z_final_ground_truth).cpu()
    
    print(f"    Generated samples for {len(generated_samples)} time points")
    return generated_samples


def generate_comparative_backward_samples(
    bridge, 
    marginal_data: Dict[float, Tensor], 
    n_samples: int = 64, 
    n_steps: int = 100, 
    device: str = 'cpu',
    solver: Callable = solve_backward_sde_euler_maruyama
) -> Tuple[Dict[float, Tensor], Dict[float, Tensor]]:
    """
    Generate backward samples starting from ground truth data at the final time.
    
    This function generates backward trajectories starting from ground truth data,
    which better represents the true macroscale evolution.
    
    Args:
        bridge: The trained bridge model
        marginal_data: Dictionary of time -> samples for marginal constraints
        n_samples: Number of samples to use from ground truth
        n_steps: Number of SDE integration steps
        device: Device to run computations on
        solver: SDE solver function to use
    
    Returns:
        Tuple of (ground_truth_samples, ground_truth_backward_samples) dictionaries
    """
    print("\n--- Generating Backward Samples from Ground Truth ---")
    
    sorted_times = sorted(marginal_data.keys())
    final_time = max(sorted_times)
    
    # Get ground truth samples at final time
    ground_truth_final = marginal_data[final_time]
    
    # Limit to n_samples if we have more
    if ground_truth_final.shape[0] > n_samples:
        indices = torch.randperm(ground_truth_final.shape[0])[:n_samples]
        ground_truth_final_subset = ground_truth_final[indices]
    else:
        ground_truth_final_subset = ground_truth_final
        n_samples = ground_truth_final_subset.shape[0]
    
    print(f"Using {n_samples} samples for backward generation")
    
    # Generate backward samples starting from ground truth final data
    ground_truth_backward = generate_backward_samples(
        bridge, marginal_data, 
        n_samples=n_samples,  # This will be ignored since we provide z_final_ground_truth
        n_steps=n_steps, 
        device=device,
        solver=solver,
        z_final_ground_truth=ground_truth_final_subset
    )
    
    print("--- Backward Sample Generation Complete ---\n")
    
    # Return ground truth data and backward samples
    # We return the original marginal_data as the first element for consistency
    return marginal_data, ground_truth_backward
