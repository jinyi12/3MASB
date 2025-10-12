"""
Validation Utilities for Asymmetric Bridges
===========================================

This module provides functions for validating the asymmetric bridge models,
including quantitative metrics, asymmetric consistency validation, and
statistical comparisons.
"""

import torch
from torch import Tensor
from torch import distributions as D
import torch.fft as fft
import math
import traceback
from typing import Dict, Any, Tuple

# Try to import optional transport library
try:
    import ot
except ImportError:
    print("Warning: Python Optimal Transport (POT) library not found. Wasserstein distance validation will be unavailable.")
    ot = None

from .simulation import solve_gaussian_bridge_reverse_sde


# ============================================================================
# Spatial Correlation Analysis
# ============================================================================

def compute_spatial_acf_2d(samples: Tensor, resolution: int) -> Tensor:
    """
    Computes the normalized spatial ACF for a batch of 2D fields using FFT (Wiener-Khinchin).
    Assumes stationarity, periodicity, and ergodicity.
    """
    B = samples.shape[0]
    fields = samples.reshape(B, resolution, resolution)

    # 1. Center the data (remove spatial mean per sample)
    spatial_mean = fields.mean(dim=[1, 2], keepdim=True)
    centered_fields = fields - spatial_mean

    # 2. Compute Power Spectral Density (PSD)
    freq_domain = fft.fft2(centered_fields, norm="ortho")
    psd = torch.abs(freq_domain)**2

    # 3. Compute Autocovariance (ACVF) = IFFT(PSD)
    autocovariance = fft.ifft2(psd, norm="ortho").real

    # 4. Average over the batch to get the ensemble estimate
    avg_autocovariance = autocovariance.mean(dim=0)

    # 5. Normalize by the variance (zero lag, index [0, 0])
    variance = avg_autocovariance[0, 0]
    if variance < 1e-9:
        return torch.zeros_like(avg_autocovariance)
    
    normalized_acf = avg_autocovariance / torch.clamp(variance, min=1e-9)
    
    # Clamp for numerical stability
    normalized_acf = torch.clamp(normalized_acf, min=-1.0, max=1.0)

    # 6. Shift zero lag to center for visualization
    return fft.fftshift(normalized_acf)


def radial_average(data_2d: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Compute the radial average of a centered 2D array (e.g., ACF) using torch.bincount.
    """
    H, W = data_2d.shape
    center_y, center_x = H // 2, W // 2
    
    # Create coordinate grids
    y, x = torch.meshgrid(torch.arange(H, device=data_2d.device), 
                          torch.arange(W, device=data_2d.device), indexing='ij')
    
    # Calculate radial distance from center
    r = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Binning by integer radius
    r_int = r.round().long()
    max_radius = min(center_x, center_y)
    
    # Flatten arrays for accumulation
    r_flat = r_int.flatten()
    data_flat = data_2d.flatten()
    
    # Use torch.bincount for efficient averaging
    sum_in_bin = torch.bincount(r_flat, weights=data_flat)
    count_in_bin = torch.bincount(r_flat)
    
    # Calculate average
    avg_in_bin = sum_in_bin / torch.clamp(count_in_bin, min=1)
    
    # Truncate to the meaningful radius
    radii = torch.arange(max_radius + 1, device=data_2d.device)
    radial_profile = avg_in_bin[:max_radius + 1]
    
    return radii, radial_profile


# ============================================================================
# Statistical Covariance Analysis Utilities
# ============================================================================

def compute_sample_covariance_matrix(samples: Tensor) -> Tensor:
    """
    Computes the full DxD sample covariance matrix for a batch of samples.
    Args:
        samples: Tensor of shape [N, D] (N samples, D dimensions).
    Returns:
        Covariance matrix of shape [D, D].
    """
    N, D = samples.shape
    if N <= 1:
        print(f"Warning: Insufficient samples (N={N}) to compute covariance. Returning zero matrix.")
        return torch.zeros(D, D, device=samples.device)
    try:
        covariance_matrix = torch.cov(samples.T)
    except Exception as e:
        print(f"Warning: Covariance matrix computation failed: {e}")
        return torch.zeros(D, D, device=samples.device)
    return covariance_matrix


def relative_covariance_frobenius_distance(cov_target: Tensor, cov_gen: Tensor) -> float:
    """
    Computes the Relative Frobenius distance between two covariance matrices.
    L_rel = || Cov_target - Cov_gen ||_F / || Cov_target ||_F
    """
    if cov_target.shape != cov_gen.shape:
        raise ValueError("Covariance matrices must have the same shape.")
    target_norm = torch.norm(cov_target, p='fro')
    diff_norm = torch.norm(cov_target - cov_gen, p='fro')
    if target_norm < 1e-9:
        return 0.0 if diff_norm < 1e-9 else float('inf')
    return (diff_norm / target_norm).item()


def compute_sample_correlation_matrix(samples: Tensor) -> Tensor:
    """
    Computes the full DxD sample correlation matrix for a batch of samples.
    Args:
        samples: Tensor of shape [N, D] (N samples, D dimensions).
    Returns:
        Correlation matrix of shape [D, D].
    """
    N, D = samples.shape
    if N <= 1:
        print(f"Warning: Insufficient samples (N={N}) to compute correlation. Returning identity matrix.")
        return torch.eye(D, device=samples.device)
    try:
        # Compute covariance matrix first
        cov_matrix = torch.cov(samples.T)
        # Extract standard deviations
        std_devs = torch.sqrt(torch.diag(cov_matrix))
        # Handle zero standard deviations
        std_devs = torch.clamp(std_devs, min=1e-9)
        # Compute correlation matrix: Corr = D^(-1/2) * Cov * D^(-1/2)
        inv_std = 1.0 / std_devs
        correlation_matrix = cov_matrix * inv_std.unsqueeze(0) * inv_std.unsqueeze(1)
        # Clamp to valid correlation range
        correlation_matrix = torch.clamp(correlation_matrix, min=-1.0, max=1.0)
    except Exception as e:
        print(f"Warning: Correlation matrix computation failed: {e}")
        return torch.eye(D, device=samples.device)
    return correlation_matrix


def relative_correlation_frobenius_distance(corr_target: Tensor, corr_gen: Tensor) -> float:
    """
    Computes the Relative Frobenius distance between two correlation matrices.
    L_rel = || Corr_target - Corr_gen ||_F / || Corr_target ||_F
    """
    if corr_target.shape != corr_gen.shape:
        raise ValueError("Correlation matrices must have the same shape.")
    target_norm = torch.norm(corr_target, p='fro')
    diff_norm = torch.norm(corr_target - corr_gen, p='fro')
    if target_norm < 1e-9:
        return 0.0 if diff_norm < 1e-9 else float('inf')
    return (diff_norm / target_norm).item()


# ============================================================================
# Quantitative Validation Metrics
# ============================================================================

def calculate_validation_metrics(marginal_data: Dict[float, Tensor], generated_samples: Dict[float, Tensor]) -> Dict[str, Any]:
    """Calculate quantitative validation metrics (W2, MSE_ACF, Covariance metrics)."""
    print("  - Calculating quantitative validation metrics (W2, MSE_ACF)...")
    
    sorted_times = sorted(marginal_data.keys())
    metrics = {
        'times': sorted_times,
        'w2_distances': [],
        'mse_acf': [],
        'rel_fro_cov': [],  # Now contains correlation metrics but keeping same key for compatibility
    }

    # Determine resolution
    if not sorted_times:
        return metrics
    data_dim = marginal_data[sorted_times[0]].shape[1]
    resolution = int(math.sqrt(data_dim))
    is_square = (resolution * resolution == data_dim)
    
    for t_val in sorted_times:
        target_data_cpu = marginal_data[t_val].cpu()
        gen_data_cpu = generated_samples[t_val].cpu()
        
        # 1. Wasserstein Distance (Sliced)
        w2_dist = float('nan')
        if ot:
            try:
                # Ensure same number of samples
                min_samples = min(target_data_cpu.shape[0], gen_data_cpu.shape[0])
                # Use numpy arrays for POT library
                w2_dist = ot.sliced_wasserstein_distance(
                    target_data_cpu[:min_samples].numpy(), 
                    gen_data_cpu[:min_samples].numpy(), 
                    n_projections=1000, seed=42
                )
            except Exception as e:
                print(f"    Warning: W2 calculation failed at t={t_val:.2f}: {e}")
        metrics['w2_distances'].append(w2_dist)

        # 2. MSE ACF (only for square 2D fields)
        mse_acf = float('nan')
        if is_square:
            try:
                acf_target = compute_spatial_acf_2d(target_data_cpu, resolution)
                acf_gen = compute_spatial_acf_2d(gen_data_cpu, resolution)
                mse_acf = torch.mean((acf_target - acf_gen)**2).item()
            except Exception as e:
                print(f"    Warning: MSE ACF calculation failed at t={t_val:.2f}: {e}")
        metrics['mse_acf'].append(mse_acf)

        # 3. Full correlation comparison metrics (works for any D)
        try:
            corr_data = compute_sample_correlation_matrix(target_data_cpu)
            corr_gen = compute_sample_correlation_matrix(gen_data_cpu)
            rel_f = relative_correlation_frobenius_distance(corr_data, corr_gen)
            metrics['rel_fro_cov'].append(rel_f)  # Note: keeping same key for compatibility

        except Exception as e:
            print(f"    Warning: Correlation metric calculation failed at t={t_val:.2f}: {e}")
            metrics['rel_fro_cov'].append(float('nan'))
        print(f"    t={t_val:.2f}: W2={w2_dist:.4f}, MSE_ACF={mse_acf:.4e}, RelF(Corr)={metrics['rel_fro_cov'][-1]:.4f}")

    return metrics


# ============================================================================
# Asymmetric Consistency Validation
# ============================================================================

def validate_asymmetric_consistency(
    bridge,
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
        t_start_tensor = torch.tensor([[start_time]], device=device, dtype=torch.float32)
        mu_start, gamma_start = bridge.get_params(t_start_tensor)
        
        # Sample initial particles (z_start ~ N(mu_T, gamma_T^2))
        z_start = mu_start + gamma_start * torch.randn(n_particles, bridge.data_dim, device=device)
        
        print(f"\nStarting reverse SDE simulation from t={start_time:.4f} to t=0.0")
        
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
            
            print(f"\nValidating at t={t_val:.2f} (closest simulated: t={actual_time:.2f})")
            
            simulated_particles = path[closest_idx].cpu()
            
            # Get analytical ground truth (forward distribution)
            t_tensor = torch.tensor([[t_val]], device=device, dtype=torch.float32)
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


# ============================================================================
# Statistical Comparison Utilities
# ============================================================================

def compute_marginal_statistics(samples: Dict[float, Tensor]) -> Dict[str, list]:
    """
    Compute basic statistics (mean, std) for each time point in the samples.
    """
    sorted_times = sorted(samples.keys())
    
    means = []
    stds = []
    
    for t_val in sorted_times:
        data = samples[t_val].cpu()
        mean_val = torch.mean(data, dim=0).mean().item()  # Average over spatial dimensions
        std_val = torch.std(data, dim=0).mean().item()
        means.append(mean_val)
        stds.append(std_val)
    
    return {
        'times': sorted_times,
        'means': means,
        'stds': stds
    }


def compare_distributions_ks_test(samples1: Tensor, samples2: Tensor) -> float:
    """
    Perform Kolmogorov-Smirnov test on flattened samples.
    Returns the KS statistic (0 = identical, 1 = completely different).
    """
    try:
        from scipy.stats import ks_2samp
        
        # Flatten the samples
        flat1 = samples1.cpu().numpy().flatten()
        flat2 = samples2.cpu().numpy().flatten()
        
        # Perform KS test
        ks_stat, _ = ks_2samp(flat1, flat2)
        return ks_stat
        
    except ImportError:
        print("Warning: scipy not available for KS test")
        return float('nan')
    except Exception as e:
        print(f"Warning: KS test failed: {e}")
        return float('nan')