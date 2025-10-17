"""
Visualization Utilities for Asymmetric Bridges
==============================================

This module provides a set of functions for visualizing the results of
the asymmetric bridge models, including trajectory plots, marginal
distribution comparisons, and data fitting analysis.
"""

import torch
from torch import Tensor
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import os
import math
import torch.fft as fft
from scipy.integrate import solve_ivp
from scipy.stats import gaussian_kde
from typing import Dict, Tuple

from .simulation import solve_gaussian_bridge_reverse_sde, solve_backward_sde_euler_maruyama, generate_backward_samples
# Correlation functions are defined inline below

# Add correlation computation functions
def compute_sample_correlation_matrix(samples: Tensor, truncate: bool = True, variance_threshold: float = 0.999) -> Tensor:
    """
    Computes the DxD sample correlation matrix for a batch of samples.
    Args:
        samples: Tensor of shape [N, D] (N samples, D dimensions).
        truncate: Whether to truncate to retain specified variance fraction.
        variance_threshold: Fraction of variance to retain if truncating.
    Returns:
        Correlation matrix of shape [D, D] (possibly truncated).
    """
    corr_matrix, _ = compute_sample_correlation_matrix_with_eigen(samples, truncate, variance_threshold)
    return corr_matrix


def compute_sample_correlation_matrix_with_eigen(samples: Tensor, truncate: bool = True, variance_threshold: float = 0.999) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    Computes the DxD sample correlation matrix for a batch of samples,
    returning both the matrix and eigendecomposition information.
    
    Args:
        samples: Tensor of shape [N, D] (N samples, D dimensions).
        truncate: Whether to truncate to retain specified variance fraction.
        variance_threshold: Fraction of variance to retain if truncating.
    Returns:
        Tuple of (correlation_matrix, eigen_info) where eigen_info contains eigendecomposition data.
    """
    # Import truncation function from validation module
    from .validation import truncate_correlation_matrix
    
    N, D = samples.shape
    if N <= 1:
        print(f"Warning: Insufficient samples (N={N}) to compute correlation. Returning identity matrix.")
        return torch.eye(D, device=samples.device), {}
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
        
        if truncate:
            correlation_matrix, eigen_info = truncate_correlation_matrix(correlation_matrix, variance_threshold)
        else:
            # Still compute eigendecomposition for non-truncated case
            eigenvals, eigenvecs = torch.linalg.eigh(correlation_matrix)
            sorted_indices = torch.argsort(eigenvals, descending=True)
            eigenvals = eigenvals[sorted_indices]
            eigenvecs = eigenvecs[:, sorted_indices]
            cumsum_variance = torch.cumsum(eigenvals, dim=0)
            variance_ratio = cumsum_variance / torch.sum(eigenvals)
            eigen_info = {
                'eigenvalues': eigenvals,
                'eigenvectors': eigenvecs,
                'n_components': len(eigenvals),
                'variance_ratio': variance_ratio
            }
    except Exception as e:
        print(f"Warning: Correlation matrix computation failed: {e}")
        return torch.eye(D, device=samples.device), {}
    return correlation_matrix, eigen_info


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
# Paper Formatting
# ============================================================================

def format_for_paper():
    """Standard formatting for publication-ready figures."""
    plt.rcParams.update({'image.cmap': 'viridis'})
    plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif',
                                        'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook',
                                        'Century Schoolbook L',  'Utopia', 'ITC Bookman', 'Bookman',
                                        'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif']})
    plt.rcParams.update({'font.family': 'serif'})
    plt.rcParams.update({'font.size': 10})
    plt.rcParams.update({'mathtext.fontset': 'custom'})
    plt.rcParams.update({'mathtext.rm': 'serif'})
    plt.rcParams.update({'mathtext.it': 'serif:italic'})
    plt.rcParams.update({'mathtext.bf': 'serif:bold'})
    plt.close('all')

# ============================================================================
# Plotting Helpers
# ============================================================================

def plot_confidence_ellipse(ax, mu, cov, n_std=2.0, **kwargs):
    """Plots a 2D confidence ellipse for a Gaussian distribution."""
    mu_2d = mu[:2]
    cov_2d = cov[:2, :2]

    lambda_, v = np.linalg.eigh(cov_2d)
    lambda_ = np.sqrt(np.maximum(lambda_, 0))
    angle = np.rad2deg(np.arctan2(*v[:, 0][::-1]))

    ell = Ellipse(xy=(mu_2d[0], mu_2d[1]),
                  width=lambda_[0]*n_std*2, height=lambda_[1]*n_std*2,
                  angle=angle, **kwargs)
    ell.set_facecolor('none')
    ax.add_patch(ell)

# ============================================================================
# Core Visualization Functions
# ============================================================================

def _plot_marginal_distribution_comparison(
    bridge, T, n_viz_particles, n_sde_steps, output_dir, device, solver=None
):
    format_for_paper()
    print("  - Plotting marginal distribution comparisons...")
    validation_times = [T * f for f in [0.25, 0.5, 0.75]]

    # Forward ODE Simulation
    z0 = bridge.base_dist.sample((n_viz_particles,)).to(device)
    if hasattr(bridge, 'flow'): # Expressive flow
        z0, _ = bridge.flow.forward(z0, torch.zeros(n_viz_particles, 1, device=device))
    else: # Gaussian flow
        mu0, gamma0 = bridge.get_params(torch.tensor([[0.0]], device=device, dtype=torch.float32))
        z0 = (mu0 + gamma0 * z0).to(device)

    def forward_ode_func(t, z_np):
        z = torch.from_numpy(z_np).float().to(device).reshape(n_viz_particles, -1)
        with torch.no_grad():
            v = bridge.forward_velocity(z, torch.tensor([[t]], device=device, dtype=torch.float32))
        return v.cpu().numpy().flatten()

    times_eval = np.linspace(0, T, n_sde_steps + 1)
    sol = solve_ivp(fun=forward_ode_func, t_span=[0, T], y0=z0.cpu().numpy().flatten(), method='RK45', t_eval=times_eval)
    forward_path = torch.from_numpy(sol.y.T).float().reshape(len(times_eval), n_viz_particles, -1)

    # Reverse SDE Simulation
    zT = bridge.base_dist.sample((n_viz_particles,)).to(device)
    if hasattr(bridge, 'flow'):
        zT, _ = bridge.flow.forward(zT, torch.full((n_viz_particles, 1), T, device=device))
    else:
        muT, gammaT = bridge.get_params(torch.tensor([[T]], device=device, dtype=torch.float32))
        zT = (muT + gammaT * zT).to(device)
        
    # Use the provided solver or default to Gaussian
    if solver is None:
        solver = solve_gaussian_bridge_reverse_sde
    reverse_path = solver(bridge, zT, T, 0.0, n_sde_steps).cpu()

    fig, axes = plt.subplots(2, len(validation_times), figsize=(12, 8), sharex=True, sharey=True)
    fig.suptitle("Comparison of Marginal Distributions p_t(z)", fontsize=12)

    for i, t_val in enumerate(validation_times):
        ax_scatter = axes[0, i]
        ax_contour = axes[1, i]

        # Find closest time index
        forward_idx = np.argmin(np.abs(times_eval - t_val))
        reverse_times = np.linspace(T, 0, n_sde_steps + 1)
        reverse_idx = np.argmin(np.abs(reverse_times - t_val))

        # Get particles
        fwd_particles = forward_path[forward_idx].cpu().numpy()
        rev_particles = reverse_path[reverse_idx].cpu().numpy()

        # Get analytical distribution
        t_tensor = torch.tensor([[t_val]], device=device, dtype=torch.float32)
        if hasattr(bridge, 'get_params'):
            mu_gt, gamma_gt = bridge.get_params(t_tensor)
            mu_gt = mu_gt[0].cpu().numpy()
            cov_gt = torch.diag(gamma_gt[0]**2).cpu().numpy()
        else:
            # For expressive flows, estimate from forward particles
            mu_gt = np.mean(fwd_particles, axis=0)
            cov_gt = np.cov(fwd_particles.T)

        # Scatter Plot (Top Row)
        ax_scatter.scatter(fwd_particles[:, 0], fwd_particles[:, 1], alpha=0.3, label='Fwd ODE', color='blue', s=10)
        ax_scatter.scatter(rev_particles[:, 0], rev_particles[:, 1], alpha=0.3, label='Rev SDE', color='green', s=10)
        ax_scatter.scatter(mu_gt[0], mu_gt[1], marker='x', color='red', s=100, label='Learned Mean')
        
        if fwd_particles.shape[1] >= 2:
            plot_confidence_ellipse(ax_scatter, mu_gt, cov_gt, n_std=2.0, edgecolor='red', linewidth=2, linestyle='--')
        
        ax_scatter.set_title(f't = {t_val:.2f}')
        if i == 0:
            ax_scatter.set_ylabel('z₂ (Scatter)')
        ax_scatter.grid(True, linestyle='--')
        ax_scatter.legend()
        ax_scatter.set_aspect('equal', adjustable='box')

        # KDE Contour Plot (Bottom Row)
        if fwd_particles.shape[1] >= 2:
            try:
                all_particles = np.vstack([fwd_particles, rev_particles])
                xmin, xmax = all_particles[:, 0].min(), all_particles[:, 0].max()
                ymin, ymax = all_particles[:, 1].min(), all_particles[:, 1].max()
                x_range = xmax - xmin
                y_range = ymax - ymin
                grid_x, grid_y = np.mgrid[xmin - 0.1*x_range:xmax + 0.1*x_range:100j, ymin - 0.1*y_range:ymax + 0.1*y_range:100j]
                grid_pts = np.vstack([grid_x.ravel(), grid_y.ravel()])

                kde_fwd = gaussian_kde(fwd_particles[:, :2].T)
                kde_rev = gaussian_kde(rev_particles[:, :2].T)
                density_fwd = kde_fwd(grid_pts).reshape(grid_x.shape)
                density_rev = kde_rev(grid_pts).reshape(grid_y.shape)

                ax_contour.contour(grid_x, grid_y, density_fwd, colors='blue', linestyles='--', levels=5)
                ax_contour.contour(grid_x, grid_y, density_rev, colors='green', linestyles='-', levels=5)
                
                from matplotlib.lines import Line2D
                legend_elements = [Line2D([0], [0], color='blue', ls='--', label='Fwd ODE'),
                                   Line2D([0], [0], color='green', ls='-', label='Rev SDE')]
                ax_contour.legend(handles=legend_elements)

            except np.linalg.LinAlgError:
                ax_contour.text(0.5, 0.5, "KDE failed (singular matrix)", ha='center', va='center', transform=ax_contour.transAxes)

        ax_contour.set_title('KDE Contour Comparison')
        ax_contour.set_xlabel('z₁')
        if i == 0:
            ax_contour.set_ylabel('z₂ (KDE)')
        ax_contour.grid(True, linestyle=':')
        ax_contour.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_marginal_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()


def _plot_grf_marginals(marginal_data, output_dir, title="GRF Data Marginals"):
    """Visualizes samples from the GRF marginal data."""
    format_for_paper()
    print(f"  - Plotting {title}...")
    
    sorted_times = sorted(marginal_data.keys())
    n_marginals = len(sorted_times)
    n_samples_to_show = 5
    
    if not sorted_times:
        return
    data_dim = marginal_data[sorted_times[0]].shape[1]
    resolution = int(math.sqrt(data_dim))
    
    if resolution * resolution != data_dim:
        print(f"Warning: Data dimension {data_dim} is not a perfect square.")
        return

    fig, axes = plt.subplots(n_samples_to_show, n_marginals, figsize=(min(12, 3 * n_marginals), min(10, 2.5 * n_samples_to_show)))
    if axes.ndim == 1:
        axes = np.array([axes])
    
    all_data = torch.cat([marginal_data[t] for t in sorted_times], dim=0).cpu().numpy()
    vmin, vmax = all_data.min(), all_data.max()

    for i in range(n_samples_to_show):
        for j, t_k in enumerate(sorted_times):
            ax = axes[i, j]
            if i < marginal_data[t_k].shape[0]:
                 sample = marginal_data[t_k][i].cpu().numpy().reshape(resolution, resolution)
                 ax.imshow(sample, cmap='viridis', vmin=vmin, vmax=vmax, extent=[0, 1, 0, 1])
            ax.axis('off')
            if i == 0:
                ax.set_title(f"t = {t_k:.2f}")

    plt.tight_layout()
    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

def _visualize_ground_truth_samples(marginal_data: Dict[float, Tensor], output_dir: str, sample_indices: Tensor = None):
    """
    Visualize ground truth marginal data samples across time points.
    """
    format_for_paper()
    print("  - Plotting ground truth samples...")
    
    sorted_times = sorted(marginal_data.keys())
    n_time_points = len(sorted_times)
    n_samples_show = 3  # Reduced number of samples for cleaner visualization
    
    if not sorted_times:
        return
    data_dim = marginal_data[sorted_times[0]].shape[1]
    resolution = int(math.sqrt(data_dim))
    
    if resolution * resolution != data_dim:
        print(f"    Warning: Data dimension {data_dim} is not a perfect square. Skipping visualization.")
        return

    # Use provided sample indices or generate consistent ones
    if sample_indices is not None:
        selected_indices = sample_indices
        n_samples_show = len(selected_indices)
    else:
        final_time = max(sorted_times)
        n_available_samples = marginal_data[final_time].shape[0]
        n_samples_show = min(n_samples_show, n_available_samples)
        selected_indices = torch.randperm(n_available_samples)[:n_samples_show]

    # Create compact figure for ground truth samples
    fig, axes = plt.subplots(n_samples_show, n_time_points, figsize=(min(10, 2.5 * n_time_points), min(6, 2 * n_samples_show)))
    if n_samples_show == 1 or n_time_points == 1:
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        if n_samples_show > 1:
            axes = axes.reshape(-1, 1)
        elif n_time_points > 1:
            axes = axes.reshape(1, -1)
    
    fig.suptitle("Ground Truth Samples Evolution", fontsize=12)
    
    # Determine global color scale
    all_data = torch.cat([marginal_data[t] for t in sorted_times], dim=0).cpu().numpy()
    vmin, vmax = all_data.min(), all_data.max()
    
    for j, t_val in enumerate(sorted_times):
        data = marginal_data[t_val].cpu().numpy()
        
        for i in range(n_samples_show):
            sample_idx = selected_indices[i].item()
            ax = axes[i, j] if axes.ndim == 2 else axes[max(i, j)]
            
            if sample_idx < data.shape[0]:
                sample = data[sample_idx].reshape(resolution, resolution)
                ax.imshow(sample, cmap='viridis', vmin=vmin, vmax=vmax, extent=[0, 1, 0, 1])
            ax.axis('off')
            
            if j == 0:
                ax.set_ylabel(f'Sample {i+1}', rotation=90, labelpad=15)
            if i == 0:
                ax.set_title(f't = {t_val:.2f}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ground_truth_samples_evolution.png"), dpi=300, bbox_inches='tight')
    plt.close()


def _visualize_generated_backward_samples(generated_samples: Dict[float, Tensor], output_dir: str, sample_indices: Tensor = None):
    """
    Visualize generated backward samples across time points.
    """
    format_for_paper()
    print("  - Plotting generated backward samples...")
    
    sorted_times = sorted(generated_samples.keys())
    n_time_points = len(sorted_times)
    n_samples_show = 3  # Reduced number of samples for cleaner visualization
    
    if not sorted_times:
        return
    data_dim = generated_samples[sorted_times[0]].shape[1]
    resolution = int(math.sqrt(data_dim))
    
    if resolution * resolution != data_dim:
        print(f"    Warning: Data dimension {data_dim} is not a perfect square. Skipping visualization.")
        return

    # Use provided sample indices or generate consistent ones
    if sample_indices is not None:
        selected_indices = sample_indices
        n_samples_show = len(selected_indices)
    else:
        final_time = max(sorted_times)
        n_available_samples = generated_samples[final_time].shape[0]
        n_samples_show = min(n_samples_show, n_available_samples)
        selected_indices = torch.randperm(n_available_samples)[:n_samples_show]

    # Create compact figure for generated samples
    fig, axes = plt.subplots(n_samples_show, n_time_points, figsize=(min(10, 2.5 * n_time_points), min(6, 2 * n_samples_show)))
    if n_samples_show == 1 or n_time_points == 1:
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        if n_samples_show > 1:
            axes = axes.reshape(-1, 1)
        elif n_time_points > 1:
            axes = axes.reshape(1, -1)
    
    fig.suptitle("Generated Backward Samples Evolution", fontsize=12)
    
    # Determine global color scale
    all_data = torch.cat([generated_samples[t] for t in sorted_times], dim=0).cpu().numpy()
    vmin, vmax = all_data.min(), all_data.max()
    
    for j, t_val in enumerate(sorted_times):
        data = generated_samples[t_val].cpu().numpy()
        
        for i in range(n_samples_show):
            sample_idx = selected_indices[i].item()
            ax = axes[i, j] if axes.ndim == 2 else axes[max(i, j)]
            
            if sample_idx < data.shape[0]:
                sample = data[sample_idx].reshape(resolution, resolution)
                ax.imshow(sample, cmap='viridis', vmin=vmin, vmax=vmax, extent=[0, 1, 0, 1])
            ax.axis('off')
            
            if j == 0:
                ax.set_ylabel(f'Sample {i+1}', rotation=90, labelpad=15)
            if i == 0:
                ax.set_title(f't = {t_val:.2f}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "generated_backward_samples_evolution.png"), dpi=300, bbox_inches='tight')
    plt.close()


def _visualize_backward_samples_comparison(marginal_data: Dict[float, Tensor], generated_samples: Dict[float, Tensor], output_dir: str, sample_indices: Tensor = None):
    """
    Wrapper function to generate both ground truth and generated backward samples visualizations.
    Creates separate, cleaner figures for better readability.
    """
    # Create separate visualizations for better readability
    _visualize_ground_truth_samples(marginal_data, output_dir, sample_indices)
    _visualize_generated_backward_samples(generated_samples, output_dir, sample_indices)


def _visualize_marginal_statistics_comparison(marginal_data: Dict[float, Tensor], generated_samples: Dict[float, Tensor], output_dir: str):
    """
    Compare statistical properties (mean, std, distribution) between original and generated samples.
    """
    format_for_paper()
    print("  - Plotting marginal statistics comparison...")
    
    sorted_times = sorted(marginal_data.keys())
    
    original_means = []
    original_stds = []
    generated_means = []
    generated_stds = []
    
    for t_val in sorted_times:
        orig_data = marginal_data[t_val].cpu()
        orig_mean = torch.mean(orig_data, dim=0).mean().item()
        orig_std = torch.std(orig_data, dim=0).mean().item()
        original_means.append(orig_mean)
        original_stds.append(orig_std)
        
        gen_data = generated_samples[t_val].cpu()
        gen_mean = torch.mean(gen_data, dim=0).mean().item()
        gen_std = torch.std(gen_data, dim=0).mean().item()
        generated_means.append(gen_mean)
        generated_stds.append(gen_std)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Statistical Comparison: Original vs Generated Samples", fontsize=14)
    
    ax1 = axes[0]
    ax1.plot(sorted_times, original_means, 'o-', label='Original', linewidth=2, markersize=6)
    ax1.plot(sorted_times, generated_means, 's--', label='Generated', linewidth=2, markersize=6)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Mean Value')
    ax1.set_title('Mean Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(sorted_times, original_stds, 'o-', label='Original', linewidth=2, markersize=6)
    ax2.plot(sorted_times, generated_stds, 's--', label='Generated', linewidth=2, markersize=6)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Standard Deviation')
    ax2.set_title('Std Dev Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[2]
    ax3.scatter(original_means, original_stds, c=sorted_times, cmap='viridis', 
               s=80, alpha=0.7, label='Original', marker='o')
    ax3.scatter(generated_means, generated_stds, c=sorted_times, cmap='plasma', 
               s=80, alpha=0.7, label='Generated', marker='s')
    ax3.set_xlabel('Mean Value')
    ax3.set_ylabel('Standard Deviation')
    ax3.set_title('Mean vs Std (colored by time)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(sorted_times), vmax=max(sorted_times)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax3)
    cbar.set_label('Time')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "marginal_statistics_comparison.png"), dpi=300)
    plt.close()


def _visualize_sample_distributions(marginal_data: Dict[float, Tensor], generated_samples: Dict[float, Tensor], output_dir: str):
    """
    Visualize the pixel-wise distributions for select time points to assess distributional match.
    """
    format_for_paper()
    print("  - Plotting sample distributions comparison...")
    
    sorted_times = sorted(marginal_data.keys())
    n_plots = min(4, len(sorted_times))
    selected_times = [sorted_times[i] for i in np.linspace(0, len(sorted_times)-1, n_plots, dtype=int)]
    
    fig, axes = plt.subplots(2, n_plots, figsize=(min(12, 3.5 * n_plots), 7))
    if n_plots == 1:
        axes = axes.reshape(-1, 1)
    fig.suptitle("Pixel Value Distributions: Original vs Generated", fontsize=14)
    
    for i, t_val in enumerate(selected_times):
        orig_data = marginal_data[t_val].cpu().numpy().flatten()
        gen_data = generated_samples[t_val].cpu().numpy().flatten()
        
        ax_hist = axes[0, i]
        ax_hist.hist(orig_data, bins=50, alpha=0.7, density=True, label='Original', color='blue')
        ax_hist.hist(gen_data, bins=50, alpha=0.7, density=True, label='Generated', color='red')
        ax_hist.set_title(f'Distributions at t={t_val:.2f}')
        ax_hist.set_xlabel('Pixel Value')
        ax_hist.set_ylabel('Density')
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        
        ax_qq = axes[1, i]
        orig_sorted = np.sort(orig_data)
        gen_sorted = np.sort(gen_data)
        
        n_points = min(len(orig_sorted), len(gen_sorted))
        orig_interp = np.interp(np.linspace(0, 1, n_points), 
                               np.linspace(0, 1, len(orig_sorted)), orig_sorted)
        gen_interp = np.interp(np.linspace(0, 1, n_points), 
                              np.linspace(0, 1, len(gen_sorted)), gen_sorted)
        
        ax_qq.scatter(orig_interp, gen_interp, alpha=0.5, s=1)
        
        min_val = min(orig_interp.min(), gen_interp.min())
        max_val = max(orig_interp.max(), gen_interp.max())
        ax_qq.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Match')
        
        ax_qq.set_title(f'Q-Q Plot at t={t_val:.2f}')
        ax_qq.set_xlabel('Original Quantiles')
        ax_qq.set_ylabel('Generated Quantiles')
        ax_qq.legend()
        ax_qq.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "sample_distributions_comparison.png"), dpi=300)
    plt.close()


def _visualize_variance_fields_comparison(
    marginal_data: Dict[float, Tensor], 
    generated_samples: Dict[float, Tensor], 
    output_dir: str
):
    """Visualizes correlation comparison between original and generated samples using diagonal values (self-correlation)."""
    format_for_paper()
    print("  - Plotting correlation fields comparison...")
    
    sorted_times = sorted(marginal_data.keys())
    if not sorted_times:
        return

    n_time_points = min(3, len(sorted_times))
    selected_indices = np.linspace(0, len(sorted_times)-1, n_time_points, dtype=int)
    selected_times = [sorted_times[i] for i in selected_indices]

    data_dim = marginal_data[selected_times[0]].shape[1]
    resolution = int(math.sqrt(data_dim))
    is_square = (resolution * resolution == data_dim)

    corr_data_list, corr_gen_list, metrics = [], [], {}

    for t_val in selected_times:
        samples_data = marginal_data[t_val].cpu()
        samples_gen = generated_samples[t_val].cpu()
        
        print(f"    Computing truncated correlation matrices for t={t_val:.2f} (99.9% variance)...")
        corr_data = compute_sample_correlation_matrix(samples_data, truncate=True, variance_threshold=0.999)
        corr_gen = compute_sample_correlation_matrix(samples_gen, truncate=True, variance_threshold=0.999)
        corr_data_list.append(corr_data)
        corr_gen_list.append(corr_gen)

        rel_f_dist = relative_correlation_frobenius_distance(corr_data, corr_gen)
        metrics[t_val] = {'Rel_F_dist': rel_f_dist}

    # Create correlation fields figure - reasonable paper size
    fig_var, axes_var = plt.subplots(2, n_time_points, figsize=(10, 6), sharex=True, sharey=True)
    if n_time_points == 1:
        axes_var = axes_var.reshape(2, 1)
    fig_var.suptitle("Truncated Correlation Field Comparison (99.9% Variance)", fontsize=12)

    # Note: Correlation diagonal elements are always 1.0 by definition
    # We'll visualize the off-diagonal mean correlation instead
    max_corr = 1.0  # Correlation values are bounded [-1, 1]

    for i, t_val in enumerate(selected_times):
        corr_data = corr_data_list[i]
        corr_gen = corr_gen_list[i]
        # Use mean off-diagonal correlations instead of diagonal (which are always 1)
        mask = ~torch.eye(corr_data.shape[0], dtype=bool, device=corr_data.device)
        mean_corr_data = torch.full((corr_data.shape[0],), corr_data[mask].abs().mean().item())
        mean_corr_gen = torch.full((corr_gen.shape[0],), corr_gen[mask].abs().mean().item())
        
        ax_target = axes_var[0, i]
        ax_gen = axes_var[1, i]

        if is_square:
            # Plot Target Mean Correlation (Top Row)
            arr_data = mean_corr_data.reshape(resolution, resolution).numpy()
            im_target = ax_target.imshow(arr_data, cmap='plasma', vmin=0, vmax=max_corr)
            
            # Plot Generated Mean Correlation (Bottom Row)
            arr_gen = mean_corr_gen.reshape(resolution, resolution).numpy()
            im_gen = ax_gen.imshow(arr_gen, cmap='plasma', vmin=0, vmax=max_corr)
            
            # Add individual colorbars for each image
            plt.colorbar(im_target, ax=ax_target, fraction=0.046, pad=0.04)
            plt.colorbar(im_gen, ax=ax_gen, fraction=0.046, pad=0.04)

        else: # 1D data case
            ax_target.plot(mean_corr_data.numpy(), label='Target')
            ax_gen.plot(mean_corr_gen.numpy(), '--', label='Generated')
            ax_gen.set_xlabel("Dimension Index")
            if i == 0:
                ax_target.legend()
                ax_gen.legend()

        # Set titles and labels
        if i == 0:
            ax_target.set_ylabel("Target")
            ax_gen.set_ylabel("Generated")
        ax_target.set_title(f"t = {t_val:.2f}\nRel F-Dist = {metrics[t_val]['Rel_F_dist']:.3f}")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "statistical_correlation_fields.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_var)


def _visualize_eigenvalue_spectra_comparison(
    marginal_data: Dict[float, Tensor], 
    generated_samples: Dict[float, Tensor], 
    output_dir: str
):
    """
    Visualizes full eigenvalue spectra comparison showing the complete spectrum 
    with truncation threshold indicated. Avoids redundant eigendecomposition by 
    reusing results from correlation matrix computation.
    """
    format_for_paper()
    print("  - Plotting full correlation eigenvalue spectra comparison...")
    
    sorted_times = sorted(marginal_data.keys())
    if not sorted_times:
        return

    n_time_points = min(3, len(sorted_times))
    selected_indices = np.linspace(0, len(sorted_times)-1, n_time_points, dtype=int)
    selected_times = [sorted_times[i] for i in selected_indices]

    # Create figure with two rows: eigenvalue spectrum and cumulative variance
    fig, axes = plt.subplots(2, n_time_points, figsize=(12, 8))
    if n_time_points == 1:
        axes = axes.reshape(-1, 1)
    fig.suptitle("Full Correlation Matrix Eigenvalue Spectrum Analysis", fontsize=12)

    for i, t_val in enumerate(selected_times):
        samples_data = marginal_data[t_val].cpu()
        samples_gen = generated_samples[t_val].cpu()
        
        # Compute correlation matrices WITH eigendecomposition (no redundant computation)
        print(f"    Computing correlation eigendecomposition for t={t_val:.2f}...")
        _, eigen_info_data = compute_sample_correlation_matrix_with_eigen(
            samples_data, truncate=True, variance_threshold=0.999
        )
        _, eigen_info_gen = compute_sample_correlation_matrix_with_eigen(
            samples_gen, truncate=True, variance_threshold=0.999
        )
        
        if not eigen_info_data or not eigen_info_gen:
            print(f"    Warning: Eigendecomposition failed for t={t_val:.2f}")
            continue
        
        # Extract eigenvalues and variance info
        eigvals_data = eigen_info_data['eigenvalues'].cpu().numpy()
        eigvals_gen = eigen_info_gen['eigenvalues'].cpu().numpy()
        n_components_data = eigen_info_data['n_components']
        n_components_gen = eigen_info_gen['n_components']
        variance_ratio_data = eigen_info_data['variance_ratio'].cpu().numpy()
        variance_ratio_gen = eigen_info_gen['variance_ratio'].cpu().numpy()
        
        # Top row: Eigenvalue spectrum (log-log plot)
        ax_eigen = axes[0, i]
        
        # Plot full spectrum
        ax_eigen.loglog(range(1, len(eigvals_data) + 1), eigvals_data, 
                       'b-', label='Target', linewidth=2, alpha=0.7)
        ax_eigen.loglog(range(1, len(eigvals_gen) + 1), eigvals_gen, 
                       'r--', label='Generated', linewidth=2, alpha=0.7)
        
        # Mark truncation cutoff with vertical lines
        ax_eigen.axvline(n_components_data, color='blue', linestyle=':', 
                        linewidth=1.5, alpha=0.5, label=f'99.9% cutoff (Target: {n_components_data})')
        ax_eigen.axvline(n_components_gen, color='red', linestyle=':', 
                        linewidth=1.5, alpha=0.5, label=f'99.9% cutoff (Gen: {n_components_gen})')
        
        ax_eigen.set_xlabel("Mode Index")
        ax_eigen.set_ylabel("Eigenvalue")
        ax_eigen.set_title(f"t = {t_val:.2f}")
        ax_eigen.grid(True, which='both', alpha=0.3)
        if i == 0:
            ax_eigen.legend(fontsize=8, loc='best')
        
        # Bottom row: Cumulative variance explained
        ax_var = axes[1, i]
        
        ax_var.semilogx(range(1, len(variance_ratio_data) + 1), variance_ratio_data * 100, 
                       'b-', label='Target', linewidth=2, alpha=0.7)
        ax_var.semilogx(range(1, len(variance_ratio_gen) + 1), variance_ratio_gen * 100, 
                       'r--', label='Generated', linewidth=2, alpha=0.7)
        
        # Mark 99.9% threshold
        ax_var.axhline(99.9, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='99.9% threshold')
        ax_var.axvline(n_components_data, color='blue', linestyle=':', linewidth=1.5, alpha=0.5)
        ax_var.axvline(n_components_gen, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
        
        ax_var.set_xlabel("Number of Components")
        ax_var.set_ylabel("Cumulative Variance (%)")
        ax_var.set_ylim([90, 100.5])
        ax_var.grid(True, which='both', alpha=0.3)
        if i == 0:
            ax_var.legend(fontsize=8, loc='lower right')
        
        # Add text annotation with effective dimensions
        ax_var.text(0.98, 0.05, f'Effective dim:\nTarget: {n_components_data}\nGen: {n_components_gen}',
                   transform=ax_var.transAxes, fontsize=8,
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "statistical_correlation_eigen_spectrum.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)


def _visualize_covariance_heatmaps_comparison(
    marginal_data: Dict[float, Tensor], 
    generated_samples: Dict[float, Tensor], 
    output_dir: str,
    max_dim_for_heatmap: int = 2048
):
    """Visualizes correlation heatmaps comparison between original and generated samples."""
    format_for_paper()
    print("  - Plotting correlation heatmaps comparison...")
    
    sorted_times = sorted(marginal_data.keys())
    if not sorted_times:
        return

    data_dim = marginal_data[sorted_times[0]].shape[1]
    if data_dim > max_dim_for_heatmap:
        print(f"    Skipping heatmaps: dimension {data_dim} > {max_dim_for_heatmap}")
        return

    n_time_points = min(3, len(sorted_times))
    selected_indices = np.linspace(0, len(sorted_times)-1, n_time_points, dtype=int)
    selected_times = [sorted_times[i] for i in selected_indices]

    global_vmax_corr = 1.0  # Correlations are bounded [-1, 1]
    global_vmax_diff = 0
    corr_data_list, corr_gen_list = [], []

    for t_val in selected_times:
        samples_data = marginal_data[t_val].cpu()
        samples_gen = generated_samples[t_val].cpu()
        
        corr_data = compute_sample_correlation_matrix(samples_data)
        corr_gen = compute_sample_correlation_matrix(samples_gen)
        corr_data_list.append(corr_data)
        corr_gen_list.append(corr_gen)

        diff = corr_data - corr_gen
        global_vmax_diff = max(global_vmax_diff, torch.abs(diff).max().item())

    # Create correlation heatmaps figure - reasonable paper size
    cmap_corr = 'coolwarm'
    cmap_diff = 'RdBu_r'
    fig_hm, axes_hm = plt.subplots(3, n_time_points, figsize=(10, 8))
    if n_time_points == 1:
        axes_hm = axes_hm.reshape(3, 1)
    fig_hm.suptitle("Correlation Matrix Heatmaps", fontsize=12)

    for i, t_val in enumerate(selected_times):
        corr_data = corr_data_list[i]
        corr_gen = corr_gen_list[i]

        ax_t = axes_hm[0, i]
        im_target = ax_t.imshow(corr_data.numpy(), cmap=cmap_corr, vmin=-global_vmax_corr, vmax=global_vmax_corr)
        ax_t.set_title(f"Target\nt = {t_val:.2f}")
        if i == 0:
            ax_t.set_ylabel("Target")

        ax_g = axes_hm[1, i]
        im_gen = ax_g.imshow(corr_gen.numpy(), cmap=cmap_corr, vmin=-global_vmax_corr, vmax=global_vmax_corr)
        ax_g.set_title("Generated")
        if i == 0:
            ax_g.set_ylabel("Generated")

        ax_d = axes_hm[2, i]
        diff = corr_data - corr_gen
        im_diff = ax_d.imshow(diff.numpy(), cmap=cmap_diff, vmin=-global_vmax_diff, vmax=global_vmax_diff)
        ax_d.set_title("Difference")
        if i == 0:
            ax_d.set_ylabel("Difference")
            
        # Add individual colorbars for each heatmap
        plt.colorbar(im_target, ax=ax_t, fraction=0.046, pad=0.04)
        plt.colorbar(im_gen, ax=ax_g, fraction=0.046, pad=0.04)
        plt.colorbar(im_diff, ax=ax_d, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "statistical_correlation_heatmaps.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_hm)


def _visualize_statistical_covariance_comparison(
    marginal_data: Dict[float, Tensor], 
    generated_samples: Dict[float, Tensor], 
    output_dir: str,
    max_dim_for_heatmap: int = 2048
):
    """
    Wrapper function to generate separate statistical correlation comparison figures:
    1) Correlation fields comparison (mean off-diagonal correlations)
    2) Correlation eigenvalue spectra comparison  
    3) Correlation heatmaps comparison (optional)
    """
    _visualize_variance_fields_comparison(marginal_data, generated_samples, output_dir)
    _visualize_eigenvalue_spectra_comparison(marginal_data, generated_samples, output_dir)
    _visualize_covariance_heatmaps_comparison(marginal_data, generated_samples, output_dir, max_dim_for_heatmap)


def _plot_marginal_data_fit(bridge, marginal_data, T, output_dir, device):
    """Plot marginal data fit comparison."""
    format_for_paper()
    print("  - Plotting marginal data fit...")

    sorted_times = sorted(marginal_data.keys())
    n_marginals = len(sorted_times)

    fig, axes = plt.subplots(2, n_marginals, figsize=(min(12, 4 * n_marginals), 8), sharex=True, sharey=True)
    if n_marginals == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    fig.suptitle("Qualitative Fit to Marginal Data Constraints", fontsize=12)

    for i, t_k in enumerate(sorted_times):
        ax_scatter = axes[0, i]
        ax_contour = axes[1, i]
        samples_k = marginal_data[t_k].cpu().numpy()

        # Get analytical distribution at time t_k
        t_k_tensor = torch.tensor([[t_k]], device=device, dtype=torch.float32)
        
        if hasattr(bridge, 'get_params'):  # Gaussian bridge
            mu_k, gamma_k = bridge.get_params(t_k_tensor)
            mu_k_np = mu_k[0].cpu().numpy()
            cov_k = torch.diag(gamma_k[0]**2).cpu().numpy()
        else:  # Expressive flow bridge - sample from learned distribution
            # For expressive flows, we don't have analytical mean/covariance
            # Instead, we'll generate samples and estimate statistics
            epsilon = bridge.base_dist.sample((samples_k.shape[0],)).to(device)
            with torch.no_grad():
                learned_samples, _ = bridge.flow.forward(epsilon, t_k_tensor.expand(samples_k.shape[0], -1))
            learned_samples = learned_samples.cpu().numpy()
            mu_k_np = np.mean(learned_samples, axis=0)
            cov_k = np.cov(learned_samples.T)

        # Scatter Plot (Top Row)
        ax_scatter.scatter(samples_k[:, 0], samples_k[:, 1], alpha=0.5, label='Data Samples', s=15, color='gray')
        ax_scatter.scatter(mu_k_np[0], mu_k_np[1], marker='P', color='orange', s=150, label='Learned Mean', edgecolors='black')
        
        if samples_k.shape[1] >= 2:  # Only plot ellipse for 2D+
            plot_confidence_ellipse(ax_scatter, mu_k_np, cov_k, n_std=2.0, edgecolor='orange', linewidth=2.5, linestyle='-')
        
        ax_scatter.set_title(f't = {t_k:.2f}')
        if i == 0:
            ax_scatter.set_ylabel('z₂ (Scatter)')
        ax_scatter.grid(True, linestyle='--')
        ax_scatter.legend()
        ax_scatter.set_aspect('equal', adjustable='box')

        # KDE Contour Plot (Bottom Row)
        if hasattr(bridge, 'get_params'):  # Gaussian bridge
            from torch.distributions import MultivariateNormal
            learned_dist = MultivariateNormal(torch.from_numpy(mu_k_np).float(), torch.from_numpy(cov_k).float())
            learned_samples_for_kde = learned_dist.sample((samples_k.shape[0],)).numpy()
        else:  # Use the samples we already generated above
            learned_samples_for_kde = learned_samples

        if samples_k.shape[1] >= 2:  # Only create KDE for 2D+
            try:
                all_particles = np.vstack([samples_k, learned_samples_for_kde])
                xmin, xmax = all_particles[:, 0].min(), all_particles[:, 0].max()
                ymin, ymax = all_particles[:, 1].min(), all_particles[:, 1].max()
                x_range = xmax - xmin
                y_range = ymax - ymin
                grid_x, grid_y = np.mgrid[xmin - 0.1*x_range:xmax + 0.1*x_range:100j, ymin - 0.1*y_range:ymax + 0.1*y_range:100j]
                grid_pts = np.vstack([grid_x.ravel(), grid_y.ravel()])

                kde_data = gaussian_kde(samples_k[:, :2].T)
                kde_learned = gaussian_kde(learned_samples_for_kde[:, :2].T)
                density_data = kde_data(grid_pts).reshape(grid_x.shape)
                density_learned = kde_learned(grid_pts).reshape(grid_y.shape)

                ax_contour.contour(grid_x, grid_y, density_data, colors='gray', linestyles='-', levels=5)
                ax_contour.contour(grid_x, grid_y, density_learned, colors='orange', linestyles='--', levels=5)

                from matplotlib.lines import Line2D
                legend_elements = [Line2D([0], [0], color='gray', ls='-', label='Data'),
                                   Line2D([0], [0], color='orange', ls='--', label='Learned')]
                ax_contour.legend(handles=legend_elements)

            except np.linalg.LinAlgError:
                ax_contour.text(0.5, 0.5, "KDE failed (singular matrix)", ha='center', va='center', transform=ax_contour.transAxes)

        ax_contour.set_title('KDE Contour Comparison')
        ax_contour.set_xlabel('z₁')
        if i == 0:
            ax_contour.set_ylabel('z₂ (KDE)')
        ax_contour.grid(True, linestyle=':')
        ax_contour.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig4_marginal_fit.png"), dpi=300, bbox_inches='tight')
    plt.close()


def visualize_bridge_results(bridge, marginal_data: Dict[float, Tensor], T: float, output_dir: str, is_grf: bool = False, n_viz_particles: int = 50, n_sde_steps: int = 100, solver: str = 'euler', enable_covariance_analysis: bool = True, validation_samples: Dict[float, Tensor] = None):
    """
    Generate and save a comprehensive set of visualizations for the trained bridge.
    """
    bridge.eval()
    device = next(bridge.parameters()).device
    print("\n--- Generating and Saving Visualizations ---")
    
    # Ensure marginal_data is on the correct device first, then create CPU version for visualization
    marginal_data_on_device = {t: samples.to(device) for t, samples in marginal_data.items()}
    marginal_data_cpu = {t: samples.cpu() for t, samples in marginal_data_on_device.items()}

    # Determine which solver to use
    if solver == 'euler':
        sde_solver = solve_backward_sde_euler_maruyama
    elif solver == 'gaussian' and hasattr(bridge, 'get_params'):
        sde_solver = solve_gaussian_bridge_reverse_sde
    else:
        print(f"Warning: Solver '{solver}' not suitable for this bridge type. Falling back to Euler-Maruyama.")
        sde_solver = solve_backward_sde_euler_maruyama

    with torch.no_grad():
        if is_grf:
            _plot_grf_marginals(marginal_data_cpu, output_dir, title="Input GRF Data Marginals")
            
            print("  - Generating comparative backward samples...")
            
            # Select consistent sample indices for visualization (4 samples for display)
            final_time = max(marginal_data.keys())
            n_viz_samples = min(4, marginal_data[final_time].shape[0])
            viz_sample_indices = torch.randperm(marginal_data[final_time].shape[0])[:n_viz_samples]
            
            # Extract the exact same samples that will be used for visualization
            final_samples_for_viz = marginal_data_on_device[final_time][viz_sample_indices].to(device)
            
            # Generate backward samples starting from these exact samples
            print(f"    Using {len(viz_sample_indices)} specific samples for consistent comparison")
            ground_truth_backward = generate_backward_samples(
                bridge, marginal_data, 
                n_samples=len(viz_sample_indices),  # This will be ignored since we provide z_final_ground_truth
                n_steps=n_sde_steps, 
                device=device,
                solver=sde_solver,
                z_final_ground_truth=final_samples_for_viz,
                batch_size=64  # Small enough for safe processing
            )
            
            # Create original_data dict using the same sample indices
            original_data = {}
            for t_val in sorted(marginal_data_on_device.keys()):
                original_data[t_val] = marginal_data_on_device[t_val][viz_sample_indices]
            
            # Convert to CPU for visualization
            original_data_cpu = {t: samples.cpu() for t, samples in original_data.items()}
            ground_truth_backward_cpu = {t: samples.cpu() for t, samples in ground_truth_backward.items()}
            
            # Denormalize both datasets (ensure proper device handling)
            device = next(bridge.parameters()).device
            for t, samples in original_data_cpu.items():
                original_data_cpu[t] = bridge.denormalize(samples.to(device)).cpu()
            for t, samples in ground_truth_backward_cpu.items():
                ground_truth_backward_cpu[t] = bridge.denormalize(samples.to(device)).cpu()

            # Add visualizations for comparative backward samples (subset)
            # Note: original_data_cpu and ground_truth_backward_cpu were created by
            # subsetting with `viz_sample_indices`, so pass local indices (0..n_viz_samples-1)
            local_indices = torch.arange(len(viz_sample_indices))
            _visualize_backward_samples_comparison(original_data_cpu, ground_truth_backward_cpu, output_dir, local_indices)
            _visualize_marginal_statistics_comparison(original_data_cpu, ground_truth_backward_cpu, output_dir)
            _visualize_sample_distributions(original_data_cpu, ground_truth_backward_cpu, output_dir)

            # For statistically accurate covariance comparisons, use FULL batches
            # anchored at the ground-truth final samples so t=1 covariances are identical.
            if validation_samples is not None:
                # Reuse validation samples (avoid duplicate SDE integration)
                print("  - Reusing validation samples for statistical analysis (saves ~50% time)...")
                generated_full = validation_samples
            else:
                # Generate new samples if not provided
                print("  - Generating new samples for statistical analysis...")
                z_final_full_gt = marginal_data_on_device[final_time].to(device)  # all available GT samples at t=final
                generated_full = generate_backward_samples(
                    bridge, marginal_data,
                    n_samples=z_final_full_gt.shape[0],  # ignored when z_final_ground_truth is provided
                    n_steps=n_sde_steps,
                    device=device,
                    solver=sde_solver,
                    z_final_ground_truth=z_final_full_gt,
                    batch_size=32  # Very conservative for large batches (1024 samples)
                )

            # Prepare full-batch CPU, denormalized
            marginal_full_cpu = {t: bridge.denormalize(marginal_data_on_device[t].to(device)).cpu() for t in sorted(marginal_data_on_device.keys())}
            generated_full_cpu = {t: bridge.denormalize(generated_full[t].to(device)).cpu() for t in sorted(generated_full.keys())}

            # Correlation comparisons (ACF + full statistical correlation) with full batches
            if enable_covariance_analysis:
                _visualize_covariance_comparison(marginal_full_cpu, generated_full_cpu, output_dir)
                _visualize_statistical_covariance_comparison(marginal_full_cpu, generated_full_cpu, output_dir)
            else:
                print("Skipping correlation visualizations (enable_covariance_analysis=False).")
            
        else:
            # ... standard spiral data visualizations
            _plot_marginal_distribution_comparison(bridge, T, n_viz_particles, n_sde_steps, output_dir, device, sde_solver)
            _plot_marginal_data_fit(bridge, marginal_data_cpu, T, output_dir, device)

    print(f"Visualizations saved to '{output_dir}' directory.")


# ============================================================================
# GRF-Specific Visualization Functions
# ============================================================================

def _plot_grf_marginals(marginal_data, output_dir, title="GRF Data Marginals"):
    """Visualizes samples from the GRF marginal data."""
    format_for_paper()
    print(f"  - Plotting {title}...")
    
    sorted_times = sorted(marginal_data.keys())
    n_marginals = len(sorted_times)
    n_samples_to_show = 5
    
    # Determine resolution
    if not sorted_times:
        return
    data_dim = marginal_data[sorted_times[0]].shape[1]
    resolution = int(math.sqrt(data_dim))
    
    if resolution * resolution != data_dim:
        print(f"Warning: Data dimension {data_dim} is not a perfect square. Cannot visualize as 2D field.")
        return

    fig, axes = plt.subplots(n_samples_to_show, n_marginals, figsize=(3 * n_marginals, 3 * n_samples_to_show))
    
    # Handle potential dimension issues
    if n_samples_to_show == 1 or n_marginals == 1:
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        if n_samples_to_show > 1:
            axes = axes.reshape(-1, 1)
        elif n_marginals > 1:
            axes = axes.reshape(1, -1)
        elif n_samples_to_show == 1 and n_marginals == 1:
            axes = axes.reshape(1, 1)
    elif axes.ndim == 1:
        axes = np.array([axes])
    
    # Determine global vmin/vmax
    all_data = torch.cat([marginal_data[t] for t in sorted_times], dim=0).cpu().numpy()
    vmin, vmax = all_data.min(), all_data.max()

    for i in range(n_samples_to_show):
        for j, t_k in enumerate(sorted_times):
            ax = axes[i, j]
            # Check if sample index exists
            if i < marginal_data[t_k].shape[0]:
                sample = marginal_data[t_k][i].cpu().numpy().reshape(resolution, resolution)
                ax.imshow(sample, cmap='viridis', vmin=vmin, vmax=vmax, extent=[0, 1, 0, 1])
            ax.axis('off')
            if i == 0:
                ax.set_title(f"t = {t_k:.2f}")

    plt.tight_layout()
    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


def _visualize_backward_samples_comparison(marginal_data: Dict[float, Tensor], generated_samples: Dict[float, Tensor], output_dir: str, sample_indices: Tensor = None):
    """
    Wrapper function to generate both ground truth and generated backward samples visualizations.
    Creates separate, cleaner figures for better readability.
    """
    # Create separate visualizations for better readability - use the functions defined above
    _visualize_ground_truth_samples(marginal_data, output_dir, sample_indices)
    _visualize_generated_backward_samples(generated_samples, output_dir, sample_indices)


def _visualize_marginal_statistics_comparison(marginal_data: Dict[float, Tensor], generated_samples: Dict[float, Tensor], output_dir: str):
    """
    Compare statistical properties (mean, std, distribution) between original and generated samples.
    """
    format_for_paper()
    print("  - Plotting marginal statistics comparison...")
    
    sorted_times = sorted(marginal_data.keys())
    
    # Calculate statistics for each time point
    original_means = []
    original_stds = []
    generated_means = []
    generated_stds = []
    
    for t_val in sorted_times:
        # Original data statistics
        orig_data = marginal_data[t_val].cpu()
        orig_mean = torch.mean(orig_data, dim=0).mean().item()
        orig_std = torch.std(orig_data, dim=0).mean().item()
        original_means.append(orig_mean)
        original_stds.append(orig_std)
        
        # Generated data statistics
        gen_data = generated_samples[t_val].cpu()
        gen_mean = torch.mean(gen_data, dim=0).mean().item()
        gen_std = torch.std(gen_data, dim=0).mean().item()
        generated_means.append(gen_mean)
        generated_stds.append(gen_std)
    
    # Create comparison plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Statistical Comparison: Original vs Generated Samples", fontsize=14)
    
    # Plot 1: Mean evolution
    ax1 = axes[0]
    ax1.plot(sorted_times, original_means, 'o-', label='Original', linewidth=2, markersize=6)
    ax1.plot(sorted_times, generated_means, 's--', label='Generated', linewidth=2, markersize=6)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Mean Value')
    ax1.set_title('Mean Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Standard deviation evolution
    ax2 = axes[1]
    ax2.plot(sorted_times, original_stds, 'o-', label='Original', linewidth=2, markersize=6)
    ax2.plot(sorted_times, generated_stds, 's--', label='Generated', linewidth=2, markersize=6)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Standard Deviation')
    ax2.set_title('Std Dev Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Scatter plot of means vs stds
    ax3 = axes[2]
    ax3.scatter(original_means, original_stds, c=sorted_times, cmap='viridis', 
               s=80, alpha=0.7, label='Original', marker='o')
    ax3.scatter(generated_means, generated_stds, c=sorted_times, cmap='plasma', 
               s=80, alpha=0.7, label='Generated', marker='s')
    ax3.set_xlabel('Mean Value')
    ax3.set_ylabel('Standard Deviation')
    ax3.set_title('Mean vs Std (colored by time)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "marginal_statistics_comparison.png"), dpi=300)
    plt.close()


def _visualize_sample_distributions(marginal_data: Dict[float, Tensor], generated_samples: Dict[float, Tensor], output_dir: str):
    """
    Visualize the pixel-wise distributions for select time points.
    """
    format_for_paper()
    print("  - Plotting sample distributions comparison...")
    
    sorted_times = sorted(marginal_data.keys())
    # Select a few representative time points
    n_plots = min(4, len(sorted_times))
    selected_times = [sorted_times[i] for i in np.linspace(0, len(sorted_times)-1, n_plots, dtype=int)]
    
    fig, axes = plt.subplots(2, n_plots, figsize=(min(12, 3.5 * n_plots), 7))
    if n_plots == 1:
        axes = axes.reshape(-1, 1)
    fig.suptitle("Pixel Value Distributions: Original vs Generated", fontsize=12)
    
    for i, t_val in enumerate(selected_times):
        # Get flattened data for histogram
        orig_data = marginal_data[t_val].cpu().numpy().flatten()
        gen_data = generated_samples[t_val].cpu().numpy().flatten()
        
        # Plot histograms
        ax_hist = axes[0, i]
        ax_hist.hist(orig_data, bins=50, alpha=0.7, density=True, label='Original', color='blue')
        ax_hist.hist(gen_data, bins=50, alpha=0.7, density=True, label='Generated', color='red')
        ax_hist.set_title(f'Distributions at t={t_val:.2f}')
        ax_hist.set_xlabel('Pixel Value')
        ax_hist.set_ylabel('Density')
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        
        # Q-Q plot for distributional comparison
        ax_qq = axes[1, i]
        # Sort both datasets for Q-Q plot
        orig_sorted = np.sort(orig_data)
        gen_sorted = np.sort(gen_data)
        
        # Interpolate to same length for comparison
        n_points = min(len(orig_sorted), len(gen_sorted))
        orig_interp = np.interp(np.linspace(0, 1, n_points), 
                               np.linspace(0, 1, len(orig_sorted)), orig_sorted)
        gen_interp = np.interp(np.linspace(0, 1, n_points), 
                              np.linspace(0, 1, len(gen_sorted)), gen_sorted)
        
        ax_qq.scatter(orig_interp, gen_interp, alpha=0.5, s=1)
        
        # Add diagonal line for perfect match
        min_val = min(orig_interp.min(), gen_interp.min())
        max_val = max(orig_interp.max(), gen_interp.max())
        ax_qq.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Match')
        
        ax_qq.set_title(f'Q-Q Plot at t={t_val:.2f}')
        ax_qq.set_xlabel('Original Quantiles')
        ax_qq.set_ylabel('Generated Quantiles')
        ax_qq.legend()
        ax_qq.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "sample_distributions_comparison.png"), dpi=300)
    plt.close()
    

# ============================================================================
# Autocovariance Analysis Utilities (for GRF comparison)
# ============================================================================


def compute_spatial_acf_2d(samples: Tensor, resolution: int) -> Tensor:
    """
    Computes the normalized spatial ACF for a batch of 2D fields using FFT (Wiener-Khinchin).
    Assumes stationarity, periodicity, and ergodicity.
    """
    # Ensure computation is done on CPU for visualization
    samples = samples.cpu()
    B = samples.shape[0]
    fields = samples.reshape(B, resolution, resolution)

    # 1. Center the data (remove spatial mean per sample)
    spatial_mean = fields.mean(dim=[1, 2], keepdim=True)
    centered_fields = fields - spatial_mean

    # 2. Compute Power Spectral Density (PSD)
    # Use norm="ortho" to ensure consistency (Parseval's theorem)
    freq_domain = fft.fft2(centered_fields, norm="ortho")
    psd = torch.abs(freq_domain)**2

    # 3. Compute Autocovariance (ACVF) = IFFT(PSD)
    autocovariance = fft.ifft2(psd, norm="ortho").real # [B, H, W]

    # 4. Average over the batch to get the ensemble estimate
    avg_autocovariance = autocovariance.mean(dim=0) # [H, W]

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
    device = data_2d.device
    
    # Create coordinate grids with explicit device specification
    y, x = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32), 
                          torch.arange(W, device=device, dtype=torch.float32), indexing='ij')
    
    # Calculate radial distance from center
    r = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Binning by integer radius
    r_int = r.round().long()
    max_radius = min(center_x, center_y)
    
    # Flatten arrays for accumulation
    r_flat = r_int.flatten()
    data_flat = data_2d.flatten()
    
    # Use torch.bincount for efficient averaging
    # Sum values in each radial bin
    sum_in_bin = torch.bincount(r_flat, weights=data_flat)
    # Count elements in each radial bin
    count_in_bin = torch.bincount(r_flat)
    
    # Calculate average
    avg_in_bin = sum_in_bin / torch.clamp(count_in_bin, min=1)
    
    # Truncate to the meaningful radius
    radii = torch.arange(max_radius + 1, device=device, dtype=torch.float32)
    radial_profile = avg_in_bin[:max_radius + 1]
    
    return radii, radial_profile


def _visualize_radial_acf_comparison(marginal_data: Dict[float, Tensor], generated_samples: Dict[float, Tensor], output_dir: str):
    """Visualizes 1D radial ACF comparison between original and generated samples."""
    format_for_paper()
    print("  - Plotting 1D radial ACF comparison...")
    
    sorted_times = sorted(marginal_data.keys())
    # Select representative time points (e.g., 4 points including start and end)
    n_time_points = min(4, len(sorted_times))
    selected_indices = np.linspace(0, len(sorted_times)-1, n_time_points, dtype=int)
    selected_times = [sorted_times[i] for i in selected_indices]

    # Determine resolution
    if not selected_times: 
        return
    data_dim = marginal_data[selected_times[0]].shape[1]
    resolution = int(math.sqrt(data_dim))
    if resolution * resolution != data_dim:
        print("    Warning: Data is not square. Skipping 1D radial ACF analysis.")
        return

    # Create single figure for 1D radial ACF - reasonable paper size
    fig, axes = plt.subplots(1, n_time_points, figsize=(12, 3.5))
    if n_time_points == 1: 
        axes = [axes]
    fig.suptitle("Radial Autocorrelation Function Comparison", fontsize=12)
    
    for i, t_val in enumerate(selected_times):
        # Compute ACFs (ensure data is on CPU for computation)
        acf_target = compute_spatial_acf_2d(marginal_data[t_val].cpu(), resolution)
        acf_gen = compute_spatial_acf_2d(generated_samples[t_val].cpu(), resolution)
        
        # Calculate MSE_ACF
        mse_acf = torch.mean((acf_target - acf_gen)**2).item()

        # Compute Radial Averages
        radii, radial_target = radial_average(acf_target)
        _, radial_gen = radial_average(acf_gen)
        
        # Convert radii to physical distance (assuming domain L=1.0)
        physical_radii = radii.numpy() / resolution

        # 1D Radial ACF Comparison
        ax = axes[i]
        ax.plot(physical_radii, radial_target.numpy(), 'b-', label='Target', linewidth=2)
        ax.plot(physical_radii, radial_gen.numpy(), 'r--', label='Generated', linewidth=2)
        ax.set_title(f't = {t_val:.2f}\nMSE = {mse_acf:.2e}')
        ax.set_xlabel('Lag Distance (r/L)')
        ax.set_ylim(-0.1, 1.05)
        ax.grid(True, alpha=0.3)
        if i == 0: 
            ax.set_ylabel('Radial ACF R(r)')
        if i == n_time_points - 1:
            ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "radial_acf_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()


def _visualize_2d_acf_comparison(marginal_data: Dict[float, Tensor], generated_samples: Dict[float, Tensor], output_dir: str):
    """Visualizes 2D spatial ACF comparison between original and generated samples."""
    format_for_paper()
    print("  - Plotting 2D ACF comparison...")
    
    sorted_times = sorted(marginal_data.keys())
    # Select representative time points (e.g., 3 points for 2D visualization)
    n_time_points = min(3, len(sorted_times))
    selected_indices = np.linspace(0, len(sorted_times)-1, n_time_points, dtype=int)
    selected_times = [sorted_times[i] for i in selected_indices]

    # Determine resolution
    if not selected_times: 
        return
    data_dim = marginal_data[selected_times[0]].shape[1]
    resolution = int(math.sqrt(data_dim))
    if resolution * resolution != data_dim:
        print("    Warning: Data is not square. Skipping 2D ACF analysis.")
        return

    # Create figure for 2D ACF comparison - reasonable paper size
    fig, axes = plt.subplots(2, n_time_points, figsize=(10, 6))
    if n_time_points == 1: 
        axes = axes.reshape(2, 1)
    fig.suptitle("2D Spatial Autocorrelation Function Comparison", fontsize=12)

    # Global color scale for 2D ACF plots
    vmin, vmax = -0.5, 1.0
    
    for i, t_val in enumerate(selected_times):
        # Compute ACFs (ensure data is on CPU for computation)
        acf_target = compute_spatial_acf_2d(marginal_data[t_val].cpu(), resolution)
        acf_gen = compute_spatial_acf_2d(generated_samples[t_val].cpu(), resolution)
        
        # Calculate MSE_ACF
        mse_acf = torch.mean((acf_target - acf_gen)**2).item()

        # Row 1: 2D Target ACF
        ax_target = axes[0, i]
        im_target = ax_target.imshow(acf_target.numpy(), cmap='viridis', vmin=vmin, vmax=vmax, extent=[-0.5, 0.5, -0.5, 0.5])
        ax_target.set_title(f'Target ACF\nt = {t_val:.2f}')
        if i == 0: 
            ax_target.set_ylabel('Y Lag')
        
        # Row 2: 2D Generated ACF
        ax_gen = axes[1, i]
        im_gen = ax_gen.imshow(acf_gen.numpy(), cmap='viridis', vmin=vmin, vmax=vmax, extent=[-0.5, 0.5, -0.5, 0.5])
        ax_gen.set_title(f'Generated ACF\nMSE = {mse_acf:.2e}')
        ax_gen.set_xlabel('X Lag')
        if i == 0: 
            ax_gen.set_ylabel('Y Lag')
            
        # Add individual colorbars for each ACF plot
        plt.colorbar(im_target, ax=ax_target, fraction=0.046, pad=0.04)
        plt.colorbar(im_gen, ax=ax_gen, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2d_acf_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()


def _visualize_covariance_comparison(marginal_data: Dict[float, Tensor], generated_samples: Dict[float, Tensor], output_dir: str):
    """Wrapper function to generate both 1D and 2D ACF comparison plots."""
    _visualize_radial_acf_comparison(marginal_data, generated_samples, output_dir)
    _visualize_2d_acf_comparison(marginal_data, generated_samples, output_dir)