"""
Visualization Utilities for Asymmetric Bridges
==============================================

This module provides a set of functions for visualizing the results of
the asymmetric bridge models, including trajectory plots, marginal
distribution comparisons, and data fitting analysis.
"""

import torch
from torch import Tensor
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
from .validation import compute_sample_covariance_matrix, relative_covariance_frobenius_distance

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
    print("  - Plotting marginal distribution comparisons...")
    validation_times = [T * f for f in [0.25, 0.5, 0.75]]

    # Forward ODE Simulation
    z0 = bridge.base_dist.sample((n_viz_particles,)).to(device)
    if hasattr(bridge, 'flow'): # Expressive flow
        z0, _ = bridge.flow.forward(z0, torch.zeros(n_viz_particles, 1, device=device))
    else: # Gaussian flow
        mu0, gamma0 = bridge.get_params(torch.tensor([[0.0]], device=device, dtype=torch.float32))
        z0 = mu0 + gamma0 * z0

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
        zT = muT + gammaT * zT
        
    # Use the provided solver or default to Gaussian
    if solver is None:
        solver = solve_gaussian_bridge_reverse_sde
    reverse_path = solver(bridge, zT, T, 0.0, n_sde_steps).cpu()

    fig, axes = plt.subplots(2, len(validation_times), figsize=(6 * len(validation_times), 11), sharex=True, sharey=True)
    fig.suptitle("Figure 3: Comparison of Marginal Distributions p_t(z)", fontsize=16)

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
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "fig3_marginal_comparison.png"), dpi=300)
    plt.close()


def _plot_grf_marginals(marginal_data, output_dir, title="GRF Data Marginals"):
    """Visualizes samples from the GRF marginal data."""
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

    fig, axes = plt.subplots(n_samples_to_show, n_marginals, figsize=(3 * n_marginals, 3 * n_samples_to_show))
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


def _visualize_comparative_backward_samples(
    marginal_data: Dict[float, Tensor], 
    ground_truth_backward: Dict[float, Tensor], 
    learned_backward: Dict[float, Tensor], 
    output_dir: str
):
    """
    Visualize comparison between original data and backward samples starting from ground truth.
    Shows how well the learned reverse dynamics reproduce the ground truth evolution.
    Uses consistent sample selection to track the same samples across time points.
    """
    print("  - Plotting comparative backward samples (original vs ground truth backward)...")
    
    sorted_times = sorted(marginal_data.keys())
    n_time_points = len(sorted_times)
    n_samples_show = 4
    
    if not sorted_times:
        return
    data_dim = marginal_data[sorted_times[0]].shape[1]
    resolution = int(math.sqrt(data_dim))
    
    if resolution * resolution != data_dim:
        print(f"    Warning: Data dimension {data_dim} is not a perfect square. Skipping visualization.")
        return

    # Select consistent sample indices from the final time point
    final_time = max(sorted_times)
    n_available_samples = min(marginal_data[final_time].shape[0], ground_truth_backward[final_time].shape[0])
    n_samples_show = min(n_samples_show, n_available_samples)
    
    # Use consistent indices across all time points
    selected_indices = torch.randperm(n_available_samples)[:n_samples_show]

    fig, axes = plt.subplots(2 * n_samples_show, n_time_points, 
                           figsize=(3 * n_time_points, 3 * 2 * n_samples_show))
    fig.suptitle("Backward Samples: Original → GT-Backward (Consistent Samples)", fontsize=16)
    
    if axes.ndim == 1:
        axes = axes.reshape(-1, 1) if n_time_points == 1 else axes.reshape(1, -1)
    
    # Get global min/max for consistent color scaling
    all_original = torch.cat([marginal_data[t] for t in sorted_times], dim=0).cpu().numpy()
    all_gt_backward = torch.cat([ground_truth_backward[t] for t in sorted_times], dim=0).cpu().numpy()
    vmin = min(all_original.min(), all_gt_backward.min())
    vmax = max(all_original.max(), all_gt_backward.max())
    
    for j, t_val in enumerate(sorted_times):
        original_data = marginal_data[t_val].cpu().numpy()
        gt_backward_data = ground_truth_backward[t_val].cpu().numpy()
        
        for i in range(n_samples_show):
            # Use consistent sample indices
            sample_idx = selected_indices[i].item()
            
            # Original data (row 0, 2, 4, ...)
            row_orig = 2 * i
            if sample_idx < original_data.shape[0]:
                field_orig = original_data[sample_idx].reshape(resolution, resolution)
                axes[row_orig, j].imshow(field_orig, vmin=vmin, vmax=vmax, cmap='viridis')
            axes[row_orig, j].axis('off')
            if j == 0:
                axes[row_orig, j].set_ylabel(f'Original {i+1}', rotation=90, labelpad=15)
            if i == 0:
                axes[row_orig, j].set_title(f't = {t_val:.2f}')
            
            # Ground truth backward (row 1, 3, 5, ...)
            row_gt = 2 * i + 1
            if sample_idx < gt_backward_data.shape[0]:
                field_gt = gt_backward_data[sample_idx].reshape(resolution, resolution)
                axes[row_gt, j].imshow(field_gt, vmin=vmin, vmax=vmax, cmap='viridis')
            axes[row_gt, j].axis('off')
            if j == 0:
                axes[row_gt, j].set_ylabel(f'GT-Backward {i+1}', rotation=90, labelpad=15)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(output_dir, "comparative_backward_samples.png"), dpi=300, bbox_inches='tight')
    plt.close()


def _visualize_backward_samples_comparison(marginal_data: Dict[float, Tensor], generated_samples: Dict[float, Tensor], output_dir: str, sample_indices: Tensor = None):
    """
    Visualize comparison between original marginal data and backward generated samples.
    Shows multiple samples side by side for qualitative assessment.
    Uses consistent sample selection to track the same samples across time points.
    """
    print("  - Plotting backward samples vs marginal data comparison...")
    
    sorted_times = sorted(marginal_data.keys())
    n_time_points = len(sorted_times)
    n_samples_show = 4
    
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
        # Fallback: Select consistent sample indices from the final time point
        final_time = max(sorted_times)
        n_available_samples = min(marginal_data[final_time].shape[0], generated_samples[final_time].shape[0])
        n_samples_show = min(n_samples_show, n_available_samples)
        selected_indices = torch.randperm(n_available_samples)[:n_samples_show]

    fig, axes = plt.subplots(2 * n_samples_show, n_time_points, 
                           figsize=(3 * n_time_points, 3 * 2 * n_samples_show))
    fig.suptitle("Backward Generated Samples vs Original Marginal Data (Consistent Samples)", fontsize=16)
    
    if axes.ndim == 1:
        axes = axes.reshape(-1, 1) if n_time_points == 1 else axes.reshape(1, -1)
    
    all_original = torch.cat([marginal_data[t] for t in sorted_times], dim=0).cpu().numpy()
    all_generated = torch.cat([generated_samples[t] for t in sorted_times], dim=0).cpu().numpy()
    vmin = min(all_original.min(), all_generated.min())
    vmax = max(all_original.max(), all_generated.max())
    
    for j, t_val in enumerate(sorted_times):
        original_data = marginal_data[t_val].cpu().numpy()
        generated_data = generated_samples[t_val].cpu().numpy()
        
        for i in range(n_samples_show):
            # Use consistent sample indices
            sample_idx = selected_indices[i].item()
            
            row_orig = 2 * i
            if sample_idx < original_data.shape[0]:
                sample_orig = original_data[sample_idx].reshape(resolution, resolution)
                axes[row_orig, j].imshow(sample_orig, cmap='viridis', vmin=vmin, vmax=vmax, extent=[0, 1, 0, 1])
            axes[row_orig, j].axis('off')
            if j == 0:
                axes[row_orig, j].set_ylabel(f'Original {i+1}', rotation=0, ha='right', va='center')
            if i == 0:
                axes[row_orig, j].set_title(f't = {t_val:.2f}')
            
            row_gen = 2 * i + 1
            if sample_idx < generated_data.shape[0]:
                sample_gen = generated_data[sample_idx].reshape(resolution, resolution)
                axes[row_gen, j].imshow(sample_gen, cmap='viridis', vmin=vmin, vmax=vmax, extent=[0, 1, 0, 1])
            axes[row_gen, j].axis('off')
            if j == 0:
                axes[row_gen, j].set_ylabel(f'Generated {i+1}', rotation=0, ha='right', va='center')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(output_dir, "backward_samples_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()


def _visualize_marginal_statistics_comparison(marginal_data: Dict[float, Tensor], generated_samples: Dict[float, Tensor], output_dir: str):
    """
    Compare statistical properties (mean, std, distribution) between original and generated samples.
    """
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
    print("  - Plotting sample distributions comparison...")
    
    sorted_times = sorted(marginal_data.keys())
    n_plots = min(4, len(sorted_times))
    selected_times = [sorted_times[i] for i in np.linspace(0, len(sorted_times)-1, n_plots, dtype=int)]
    
    fig, axes = plt.subplots(2, n_plots, figsize=(4 * n_plots, 8))
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


def _visualize_statistical_covariance_comparison(
    marginal_data: Dict[float, Tensor], 
    generated_samples: Dict[float, Tensor], 
    output_dir: str,
    max_dim_for_heatmap: int = 2048
):
    """
    Visualizes and compares covariance statistics, saving THREE separate figures:
    1) Variance fields (target as color, generated as contours)
    2) Eigenvalue spectra
    3) Covariance heatmaps (target, generated, difference) [optional]
    """
    print("  - Plotting statistical covariance comparison (3 figures)...")
    sorted_times = sorted(marginal_data.keys())
    if not sorted_times:
        return

    n_time_points = min(3, len(sorted_times))
    selected_indices = np.linspace(0, len(sorted_times)-1, n_time_points, dtype=int)
    selected_times = [sorted_times[i] for i in selected_indices]

    data_dim = marginal_data[selected_times[0]].shape[1]
    resolution = int(math.sqrt(data_dim))
    is_square = (resolution * resolution == data_dim)

    visualize_heatmaps = (data_dim <= max_dim_for_heatmap)

    global_vmax_cov = 0
    global_vmax_diff = 0
    cov_data_list, cov_gen_list, metrics = [], [], {}

    for t_val in selected_times:
        samples_data = marginal_data[t_val].cpu()
        samples_gen = generated_samples[t_val].cpu()
        N_samples = min(samples_data.shape[0], samples_gen.shape[0])
        if N_samples < data_dim:
            print(f"    CRITICAL Warning: Insufficient samples (N={N_samples}) relative to D={data_dim} at t={t_val:.2f}. Estimates may be rank-deficient.")

        cov_data = compute_sample_covariance_matrix(samples_data)
        cov_gen = compute_sample_covariance_matrix(samples_gen)
        cov_data_list.append(cov_data)
        cov_gen_list.append(cov_gen)

        rel_f_dist = relative_covariance_frobenius_distance(cov_data, cov_gen)
        metrics[t_val] = {'Rel_F_dist': rel_f_dist}

        if visualize_heatmaps:
            max_abs_val = max(torch.abs(cov_data).max().item(), torch.abs(cov_gen).max().item())
            global_vmax_cov = max(global_vmax_cov, max_abs_val)
            diff = cov_data - cov_gen
            global_vmax_diff = max(global_vmax_diff, torch.abs(diff).max().item())

    # ---------------------------- Figure 1: Variance Fields ----------------------------
    # Create a figure with 2 rows (Target, Generated) and n_time_points columns
    fig_var, axes_var = plt.subplots(2, n_time_points, figsize=(6 * n_time_points, 9), sharex=True, sharey=True)
    if n_time_points == 1:
        axes_var = axes_var.reshape(2, 1)
    fig_var.suptitle("Variance Field Comparison", fontsize=16)

    # Compute a global max variance across all times for consistent scale
    if is_square:
        max_var = max(torch.diag(c).max().item() for c in cov_data_list + cov_gen_list if c.numel() > 0)
    else:
        max_var = None

    im_data = None  # keep reference to the last image mappable for a single shared colorbar

    for i, t_val in enumerate(selected_times):
        cov_data = cov_data_list[i]
        cov_gen = cov_gen_list[i]
        var_data = torch.diag(cov_data)
        var_gen = torch.diag(cov_gen)
        title_str = f"t={t_val:.2f} (Rel F-Dist={metrics[t_val]['Rel_F_dist']:.3f})"
        
        ax_target = axes_var[0, i]
        ax_gen = axes_var[1, i]

        if is_square:
            # Plot Target Variance (Top Row)
            arr_data = var_data.reshape(resolution, resolution).numpy()
            im_data = ax_target.imshow(arr_data, cmap='plasma', vmin=0, vmax=max_var)
            
            # Plot Generated Variance (Bottom Row)
            arr_gen = var_gen.reshape(resolution, resolution).numpy()
            ax_gen.imshow(arr_gen, cmap='plasma', vmin=0, vmax=max_var)

        else: # 1D data case
            ax_target.plot(var_data.numpy(), label='Target')
            ax_gen.plot(var_gen.numpy(), '--', label='Generated')
            ax_gen.set_xlabel("Dimension Index")
            if i == 0:
                ax_target.legend()
                ax_gen.legend()

        # Set titles and labels
        if i == 0:
            ax_target.set_ylabel("Target Variance")
            ax_gen.set_ylabel("Generated Variance")
        ax_target.set_title(title_str)


    # If square fields were plotted, place a single shared colorbar to the right
    if is_square and im_data is not None:
        fig_var.subplots_adjust(right=0.85)
        cax = fig_var.add_axes([0.88, 0.15, 0.04, 0.7])
        fig_var.colorbar(im_data, cax=cax, label="Variance")

    plt.tight_layout(rect=[0, 0.03, 0.85, 0.95])
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "statistical_covariance_variance_fields.png"), dpi=300)
    plt.close(fig_var)


    # ------------------------- Figure 2: Eigenvalue Spectra -------------------------
    fig_eig, axes_eig = plt.subplots(1, n_time_points, figsize=(6 * n_time_points, 5))
    if n_time_points == 1:
        axes_eig = np.array([axes_eig])
    fig_eig.suptitle("Eigenvalue Spectrum Comparison", fontsize=16)

    for i, t_val in enumerate(selected_times):
        cov_data = cov_data_list[i]
        cov_gen = cov_gen_list[i]
        ax_eig = axes_eig[i]
        try:
            eigvals_data = torch.linalg.eigvalsh(cov_data.double())
            eigvals_gen = torch.linalg.eigvalsh(cov_gen.double())
            eigvals_data = torch.sort(eigvals_data, descending=True)[0].float()
            eigvals_gen = torch.sort(eigvals_gen, descending=True)[0].float()
            eigvals_data = torch.clamp(eigvals_data, min=1e-12)
            eigvals_gen = torch.clamp(eigvals_gen, min=1e-12)
            ax_eig.loglog(eigvals_data.numpy(), 'b-', label='Target')
            ax_eig.loglog(eigvals_gen.numpy(), 'r--', label='Generated')
            ax_eig.set_title(f"t={t_val:.2f}")
            ax_eig.set_xlabel("Mode Index")
            if i == 0:
                ax_eig.set_ylabel("Eigenvalue (Log-Log)")
                ax_eig.legend()
            ax_eig.grid(True, which="both", ls="--", alpha=0.3)
        except Exception as e:
            print(f"    Warning: Eigenvalue computation failed at t={t_val:.2f}: {e}")
            ax_eig.text(0.5, 0.5, "Eig computation failed", ha='center', va='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "statistical_covariance_eigen_spectrum.png"), dpi=300)
    plt.close(fig_eig)

    # ---------------------- Figure 3: Covariance Heatmaps (opt) ----------------------
    if visualize_heatmaps:
        cmap_cov = 'coolwarm'
        cmap_diff = 'RdBu_r'
        fig_hm, axes_hm = plt.subplots(3, n_time_points, figsize=(4 * n_time_points, 12))
        if n_time_points == 1:
            axes_hm = axes_hm.reshape(3, 1)
        fig_hm.suptitle("Covariance Heatmaps (Target, Generated, Difference)", fontsize=16)

        im_target = None
        im_diff = None

        for i, t_val in enumerate(selected_times):
            cov_data = cov_data_list[i]
            cov_gen = cov_gen_list[i]

            ax_t = axes_hm[0, i]
            im_target = ax_t.imshow(cov_data.numpy(), cmap=cmap_cov, vmin=-global_vmax_cov, vmax=global_vmax_cov)
            ax_t.set_title(f"t={t_val:.2f} Target")
            if i == 0:
                ax_t.set_ylabel("Target")

            ax_g = axes_hm[1, i]
            ax_g.imshow(cov_gen.numpy(), cmap=cmap_cov, vmin=-global_vmax_cov, vmax=global_vmax_cov)
            ax_g.set_title("Generated")
            if i == 0:
                ax_g.set_ylabel("Generated")

            ax_d = axes_hm[2, i]
            diff = cov_data - cov_gen
            im_diff = ax_d.imshow(diff.numpy(), cmap=cmap_diff, vmin=-global_vmax_diff, vmax=global_vmax_diff)
            ax_d.set_title("Difference")
            if i == 0:
                ax_d.set_ylabel("Difference")

        # Reserve space on the right for two colorbars and avoid overlap with subplots
        fig_hm.subplots_adjust(right=0.82, hspace=0.35, wspace=0.3)

        # Add two dedicated colorbar axes on the right of the figure with controlled placement:
        # - cax_cov spans approximately the vertical space of the top two rows (Target + Generated)
        # - cax_diff sits below it aligned to the bottom row (Difference)
        cax_cov = fig_hm.add_axes([0.85, 0.35, 0.03, 0.52])   # [left, bottom, width, height]
        cax_diff = fig_hm.add_axes([0.85, 0.08, 0.03, 0.22])

        # Attach colorbars to the corresponding mappable objects
        if im_target is not None:
            fig_hm.colorbar(im_target, cax=cax_cov, label="Covariance")
        if im_diff is not None:
            fig_hm.colorbar(im_diff, cax=cax_diff, label="Difference")

        # Final layout adjustments and save
        plt.tight_layout(rect=[0, 0.03, 0.84, 0.95])
        plt.savefig(os.path.join(output_dir, "statistical_covariance_heatmaps.png"), dpi=300)
        plt.close(fig_hm)


def _plot_marginal_data_fit(bridge, marginal_data, T, output_dir, device):
    """Plot marginal data fit comparison."""
    print("  - Plotting marginal data fit...")

    sorted_times = sorted(marginal_data.keys())
    n_marginals = len(sorted_times)

    fig, axes = plt.subplots(2, n_marginals, figsize=(6 * n_marginals, 11), sharex=True, sharey=True)
    if n_marginals == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    fig.suptitle("Figure 4: Qualitative Fit to Marginal Data Constraints", fontsize=16)

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
            epsilon = bridge.base_dist.sample((samples_k.shape[0],))
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

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "fig4_marginal_fit.png"), dpi=300)
    plt.close()


def visualize_bridge_results(bridge, marginal_data: Dict[float, Tensor], T: float, output_dir: str, is_grf: bool = False, n_viz_particles: int = 50, n_sde_steps: int = 100, solver: str = 'euler'):
    """
    Generate and save a comprehensive set of visualizations for the trained bridge.
    """
    bridge.eval()
    device = next(bridge.parameters()).device
    print("\n--- Generating and Saving Visualizations ---")
    
    marginal_data_cpu = {t: samples.cpu() for t, samples in marginal_data.items()}

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
            final_samples_for_viz = marginal_data[final_time][viz_sample_indices]
            
            # Generate backward samples starting from these exact samples
            print(f"    Using {len(viz_sample_indices)} specific samples for consistent comparison")
            ground_truth_backward = generate_backward_samples(
                bridge, marginal_data, 
                n_samples=len(viz_sample_indices),  # This will be ignored since we provide z_final_ground_truth
                n_steps=n_sde_steps, 
                device=device,
                solver=sde_solver,
                z_final_ground_truth=final_samples_for_viz
            )
            
            # Create original_data dict using the same sample indices
            original_data = {}
            for t_val in sorted(marginal_data.keys()):
                original_data[t_val] = marginal_data[t_val][viz_sample_indices]
            
            # Convert to CPU for visualization
            original_data_cpu = {t: samples.cpu() for t, samples in original_data.items()}
            ground_truth_backward_cpu = {t: samples.cpu() for t, samples in ground_truth_backward.items()}
            
            # Denormalize both datasets
            for t, samples in original_data_cpu.items():
                original_data_cpu[t] = bridge.denormalize(samples)
            for t, samples in ground_truth_backward_cpu.items():
                ground_truth_backward_cpu[t] = bridge.denormalize(samples)

            # Add visualizations for comparative backward samples (subset)
            # Note: original_data_cpu and ground_truth_backward_cpu were created by
            # subsetting with `viz_sample_indices`, so pass local indices (0..n_viz_samples-1)
            local_indices = torch.arange(len(viz_sample_indices))
            _visualize_backward_samples_comparison(original_data_cpu, ground_truth_backward_cpu, output_dir, local_indices)
            _visualize_marginal_statistics_comparison(original_data_cpu, ground_truth_backward_cpu, output_dir)
            _visualize_sample_distributions(original_data_cpu, ground_truth_backward_cpu, output_dir)

            # For statistically accurate covariance comparisons, use FULL batches
            # anchored at the ground-truth final samples so t=1 covariances are identical.
            z_final_full_gt = marginal_data[final_time]  # all available GT samples at t=final
            generated_full = generate_backward_samples(
                bridge, marginal_data,
                n_samples=z_final_full_gt.shape[0],  # ignored when z_final_ground_truth is provided
                n_steps=n_sde_steps,
                device=device,
                solver=sde_solver,
                z_final_ground_truth=z_final_full_gt
            )

            # Prepare full-batch CPU, denormalized
            marginal_full_cpu = {t: bridge.denormalize(marginal_data[t].cpu()) for t in sorted(marginal_data.keys())}
            generated_full_cpu = {t: bridge.denormalize(generated_full[t].cpu()) for t in sorted(generated_full.keys())}

            # Covariance comparisons (ACF + full statistical covariance) with full batches
            _visualize_covariance_comparison(marginal_full_cpu, generated_full_cpu, output_dir)
            _visualize_statistical_covariance_comparison(marginal_full_cpu, generated_full_cpu, output_dir)
            
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
    Visualize comparison between original marginal data and backward generated samples.
    Uses consistent sample selection to track the same samples across time points.
    """
    print("  - Plotting backward samples vs marginal data comparison...")
    
    sorted_times = sorted(marginal_data.keys())
    n_time_points = len(sorted_times)
    n_samples_show = 4
    
    # Determine resolution
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
        # Fallback: Select consistent sample indices from the final time point
        final_time = max(sorted_times)
        n_available_samples = min(marginal_data[final_time].shape[0], generated_samples[final_time].shape[0])
        n_samples_show = min(n_samples_show, n_available_samples)
        selected_indices = torch.randperm(n_available_samples)[:n_samples_show]

    # Create figure with 2 rows (original vs generated) and multiple columns
    fig, axes = plt.subplots(2 * n_samples_show, n_time_points, 
                           figsize=(3 * n_time_points, 3 * 2 * n_samples_show))
    fig.suptitle("Backward Generated Samples vs Original Marginal Data (Consistent Samples)", fontsize=16)
    
    # Handle dimension edge cases
    if axes.ndim == 1:
        axes = axes.reshape(-1, 1) if n_time_points == 1 else axes.reshape(1, -1)
    
    # Determine global color scale for consistency
    all_original = torch.cat([marginal_data[t] for t in sorted_times], dim=0).cpu().numpy()
    all_generated = torch.cat([generated_samples[t] for t in sorted_times], dim=0).cpu().numpy()
    vmin = min(all_original.min(), all_generated.min())
    vmax = max(all_original.max(), all_generated.max())
    
    for j, t_val in enumerate(sorted_times):
        original_data = marginal_data[t_val].cpu().numpy()
        generated_data = generated_samples[t_val].cpu().numpy()
        
        for i in range(n_samples_show):
            # Use consistent sample indices
            sample_idx = selected_indices[i].item()
            
            # Original data samples
            row_orig = 2 * i
            if sample_idx < original_data.shape[0]:
                sample_orig = original_data[sample_idx].reshape(resolution, resolution)
                axes[row_orig, j].imshow(sample_orig, cmap='viridis', vmin=vmin, vmax=vmax, extent=[0, 1, 0, 1])
            axes[row_orig, j].axis('off')
            if j == 0:
                axes[row_orig, j].set_ylabel(f'Original {i+1}', rotation=0, ha='right', va='center')
            if i == 0:
                axes[row_orig, j].set_title(f't = {t_val:.2f}')
            
            # Generated data samples  
            row_gen = 2 * i + 1
            if sample_idx < generated_data.shape[0]:
                sample_gen = generated_data[sample_idx].reshape(resolution, resolution)
                axes[row_gen, j].imshow(sample_gen, cmap='viridis', vmin=vmin, vmax=vmax, extent=[0, 1, 0, 1])
            axes[row_gen, j].axis('off')
            if j == 0:
                axes[row_gen, j].set_ylabel(f'Generated {i+1}', rotation=0, ha='right', va='center')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(output_dir, "backward_samples_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()


def _visualize_marginal_statistics_comparison(marginal_data: Dict[float, Tensor], generated_samples: Dict[float, Tensor], output_dir: str):
    """
    Compare statistical properties (mean, std, distribution) between original and generated samples.
    """
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
    print("  - Plotting sample distributions comparison...")
    
    sorted_times = sorted(marginal_data.keys())
    # Select a few representative time points
    n_plots = min(4, len(sorted_times))
    selected_times = [sorted_times[i] for i in np.linspace(0, len(sorted_times)-1, n_plots, dtype=int)]
    
    fig, axes = plt.subplots(2, n_plots, figsize=(4 * n_plots, 8))
    if n_plots == 1:
        axes = axes.reshape(-1, 1)
    fig.suptitle("Pixel Value Distributions: Original vs Generated", fontsize=14)
    
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
    # Sum values in each radial bin
    sum_in_bin = torch.bincount(r_flat, weights=data_flat)
    # Count elements in each radial bin
    count_in_bin = torch.bincount(r_flat)
    
    # Calculate average
    avg_in_bin = sum_in_bin / torch.clamp(count_in_bin, min=1)
    
    # Truncate to the meaningful radius
    radii = torch.arange(max_radius + 1, device=data_2d.device)
    radial_profile = avg_in_bin[:max_radius + 1]
    
    return radii, radial_profile


def _visualize_covariance_comparison(marginal_data: Dict[float, Tensor], generated_samples: Dict[float, Tensor], output_dir: str):
    """Visualizes and compares the spatial ACF between original and generated samples."""
    print("  - Plotting spatial covariance (ACF) comparison...")
    
    sorted_times = sorted(marginal_data.keys())
    # Select representative time points (e.g., 5 points including start and end)
    n_time_points = min(5, len(sorted_times))
    selected_indices = np.linspace(0, len(sorted_times)-1, n_time_points, dtype=int)
    selected_times = [sorted_times[i] for i in selected_indices]

    # Determine resolution
    if not selected_times: 
        return
    data_dim = marginal_data[selected_times[0]].shape[1]
    resolution = int(math.sqrt(data_dim))
    if resolution * resolution != data_dim:
        print("    Warning: Data is not square. Skipping covariance analysis.")
        return

    # Setup figure: 3 rows (1D Radial ACF, 2D Target ACF, 2D Gen ACF)
    fig, axes = plt.subplots(3, n_time_points, figsize=(4 * n_time_points, 12))
    if n_time_points == 1: 
        axes = axes.reshape(3, 1)
    fig.suptitle("Spatial Autocorrelation Function (ACF) Comparison (Second-Order Statistics)", fontsize=16)

    # Global color scale for 2D ACF plots
    vmin, vmax = -0.5, 1.0
    im = None
    
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

        # Row 1: 1D Radial ACF Comparison
        ax_1d = axes[0, i]
        ax_1d.plot(physical_radii, radial_target.numpy(), 'b-', label='Target', linewidth=2)
        ax_1d.plot(physical_radii, radial_gen.numpy(), 'r--', label='Generated', linewidth=2)
        ax_1d.set_title(f't={t_val:.2f} (MSE={mse_acf:.2e})')
        ax_1d.set_xlabel('Lag Distance (r/L)')
        ax_1d.set_ylim(-0.1, 1.05)
        ax_1d.grid(True, alpha=0.3)
        if i == 0: 
            ax_1d.set_ylabel('Radial ACF R(r)')
        if i == n_time_points - 1:
            ax_1d.legend()

        # Row 2: 2D Target ACF
        ax_target = axes[1, i]
        im = ax_target.imshow(acf_target.numpy(), cmap='viridis', vmin=vmin, vmax=vmax, extent=[-0.5, 0.5, -0.5, 0.5])
        ax_target.set_title('Target 2D ACF')
        if i == 0: 
            ax_target.set_ylabel('Y Lag')
        
        # Row 3: 2D Generated ACF
        ax_gen = axes[2, i]
        ax_gen.imshow(acf_gen.numpy(), cmap='viridis', vmin=vmin, vmax=vmax, extent=[-0.5, 0.5, -0.5, 0.5])
        ax_gen.set_title('Generated 2D ACF')
        ax_gen.set_xlabel('X Lag')
        if i == 0: 
            ax_gen.set_ylabel('Y Lag')

    # Add colorbar
    if im is not None:
        fig.colorbar(im, ax=axes[1:, :].ravel().tolist(), fraction=0.046, pad=0.04, label="Correlation")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "covariance_comparison.png"), dpi=300)
    plt.close()