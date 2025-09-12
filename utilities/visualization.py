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
from matplotlib.lines import Line2D
import numpy as np
import os
import math
from scipy.integrate import solve_ivp
from scipy.stats import gaussian_kde
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ..asymmetric_bridge_distributions_grf import NeuralGaussianBridge
    from ..asymmetric_bridge_expressive_flow import NeuralBridgeExpressive

from .simulation import solve_gaussian_bridge_reverse_sde, solve_backward_sde_euler_maruyama, generate_backward_samples, generate_comparative_backward_samples

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
        mu0, gamma0 = bridge.get_params(torch.tensor([[0.0]], device=device))
        z0 = mu0 + gamma0 * z0

    def forward_ode_func(t, z_np):
        z = torch.from_numpy(z_np).float().to(device).reshape(n_viz_particles, -1)
        with torch.no_grad():
            v = bridge.forward_velocity(z, torch.tensor([[t]], device=device))
        return v.cpu().numpy().flatten()

    times_eval = np.linspace(0, T, n_sde_steps + 1)
    sol = solve_ivp(fun=forward_ode_func, t_span=[0, T], y0=z0.cpu().numpy().flatten(), method='RK45', t_eval=times_eval)
    forward_path = torch.from_numpy(sol.y.T).float().reshape(len(times_eval), n_viz_particles, -1)

    # Reverse SDE Simulation
    zT = bridge.base_dist.sample((n_viz_particles,)).to(device)
    if hasattr(bridge, 'flow'):
        zT, _ = bridge.flow.forward(zT, torch.full((n_viz_particles, 1), T, device=device))
    else:
        muT, gammaT = bridge.get_params(torch.tensor([[T]], device=device))
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
        t_tensor = torch.tensor([[t_val]], device=device)
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
    
    if not sorted_times: return
    data_dim = marginal_data[sorted_times[0]].shape[1]
    resolution = int(math.sqrt(data_dim))
    
    if resolution * resolution != data_dim:
        print(f"Warning: Data dimension {data_dim} is not a perfect square.")
        return

    fig, axes = plt.subplots(n_samples_to_show, n_marginals, figsize=(3 * n_marginals, 3 * n_samples_to_show))
    if axes.ndim == 1: axes = np.array([axes])
    
    all_data = torch.cat([marginal_data[t] for t in sorted_times], dim=0).cpu().numpy()
    vmin, vmax = all_data.min(), all_data.max()

    for i in range(n_samples_to_show):
        for j, t_k in enumerate(sorted_times):
            ax = axes[i, j]
            if i < marginal_data[t_k].shape[0]:
                 sample = marginal_data[t_k][i].cpu().numpy().reshape(resolution, resolution)
                 ax.imshow(sample, cmap='viridis', vmin=vmin, vmax=vmax, extent=[0, 1, 0, 1])
            ax.axis('off')
            if i == 0: ax.set_title(f"t = {t_k:.2f}")

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

    fig, axes = plt.subplots(2 * n_samples_show, n_time_points, 
                           figsize=(3 * n_time_points, 3 * 2 * n_samples_show))
    fig.suptitle("Backward Samples: Original → GT-Backward", fontsize=16)
    
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
            # Original data (row 0, 2, 4, ...)
            row_orig = 2 * i
            if i < original_data.shape[0]:
                field_orig = original_data[i].reshape(resolution, resolution)
                axes[row_orig, j].imshow(field_orig, vmin=vmin, vmax=vmax, cmap='viridis')
            axes[row_orig, j].axis('off')
            if j == 0:
                axes[row_orig, j].set_ylabel(f'Original {i+1}', rotation=90, labelpad=15)
            if i == 0:
                axes[row_orig, j].set_title(f't = {t_val:.2f}')
            
            # Ground truth backward (row 1, 3, 5, ...)
            row_gt = 2 * i + 1
            if i < gt_backward_data.shape[0]:
                field_gt = gt_backward_data[i].reshape(resolution, resolution)
                axes[row_gt, j].imshow(field_gt, vmin=vmin, vmax=vmax, cmap='viridis')
            axes[row_gt, j].axis('off')
            if j == 0:
                axes[row_gt, j].set_ylabel(f'GT-Backward {i+1}', rotation=90, labelpad=15)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(output_dir, "comparative_backward_samples.png"), dpi=300, bbox_inches='tight')
    plt.close()


def _visualize_backward_samples_comparison(marginal_data: Dict[float, Tensor], generated_samples: Dict[float, Tensor], output_dir: str):
    """
    Visualize comparison between original marginal data and backward generated samples.
    Shows multiple samples side by side for qualitative assessment.
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

    fig, axes = plt.subplots(2 * n_samples_show, n_time_points, 
                           figsize=(3 * n_time_points, 3 * 2 * n_samples_show))
    fig.suptitle("Backward Generated Samples vs Original Marginal Data", fontsize=16)
    
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
            row_orig = 2 * i
            if i < original_data.shape[0]:
                sample_orig = original_data[i].reshape(resolution, resolution)
                axes[row_orig, j].imshow(sample_orig, cmap='viridis', vmin=vmin, vmax=vmax, extent=[0, 1, 0, 1])
            axes[row_orig, j].axis('off')
            if j == 0:
                axes[row_orig, j].set_ylabel(f'Original {i+1}', rotation=0, ha='right', va='center')
            if i == 0:
                axes[row_orig, j].set_title(f't = {t_val:.2f}')
            
            row_gen = 2 * i + 1
            if i < generated_data.shape[0]:
                sample_gen = generated_data[i].reshape(resolution, resolution)
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
        t_k_tensor = torch.tensor([[t_k]], device=device)
        
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

        ax_contour.set_title(f'KDE Contour Comparison')
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
            n_backward_samples = min(128, list(marginal_data.values())[0].shape[0])
            
            # Generate backward samples starting from ground truth data
            original_data, ground_truth_backward = generate_comparative_backward_samples(
                bridge, marginal_data, 
                n_samples=n_backward_samples, 
                n_steps=n_sde_steps, 
                device=device,
                solver=sde_solver
            )
            
            # Convert to CPU for visualization
            ground_truth_backward_cpu = {t: samples.cpu() for t, samples in ground_truth_backward.items()}
            # Use the original marginal_data_cpu that's already denormalized for the first argument
            
            # denormalize the backward samples
            for t, samples in ground_truth_backward_cpu.items():
                ground_truth_backward_cpu[t] = bridge.denormalize(samples)
            for t, samples in marginal_data_cpu.items():
                marginal_data_cpu[t] = bridge.denormalize(samples)
            

            # Add visualizations for comparative backward samples
            _visualize_comparative_backward_samples(marginal_data_cpu, ground_truth_backward_cpu, ground_truth_backward_cpu, output_dir)
            _visualize_backward_samples_comparison(marginal_data_cpu, ground_truth_backward_cpu, output_dir)
            _visualize_marginal_statistics_comparison(marginal_data_cpu, ground_truth_backward_cpu, output_dir)
            _visualize_sample_distributions(marginal_data_cpu, ground_truth_backward_cpu, output_dir)
        else:
            # ... standard spiral data visualizations
            _plot_marginal_distribution_comparison(bridge, T, n_viz_particles, n_sde_steps, output_dir, device, sde_solver)
            _plot_marginal_data_fit(bridge, marginal_data_cpu, T, output_dir, device)

    print(f"Visualizations saved to '{output_dir}' directory.")
