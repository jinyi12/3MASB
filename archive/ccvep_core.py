#!/usr/bin/env python3
"""
Coupled Consistent Velocity-Energy Parameterization (C-CVEP) Core Components
============================================================================

This module implements the core components for the C-CVEP framework, integrating
Dual Score Matching (DSM) and Metric Flow Matching (MFM) for simulation-free
training of neural bridges, using a two-stage optimization approach.

The implementation supports multi-marginal interpolation following Neklyudov et al. [2023b],
where the interpolation between marginals uses a shared neural network φ_η with global
endpoint information for parameter sharing across timesteps.

Key Components:
- VelocityNetwork: Parameterizes velocity field v_theta(x, s)
- EnergyNetwork: Parameterizes energy U_phi(x, s) using homogeneous architecture
- InterpolatorNetwork: Parameterizes MFM trajectory correction term phi_eta(t, x0, x1)
  using multi-marginal interpolation with shared parameters across timesteps
- TrajectoryHandler: Manages parameterized multi-marginal interpolation
- RBFMetricHandler: Learns data-dependent Riemannian metrics for enhanced geodesic computation
- Loss functions: Trajectory matching and consistency losses
- Training steps: Three-stage RBF+MFM+DSM optimization

Multi-Marginal Interpolation Formula:
x_t = ((t_{i+1} - t)/(t_{i+1} - t_i)) * x_{t_i} + ((t - t_i)/(t_{i+1} - t_i)) * x_{t_{i+1}} +
      (1 - ((t_{i+1} - t)/(t_{i+1} - t_i))^2 - ((t - t_i)/(t_{i+1} - t_i))^2) * φ_η(t, x_0, x_1)

where x_0, x_1 are global endpoints and φ_η parameters are shared across all timesteps.
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, Callable, Any, List

# Import spatial components from backbones.py
try:
    from backbones import TimeConditionedUNet
except ImportError:
    print("Warning: Could not import spatial components from backbones.py.")

# Import KMeans for RBF initialization
try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None


# =============================================================================
# 0. Utility Functions for Automatic Differentiation
# =============================================================================

def jvp(f: Callable[[torch.Tensor], Any], x: torch.Tensor, v: torch.Tensor) -> Tuple[Any, Any]:
    """Compute Jacobian-vector product. Used for time derivatives."""
    # torch.autograd.functional.jvp returns (primal_output, tangent_output)
    return torch.autograd.functional.jvp(
        f, x, v,
        create_graph=torch.is_grad_enabled()
    )

def t_dir(f: Callable[[torch.Tensor], Any], t: torch.Tensor) -> Tuple[Any, Any]:
    """Compute the time derivative of f(t) by using jvp with v=1. Returns (f(t), df/dt)."""
    # Wrap the output in tuples to match the expected return format
    primal, tangent = jvp(f, t, torch.ones_like(t))
    return (primal,), (tangent,)


# =============================================================================
# 1. Core Network Architectures (Velocity and Energy)
# =============================================================================

class VelocityNetwork(nn.Module):
    """
    Parameterizes the velocity field v_theta(x, s).
    Includes functionality for efficient divergence estimation.
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.net = backbone

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # Ensure s is correctly shaped if the backbone expects specific conditioning
        if s.ndim == 1:
            s = s.view(-1, 1)
        return self.net(x, s)

    def compute_divergence(self, x: torch.Tensor, s: torch.Tensor, noise_type: str = 'rademacher') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimates the divergence nabla_x . v_theta(x, s) using the Skilling-Hutchinson trace estimator.
        Returns both the divergence and the velocity output.
        """
        # We use torch.enable_grad() and clone the input to ensure the computation graph
        # is correctly constructed for higher-order gradients.
        with torch.enable_grad():
            # Clone and require grad to make it a leaf node for this computation
            x_inp = x.clone().requires_grad_(True)
            
            # 1. Forward pass
            velocity = self(x_inp, s)
            
            # 2. Generate noise vector eta (probe vector)
            if noise_type == 'rademacher':
                # Rademacher noise often provides lower variance estimates
                eta = torch.randint(0, 2, size=x.shape, device=x.device, dtype=x.dtype) * 2 - 1
            else: # Gaussian
                eta = torch.randn_like(x)

            # 3. Compute Vector-Jacobian Product (VJP): J^T eta
            # create_graph=True is essential as the divergence is part of the Consistency Loss.
            vjp = autograd.grad(
                outputs=velocity,
                inputs=x_inp,
                grad_outputs=eta,
                create_graph=True
            )[0]
        
        # 4. Compute divergence estimate: <eta, VJP>
        dims = tuple(range(1, x.ndim))
        # Output shape: (B, 1)
        divergence = torch.sum(vjp * eta, dim=dims, keepdim=True)
        
        # We return the velocity computed on x_inp as it tracks gradients needed for the consistency loss
        return divergence, velocity


class EnergyNetwork(nn.Module):
    """
    Parameterizes the Energy U_phi(x, s) using the homogeneous architecture (DSM inspiration):
    U_phi(x, s) = 0.5 * <x, S_phi(x, s)>.
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.S_phi = backbone # The backbone represents the score S_phi

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Computes the scalar energy value U_phi(x, s)."""
         # Ensure s is correctly shaped for conditioning
        if s.ndim == 1:
            s_cond = s.view(-1, 1)
        else:
            s_cond = s

        S_phi_out = self.S_phi(x, s_cond)
        
        # Compute <x, S_phi>
        dims = tuple(range(1, x.ndim))
        # Output shape: (B, 1)
        energy = 0.5 * torch.sum(x * S_phi_out, dim=dims, keepdim=True)
        return energy

    def compute_scores(self, x: torch.Tensor, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Efficiently computes spatial score (nabla_x U_phi) and scale score (partial_s U_phi).
        """
        # Robust setup for higher-order gradients.
        with torch.enable_grad():
            # Clone and require grad to make them leaf nodes for this computation
            x_inp = x.clone().requires_grad_(True)
            s_inp = s.clone().requires_grad_(True)
            
            energy = self(x_inp, s_inp)
            
            # Compute gradients. create_graph=True is crucial for the Consistency Loss.
            grads = autograd.grad(
                outputs=energy,
                inputs=(x_inp, s_inp),
                grad_outputs=torch.ones_like(energy),
                create_graph=True
            )
        
        spatial_score = grads[0] # nabla_x U
        scale_score = grads[1]   # partial_s U

        # Ensure scale_score has the shape (B, 1) matching the energy output
        if scale_score.numel() == x.shape[0] and scale_score.shape != energy.shape:
             scale_score = scale_score.view_as(energy)

        return spatial_score, scale_score


# =============================================================================
# 1.5. MFM Components (Interpolator Network)
# =============================================================================

class InterpolatorNetwork(nn.Module):
    """
    Parameterizes the MFM trajectory correction term phi_eta(t, x0, x1).
    Implements the multi-marginal parameterized interpolant following Neklyudov et al. [2023b]:
    
    x_t = ((t_{i+1} - t)/(t_{i+1} - t_i)) * x_{t_i} + ((t - t_i)/(t_{i+1} - t_i)) * x_{t_{i+1}} +
          (1 - ((t_{i+1} - t)/(t_{i+1} - t_i))^2 - ((t - t_i)/(t_{i+1} - t_i))^2) * phi_eta(t, x_0, x_1)
    
    where x_0, x_1 are the global endpoints and parameters eta are shared across timesteps.
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.net = backbone

    def forward(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Computes the correction term phi_eta(t, x0, x1) using global endpoints."""
        return self.net(t, x0, x1)

    def compute_trajectory(self, t: torch.Tensor, xk: torch.Tensor, xk1: torch.Tensor, 
                         tk: torch.Tensor, hk: torch.Tensor, 
                         x0: torch.Tensor, x1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the multi-marginal trajectory x_t and its time derivative using the enhanced formula.
        
        Args:
            t: Global time (B,).
            xk, xk1: Segment endpoints (B, D...).
            tk: Segment start time (B,).
            hk: Segment duration (B,).
            x0, x1: Global trajectory endpoints (B, D...).
        """
        B = xk.shape[0]
        
        # Ensure time tensors are (B,) for AD calculation and network input
        if t.dim() > 1:
            t = t.view(-1)
        if tk.dim() > 1:
            tk = tk.view(-1)
        if hk.dim() > 1:
            hk = hk.view(-1)

        # Calculate normalized time tau = (t - tk) / hk for spatial broadcasting
        view_shape = [B] + [1] * (xk.dim() - 1)
        
        t_exp = t.view(view_shape)
        tk_exp = tk.view(view_shape)
        hk_exp = hk.view(view_shape)
        
        # tau = (t - t_i) / (t_{i+1} - t_i)
        tau = (t_exp - tk_exp) / hk_exp
        tau_comp = 1.0 - tau  # (t_{i+1} - t) / (t_{i+1} - t_i)

        # Define the function for time derivative calculation
        def phi_func(t_in):
            # Network uses global endpoints x0, x1 (shared parameters)
            return self(t_in, x0, x1)

        # Compute phi and dot_phi (wrt global time t) simultaneously using JVP
        (phi,), (dot_phi,) = t_dir(phi_func, t)

        # Multi-marginal interpolation formula:
        # x_t = tau_comp * x_k + tau * x_{k+1} + (1 - tau_comp^2 - tau^2) * phi_eta(t, x_0, x_1)
        
        # Linear interpolation term
        linear_term = tau_comp * xk + tau * xk1
        
        # Correction weight: (1 - tau_comp^2 - tau^2)
        correction_weight = 1.0 - tau_comp**2 - tau**2
        
        # Full trajectory
        x_t = linear_term + correction_weight * phi
        
        # Calculate velocity dot_x_t using chain rule
        # d/dt[tau_comp * x_k + tau * x_{k+1}] = (1/hk) * (x_{k+1} - x_k)
        linear_velocity = (xk1 - xk) / hk_exp
        
        # d/dt[correction_weight] = d/dt[1 - tau_comp^2 - tau^2] = -2*tau_comp*(-1/hk) - 2*tau*(1/hk) = (2/hk)*(tau_comp - tau)
        correction_weight_dot = (2.0 / hk_exp) * (tau_comp - tau)
        
        # Total velocity
        dot_x_t = linear_velocity + correction_weight_dot * phi + correction_weight * dot_phi
        
        return x_t, dot_x_t

class InterpolatorBackboneMLP(nn.Module):
    """
    Simple MLP backbone for InterpolatorNetwork phi_eta(t, x0, x1).
    Handles input concatenation and flattening for high-dimensional data.
    Uses global endpoints x0, x1 with shared parameters across timesteps.
    """
    def __init__(self, data_dim: int, hidden_dims: list):
        super().__init__()
        # Input dim: 1 (t) + 2 * data_dim (x0, x1). data_dim is the flattened dimension.
        self.data_dim = data_dim
        input_dim = 1 + 2 * data_dim
        
        dims = [input_dim] + hidden_dims + [data_dim]
        layers = []
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                # Use SiLU activation for smoothness
                layers.append(nn.SiLU())
        
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Computes phi_eta(t, x0, x1) using global endpoints."""
        B = x0.shape[0]
        
        # Ensure t is (B, 1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        elif t.dim() == 0:
             t = t.view(1, 1).expand(B, 1)

        # Flatten inputs if they have spatial dimensions
        original_shape = x0.shape[1:]
        if x0.dim() > 2:
            x0_flat = x0.view(B, -1)
            x1_flat = x1.view(B, -1)
        else:
            x0_flat, x1_flat = x0, x1

        # Input is concatenated (t, x0, x1)
        inp = torch.cat([t, x0_flat, x1_flat], dim=-1)
        output = self.net(inp)
        
        # Reshape output back to original spatial dimensions
        if x0.dim() > 2:
            output = output.view(B, *original_shape)

        return output


class InterpolatorBackboneUNet(nn.Module):
    """
    U-Net backbone for InterpolatorNetwork. Preserves spatial correlations.
    Designed for spatial data like GRF fields. Uses global endpoints x0, x1.
    """
    def __init__(self, resolution: int, channels: int = 1, hidden_dims: List[int] = [64, 128], time_embedding_dim: int = 64):
        super().__init__()
        self.resolution = resolution
        self.channels = channels

        # Input channels = 2 * C (x0 and x1 concatenated), Output channels = C
        self.unet = TimeConditionedUNet(
            data_dim=resolution*resolution*channels*2,  # Proxy data_dim
            resolution=resolution,
            in_channels=channels * 2,
            out_channels=channels,
            hidden_dims=hidden_dims,
            time_embedding_dim=time_embedding_dim
        )

    def _format_input(self, x: torch.Tensor) -> torch.Tensor:
        """Format input to spatial dimensions."""
        B = x.shape[0]
        if x.dim() == 2:
            return x.view(B, self.channels, self.resolution, self.resolution)
        elif x.dim() == 4:
            return x
        else:
            raise ValueError(f"Unsupported input dimensions: {x.dim()}.")

    def forward(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Computes the correction term phi_eta(t, x0, x1) using global endpoints and U-Net."""
        x0_spatial = self._format_input(x0)
        x1_spatial = self._format_input(x1)
        original_shape = x0.shape

        # Concatenate global endpoints along channel dimension
        x_spatial = torch.cat([x0_spatial, x1_spatial], dim=1)

        if t.dim() > 1: 
            t = t.view(-1)
        
        out = self.unet(x_spatial, t)
        
        if len(original_shape) == 2:
            return out.view(original_shape)
        return out


# =============================================================================
# 2. Metric Flow Matching: RBF Metric Handler
# =============================================================================

class RBFMetricHandler(nn.Module):
    """
    Handles the learning (Stage 0) and computation of the RBF metric G_RBF(x).
    Implements data-dependent Riemannian metric for improved geodesic computation.
    
    Supports both isotropic and anisotropic kernels:
    
    ISOTROPIC MODE (default, legacy behavior):
    - Scalar output h(x) per point (not per-dimension)
    - Euclidean distances to cluster centers  
    - Single bandwidth per cluster across all dimensions
    - Diagonal metric tensor: M_ii = 1/(h(x) + epsilon)^alpha
    
    ANISOTROPIC MODE (new, dimension-aware):
    - Vector output h(x) with per-dimension values
    - Per-dimension bandwidths: lambda_k_d = 0.5 / ((kappa * sigma_k_d)**2 + 1e-6)
    - Mahalanobis-like distance: sum_d(lambda_k_d * (x_d - c_k_d)^2)
    - Diagonal metric tensor: M_ii(x) = 1/(h_i(x) + epsilon)^alpha (different per dimension)
    
    Reference: https://github.com/kksniak/metric-flow-matching/blob/main/mfm/geo_metrics/rbf.py
    
    CRITICAL: Trained on original data space but applied to normalized data.
    """
    def __init__(self, data_dim: int, num_clusters: int = 50, epsilon: float = 1e-3, 
                 alpha: float = 1.0, device: str = 'cpu',
                 data_mean: torch.Tensor = None, data_std: torch.Tensor = None,
                 anisotropic: bool = False):
        super().__init__()
        self.D = data_dim
        self.K = num_clusters
        self.epsilon = epsilon
        self.alpha = alpha  # Steepness parameter for metric tensor
        self.device = device
        self.anisotropic = anisotropic  # Control isotropic vs anisotropic behavior
        
        # Learnable weights w_k,d. Use softplus for positivity.
        self.raw_weights = nn.Parameter(torch.zeros(self.K, self.D))
        
        # Fixed parameters initialized during setup (Stage 0a)
        self.register_buffer('centers', torch.zeros(self.K, self.D))
        self.register_buffer('bandwidths', torch.ones(self.K, self.D))
        self.is_initialized = False
        
        # CRITICAL FIX: Store normalization parameters for space transformation
        if data_mean is not None and data_std is not None:
            self.register_buffer('data_mean', data_mean.to(device))
            self.register_buffer('data_std', data_std.to(device))
            self.has_normalization = True
        else:
            self.register_buffer('data_mean', torch.zeros(data_dim, device=device))
            self.register_buffer('data_std', torch.ones(data_dim, device=device))
            self.has_normalization = False

    def _denormalize(self, x_normalized: torch.Tensor) -> torch.Tensor:
        """Transform normalized data back to original space for metric computation."""
        if not self.has_normalization:
            return x_normalized
        return x_normalized * self.data_std + self.data_mean

    def _normalize(self, x_original: torch.Tensor) -> torch.Tensor:
        """Transform original data to normalized space."""
        if not self.has_normalization:
            return x_original
        return (x_original - self.data_mean) / self.data_std

    def initialize_from_data(self, dataset: torch.Tensor, kappa: float = 1.0, image_data: bool = True):
        """Stage 0a: Initialize centers using k-means and calculate bandwidths."""
        if dataset.dim() > 2:
            dataset = dataset.view(dataset.shape[0], -1)
            
        N, D = dataset.shape

        # 1. K-Means Clustering
        if KMeans is None:
            raise ImportError("scikit-learn is required for RBF initialization. Please install it.")
        
        dataset_np = dataset.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.K, random_state=42, n_init=10)
        kmeans.fit(dataset_np)
        centers = torch.tensor(kmeans.cluster_centers_, device=self.device, dtype=dataset.dtype)
        labels = torch.tensor(kmeans.labels_, device=self.device)

        self.centers.copy_(centers)

        # 2. Calculate Bandwidths
        # Reference: https://github.com/kksniak/metric-flow-matching/blob/main/mfm/geo_metrics/rbf.py
        
        if self.anisotropic:
            # ANISOTROPIC MODE: Store per-dimension bandwidths (K, D)
            bandwidths = torch.ones(self.K, self.D, device=self.device, dtype=dataset.dtype)
            
            for k in range(self.K):
                mask = (labels == k)
                if mask.sum() > 1:
                    cluster_data = dataset[mask]
                    # Variance per dimension relative to the cluster center
                    cluster_var_per_dim = torch.mean((cluster_data - self.centers[k])**2, dim=0)  # (D,)
                    
                    # Calculate per-dimension standard deviations
                    sigma_k_d = torch.sqrt(cluster_var_per_dim + 1e-8)  # (D,)
                    
                    # Calculate per-dimension bandwidths: lambda_k_d = 0.5 / ((kappa * sigma_k_d)^2 + 1e-6)
                    lambda_k_d = 0.5 / ((kappa * sigma_k_d)**2 + 1e-6)  # (D,)
                    
                    bandwidths[k] = lambda_k_d
                else:
                    # Fallback for single-point or empty clusters
                    bandwidths[k] = 1.0
            
            self.bandwidths.copy_(bandwidths)
            print(f"RBF metric initialized with {self.K} clusters (ANISOTROPIC mode).")
            
        else:
            # ISOTROPIC MODE: Scalar bandwidth per cluster, broadcast to all dimensions
            bandwidths = torch.ones(self.K, 1, device=self.device, dtype=dataset.dtype)
            
            for k in range(self.K):
                mask = (labels == k)
                if mask.sum() > 1:
                    cluster_data = dataset[mask]
                    # Variance per dimension relative to the cluster center
                    cluster_var_per_dim = torch.mean((cluster_data - self.centers[k])**2, dim=0) # (D,)
                    
                    # Aggregate variance to a scalar sigma
                    if image_data:
                        # For image data, sum variances across dimensions
                        total_variance = cluster_var_per_dim.sum()
                    else:
                        # For other data, mean variance across dimensions
                        total_variance = cluster_var_per_dim.mean()
                    
                    sigma_k = torch.sqrt(total_variance)
                    
                    # Calculate scalar bandwidth (precision) lambda_k = 1 / (2 * (kappa * sigma_k)^2)
                    lambda_k = 0.5 / ((kappa * sigma_k)**2 + 1e-6)
                    
                    bandwidths[k] = lambda_k
                else:
                    # Fallback for single-point or empty clusters
                    bandwidths[k] = 1.0
            
            # Store as (K,1) and broadcast it in compute_h
            # To maintain compatibility with raw_weights (K,D), we expand it
            self.bandwidths.copy_(bandwidths.expand(-1, self.D))
            print(f"RBF metric initialized with {self.K} clusters (ISOTROPIC mode).")
        
        self.is_initialized = True

    def compute_h(self, x_flat: torch.Tensor) -> torch.Tensor:
        """
        Computes the RBF network output h(x).
        
        ISOTROPIC MODE: Returns scalar h(x) broadcast to all dimensions.
        ANISOTROPIC MODE: Returns vector h(x) with per-dimension values.
        
        Reference: https://github.com/kksniak/metric-flow-matching/blob/main/mfm/geo_metrics/rbf.py
        """
        if self.anisotropic:
            # ANISOTROPIC MODE: Compute per-dimension h(x)
            # 1. Element-wise squared distance: (B, K, D)
            sq_dist = (x_flat.unsqueeze(1) - self.centers.unsqueeze(0))**2  # Shape: [B, K, D]
            
            # 2. Apply anisotropic bandwidths (per-dimension precision)
            # self.bandwidths is (K, D) with potentially different values per dimension
            scaled_sq_dist = sq_dist * self.bandwidths.unsqueeze(0)  # Shape: [B, K, D]
            
            # # 3. Sum over dimensions to get anisotropic (Mahalanobis-like) distance
            # anisotropic_dist = torch.sum(scaled_sq_dist, dim=2)  # Shape: [B, K]
            # sum causes underflow and vanishing gradients
            anisotropic_dist = torch.mean(scaled_sq_dist, dim=2)  # Shape: [B, K]
            
            # 4. Compute RBF kernel values
            rbf_kernel = torch.exp(-anisotropic_dist)  # Shape: [B, K]
            
            # 5. Apply per-dimension weights (K, D)
            # self.raw_weights is (K, D), apply softplus for positivity
            weights = F.softplus(self.raw_weights)  # Shape: (K, D)
            
            # 6. Compute weighted sum to get per-dimension output h(x)
            # rbf_kernel: (B, K), weights: (K, D)
            # We want: sum_k rbf_kernel[b,k] * weights[k,d] for each dimension d
            h_x = torch.einsum('bk,kd->bd', rbf_kernel, weights)  # Shape: (B, D)
            
        else:
            # ISOTROPIC MODE: Compute scalar h(x) and broadcast
            # 1. Compute squared Euclidean distance to centers (B, K)
            sq_dist = torch.cdist(x_flat, self.centers) ** 2
            
            # 2. Get scalar bandwidths (precision) per cluster (K,)
            # self.bandwidths is (K, D) with identical values per row. Take the first column.
            scalar_bandwidths = self.bandwidths[:, 0]  # Shape: (K,)
            
            # 3. Apply scalar bandwidths to scalar distances
            # (B, K) * (K,) -> (B, K)
            scaled_sq_dist = sq_dist * scalar_bandwidths.unsqueeze(0)
            
            # 4. Compute RBF kernel values (phi_x in reference)
            rbf_kernel = torch.exp(-scaled_sq_dist)  # Shape: (B, K)
            
            # 5. Get scalar weights per cluster (W in reference)
            # self.raw_weights is (K, D). We average to get a scalar weight per cluster.
            scalar_weights = F.softplus(self.raw_weights).mean(dim=1)  # Shape: (K,)
            
            # 6. Weighted sum to get a SCALAR output h(x) per data point
            # (B, K) * (K,) -> sum over K -> (B,)
            h_x_scalar = torch.sum(rbf_kernel * scalar_weights.unsqueeze(0), dim=1)
            
            # 7. Broadcast scalar h(x) to match the data dimension (D)
            # This is for constructing the diagonal metric where each diagonal element is the same.
            h_x = h_x_scalar.unsqueeze(1).expand(-1, x_flat.shape[1])  # Shape: (B, D)
        
        return h_x

    def compute_metric(self, x: torch.Tensor, alpha: float = None) -> torch.Tensor:
        """Compute the diagonal RBF metric G_RBF(x) = (h(x) + epsilon)^(-alpha).
        
        Following the authors' formulation where alpha controls the steepness of the metric tensor.
        Higher alpha makes the metric more sensitive to deviations from data points.
        
        ISOTROPIC MODE: h(x) is scalar, broadcast to all dimensions (M_ii identical)
        ANISOTROPIC MODE: h(x) is vector, different M_ii per dimension
        
        Reference: https://github.com/kksniak/metric-flow-matching/blob/main/mfm/geo_metrics/rbf.py#L143-152
        
        Args:
            x: Input tensor (in normalized space if normalization is used)
            alpha: Steepness parameter. If None, uses instance alpha. Authors sometimes use different values.
            
        Returns:
            Diagonal metric tensor with shape matching input
            
        CRITICAL: x is in normalized space, but metric is computed in original space.
        """
        # Use instance alpha if not specified
        if alpha is None:
            alpha = self.alpha
            
        original_shape = x.shape
        B = x.shape[0]
        x_flat_normalized = x.view(B, -1)
        
        # CRITICAL FIX: Transform to original space for metric computation
        x_flat_original = self._denormalize(x_flat_normalized)
        
        # Compute RBF network output h(x)
        # ISOTROPIC: (B, D) with identical values per row
        # ANISOTROPIC: (B, D) with different values per dimension
        h_x = self.compute_h(x_flat_original)
        
        # assume image data, following authors
        h_x = 1 - torch.abs(1 - h_x)
        
        # Authors' metric formula: M(x) = 1 / (h(x) + epsilon)^alpha
        metric_diag = 1.0 / (h_x + self.epsilon) ** alpha
        
        # IMPORTANT: Scale metric by normalization factor
        # The metric should be scaled by 1/std^2 to account for coordinate scaling
        if self.has_normalization:
            std_scaling = self.data_std.view(1, -1) ** 2
            metric_diag = metric_diag * (std_scaling + 1e-8)
        
        return metric_diag.view(original_shape)
        
        # BACKWARD COMPATIBILITY: Original approach (commented)
        # Original approach without alpha parameter:
        # metric_diag = 1.0 / (h_x + self.epsilon)
        # 
        # This corresponds to alpha=1.0 in the authors' formulation

    def train_metric(self, dataset: torch.Tensor, epochs: int = 100, lr: float = 1e-3):
        """Stage 0b: Train weights w_k,d to enforce h(x_i) ≈ 1.
        
        ISOTROPIC MODE: Trains to make scalar h(x) ≈ 1
        ANISOTROPIC MODE: Trains to make per-dimension h_d(x) ≈ 1
        
        Args:
            dataset: Training data
            epochs: Number of training epochs
            lr: Learning rate
        """
        if not self.is_initialized:
            raise RuntimeError("Must initialize RBF metric before training.")

        mode_str = "ANISOTROPIC" if self.anisotropic else "ISOTROPIC"
        print(f"Training RBF metric ({mode_str} mode) for {epochs} epochs...")
        if dataset.dim() > 2:
            dataset = dataset.view(dataset.shape[0], -1)
            
        optimizer = torch.optim.Adam([self.raw_weights], lr=lr)
        
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Compute h(x) for all data points
            # ISOTROPIC: (N, D) with identical values per row
            # ANISOTROPIC: (N, D) with different values per dimension
            h_x = self.compute_h(dataset)
            
            if self.anisotropic:
                # ANISOTROPIC: Monitor per-dimension statistics
                if epoch % (epochs // 5) == 0:
                    h_x_mean_per_dim = h_x.mean(dim=0)  # (D,) - mean h for each dimension
                    print(f"  h(x) per-dim mean: min={h_x_mean_per_dim.min().item():.4f}, "
                          f"max={h_x_mean_per_dim.max().item():.4f}, "
                          f"avg={h_x_mean_per_dim.mean().item():.4f}")
                    print(f"  h(x) overall: mean={h_x.mean().item():.4f}, "
                          f"std={h_x.std().item():.4f}")
                
                # Loss: enforce h_d(x_i) ≈ 1 for all dimensions
                target = torch.ones_like(h_x)
                loss = F.mse_loss(h_x, target)
            else:
                # ISOTROPIC: Monitor scalar h(x) values (take first dimension since all are identical)
                h_x_scalar = h_x[:, 0]  # (N,) - scalar values
                
                if epoch % (epochs // 5) == 0:
                    print(f"  h(x) stats: mean={h_x_scalar.mean().item():.4f}, "
                          f"std={h_x_scalar.std().item():.4f}, "
                          f"min={h_x_scalar.min().item():.4f}, "
                          f"max={h_x_scalar.max().item():.4f}")
                
                # Loss: enforce h(x_i) ≈ 1 (scalar target)
                target_scalar = torch.ones_like(h_x_scalar)
                loss = F.mse_loss(h_x_scalar, target_scalar)
            
            loss.backward()
            optimizer.step()
            
            if epoch % (epochs // 5) == 0:
                print(f"  Epoch {epoch:4d}: Loss = {loss.item():.6f}")
        
        self.eval()
        print("RBF metric training completed.")
        if self.anisotropic:
            print(f"Final h(x) mean: {h_x.mean().item():.4f}, target=1.0 (per dimension)")
        else:
            print(f"Final h(x) mean: {h_x_scalar.mean().item():.4f}, target=1.0")

        # BACKWARD COMPATIBILITY: Original per-dimension training approach (commented)
        # Original approach monitored h_x.mean(dim=0) and used per-dimension targets:
        # target = torch.ones_like(h_x)  # (N, D) target
        # loss = F.mse_loss(h_x, target)  # Per-dimension MSE


# =============================================================================
# 3. Metric Flow Matching: Trajectory Handling (UPDATED)
# =============================================================================

class TrajectoryHandler:
    """
    Handles the construction and sampling of parameterized trajectories (MFM interpolants) 
    between coupled data points across multiple time points (s0, ..., sK).
    Updated to support spatial data and RBF metrics.
    """
    def __init__(self, 
                 interpolator_network: InterpolatorNetwork,
                 couplings: Optional[Tuple[torch.Tensor, ...]] = None, 
                 time_points: Optional[torch.Tensor] = None,
                 metric_handler: Optional[RBFMetricHandler] = None):
        
        self.phi_eta = interpolator_network
        self.metric_handler = metric_handler
        self.use_metric = (metric_handler is not None)
        self.couplings = couplings
        self.mode = 'parameterized_multi_marginal'

        if couplings is not None and len(couplings) > 0:
            self._initialize_dimensions(couplings, time_points)
        else:
            self.N = self.K = 0
            self.time_points = None

    def _initialize_dimensions(self, couplings: Tuple[torch.Tensor, ...], time_points: Optional[torch.Tensor]):
        """Initialize dimensions, device, and preserves spatial structure."""
        self.N = couplings[0].shape[0]  # Number of samples
        self.K_plus_1 = len(couplings)  # Number of time points (K+1)
        self.K = self.K_plus_1 - 1      # Number of segments (K)
        self.device = couplings[0].device

        # MODIFIED: Do not flatten spatial dimensions. Preserve original shape.
        self.couplings = couplings
        
        if self.couplings:
            # For spatial data, compute flattened dimension for metric calculations
            sample_shape = self.couplings[0].shape[1:]
            self.D = torch.prod(torch.tensor(sample_shape)).item()  # Flattened dimension

        # Initialize time points (knots)
        if time_points is not None:
            if len(time_points) != self.K_plus_1:
                raise ValueError("Number of time points must match the number of couplings.")
            self.time_points = time_points.to(self.device)
        else:
            # Default to uniform spacing [0, 1] if not provided
            self.time_points = torch.linspace(0.0, 1.0, self.K_plus_1, device=self.device)
        
        if torch.any(self.time_points[1:] <= self.time_points[:-1]):
            raise ValueError("Time points must be strictly increasing.")

    def sample_trajectory_batch(self, batch_size: int, compute_geodesic_loss: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Samples a batch from the parameterized trajectories.
        Returns (x_t, s_t, target_v_t, geodesic_loss_t).

        Args:
            batch_size: Number of samples.
            compute_geodesic_loss: If True (Stage 1), computes the loss and allows gradients wrt phi_eta.
                                   If False (Stage 2), detaches x_t/v_t and returns None for loss.
        """
        if self.couplings is None:
            raise RuntimeError("TrajectoryHandler requires initialized couplings.")

        if batch_size > self.N:
            batch_size = self.N

        # Handle the static case (K=0)
        if self.K < 1:
            x_t = self.couplings[0][:batch_size]
            s_t = torch.full((batch_size,), self.time_points[0].item(), device=self.device)
            target_v_t = torch.zeros_like(x_t)
            geodesic_loss = torch.tensor(0.0, device=self.device) if compute_geodesic_loss else None
            return x_t, s_t, target_v_t, geodesic_loss

        # 1. Select Couplings (Indices)
        indices = torch.randint(0, self.N, (batch_size,), device=self.device)

        # 2. Sample global time s_t uniformly
        T_start, T_end = self.time_points[0], self.time_points[-1]
        s_t = T_start + torch.rand(batch_size, device=self.device) * (T_end - T_start)  # (B,)

        # 3. Find the corresponding segment k for each s_t
        k_indices = torch.searchsorted(self.time_points, s_t, right=True) - 1
        k_indices = torch.clamp(k_indices, 0, self.K - 1)

        # 4. Gather data points and time info for the segment
        t_k = self.time_points[k_indices]  # (B,)
        h_k = self.time_points[k_indices+1] - t_k  # (B,)

        # Stack couplings (K+1, N, D...)
        X = torch.stack(self.couplings, dim=0) 
        # Select trajectories (K+1, B, D...)
        X_selected = X[:, indices, :] 
        
        # Select segment endpoints (B, D...)
        b_range = torch.arange(batch_size, device=self.device)
        X_k = X_selected[k_indices, b_range, :] 
        X_k1 = X_selected[k_indices+1, b_range, :]
        
        # Global endpoints for multi-marginal interpolation
        X_0 = X_selected[0, b_range, :]  # First time point
        X_1 = X_selected[-1, b_range, :] # Last time point

        # 5. Compute parameterized trajectory and velocity
        # Control gradient flow based on the training stage.
        if not compute_geodesic_loss:
            # Stage 2: phi_eta is fixed. Use no_grad to compute trajectories efficiently.
            with torch.no_grad():
                x_t, target_v_t = self.phi_eta.compute_trajectory(s_t, X_k, X_k1, t_k, h_k, X_0, X_1)
            geodesic_loss = None
        else:
            # Stage 1: Optimizing phi_eta. Gradients are required.
            x_t, target_v_t = self.phi_eta.compute_trajectory(s_t, X_k, X_k1, t_k, h_k, X_0, X_1)
            
            # 6. Compute Geodesic Loss (UPDATED for RBF Metric)
            # L_g = E[ v_t^T G(x_t) v_t ]
            
            if self.use_metric:
                # Use data-dependent RBF metric G(x_t) (diagonal)
                G_x_t = self.metric_handler.compute_metric(x_t)
                # Kinetic energy: sum(v_t * G_x_t * v_t) (element-wise)
                kinetic_energy_g = G_x_t * (target_v_t**2)
            else:
                # Use Euclidean metric (G=I)
                kinetic_energy_g = target_v_t**2

            dims = tuple(range(1, target_v_t.ndim))
            geodesic_loss = torch.sum(kinetic_energy_g, dim=dims).mean()

        # MODIFIED: No reshaping needed as we preserve spatial structure
        return x_t, s_t, target_v_t, geodesic_loss


# =============================================================================
# 3. Loss Functions
# =============================================================================

def trajectory_matching_loss(
    v_theta_model: VelocityNetwork,
    x_t: torch.Tensor,
    s_t: torch.Tensor,
    target_velocity: torch.Tensor,
    metric_handler: Optional[RBFMetricHandler] = None
) -> torch.Tensor:
    """
    L_Trajectory (MFM aspect): E[ || v_theta(x_t, s_t) - target_velocity ||^2_{metric} ]
    
    If metric_handler is provided, the loss is weighted by the RBF metric g(x_t, η*):
    L = E[ g(x_t, η*) * || v_theta(x_t, s_t) - target_velocity ||^2 ]
    """
    predicted_velocity = v_theta_model(x_t, s_t)
    velocity_diff = predicted_velocity - target_velocity
    
    if metric_handler is not None:
        # Weight the loss by the RBF metric g(x_t, η*)
        metric_weights = metric_handler.compute_metric(x_t)  # Shape: (B, D...)
        weighted_diff_squared = metric_weights * (velocity_diff ** 2)
        # Sum over spatial dimensions and take mean over batch
        dims = tuple(range(1, velocity_diff.ndim))
        loss = torch.sum(weighted_diff_squared, dim=dims).mean()
    else:
        # Standard MSE loss (Euclidean metric)
        loss = F.mse_loss(predicted_velocity, target_velocity, reduction='mean')
    # loss = F.mse_loss(predicted_velocity, target_velocity, reduction='mean')
    
    return loss


# def consistency_loss(
#     v_theta_model: VelocityNetwork,
#     U_phi_model: EnergyNetwork,
#     x_t: torch.Tensor,
#     s_t: torch.Tensor,
#     metric_handler: Optional[RBFMetricHandler] = None
# ) -> torch.Tensor:
#     """
#     L_CE (DSM aspect): Enforces the Continuity Equation in energy form:
#     partial_s U_phi = nabla_x . v_theta - <v_theta, nabla_x U_phi>
    
#     If metric_handler is provided, the loss is weighted by the RBF metric g(x_t, η*):
#     L = E[ g(x_t, η*) * || partial_s U_phi - (nabla_x . v_theta - <v_theta, nabla_x U_phi>) ||^2 ]
#     """
    
#     # 1. Compute Spatial and Scale Scores (Requires gradients)
#     spatial_score, scale_score = U_phi_model.compute_scores(x_t, s_t)
    
#     # 2. Compute Divergence and Velocity (Requires gradients)
#     # The returned velocity tracks gradients necessary for the coupled optimization.
#     divergence, velocity = v_theta_model.compute_divergence(x_t, s_t)

#     # 3. Compute the transport/convection term: <v_theta, nabla_x U_phi>
#     dims = tuple(range(1, x_t.ndim))
#     # Output shape: (B, 1)
#     transport_term = torch.sum(velocity * spatial_score, dim=dims, keepdim=True)
    
#     # 4. Calculate the Spatial Dynamics component (RHS): (nabla_x . v - <v, nabla_x U>)
#     spatial_dynamics = divergence - transport_term
    
#     # 5. Compute the loss (MSE between LHS and RHS)
#     # This involves double backpropagation.
#     assert scale_score.shape == spatial_dynamics.shape
    
#     if metric_handler is not None:
#         # Weight the loss by the RBF metric g(x_t, η*)
#         # Note: scale_score and spatial_dynamics have shape (B, 1), so we need the flattened metric
#         x_t_flat = x_t.view(x_t.shape[0], -1)
#         metric_weights_flat = metric_handler.compute_h(x_t_flat)  # Shape: (B, D_flat)
#         # Take mean over spatial dimensions to get scalar weight per sample
#         metric_weights_scalar = metric_weights_flat.mean(dim=1, keepdim=True)  # Shape: (B, 1)
        
#         # Apply metric weighting
#         residual = scale_score - spatial_dynamics  # Shape: (B, 1)
#         weighted_residual_squared = metric_weights_scalar * (residual ** 2)
#         loss = weighted_residual_squared.mean()
#     else:
#         # Standard MSE loss (Euclidean metric)
#         loss = F.mse_loss(scale_score, spatial_dynamics, reduction='mean')
    
#     return loss

def consistency_loss(
    v_theta_model: VelocityNetwork,
    U_phi_model: EnergyNetwork,
    x_t: torch.Tensor,
    s_t: torch.Tensor,
    # CRITICAL: Add target velocity from MFM for stabilization
    target_velocity: torch.Tensor, 
    metric_handler: Optional[RBFMetricHandler] = None
) -> torch.Tensor:
    
    # 1. Compute Scores and Divergence
    spatial_score, scale_score = U_phi_model.compute_scores(x_t, s_t)
    divergence, _ = v_theta_model.compute_divergence(x_t, s_t)

    # 2. Compute Stabilized Transport term: <target_velocity, nabla_x U_phi>
    dims = tuple(range(1, x_t.ndim))
    # Use the fixed MFM velocity (detached for safety)
    stabilized_transport_term = torch.sum(target_velocity.detach() * spatial_score, dim=dims, keepdim=True)
    
    # 3. Calculate the ICoV residual: 
    # LHS: Total time derivative (d/dt U = partial_s U + target_velocity . nabla_x U)
    total_time_derivative = scale_score + stabilized_transport_term
    
    # RHS: Divergence of the parameterized velocity
    spatial_dynamics = divergence
    
    # 4. Compute the loss (MSE between LHS and RHS)
    # Dont need the weighting because we are matching energies directly
    loss = F.mse_loss(total_time_derivative, spatial_dynamics, reduction='mean')

    return loss


# =============================================================================
# 4. Training Steps (Two-Stage Optimization)
# =============================================================================

def training_step_mfm_interpolator(
    trajectory_handler: TrajectoryHandler,
    optimizer: torch.optim.Optimizer,
    batch_size: int
) -> Dict[str, float]:
    """
    Stage 1: MFM Interpolator Pre-training (Algorithm 1).
    Optimizes phi_eta by minimizing the Geodesic Loss (Kinetic Energy).
    """
    optimizer.zero_grad()

    # 1. Sample batch and compute Geodesic Loss (L_G)
    try:
        # compute_geodesic_loss=True enables gradients for phi_eta.
        _, _, _, loss_geodesic = trajectory_handler.sample_trajectory_batch(batch_size, compute_geodesic_loss=True)
    except Exception as e:
        print(f"Skipping MFM pre-training step due to error: {e}")
        return {}

    if loss_geodesic is None:
        # This might happen if K=0 (static data)
        return {"loss_geodesic": 0.0}

    # 2. Backpropagate (updates phi_eta)
    loss_geodesic.backward()
    
    # NOTE: Optimizer step handled in the main loop for consistency (e.g., clipping)
    
    return {
        "loss_geodesic": loss_geodesic.item()
    }


def training_step_ccvep(
    trajectory_handler: TrajectoryHandler,
    v_theta_model: VelocityNetwork,
    U_phi_model: EnergyNetwork,
    optimizer: torch.optim.Optimizer,
    lambda_CE: float,
    batch_size: int
) -> Dict[str, float]:
    """
    Stage 2: C-CVEP Training (Algorithm 2).
    Optimizes v_theta and U_phi using fixed, pre-trained interpolants phi_eta.
    """
    
    # 1. Sample batch from optimized trajectories
    try:
        # compute_geodesic_loss=False ensures phi_eta is fixed (no gradients).
        x_t, s_t, target_v_t, _ = trajectory_handler.sample_trajectory_batch(batch_size, compute_geodesic_loss=False)
    except Exception as e:
        print(f"Skipping C-CVEP training step due to error during sampling: {e}")
        optimizer.zero_grad(set_to_none=True)
        return {}

    optimizer.zero_grad()

    # Extract metric handler from trajectory_handler for loss weighting
    metric_handler = trajectory_handler.metric_handler if hasattr(trajectory_handler, 'metric_handler') else None

    # 2. Trajectory Matching Loss (L_Traj)
    # Updates v_theta.
    loss_trajectory = trajectory_matching_loss(v_theta_model, x_t, s_t, target_v_t, metric_handler)

    # 3. Consistency Loss (L_CE)
    # Updates v_theta and U_phi.
    # Note: The loss functions internally handle the required gradients wrt x_t via cloning.
    loss_consistency = consistency_loss(v_theta_model, U_phi_model, x_t, s_t, target_v_t, metric_handler)

    # Total Loss: L = L_Traj + lambda_CE * L_CE
    total_loss = loss_trajectory + lambda_CE * loss_consistency
    
    total_loss.backward()
        
    return {
        "loss_total": total_loss.item(),
        "loss_trajectory": loss_trajectory.item(),
        "loss_consistency": loss_consistency.item(),
    }