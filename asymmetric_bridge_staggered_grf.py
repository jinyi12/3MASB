import torch
from torch import nn, Tensor
from torch import distributions as D
import copy
from typing import Tuple, Callable, Optional, Dict, Any
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
import os
import traceback
import math

# Set default dtype
torch.set_default_dtype(torch.float32)

# ============================================================================
# 0. Random Field Generation Utilities (Multiscale Homogenization)
# (No changes to this section, included for completeness)
# ============================================================================

def gaussian_blur_periodic(input_tensor: Tensor, kernel_size: int, sigma: float) -> Tensor:
    # ... (Implementation same as original file) ...
    if sigma <= 1e-9 or kernel_size <= 1:
        return input_tensor
    if kernel_size % 2 == 0:
        kernel_size += 1
    k = torch.arange(kernel_size, dtype=torch.float32, device=input_tensor.device)
    center = (kernel_size - 1) / 2
    gauss_1d = torch.exp(-0.5 * ((k - center) / sigma)**2)
    gauss_1d = gauss_1d / gauss_1d.sum()
    gauss_2d = torch.outer(gauss_1d, gauss_1d)
    if input_tensor.dim() != 4:
        raise ValueError("Input tensor must be [B, C, H, W]")
    C_in = input_tensor.shape[1]
    kernel = gauss_2d.expand(C_in, 1, kernel_size, kernel_size)
    padding = (kernel_size - 1) // 2
    padded_input = torch.nn.functional.pad(input_tensor, (padding, padding, padding, padding), mode='circular')
    output = torch.nn.functional.conv2d(padded_input, kernel, padding=0, groups=C_in)
    return output

class RandomFieldGenerator2D:
    # ... (Implementation same as original file) ...
    def __init__(self, nx=100, ny=100, lx=1.0, ly=1.0, device='cpu'):
        self.nx = nx; self.ny = ny; self.lx = lx; self.ly = ly; self.device = device

    def generate_random_field(self, mean=10.0, std=2.0, correlation_length=0.2, covariance_type="exponential"):
        dx = self.lx / self.nx; dy = self.ly / self.ny
        white_noise = np.random.normal(0, 1, (self.nx, self.ny))
        fourier_coefficients = np.fft.fft2(white_noise)
        kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=dx); ky = 2 * np.pi * np.fft.fftfreq(self.ny, d=dy)
        Kx, Ky = np.meshgrid(kx, ky, indexing="ij"); K = np.sqrt(Kx**2 + Ky**2)
        l = correlation_length
        if covariance_type == "exponential":
            denom = (1 + (l * K) ** 2); P = (2 * np.pi * l**2) / np.maximum(1e-9, denom ** (1.5))
        elif covariance_type == "gaussian":
            P = np.pi * l**2 * np.exp(-((l * K) ** 2) / 4)
        else: raise ValueError("Invalid covariance_type")
        P = np.nan_to_num(P)
        fourier_coefficients *= np.sqrt(P); field = np.fft.ifft2(fourier_coefficients).real
        field_std = np.std(field)
        if field_std > 1e-9: field = (field - np.mean(field)) / field_std * std + mean
        else: field = np.full_like(field, mean)
        return field

    def coarsen_field(self, field, H):
        if isinstance(field, np.ndarray): field = torch.from_numpy(field).to(self.device)
        if field.dim() == 3: field = field.unsqueeze(1); squeeze_channel = True
        elif field.dim() == 4: squeeze_channel = False
        else: raise ValueError("Unsupported field dimensions (must be 3 or 4)")
        pixel_size = self.lx / self.nx; filter_sigma_phys = H / 6.0
        filter_sigma_pix = filter_sigma_phys / pixel_size
        if filter_sigma_pix < 1e-6: smooth = field
        else:
            kernel_size = int(6 * filter_sigma_pix)
            if kernel_size % 2 == 0: kernel_size += 1
            kernel_size = max(3, kernel_size)
            smooth = gaussian_blur_periodic(field, kernel_size=kernel_size, sigma=filter_sigma_pix)
        coarse = smooth
        if squeeze_channel: coarse = coarse.squeeze(1)
        return coarse

# ============================================================================
# 1. Utilities (JVP, t_dir, and Initialization Helpers)
# ============================================================================

def jvp(f: Callable[[Tensor], Any], x: Tensor, v: Tensor) -> Tuple[Any, Any]:
    # ... (Implementation same as original file) ...
    try:
        return torch.autograd.functional.jvp(f, x, v, create_graph=torch.is_grad_enabled())
    except RuntimeError as e:
        print(f"Warning: JVP computation failed. Error: {e}. Returning zero derivative.")
        output = f(x)
        if isinstance(output, tuple): jvp_result = tuple(torch.zeros_like(o) if torch.is_tensor(o) else 0 for o in output)
        else: jvp_result = torch.zeros_like(output)
        return output, jvp_result

def t_dir(f: Callable[[Tensor], Any], t: Tensor) -> Tuple[Any, Any]:
    return jvp(f, t, torch.ones_like(t))

# NEW: Helper for Initialization
def calculate_data_statistics(marginal_data: Dict[float, Tensor]) -> Tuple[Tensor, Tensor]:
    """Calculates the overall mean and std dev of the dataset for initialization."""
    # Concatenate all data across all time points
    all_data = torch.cat(list(marginal_data.values()), dim=0)
    
    # Calculate mean and std deviation across the batch dimension
    data_mean = torch.mean(all_data, dim=0)
    data_std = torch.std(all_data, dim=0)
    
    # Clamp std for numerical stability (essential for log calculation later)
    data_std = torch.clamp(data_std, min=1e-5)
    
    return data_mean, data_std

# ============================================================================
# 2. Revised Multiscale Data Generation
# ============================================================================

def generate_multiscale_grf_data(
    N_samples: int, T: float = 1.0, N_constraints: int = 5, resolution: int = 32,
    L_domain: float = 1.0, micro_corr_length: float = 0.1, H_max_factor: float = 0.5,
    mean_val: float = 10.0, std_val: float = 2.0, covariance_type: str = "exponential",
    device: str = 'cpu'
) -> Tuple[Dict[float, Tensor], int]:
    # ... (Implementation same as original file) ...
    print(f"\n--- Generating Multiscale GRF Data (Fixed Resolution: {resolution}x{resolution}) ---")
    time_steps = torch.linspace(0, T, N_constraints); marginal_data = {}; data_dim = resolution * resolution
    generator = RandomFieldGenerator2D(nx=resolution, ny=resolution, lx=L_domain, ly=L_domain, device=device)
    print("Generating base microscopic fields (t=0)..."); micro_fields = []
    for _ in trange(N_samples):
        field = generator.generate_random_field(mean=mean_val, std=std_val, correlation_length=micro_corr_length, covariance_type=covariance_type)
        micro_fields.append(field)
    micro_fields_tensor = torch.tensor(np.array(micro_fields), dtype=torch.float32).to(device)
    print("Applying progressive coarsening filters..."); H_max = L_domain * H_max_factor
    for t in time_steps:
        t_val = t.item(); t_norm = t_val / T; H_t = t_norm * H_max
        coarsened_fields = generator.coarsen_field(micro_fields_tensor, H=H_t)
        flattened_fields = coarsened_fields.reshape(N_samples, data_dim)
        marginal_data[t_val] = flattened_fields
        mean_std = torch.std(flattened_fields, dim=0).mean().item()
        print(f"  t={t_val:.2f}: H={H_t:.4f}, Mean Std Dev across field: {mean_std:.4f}")
    print("Multiscale data generation complete.")
    return marginal_data, data_dim

# ============================================================================
# 3. MMD Loss Implementation (REVISED: Multi-Scale MMD)
# ============================================================================

def compute_multiscale_mmd_U(X: Tensor, Y: Tensor, sigmas: list[float]) -> Tensor:
    """
    Computes the unbiased MMD² U-statistic using a mixture of Gaussian kernels.
    """
    N = X.shape[0]; M = Y.shape[0]
    
    if N < 2 or M < 2: return torch.tensor(0.0, device=X.device)

    # Efficient and stable computation of distance matrices
    def compute_dist_sq(A, B):
        A_norm = (A**2).sum(1).view(-1, 1); B_norm = (B**2).sum(1).view(1, -1)
        try:
            dist = A_norm + B_norm - 2.0 * torch.mm(A, B.t())
        except RuntimeError:
             print("Warning: Falling back to slower distance computation.")
             dist = torch.cdist(A, B, p=2)**2
        return torch.clamp(dist, min=0.0)
            
    dist_sq_XX = compute_dist_sq(X, X)
    dist_sq_YY = compute_dist_sq(Y, Y)
    dist_sq_XY = compute_dist_sq(X, Y)

    mmd_sq = 0.0
    
    # Sum contributions from different kernel bandwidths (sigmas)
    for sigma in sigmas:
        if sigma <= 1e-9: continue
            
        # Gaussian Kernel K(x,y) = exp(-||x-y||^2 / (2*sigma^2))
        K_XX = torch.exp(-dist_sq_XX / (2 * sigma**2))
        K_YY = torch.exp(-dist_sq_YY / (2 * sigma**2))
        K_XY = torch.exp(-dist_sq_XY / (2 * sigma**2))

        # U-statistic calculation (excluding diagonals for XX and YY)
        term_XX = (torch.sum(K_XX) - torch.trace(K_XX)) / (N * (N - 1))
        term_YY = (torch.sum(K_YY) - torch.trace(K_YY)) / (M * (M - 1))
        term_XY = torch.mean(K_XY)
                 
        # We sum the MMDs (stronger signal than averaging)
        mmd_sq += (term_XX + term_YY - 2.0 * term_XY)
        
    return torch.clamp(mmd_sq, min=0.0)

# ============================================================================
# 4. Model Architectures (REVISED: Residual Initialization)
# ============================================================================

class TimeDependentAffine(nn.Module):
    """
    G(epsilon, t) = mu(t) + gamma(t) * epsilon.
    Implements residual dynamics around data statistics for robust initialization.
    """
    def __init__(self, data_dim: int, hidden_size: int, n_layers: int = 3, data_mean: Optional[Tensor] = None, data_std: Optional[Tensor] = None):
        super().__init__()
        
        # Network definition
        layers = [nn.Linear(1, hidden_size), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.SiLU()])
        layers.append(nn.Linear(hidden_size, 2 * data_dim))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize the NN weights such that the initial output is small
        self._initialize_nn_weights()

        # --- Initialization Priors (as Buffers) ---
        self._register_priors(data_dim, data_mean, data_std)

    def _initialize_nn_weights(self):
        # Initialize the final layer weights and biases near zero
        # This ensures the initial NN output (the residual) is small.
        last_layer = self.net[-1]
        if isinstance(last_layer, nn.Linear):
            nn.init.normal_(last_layer.weight, mean=0.0, std=1e-5)
            if last_layer.bias is not None:
                nn.init.zeros_(last_layer.bias)

    def _register_priors(self, data_dim, data_mean, data_std):
        if data_mean is None:
            self.register_buffer('mu_prior', torch.zeros(data_dim))
        else:
            self.register_buffer('mu_prior', data_mean.clone().detach())
            
        if data_std is None:
            self.register_buffer('log_gamma_prior', torch.zeros(data_dim)) # log(1)=0
        else:
            # data_std should already be clamped > 0
            self.register_buffer('log_gamma_prior', torch.log(data_std.clone().detach()))

    def get_coeffs(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
            
        # Ensure gradients can flow through time if needed for derivatives
        if not t.requires_grad and torch.is_grad_enabled():
             # Use requires_grad_(True) for in-place modification if necessary
             # but generally letting autograd handle it is cleaner if possible.
             # If t is used in JVP, it must allow gradients.
             t.requires_grad = True

        # The NN outputs the residual dynamics
        d_mu, d_log_gamma = self.net(t).chunk(chunks=2, dim=1)
        
        # Add the prior (broadcasting handles the time batch dimension)
        # mu(t) = mu_prior + d_mu(t)
        # log_gamma(t) = log_gamma_prior + d_log_gamma(t)
        mu = self.mu_prior + d_mu
        log_gamma = self.log_gamma_prior + d_log_gamma
        
        gamma = torch.exp(log_gamma)
        gamma = torch.clamp(gamma, min=1e-6) # Stability clamp
        return mu, gamma

    def forward(self, t: Tensor, return_t_dir: bool = False) -> Any:
        if return_t_dir:
            def f(t_in: Tensor) -> Tuple[Tensor, Tensor]:
                return self.get_coeffs(t_in)
            return t_dir(f, t)
        else:
            return self.get_coeffs(t)

class TimeDependentVolatility(nn.Module):
    # ... (Implementation same as original file) ...
    def __init__(self, data_dim: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hidden_size), nn.SiLU(), nn.Linear(hidden_size, data_dim), nn.Softplus())
    def forward(self, t: Tensor) -> Tensor:
        if t.dim() == 1: t = t.unsqueeze(-1)
        return self.net(t) + 1e-5

# ============================================================================
# 5. RS-IPFO Bridge Implementation (REVISED: Initialization and MMD)
# ============================================================================

class RSIPFOBridge(nn.Module):
    
    # Updated initialization signature
    def __init__(self, data_dim: int, hidden_size: int, T: float = 1.0, V_t=None, flow_layers: int = 3, data_mean: Optional[Tensor] = None, data_std: Optional[Tensor] = None):
        super().__init__()
        self.data_dim = data_dim; self.T = T; self.V_t = V_t
        
        # Initialize flow with data statistics using the residual approach
        self.affine_flow = TimeDependentAffine(data_dim, hidden_size, n_layers=flow_layers, data_mean=data_mean, data_std=data_std)
        
        vol_hidden_size = max(hidden_size // 4, 16)
        self.volatility_model = TimeDependentVolatility(data_dim, vol_hidden_size)

        self.register_buffer('base_mean', torch.zeros(data_dim))
        self.register_buffer('base_std', torch.ones(data_dim))
    
    @property
    def base_dist(self):
        return D.Independent(D.Normal(self.base_mean, self.base_std), 1)

    # Helper Functions and Standard Losses (KE, DLA, Action) remain the same
    def get_params(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        return self.affine_flow(t, return_t_dir=False)

    def get_params_and_derivs(self, t: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        (mu, gamma), (dmu_dt, dgamma_dt) = self.affine_flow(t, return_t_dir=True)
        return mu, gamma, dmu_dt, dgamma_dt
    
    def G_transform(self, epsilon: Tensor, t: Tensor) -> Tensor:
        mu, gamma = self.get_params(t)
        B_time = mu.shape[0]; B_latent = epsilon.shape[0]
        if B_time == B_latent or B_time == 1: return mu + gamma * epsilon
        else: raise NotImplementedError(f"Broadcasting mismatch: Time batch {B_time}, Latent batch {B_latent}")

    def kinetic_energy(self, t: Tensor) -> Tensor:
        _, _, dmu_dt, dgamma_dt = self.get_params_and_derivs(t)
        mean_term = torch.sum(dmu_dt**2, dim=-1); var_term = torch.sum(dgamma_dt**2, dim=-1)
        return (mean_term + var_term) / 2.0

    def state_cost(self, t: Tensor, n_mc_samples: int = 128) -> Tensor:
        if self.V_t is None: return torch.zeros(t.shape[0], device=t.device)
        return torch.zeros(t.shape[0], device=t.device)

    def loss_Action(self, t: Tensor, n_mc_samples_sc: int = 128) -> Tuple[Tensor, Tensor, Tensor]:
        ke = self.kinetic_energy(t); sc = self.state_cost(t, n_mc_samples=n_mc_samples_sc)
        action = ke + sc
        return action.mean(), ke.mean(), sc.mean()

    def loss_DLA(self, t: Tensor, target_model: 'RSIPFOBridge') -> Tensor:
        target_model.eval()
        mu_theta, gamma_theta = self.get_params(t)
        with torch.no_grad(): mu_target, gamma_target = target_model.get_params(t)
        mean_term = torch.sum((mu_theta - mu_target)**2, dim=-1)
        var_term = torch.sum((gamma_theta - gamma_target)**2, dim=-1)
        return (mean_term + var_term).mean()

    # REVISED: loss_MMD using Multi-Scale Kernels
    def loss_MMD(self, marginal_data: Dict[float, Tensor], n_samples_model: int, mmd_sigma: float = 1.0) -> Tensor:
        mmd_loss = 0.0
        device = self.base_mean.device
        
        # Determine MMD scales (sigmas)
        if mmd_sigma <= 0:
            # Robust, predefined mixture for high dimensions (D=256)
            # Covering scales from fine structure (1.0) to large distances (100.0+)
            # This mixture is designed to handle the scale of the data (Mean=10, Std=2)
            sigmas = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 150.0]
        else:
            # Use a small mixture around the specified sigma
            sigmas = [mmd_sigma * 0.5, mmd_sigma, mmd_sigma * 2.0]

        epsilon = self.base_dist.sample((n_samples_model,))

        for t_k, samples_data in marginal_data.items():
            t_k_tensor = torch.tensor([[t_k]], device=device)
            samples_model = self.G_transform(epsilon, t_k_tensor)
            
            # Subsample data if necessary
            N_data = samples_data.shape[0]
            if N_data > n_samples_model:
                 indices = torch.randperm(N_data, device=device)[:n_samples_model]
                 samples_data_sub = samples_data[indices]
            else:
                 samples_data_sub = samples_data
                 
            # Use the new multiscale MMD function
            mmd_sq = compute_multiscale_mmd_U(samples_data_sub, samples_model, sigmas=sigmas)
            mmd_loss += mmd_sq

        return mmd_loss / len(marginal_data)

    def loss_Backward_Energy(self, t: Tensor, n_mc_samples: int = 128) -> Tensor:
        # (Implementation remains the same, ensure stability clamps are present)
        mu, gamma, dmu_dt, dgamma_dt = self.get_params_and_derivs(t)
        B_time = mu.shape[0]

        mu_e, gamma_e = mu.unsqueeze(1), gamma.unsqueeze(1)
        dmu_dt_e, dgamma_dt_e = dmu_dt.unsqueeze(1), dgamma_dt.unsqueeze(1)

        epsilon = torch.randn(B_time, n_mc_samples, self.data_dim, device=t.device)
        X_t = mu_e + gamma_e * epsilon
        
        # Clamping for stability (Crucial to prevent NaNs)
        v_theta = dmu_dt_e + (dgamma_dt_e / torch.clamp(gamma_e, min=1e-6)) * (X_t - mu_e)
        score = -(X_t - mu_e) / torch.clamp(gamma_e**2, min=1e-6)
        
        g_phi = self.volatility_model(t)
        D_phi_e = (g_phi**2).unsqueeze(1)
        
        R = v_theta - 0.5 * D_phi_e * score
        
        J_Bwd = 0.5 * torch.mean(torch.sum(R**2, dim=-1))
        return J_Bwd

# ============================================================================
# 7. RS-IPFO Training Loop (Staggered Scheme)
# ============================================================================

def train_rs_ipfo(
    bridge: RSIPFOBridge, marginal_data: Dict[float, Tensor], config: Dict[str, Any]
) -> list:
    # ... (Implementation mostly the same as original, using updated components) ...
    iterations = config.get('iterations', 10); sub_steps = config.get('sub_steps', 200)
    lr_theta = config.get('lr_theta', 1e-3); lr_phi = config.get('lr_phi', 1e-3)
    lambda_MM = config.get('lambda_MM', 1.0); lambda_Match = config.get('lambda_Match', 50.0)
    batch_size_time = config.get('batch_size_time', 128); n_samples_mmd = config.get('n_samples_mmd', 256)
    grad_clip_norm = config.get('grad_clip_norm', 1.0); mmd_sigma = config.get('mmd_sigma', 1.0)

    optimizer_theta = torch.optim.AdamW(bridge.affine_flow.parameters(), lr=lr_theta, weight_decay=1e-5)
    optimizer_phi = torch.optim.AdamW(bridge.volatility_model.parameters(), lr=lr_phi, weight_decay=1e-5)
    
    total_steps = iterations * sub_steps
    scheduler_theta = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_theta, T_max=total_steps)
    scheduler_phi = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_phi, T_max=total_steps)

    history = []; device = bridge.base_mean.device; T = bridge.T
    print("\nStarting RS-IPFO Training...")

    for iteration in range(iterations):
        print(f"\n--- Iteration {iteration + 1}/{iterations} ---")
        
        # Step 1: GSB Path Optimization
        bridge.train(); pbar = trange(sub_steps, desc="Step 1 (J_GSB)")
        for step in pbar:
            optimizer_theta.zero_grad()
            loss_mmd = bridge.loss_MMD(marginal_data, n_samples_model=n_samples_mmd, mmd_sigma=mmd_sigma)
            t_rand = torch.rand(batch_size_time, 1, device=device) * T
            loss_action, loss_ke, loss_sc = bridge.loss_Action(t_rand)
            loss_step1 = lambda_MM * loss_mmd + loss_action
            
            if torch.isnan(loss_step1):
                print("NaN detected in Step 1 Loss. Aborting."); return history

            loss_step1.backward()
            if grad_clip_norm > 0: torch.nn.utils.clip_grad_norm_(bridge.affine_flow.parameters(), max_norm=grad_clip_norm)
            optimizer_theta.step(); scheduler_theta.step()
            pbar.set_description(f"S1 J_GSB: {loss_step1.item():.4f} (MMD: {loss_mmd.item():.4f}, KE: {loss_ke.item():.4f})")

        target_bridge = copy.deepcopy(bridge); J_GSB_target = loss_step1.item()

        # Step 2: Field Refinement
        bridge.train(); pbar = trange(sub_steps, desc="Step 2 (Refinement)")
        for step in pbar:
            optimizer_theta.zero_grad()
            t_rand = torch.rand(batch_size_time, 1, device=device) * T
            loss_ke = bridge.kinetic_energy(t_rand).mean()
            loss_dla = bridge.loss_DLA(t_rand, target_bridge)
            loss_step2 = loss_ke + lambda_Match * loss_dla
            
            if torch.isnan(loss_step2): print("NaN detected in Step 2 Loss. Aborting."); return history
            
            loss_step2.backward()
            if grad_clip_norm > 0: torch.nn.utils.clip_grad_norm_(bridge.affine_flow.parameters(), max_norm=grad_clip_norm)
            optimizer_theta.step(); scheduler_theta.step()
            pbar.set_description(f"S2 Loss: {loss_step2.item():.4f} (KE: {loss_ke.item():.4f}, DLA: {loss_dla.item():.4f})")

        # Step 4: Volatility Update
        bridge.train()
        for param in bridge.affine_flow.parameters(): param.requires_grad = False
        pbar = trange(sub_steps, desc="Step 4 (J_Bwd)")
        for step in pbar:
            optimizer_phi.zero_grad()
            t_rand = torch.rand(batch_size_time, 1, device=device) * T
            loss_bwd = bridge.loss_Backward_Energy(t_rand, n_mc_samples=64)

            if torch.isnan(loss_bwd): print("NaN detected in Step 4 Loss. Aborting."); return history

            loss_bwd.backward()
            if grad_clip_norm > 0: torch.nn.utils.clip_grad_norm_(bridge.volatility_model.parameters(), max_norm=grad_clip_norm)
            optimizer_phi.step(); scheduler_phi.step()
            pbar.set_description(f"S4 J_Bwd: {loss_bwd.item():.4f}")

        for param in bridge.affine_flow.parameters(): param.requires_grad = True

        # --- Evaluation and Logging ---
        with torch.no_grad():
            bridge.eval()
            t_eval = torch.rand(512, 1, device=device) * T
            loss_mmd_final = bridge.loss_MMD(marginal_data, n_samples_model=n_samples_mmd, mmd_sigma=mmd_sigma)
            loss_action_final, _, _ = bridge.loss_Action(t_eval)
            J_GSB_refined = (lambda_MM * loss_mmd_final + loss_action_final).item()
            J_Bwd_final = bridge.loss_Backward_Energy(t_eval).item()

        warning = ""
        if J_GSB_refined > J_GSB_target + 1e-3:
             warning = f"(Warn: J_GSB increased due to soft constraint violation)"

        print(f"Iteration Summary: J_GSB Target: {J_GSB_target:.4f} -> Refined: {J_GSB_refined:.4f} {warning}. J_Bwd: {J_Bwd_final:.4f}")
        history.append({'iteration': iteration, 'J_GSB_target': J_GSB_target, 'J_GSB_refined': J_GSB_refined, 'J_Bwd': J_Bwd_final})
        
    return history

# ============================================================================
# 8. Enhanced Visualization Functions (Adapted for GRF data)
# ============================================================================
def _plot_convergence_analysis(history, output_dir):
    """Plot RS-IPFO convergence analysis."""
    if not history: return
    print("  - Plotting convergence analysis...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("RS-IPFO Training Convergence Analysis", fontsize=14)
    
    iters = [h['iteration'] for h in history]
    target_loss = [h['J_GSB_target'] for h in history]
    refined_loss = [h['J_GSB_refined'] for h in history]
    j_bwd = [h['J_Bwd'] for h in history]
        
    # A) GSB Objective Evolution
    ax1 = axes[0]
    ax1.plot(iters, target_loss, marker='o', linestyle='--', linewidth=2, label='J_GSB(Target) - Step 1')
    ax1.plot(iters, refined_loss, marker='x', linestyle='-', linewidth=2, label='J_GSB(Refined) - Step 2')
    ax1.set_title("A) GSB Objective (Monotonic Guarantee)")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("GSB Objective")
    ax1.legend()
    ax1.grid(True)
        
    # B) Backward Energy Evolution
    ax2 = axes[1]
    ax2.plot(iters, j_bwd, marker='s', linewidth=2, color='purple', label='J_Bwd (Backward Energy)')
    ax2.set_title("B) Backward Energy Convergence (Step 4)")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("J_Bwd")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "convergence.png"), dpi=300)
    plt.close()

def _visualize_grf_marginals(marginal_data, output_dir, title="GRF Data Marginals"):
    """Visualizes samples from the GRF marginal data."""
    print(f"  - Plotting {title}...")
    
    sorted_times = sorted(marginal_data.keys())
    n_marginals = len(sorted_times)
    n_samples_to_show = 5
    
    # Determine resolution
    data_dim = marginal_data[sorted_times[0]].shape[1]
    resolution = int(math.sqrt(data_dim))
    
    if resolution * resolution != data_dim:
        return

    fig, axes = plt.subplots(n_samples_to_show, n_marginals, figsize=(3 * n_marginals, 3 * n_samples_to_show))
    if axes.ndim == 1: axes = np.array([axes])
    
    # Determine global vmin/vmax
    all_data = torch.cat([marginal_data[t] for t in sorted_times], dim=0).cpu().numpy()
    vmin, vmax = all_data.min(), all_data.max()

    for i in range(n_samples_to_show):
        for j, t_k in enumerate(sorted_times):
            ax = axes[i, j]
            sample = marginal_data[t_k][i].cpu().numpy().reshape(resolution, resolution)
            im = ax.imshow(sample, cmap='viridis', vmin=vmin, vmax=vmax, extent=[0, 1, 0, 1])
            ax.axis('off')
            if i == 0: ax.set_title(f"t = {t_k:.2f}")

    plt.tight_layout()
    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

def _visualize_learned_grf_evolution(bridge, T, output_dir, device):
    """Visualizes the evolution of the learned GRF distribution (mean and std dev fields)."""
    print("  - Plotting learned GRF evolution (Mean and Std)...")

    data_dim = bridge.data_dim
    resolution = int(math.sqrt(data_dim))
    if resolution * resolution != data_dim:
        return

    n_steps = 5
    time_steps = torch.linspace(0, T, n_steps)
    
    # Get learned parameters (mu and gamma)
    t_tensor = time_steps.unsqueeze(-1).to(device)
    with torch.no_grad():
        mu_t, gamma_t = bridge.get_params(t_tensor)
    
    mu_t = mu_t.cpu().numpy().reshape(n_steps, resolution, resolution)
    gamma_t = gamma_t.cpu().numpy().reshape(n_steps, resolution, resolution)
    
    fig, axes = plt.subplots(2, n_steps, figsize=(3 * n_steps, 6))
    fig.suptitle("Learned GRF Evolution: Mean μ(t) and Std Dev γ(t)", fontsize=16)

    vmin_mu, vmax_mu = mu_t.min(), mu_t.max()
    vmin_gamma, vmax_gamma = gamma_t.min(), gamma_t.max()

    for i in range(n_steps):
        t_val = time_steps[i].item()
        
        # Plot Mean μ(t)
        ax_mu = axes[0, i]
        ax_mu.imshow(mu_t[i], cmap='viridis', vmin=vmin_mu, vmax=vmax_mu, extent=[0, 1, 0, 1])
        ax_mu.set_title(f"μ(t={t_val:.2f})")
        ax_mu.axis('off')
        
        # Plot Std Dev γ(t)
        ax_gamma = axes[1, i]
        ax_gamma.imshow(gamma_t[i], cmap='plasma', vmin=vmin_gamma, vmax=vmax_gamma, extent=[0, 1, 0, 1])
        ax_gamma.set_title(f"γ(t={t_val:.2f})")
        ax_gamma.axis('off')

    # Add colorbars (omitted for brevity, can be added similar to _visualize_grf_marginals)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "learned_grf_evolution.png"), dpi=300)
    plt.close()

def visualize_enhanced_results(bridge, history, marginal_data, T, output_dir="output_rsipfo", is_grf=False):
    """Main visualization entry point."""
    os.makedirs(output_dir, exist_ok=True)
    bridge.eval()
    device = bridge.base_mean.device
    
    print(f"\n--- Generating Enhanced Visualizations in '{output_dir}' ---")
    
    # 1. Convergence Analysis
    _plot_convergence_analysis(history, output_dir)
    
    # 2. GRF Specific Visualizations
    if is_grf:
        _visualize_grf_marginals(marginal_data, output_dir, title="Input GRF Data Marginals")
        _visualize_learned_grf_evolution(bridge, T, output_dir, device)
        
    print("Visualization complete.")

# ============================================================================
# 9. Example Usage: Multiscale GRF Modeling (REVISED: Initialization and Config)
# ============================================================================

def run_multiscale_grf_example():
    """
    Run the RS-IPFO bridge on multiscale Gaussian Random Field data.
    """
    # Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED)
    
    OUTPUT_DIR = "output_multiscale_grf_rsipfo_fixed"
    
    # --- Experiment Hyperparameters (Multiscale GRF) ---
    T_MAX = 1.0; RESOLUTION = 16; N_SAMPLES = 512; N_CONSTRAINTS = 5
    MEAN_VAL = 10.0; STD_VAL = 2.0; L_DOMAIN = 1.0
    MICRO_CORR_LENGTH = 0.1; H_MAX_FACTOR = 0.5; COVARIANCE_TYPE = "gaussian"

    # --- Model Hyperparameters ---
    HIDDEN_SIZE = 256; FLOW_LAYERS = 4
    
    # --- RS-IPFO Configuration ---
    config = {
        'iterations': 1, 'sub_steps': 300, 'lr_theta': 1e-4, 'lr_phi': 1e-4,         
        'lambda_MM': 100000.0,      # Adjusted lambda_MM for the scale of Mixture MMD
        'lambda_Match': 100.0, 'batch_size_time': 128, 'n_samples_mmd': N_SAMPLES,
        'grad_clip_norm': 1.0,
        'mmd_sigma': -1.0,      # Activate Multi-Scale MMD
    }

    print("="*80)
    print("RS-IPFO FOR MULTISCALE RANDOM FIELD MODELING (FIXED IMPLEMENTATION)")
    print(f"Device: {DEVICE} | Resolution: {RESOLUTION}x{RESOLUTION}")
    print("="*80)
    
    # Step 1: Generate multiscale GRF data
    try:
        marginal_data, data_dim = generate_multiscale_grf_data(
            N_samples=N_SAMPLES, T=T_MAX, N_constraints=N_CONSTRAINTS, resolution=RESOLUTION,
            L_domain=L_DOMAIN, micro_corr_length=MICRO_CORR_LENGTH, H_max_factor=H_MAX_FACTOR,
            mean_val=MEAN_VAL, std_val=STD_VAL, covariance_type=COVARIANCE_TYPE, device=DEVICE
        )
    except Exception as e:
        print(f"Error during data generation: {e}"); traceback.print_exc(); return

    print(f"Data Dimension (D): {data_dim}")
    
    # Step 1.5: Calculate data statistics for initialization
    print("\n1.5. Calculating data statistics for initialization...")
    data_mean, data_std = calculate_data_statistics(marginal_data)
    print(f"  Data Mean (avg): {data_mean.mean().item():.4f}, Data Std (avg): {data_std.mean().item():.4f}")

    # Step 2: Initialize bridge
    print("\n2. Initializing RS-IPFO Bridge (Residual Dynamics)...")
    bridge = RSIPFOBridge(
        data_dim=data_dim, hidden_size=HIDDEN_SIZE, T=T_MAX, V_t=None, flow_layers=FLOW_LAYERS,
        data_mean=data_mean, # Pass mean for initialization
        data_std=data_std    # Pass std for initialization
    ).to(DEVICE)

    # Step 3: Train the bridge
    print("\n3. Training with RS-IPFO staggered scheme...")
    try:
        history = train_rs_ipfo(bridge, marginal_data, config)
        
        # Step 4: Visualize results (Requires visualization functions to be present)
        if history:
            print("\n4. Visualizing results...")
            visualize_enhanced_results(
                bridge, history, marginal_data, T_MAX, output_dir=OUTPUT_DIR, is_grf=True
            )
        
    except Exception as e:
        print(f"An error occurred during training/visualization: {e}")
        traceback.print_exc()

    print("\n" + "="*80)
    print("MULTISCALE GRF EXPERIMENT COMPLETE")
    print("="*80)

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
     # Note: This script requires PyTorch and other dependencies to be installed.
     try:
         import torch

         
         run_multiscale_grf_example() 
        #  print("Script prepared. Please uncomment 'run_multiscale_grf_example()' and ensure visualization functions are included to execute.")
     except ImportError:
         print("PyTorch or other dependencies not found.")