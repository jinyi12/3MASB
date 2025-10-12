import torch
import torch.nn as nn
from torch.distributions import Normal
from glow_layers import (
    ActNorm, Invertible1x1Conv, TimeCondAffineCoupling, squeeze, unsqueeze,
    TimeDependentActNorm, TimeDependentInvertible1x1Conv
)
from affinelayers import AffineCoupling, NeuralFlowCoupling
import numpy as np

# ============================================================================
# Time-Conditioned Glow Model
# ============================================================================

class FlowStep(nn.Module):
    """A single, expressive, and invertible step of a Neural Flow."""
    def __init__(self, num_channels, hidden_dim):
        super().__init__()
        self.act_norm = ActNorm(num_channels)
        self.inv_conv = Invertible1x1Conv(num_channels)
        self.affine_coupling = NeuralFlowCoupling(num_channels, hidden_dim)

    def forward(self, x, t, log_det_jac, reverse=False):
        if not reverse:
            x, log_det_jac = self.act_norm(x, log_det_jac, reverse=False)
            x, log_det_jac = self.inv_conv(x, log_det_jac, reverse=False)
            x, log_det_jac = self.affine_coupling(x, t, log_det_jac, reverse=False)
        else:
            x, log_det_jac = self.affine_coupling(x, t, log_det_jac, reverse=True)
            x, log_det_jac = self.inv_conv(x, log_det_jac, reverse=True)
            x, log_det_jac = self.act_norm(x, log_det_jac, reverse=True)
        return x, log_det_jac


class FlowScale(nn.Module):
    """A sequence of FlowSteps at a single scale (Stabilized)."""
    def __init__(self, num_channels, hidden_dim, num_steps):
        super().__init__()
        self.steps = nn.ModuleList([
            FlowStep(num_channels, hidden_dim) for _ in range(num_steps)
        ])

    def forward(self, x, t, log_det_jac, reverse=False):
        flow_steps = self.steps if not reverse else reversed(self.steps)

        for step in flow_steps:
            x, log_det_jac = step(x, t, log_det_jac, reverse=reverse)
            
            # FIX: Apply strong clamping in BOTH directions to prevent explosion.
            clamp_value = 1e5
            x = torch.clamp(x, -clamp_value, clamp_value)
            
            # Failsafe check for NaN/Inf
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"Warning: NaN/Inf detected in FlowScale activation (reverse={reverse}). Replacing.")
                x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)

            if torch.isnan(log_det_jac).any():
                print(f"Warning: NaN detected in log_det_jac (reverse={reverse}). Replacing.")
                log_det_jac = torch.where(torch.isnan(log_det_jac), torch.zeros_like(log_det_jac), log_det_jac)

        return x, log_det_jac

class TimeConditionedGlow(nn.Module):
    """Time-Conditioned Glow Model with a multi-scale architecture."""
    # Initialization robustness check added:
    def __init__(self, input_shape, hidden_dim, n_blocks_flow, num_scales=2):
        super().__init__()
        self.num_scales = num_scales
        
        self.scales = nn.ModuleList()
        C, H, W = input_shape
        
        self.latent_shapes = []
        print(f"TimeConditionedGlow init: input_shape={input_shape}")

        current_C, current_H, current_W = C, H, W
        for i in range(num_scales):
            # FIX: Check dimensions before squeeze
            if current_H < 2 or current_W < 2 or current_H % 2 != 0 or current_W % 2 != 0:
                 print(f"Warning: Spatial dimensions invalid for squeeze ({current_H}x{current_W}) at scale {i}. Reducing num_scales.")
                 self.num_scales = i
                 break

            # Squeeze
            current_C *= 4
            current_H //= 2
            current_W //= 2
            print(f"Scale {i}: After squeeze -> C={current_C}, H={current_H}, W={current_W}")
            
            self.scales.append(FlowScale(current_C, hidden_dim, n_blocks_flow))
            
            if i < num_scales - 1:
                # Split
                self.latent_shapes.append((current_C // 2, current_H, current_W))
                current_C = current_C // 2
                print(f"Scale {i}: After split -> remaining C={current_C}")
            else:
                # Final scale
                self.latent_shapes.append((current_C, current_H, current_W))
                print(f"Scale {i}: Final latent shape: ({current_C}, {current_H}, {current_W})")
        
        print(f"All latent shapes: {self.latent_shapes}")
        
        self.register_buffer('base_mean', torch.zeros(1))
        self.register_buffer('base_std', torch.ones(1))

    @property
    def base_dist(self):
        # A bit of a hack to make it compatible with the old structure
        # The base distribution is now over the entire spatial latent field
        return Normal(self.base_mean, self.base_std)

    def forward(self, data, t):
        # Inference direction: data -> epsilon
        z = data
        total_log_jac_det_inv = torch.zeros(z.shape[0], device=z.device)
        
        latents = []

        for i in range(self.num_scales):
            # Squeeze
            z = squeeze(z)

            # Flow at this scale
            z, total_log_jac_det_inv = self.scales[i](z, t, total_log_jac_det_inv, reverse=False)
            
            # Split
            if i < self.num_scales - 1:
                z_this_scale, z = z.chunk(2, dim=1)
                latents.append(z_this_scale)
        
        # The final z is the latent at the coarsest scale
        latents.append(z)
        
        # For compatibility, we need to return a single epsilon tensor.
        # We can flatten and concatenate all latents.
        flat_latents = [torch.flatten(latent, start_dim=1) for latent in latents]
        epsilon = torch.cat(flat_latents, dim=1)

        return epsilon, total_log_jac_det_inv

    def inverse(self, epsilon, t):
        # GENERATION: latent (epsilon) -> data (Z_t)
        
        batch_size = epsilon.shape[0]
        
        # Check if epsilon is already in the right spatial format
        if len(epsilon.shape) == 4:  # Already spatial format (B, C, H, W)
            # Convert back to flat format for consistent processing
            epsilon = epsilon.view(batch_size, -1)
        
        # We need to reconstruct the spatial latents from the flat epsilon vector.
        # First, un-flatten the epsilon vector into the list of spatial latents.
        latents = []
        current_pos = 0
        for i, shape in enumerate(self.latent_shapes):
            num_elements = int(np.prod(shape))
            if current_pos + num_elements > epsilon.shape[1]:
                # Pad epsilon if needed
                padding_size = current_pos + num_elements - epsilon.shape[1]
                epsilon = torch.cat([epsilon, torch.zeros(batch_size, padding_size, device=epsilon.device)], dim=1)
                
            flat_latent = epsilon[:, current_pos : current_pos + num_elements]
            spatial_latent = flat_latent.view(batch_size, *shape)
            latents.append(spatial_latent)
            current_pos += num_elements

        # The last latent in the list is the one for the coarsest scale.
        z = latents.pop()
        total_log_jac_det = torch.zeros(z.shape[0], device=z.device)

        for i in reversed(range(self.num_scales)):
            
            # Un-split (if not the finest scale)
            if i < self.num_scales - 1:
                # The other half of the split is the next latent in the reversed list
                new_z = latents.pop()
                z = torch.cat([new_z, z], dim=1)

            # Flow at this scale (in reverse)
            z, log_jac_det = self.scales[i](z, t, torch.zeros(z.shape[0], device=z.device), reverse=True)
            
            # Check for numerical issues after flow
            if torch.isnan(z).any():
                z = torch.where(torch.isnan(z), torch.zeros_like(z), z)
            if torch.isinf(z).any():
                z = torch.clamp(z, min=-100.0, max=100.0)
            
            total_log_jac_det += log_jac_det

            # Un-squeeze
            z = unsqueeze(z)
            
        return z, total_log_jac_det

    def log_prob(self, Z_t, t):
        # This is tricky because the latent space is now structured.
        # The original implementation assumed a flat latent vector.
        # We need to adapt the score function logic.
        
        epsilon, log_jac_det_inv = self.forward(Z_t, t) # <--- CORRECT
        
        if len(epsilon.shape) != 2: # Should be flat [B, D]
            epsilon = torch.flatten(epsilon, start_dim=1)

        # Check for NaN values before computing log probability
        if torch.isnan(epsilon).any():
            print("Warning: NaN values found in epsilon!")
            epsilon = torch.where(torch.isnan(epsilon), torch.zeros_like(epsilon), epsilon)

        # The base distribution is a standard normal over the flattened latent space.
        log_prob_base = Normal(0, 1).log_prob(epsilon).sum(dim=1)
        
        # Check for NaN in log_jac_det_inv
        if torch.isnan(log_jac_det_inv).any():
            print("Warning: NaN values found in log_jac_det_inv!")
            log_jac_det_inv = torch.where(torch.isnan(log_jac_det_inv), torch.zeros_like(log_jac_det_inv), log_jac_det_inv)
        
        return log_prob_base + log_jac_det_inv

    def score_function(self, Z_t, t):
        """Compute nabla_z log p_t(z) via automatic differentiation."""
        Z_t_g = Z_t.detach().clone().requires_grad_(True)
        log_p = self.log_prob(Z_t_g, t)
        
        score = torch.autograd.grad(
            log_p.sum(), Z_t_g, create_graph=torch.is_grad_enabled()
        )[0]
        
        # To prevent score blowup, we can clamp the score.
        # This is a simple but effective heuristic.
        score_norm = torch.norm(score.view(score.shape[0], -1), p=2, dim=1)
        max_norm = 1000.0  # Fixed max norm instead of using undefined latent_dim
        
        # Create a mask for scores that exceed the max norm
        mask = (score_norm > max_norm).view(-1, 1, 1, 1)
        
        # Clamp the score
        clamped_score = score * (max_norm / (score_norm.view(-1, 1, 1, 1) + 1e-6))
        
        # Apply clamping only where needed
        score = torch.where(mask, clamped_score, score)
        
        # Add a final failsafe to prevent returning nan or inf
        if torch.isnan(score).any() or torch.isinf(score).any():
            print("Warning: NaN/Inf detected in score function. Replacing with zeros.")
            score = torch.where(torch.isnan(score) | torch.isinf(score), torch.zeros_like(score), score)

        return score


# ============================================================================
# Time-Independent Glow Model (G_phi for Density Estimation p_0)
# ============================================================================

class FlowStepStatic(nn.Module):
    def __init__(self, num_channels, hidden_dim):
        super().__init__()
        self.act_norm = ActNorm(num_channels)
        self.inv_conv = Invertible1x1Conv(num_channels)
        # Use the standard, time-independent coupling layer
        self.affine_coupling = AffineCoupling(num_channels, hidden_dim)

    def forward(self, x, log_det_jac, reverse=False):
        if not reverse:
            x, log_det_jac = self.act_norm(x, log_det_jac, reverse=False)
            x, log_det_jac = self.inv_conv(x, log_det_jac, reverse=False)
            x, log_det_jac = self.affine_coupling(x, log_det_jac, reverse=False)
        else:
            x, log_det_jac = self.affine_coupling(x, log_det_jac, reverse=True)
            x, log_det_jac = self.inv_conv(x, log_det_jac, reverse=True)
            x, log_det_jac = self.act_norm(x, log_det_jac, reverse=True)
        return x, log_det_jac

class FlowScaleStatic(nn.Module):
    def __init__(self, num_channels, hidden_dim, num_steps):
        super().__init__()
        self.steps = nn.ModuleList([
            FlowStepStatic(num_channels, hidden_dim) for _ in range(num_steps)
        ])

    def forward(self, x, log_det_jac, reverse=False):
        flow_steps = self.steps if not reverse else reversed(self.steps)

        for step in flow_steps:
            x, log_det_jac = step(x, log_det_jac, reverse=reverse)
            # Stability clamping (KISS stabilization)
            clamp_value = 1e5
            x = torch.clamp(x, -clamp_value, clamp_value)
            if torch.isnan(x).any() or torch.isinf(x).any():
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        return x, log_det_jac

class StaticGlow(nn.Module):
    """Time-Independent Glow Model for density estimation (p_0). Uses multi-scale architecture."""
    def __init__(self, input_shape, hidden_dim, n_blocks_flow, num_scales=2):
        super().__init__()
        self.num_scales = num_scales
        self.scales = nn.ModuleList()
        C, H, W = input_shape
        self.latent_shapes = []

        # Initialization logic (mirrors TimeConditionedGlow)
        current_C, current_H, current_W = C, H, W
        for i in range(num_scales):
            if current_H < 2 or current_W < 2:
                 self.num_scales = i
                 break

            current_C *= 4
            current_H //= 2
            current_W //= 2
            
            self.scales.append(FlowScaleStatic(current_C, hidden_dim, n_blocks_flow))
            
            # Standard GLOW splitting strategy
            if i < num_scales - 1:
                self.latent_shapes.append((current_C // 2, current_H, current_W))
                current_C = current_C // 2
            else:
                self.latent_shapes.append((current_C, current_H, current_W))
        
        self.register_buffer('base_mean', torch.zeros(1))
        self.register_buffer('base_std', torch.ones(1))

    @property
    def base_dist(self):
        return Normal(self.base_mean, self.base_std)

    def forward(self, data):
        # Inference: x_0 -> epsilon (G_phi^-1)
        z = data
        total_log_jac_det_inv = torch.zeros(z.shape[0], device=z.device)
        latents = []

        for i in range(self.num_scales):
            z = squeeze(z)
            z, total_log_jac_det_inv = self.scales[i](z, total_log_jac_det_inv, reverse=False)
            
            if i < self.num_scales - 1:
                z_this_scale, z = z.chunk(2, dim=1)
                latents.append(z_this_scale)
        
        latents.append(z)
        flat_latents = [torch.flatten(latent, start_dim=1) for latent in latents]
        epsilon = torch.cat(flat_latents, dim=1)
        return epsilon, total_log_jac_det_inv

    def inverse(self, epsilon):
        # Generation: epsilon -> x_0 (G_phi)
        batch_size = epsilon.shape[0]
        
        # Handle spatial input if provided, otherwise assume flat
        if len(epsilon.shape) == 4:
            epsilon = epsilon.view(batch_size, -1)
        
        # Un-flatten epsilon into spatial latents
        latents = []
        current_pos = 0
        for i, shape in enumerate(self.latent_shapes):
            num_elements = int(np.prod(shape))
            # Handle potential size mismatch (padding if needed)
            if current_pos + num_elements > epsilon.shape[1]:
                padding_size = current_pos + num_elements - epsilon.shape[1]
                epsilon = torch.cat([epsilon, torch.zeros(batch_size, padding_size, device=epsilon.device)], dim=1)
                
            flat_latent = epsilon[:, current_pos : current_pos + num_elements]
            spatial_latent = flat_latent.view(batch_size, *shape)
            latents.append(spatial_latent)
            current_pos += num_elements

        z = latents.pop()
        total_log_jac_det = torch.zeros(z.shape[0], device=z.device)

        for i in reversed(range(self.num_scales)):
            if i < self.num_scales - 1:
                new_z = latents.pop()
                z = torch.cat([new_z, z], dim=1)

            # FlowScaleStatic accumulates the log_det_jac
            z, total_log_jac_det = self.scales[i](z, total_log_jac_det, reverse=True)
            z = unsqueeze(z)
            
        return z, total_log_jac_det

    def log_prob(self, Z_0):
        epsilon, log_jac_det_inv = self.forward(Z_0)
        if len(epsilon.shape) != 2:
            epsilon = torch.flatten(epsilon, start_dim=1)
            
        # Stability checks
        if torch.isnan(epsilon).any() or torch.isinf(epsilon).any():
            epsilon = torch.nan_to_num(epsilon, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.isnan(log_jac_det_inv).any():
            log_jac_det_inv = torch.nan_to_num(log_jac_det_inv, nan=0.0)

        log_prob_base = Normal(0, 1).log_prob(epsilon).sum(dim=1)
        return log_prob_base + log_jac_det_inv

# ============================================================================
# Invertible Neural Flow Model (T_theta for Dynamics T(x_0, t))
# ============================================================================

# Define the efficient flow step
class EfficientFlowStep(nn.Module):
    """
    An efficient, invertible step for Neural Flow. 
    Uses only NeuralFlowCoupling for speed, stability, and guaranteed identity at t=0.
    """
    def __init__(self, num_channels, hidden_dim, reverse_split=False):
        super().__init__()
        if num_channels % 2 != 0:
            raise ValueError(f"Number of channels must be even for splitting, got {num_channels}")
            
        self.affine_coupling = NeuralFlowCoupling(num_channels, hidden_dim, reverse_split=reverse_split)

    def forward(self, x, t, log_det_jac, reverse=False):
        x, log_det_jac = self.affine_coupling(x, t, log_det_jac, reverse=reverse)
        return x, log_det_jac

# Redefine InvertibleNeuralFlow (Replacing the existing implementation)
class InvertibleNeuralFlow(nn.Module):
    """
    Efficient, dimension-preserving Invertible Neural Flow for the transport map T(x_0, t).
    Uses Squeeze -> Flow (RealNVP style) -> Unsqueeze architecture.
    """
    def __init__(self, input_shape, hidden_dim, n_blocks_flow, num_scales=2):
        super().__init__()
        # Architecture redesigned; num_scales ignored but kept for compatibility.
        
        C, H, W = input_shape
        
        # Flow operates on the squeezed representation.
        C_squeezed = C * 4
        
        self.flow_steps = nn.ModuleList()
        for i in range(n_blocks_flow):
            # Alternate the splitting strategy for channel mixing
            reverse_split = (i % 2 == 1)
            self.flow_steps.append(
                EfficientFlowStep(C_squeezed, hidden_dim, reverse_split=reverse_split)
            )

    def _ensure_time_batch(self, t, batch_size):
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=next(self.parameters()).device, dtype=torch.float32)
        if t.dim() == 0:
            t = t.view(1, 1).expand(batch_size, 1)
        elif t.shape[0] == 1:
             t = t.expand(batch_size, *t.shape[1:])
        if t.dim() == 1:
            t = t.unsqueeze(1)
        return t

    def forward(self, x_0, t):
        # T(x_0, t) -> x_t
        t_batch = self._ensure_time_batch(t, x_0.shape[0])
        log_det_jac = torch.zeros(x_0.shape[0], device=x_0.device)
        
        # 1. Squeeze
        z = squeeze(x_0)
        
        # 2. Apply Flow
        for step in self.flow_steps:
            z, log_det_jac = step(z, t_batch, log_det_jac, reverse=False)
        
        # 3. Unsqueeze (Dimension preservation)
        x_t = unsqueeze(z)
        
        return x_t, log_det_jac

    def inverse(self, x_t, t):
        # T^{-1}(x_t, t) -> x_0
        t_batch = self._ensure_time_batch(t, x_t.shape[0])
        log_det_jac_inv = torch.zeros(x_t.shape[0], device=x_t.device)
        
        # 1. Squeeze (to match the space where the flow operates)
        z = squeeze(x_t)
        
        # 2. Apply Inverse Flow
        for step in reversed(self.flow_steps):
            z, log_det_jac_inv = step(z, t_batch, log_det_jac_inv, reverse=True)
            
        # 3. Unsqueeze
        x_0 = unsqueeze(z)
            
        return x_0, log_det_jac_inv