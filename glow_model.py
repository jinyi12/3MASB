import torch
import torch.nn as nn
from torch.distributions import Normal
from glow_layers import ActNorm, Invertible1x1Conv, TimeCondAffineCoupling, squeeze, unsqueeze
import numpy as np

# ============================================================================
# Time-Conditioned Glow Model
# ============================================================================

class FlowStep(nn.Module):
    # (Implementation remains the same, relying on stabilized layers)
    def __init__(self, num_channels, hidden_dim):
        super().__init__()
        self.act_norm = ActNorm(num_channels)
        self.inv_conv = Invertible1x1Conv(num_channels)
        self.affine_coupling = TimeCondAffineCoupling(num_channels, hidden_dim)

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
