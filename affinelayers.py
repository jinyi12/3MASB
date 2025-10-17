import torch
import torch.nn as nn
from glow_layers import GatedConvNet

# ============================================================================
# Mask Creation Utilities
# ============================================================================

def create_checkerboard_mask(h, w, invert=False):
    """Create a checkerboard mask for spatial masking."""
    x, y = torch.arange(h, dtype=torch.int32), torch.arange(w, dtype=torch.int32)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    mask = torch.fmod(xx + yy, 2)
    mask = mask.to(torch.float32).view(1, 1, h, w)
    if invert:
        mask = 1 - mask
    return mask

def create_channel_mask(c_in, invert=False):
    """Create a channel-wise mask for channel splitting."""
    mask = torch.cat([torch.ones(c_in//2, dtype=torch.float32),
                      torch.zeros(c_in-c_in//2, dtype=torch.float32)])
    mask = mask.view(1, c_in, 1, 1)
    if invert:
        mask = 1 - mask
    return mask

# ============================================================================
# Additional Layers for Decoupled Approach
# ============================================================================

class AffineCoupling(nn.Module):
    """Standard Affine Coupling Layer (Stabilized, Time-Independent). Used for Density Model G_phi."""
    def __init__(self, in_channels, hidden_dim, mask=None, num_layers=3):
        """
        Args:
            in_channels: Number of input channels
            hidden_dim: Hidden dimension for the neural network
            mask: Optional binary mask (0 or 1). If None, uses channel splitting (channels // 2).
                  When provided: 1 = input to NN, 0 = transformed.
            num_layers: Number of GatedConv blocks
        """
        super().__init__()
        self.in_channels = in_channels
        self.use_mask = mask is not None
        
        # Register mask as buffer if provided
        if self.use_mask:
            self.register_buffer('mask', mask)
            # Network takes full masked input
            net_in_channels = in_channels
            net_out_channels = in_channels * 2
        else:
            # Channel splitting mode
            net_in_channels = in_channels // 2
            net_out_channels = (in_channels // 2) * 2

        self.net = GatedConvNet(
            c_in=net_in_channels,
            c_hidden=hidden_dim,
            c_out=net_out_channels,
            num_layers=num_layers
        )
        
        # Learnable scaling factor for stability
        scale_channels = in_channels if self.use_mask else in_channels // 2
        self.log_scaling_factor = nn.Parameter(torch.zeros(scale_channels, 1, 1))

    def forward(self, x, log_det_jac, reverse=False):
        if x.shape[0] == 0:
            return x, log_det_jac
        
        if self.use_mask:
            # Mask-based coupling
            x_masked = x * self.mask
            s_m = self.net(x_masked)
            log_s_raw, m = s_m.chunk(2, dim=1)
            
            # Improved stabilization
            s_fac = torch.exp(torch.clamp(self.log_scaling_factor, -3.0, 3.0))
            s = torch.tanh(log_s_raw / (s_fac + 1e-6)) * s_fac
            
            # Apply inverse mask to outputs
            s = s * (1 - self.mask)
            m = m * (1 - self.mask)
            
            # Affine transformation
            if not reverse:
                y = (x + m) * torch.exp(s)
                log_det_jac = log_det_jac + s.sum(dim=[1, 2, 3])
            else:
                y = (x * torch.exp(-s)) - m
                log_det_jac = log_det_jac - s.sum(dim=[1, 2, 3])
        else:
            # Channel splitting mode
            x_a, x_b = x.chunk(2, dim=1)
            s_m = self.net(x_b)
            log_s_raw, m = s_m.chunk(2, dim=1)
            
            # Improved stabilization
            s_fac = torch.exp(torch.clamp(self.log_scaling_factor, -3.0, 3.0))
            s = torch.tanh(log_s_raw / (s_fac + 1e-6)) * s_fac
            
            if not reverse:
                y_a = torch.exp(s) * x_a + m
                log_det_jac = log_det_jac + s.sum(dim=[1, 2, 3])
            else:
                y_a = (x_a - m) * torch.exp(-s)
                log_det_jac = log_det_jac - s.sum(dim=[1, 2, 3])
            
            y = torch.cat([y_a, x_b], dim=1)
            
        return y, log_det_jac

class NeuralFlowCoupling(nn.Module):
    """
    Time-Conditioned Affine Coupling Layer for Invertible Neural Flow (Dynamics Model T_theta).
    Ensures identity map at t=0 by scaling the transformation parameters (s, m) by t.
    Supports alternating channel splits.
    """
    # Update the __init__ method to accept reverse_split
    def __init__(self, in_channels, hidden_dim, mask=None, num_layers=3, reverse_split=False):
        super().__init__()
        self.in_channels = in_channels
        self.use_mask = mask is not None
        self.reverse_split = reverse_split # Added parameter

        # Register mask as buffer if provided
        if self.use_mask:
            self.register_buffer('mask', mask)
            # Network takes full masked input + time
            net_in_channels = in_channels + 1
            net_out_channels = in_channels * 2
        else:
            # Channel splitting mode + time
            net_in_channels = in_channels // 2 + 1
            net_out_channels = (in_channels // 2) * 2

        self.net = GatedConvNet(
            c_in=net_in_channels,
            c_hidden=hidden_dim,
            c_out=net_out_channels,
            num_layers=num_layers
        )
        
        # Learnable scaling factor for stability
        scale_channels = in_channels if self.use_mask else in_channels // 2
        self.log_scaling_factor = nn.Parameter(torch.zeros(scale_channels, 1, 1))

    # Update the forward method
    def forward(self, x, t, log_det_jac, reverse=False):
        if x.shape[0] == 0:
            return x, log_det_jac
        
        # Time conditioning setup
        if t.shape[0] != x.shape[0]:
            if t.shape[0] == 1:
                t = t.expand(x.shape[0], *t.shape[1:])
            else:
                raise ValueError(f"Time tensor batch size mismatch. Expected {x.shape[0]} or 1, got {t.shape[0]}.")
        
        time_scale = t.view(-1, 1, 1, 1)
        
        if self.use_mask:
            # Mask-based coupling
            x_masked = x * self.mask
            t_channel = time_scale.expand(-1, 1, x.shape[2], x.shape[3])
            net_in = torch.cat([x_masked, t_channel], dim=1)
            
            s_m = self.net(net_in)
            log_s_raw, m = s_m.chunk(2, dim=1)
            
            # Improved stabilization
            s_fac = torch.exp(torch.clamp(self.log_scaling_factor, -3.0, 3.0))
            s = torch.tanh(log_s_raw / (s_fac + 1e-6)) * s_fac
            
            # Scale by time for identity at t=0
            s = s * time_scale
            m = m * time_scale
            
            # Apply inverse mask to outputs
            s = s * (1 - self.mask)
            m = m * (1 - self.mask)
            
            # Affine transformation
            if not reverse:
                y = (x + m) * torch.exp(s)
                log_det_jac = log_det_jac + s.sum(dim=[1, 2, 3])
            else:
                y = (x * torch.exp(-s)) - m
                log_det_jac = log_det_jac - s.sum(dim=[1, 2, 3])
        else:
            # Channel splitting mode
            x_a, x_b = x.chunk(2, dim=1)
            
            # MODIFICATION: Implement alternating split
            # If reverse_split=False (Default): x_a is transformed, x_b is conditioning.
            if not self.reverse_split:
                x_trans, x_cond = x_a, x_b
            else: # reverse_split=True: x_b is transformed, x_a is conditioning.
                x_trans, x_cond = x_b, x_a

            t_channel = time_scale.expand(-1, 1, x_cond.shape[2], x_cond.shape[3])
            net_in = torch.cat([x_cond, t_channel], dim=1)
            
            s_m = self.net(net_in)
            log_s_raw, m = s_m.chunk(2, dim=1)
            
            # Stabilization (remains the same)
            s_fac = torch.exp(torch.clamp(self.log_scaling_factor, -3.0, 3.0))
            s = torch.tanh(log_s_raw / (s_fac + 1e-6)) * s_fac
            
            # Scale by time for identity at t=0 (remains the same)
            s = s * time_scale
            m = m * time_scale
            
            if not reverse:
                y_trans = torch.exp(s) * x_trans + m
                log_det_jac = log_det_jac + s.sum(dim=[1, 2, 3])
            else:
                y_trans = (x_trans - m) * torch.exp(-s)
                log_det_jac = log_det_jac - s.sum(dim=[1, 2, 3])
            
            # MODIFICATION: Recombine based on split order
            if not self.reverse_split:
                y_a, y_b = y_trans, x_cond
            else:
                y_a, y_b = x_cond, y_trans
            
            y = torch.cat([y_a, y_b], dim=1)
            
        return y, log_det_jac