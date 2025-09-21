import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# FIX: Use torch.linalg instead of scipy.linalg for consistency and stability

# ============================================================================
# Glow Architecture Layers (PyTorch Implementation) - STABILIZED
# ============================================================================

class ActNorm(nn.Module):
    """Activation Normalization Layer - Stabilized."""
    def __init__(self, num_features, eps=1e-6, log_scale_clamp=5.0):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.log_scale_clamp = log_scale_clamp # FIX: Clamping factor
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
        self.log_scale = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x, log_det_jac, reverse=False):
        if not self.initialized:
            with torch.no_grad():
                # Data-dependent initialization
                mean = torch.mean(x, dim=[0, 2, 3], keepdim=True)
                std = torch.std(x, dim=[0, 2, 3], keepdim=True)
                
                # FIX: Robust initialization - ensure std is not too small (e.g. min 1e-3)
                std_clamped = torch.clamp(std, min=1e-3)
                
                self.bias.data.copy_(-mean)
                log_scale_init = torch.log(1 / (std_clamped + self.eps))
                
                # Clamp initialization
                log_scale_init = torch.clamp(log_scale_init, min=-self.log_scale_clamp, max=self.log_scale_clamp)
                self.log_scale.data.copy_(log_scale_init)
                self.initialized.fill_(1)

        # FIX: Clamping log_scale during training
        log_scale = torch.clamp(self.log_scale, min=-self.log_scale_clamp, max=self.log_scale_clamp)
        bias = self.bias

        H, W = x.shape[2], x.shape[3]

        if not reverse:
            x = torch.exp(log_scale) * (x + bias)
            d_log_det = torch.sum(log_scale) * H * W
            log_det_jac = log_det_jac + d_log_det
        else:
            x = x * torch.exp(-log_scale) - bias
            d_log_det = torch.sum(log_scale) * H * W
            log_det_jac = log_det_jac - d_log_det
            
        return x, log_det_jac

class Invertible1x1Conv(nn.Module):
    """Invertible 1x1 Convolution using PLU decomposition - Stabilized."""
    def __init__(self, num_channels, log_s_clamp=5.0):
        super().__init__()
        self.num_channels = num_channels
        self.log_s_clamp = log_s_clamp # FIX: Clamping factor
        
        # FIX: Initialize with torch.linalg
        W_init = torch.randn(num_channels, num_channels)
        Q, _ = torch.linalg.qr(W_init)
        
        P, L, U = torch.linalg.lu(Q)
        
        self.register_buffer('P', P)
        self.register_buffer('sign_s', torch.sign(torch.diag(U)))
        
        self.L = nn.Parameter(L)
        self.log_s = nn.Parameter(torch.log(torch.abs(torch.diag(U)) + 1e-6))
        self.U = nn.Parameter(torch.triu(U, diagonal=1))
        
        self.register_buffer('L_mask', torch.tril(torch.ones(num_channels, num_channels), -1))
        self.register_buffer('I', torch.eye(num_channels))

    def _get_clamped_log_s(self):
        return torch.clamp(self.log_s, min=-self.log_s_clamp, max=self.log_s_clamp)

    def _get_weight(self, reverse):
        L = self.L * self.L_mask + self.I
        
        # FIX: Use clamped log_s
        log_s_clamped = self._get_clamped_log_s()
        U = self.U * self.L_mask.T + torch.diag(self.sign_s * torch.exp(log_s_clamped))
        
        if not reverse:
            W = self.P @ L @ U
        else:
            # FIX: Robust inversion using torch.linalg.inv
            try:
                # P is orthogonal, P.T is the inverse
                W = torch.linalg.inv(U) @ torch.linalg.inv(L) @ self.P.T
            except torch.linalg.LinAlgError:
                print("Warning: Singular matrix in Invertible1x1Conv inverse. Falling back.")
                # Fallback: Recompute forward matrix and invert it
                W_fwd = self.P @ L @ U
                W = torch.linalg.inv(W_fwd)
            
        return W.view(self.num_channels, self.num_channels, 1, 1)

    def forward(self, x, log_det_jac, reverse=False):
        W = self._get_weight(reverse)
        x = F.conv2d(x, W)
        
        # FIX: Use clamped log_s for determinant calculation
        log_s_clamped = self._get_clamped_log_s()
        
        d_log_det = log_s_clamped.sum() * x.shape[2] * x.shape[3]
        if reverse:
            d_log_det *= -1
            
        log_det_jac = log_det_jac + d_log_det
        return x, log_det_jac

class TimeCondAffineCoupling(nn.Module):
    """Time-Conditioned Affine Coupling Layer (Stabilized)."""
    def __init__(self, in_channels, hidden_dim, scale_clamp=2.0):
        super().__init__()
        self.in_channels = in_channels
        self.scale_clamp = scale_clamp
        
        input_channels = in_channels // 2 + 1
        
        # Ensure the output channels match the input split size
        output_channels = (in_channels // 2) * 2

        self.net = nn.Sequential(
            nn.utils.weight_norm(nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, padding=0)),
            nn.ReLU(inplace=True),
            # FIX: Ensure output channels match expected size
            nn.utils.weight_norm(nn.Conv2d(hidden_dim, output_channels, kernel_size=3, padding=1))
        )
        # Initialize last layer to zeros
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

    def forward(self, x, t, log_det_jac, reverse=False):
        if x.shape[0] == 0:
            return x, log_det_jac
            
        x_a, x_b = x.chunk(2, dim=1)
        
        # Time conditioning setup... (remains the same)
        if t.shape[0] != x.shape[0]:
            if t.shape[0] == 1:
                t = t.expand(x.shape[0], -1)
            else:
                raise ValueError(f"Time tensor batch size mismatch.")
        
        t_channel = t.view(-1, 1, 1, 1).expand(-1, 1, x_b.shape[2], x_b.shape[3])
        net_in = torch.cat([x_b, t_channel], dim=1)
        
        s_m = self.net(net_in)
        log_s_raw, m = s_m.chunk(2, dim=1)
        
        # FIX: Standardized stabilization using tanh scaling (RealNVP style)
        s = torch.tanh(log_s_raw) * self.scale_clamp
        
        # Removed redundant clamping of m and exp(s) checks.
        
        if not reverse:
            y_a = torch.exp(s) * x_a + m
            log_det_jac = log_det_jac + s.sum(dim=[1, 2, 3])
        else:
            y_a = (x_a - m) * torch.exp(-s)
            log_det_jac = log_det_jac - s.sum(dim=[1, 2, 3])
            
        y = torch.cat([y_a, x_b], dim=1)
        return y, log_det_jac

def squeeze(x):
    # (Implementation remains the same)
    B, C, H, W = x.shape
    x = x.view(B, C, H // 2, 2, W // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(B, C * 4, H // 2, W // 2)

def unsqueeze(x):
    # (Implementation remains the same)
    B, C, H, W = x.shape
    x = x.view(B, C // 4, 2, 2, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    return x.view(B, C // 4, H * 2, W * 2)