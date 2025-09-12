"""
Training Utilities for Asymmetric Bridges
==========================================

This module provides training functions for different bridge models.
"""

import torch
from tqdm import trange
from typing import Dict, List, Any
from torch import Tensor

def train_bridge(
    bridge, 
    marginal_data: Dict[float, Tensor],
    epochs: int = 1000, 
    lr: float = 1e-3,
    lambda_path: float = 0.1,
    verbose: bool = True
) -> List[Dict[str, float]]:
    """Train any bridge model using MLE and path regularization."""
    optimizer = torch.optim.Adam(bridge.parameters(), lr=lr)
    loss_history = []
    
    if verbose:
        pbar = trange(epochs)
    else:
        pbar = range(epochs)
    
    for epoch in pbar:
        optimizer.zero_grad()
        
        # Compute loss
        total_loss, mle_loss, path_loss = bridge.loss(marginal_data, lambda_path=lambda_path)
        
        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), max_norm=1.0)

        optimizer.step()
        
        loss_history.append({
            'total': total_loss.item(),
            'mle': mle_loss.item(), 
            'path': path_loss.item()
        })
        
        if verbose and isinstance(pbar, type(trange(0))):
            pbar.set_description(
                f"Loss: {total_loss.item():.4f} (MLE: {mle_loss.item():.4f}, Path: {path_loss.item():.4f})"
            )
    
    return loss_history