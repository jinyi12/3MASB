"""
Training Utilities for Asymmetric Bridges
==========================================

This module provides training functions for different bridge models.
Includes the two-stage training procedure for C-CVEP with MFM.
"""

import torch
from tqdm import trange
from typing import Dict, List, Tuple
from torch import Tensor

# Import C-CVEP components
from ccvep_core import TrajectoryHandler, training_step_ccvep, training_step_mfm_interpolator


def extract_couplings(marginal_data: Dict[float, Tensor]) -> Tuple[Tensor, Tuple[Tensor, ...]]:
    """
    Helper to extract time points and coupled data tensors. Ensures data is paired (required for MFM).
    """
    # Sort by time
    sorted_items = sorted(marginal_data.items(), key=lambda item: item[0])
    
    time_points_list, couplings_list = [], []
    N_samples, device = None, None

    for t, samples in sorted_items:
        if device is None: 
            device = samples.device
        if N_samples is None:
            N_samples = samples.shape[0]
        elif samples.shape[0] != N_samples:
            raise ValueError(f"Data must be paired. Inconsistent sample count at time {t}.")
            
        time_points_list.append(t)
        couplings_list.append(samples)
        
    time_points = torch.tensor(time_points_list, dtype=torch.float32, device=device)
    couplings = tuple(couplings_list)
    
    return time_points, couplings

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


def train_ccvep_bridge(
    bridge,  # CCVEPBridge
    marginal_data: Dict[float, Tensor],
    # Stage 0 (RBF Metric) parameters
    use_rbf_metric: bool = False,
    rbf_params: Dict = {},
    # Stage 1 (MFM Pre-training) parameters
    mfm_epochs: int = 500,
    mfm_lr: float = 1e-3,
    # Early stopping for MFM pre-training
    early_stopping: bool = True,
    early_stopping_patience: int = 20,
    early_stopping_min_delta: float = 1e-4,
    # Stage 2 (C-CVEP Training) parameters
    ccvep_epochs: int = 1000, 
    ccvep_lr: float = 1e-4,
    lambda_CE: float = 1.0,
    # Common parameters
    batch_size: int = 64,
    grad_clip_norm: float = 1.0,
    verbose: bool = True
) -> List[Dict[str, float]]:
    """Train the C-CVEP bridge model using the three-stage RBF+MFM+DSM procedure."""
    
    # 1. Data Preparation
    print("Preparing coupled data...")
    try:
        time_points, couplings = extract_couplings(marginal_data)
    except ValueError as e:
        print(f"Error during data preparation: {e}")
        return []

    # ======================================================
    # Stage 0: RBF Metric Initialization and Pre-training
    # ======================================================
    metric_handler = getattr(bridge, 'G_rbf', None)

    if use_rbf_metric and metric_handler is not None:
        print("\n--- Stage 0: RBF Metric Initialization and Pre-training ---")
        
        # Combine all data for metric learning
        all_data = torch.cat(couplings, dim=0)

        # Initialize (K-means and bandwidths)
        metric_handler.initialize_from_data(
            all_data, 
            kappa=rbf_params.get('kappa', 1.0)
        )
        
        # Pre-train weights (Stage 0b)
        rbf_epochs = rbf_params.get('epochs', 100)
        if rbf_epochs > 0:
            metric_handler.train_metric(
                all_data, 
                epochs=rbf_epochs,
                lr=rbf_params.get('lr', 1e-3)
            )
        
        metric_handler.eval()  # Set metric handler to eval mode
        print("Stage 0 completed.")
    else:
        print("Skipping Stage 0. Using Euclidean metric (G=I).")
        metric_handler = None

    # 2. Initialize Trajectory Handler
    print("Initializing Parameterized Trajectory Handler...")
    if not hasattr(bridge, 'phi_eta') or bridge.phi_eta is None:
        raise ValueError("Bridge model does not have an initialized phi_eta (InterpolatorNetwork).")

    trajectory_handler = TrajectoryHandler(
        interpolator_network=bridge.phi_eta,
        couplings=couplings, 
        time_points=time_points,
        metric_handler=metric_handler  # Pass the trained metric handler
    )
    
    # ======================================================
    # Stage 1: MFM Interpolator Pre-training (Optimize phi_eta)
    # ======================================================
    if mfm_epochs > 0:
        print("\n--- Stage 1: MFM Pre-training (Optimizing Interpolants) ---")
        print(f"Training phi_eta for {mfm_epochs} epochs with lr={mfm_lr}")

        # Initialize optimizer targeting only phi_eta parameters
        optimizer_mfm = torch.optim.Adam(bridge.phi_eta.parameters(), lr=mfm_lr)
        
        pbar_mfm = trange(mfm_epochs) if verbose else range(mfm_epochs)
        
        # Set model mode
        bridge.train()  # Ensure phi_eta is in train mode

        # Early stopping variables
        best_loss = float('inf')
        best_epoch = -1
        best_state = None
        epochs_no_improve = 0

        for epoch in pbar_mfm:
            loss_dict = training_step_mfm_interpolator(
                trajectory_handler=trajectory_handler,
                optimizer=optimizer_mfm,
                batch_size=batch_size
            )

            if not loss_dict:
                continue

            # Apply Gradient Clipping (specific to phi_eta)
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(bridge.phi_eta.parameters(), max_norm=grad_clip_norm)

            optimizer_mfm.step()

            # Logging
            geo_loss = loss_dict.get('loss_geodesic', None)
            if verbose and isinstance(pbar_mfm, type(trange(0))) and geo_loss is not None:
                pbar_mfm.set_description(f"Geodesic Loss: {geo_loss:.6f}")

            # Early stopping check
            if early_stopping and geo_loss is not None:
                if best_loss - geo_loss > early_stopping_min_delta:
                    best_loss = geo_loss
                    best_epoch = epoch
                    # Save best phi_eta weights (CPU copy)
                    best_state = {k: v.cpu().clone() for k, v in bridge.phi_eta.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch} (best epoch {best_epoch}, best loss {best_loss:.6f}). Restoring best phi_eta weights.")
                    if best_state is not None:
                        bridge.phi_eta.load_state_dict(best_state)
                    break

        # If training finished without early stopping but best_state exists, restore best_state
        if best_state is not None:
            bridge.phi_eta.load_state_dict(best_state)

        print("Stage 1 completed.")
    else:
        print("Skipping Stage 1 (MFM Pre-training).")

    # ======================================================
    # Stage 2: C-CVEP Training (Optimize v_theta, U_phi)
    # ======================================================
    print("\n--- Stage 2: C-CVEP Training (Optimizing Velocity/Energy) ---")
    print(f"Training v_theta/U_phi for {ccvep_epochs} epochs with lr={ccvep_lr}")

    # Initialize optimizer targeting v_theta and U_phi parameters
    # We must explicitly optimize only these parameters, as phi_eta is fixed.
    ccvep_params = list(bridge.v_theta.parameters()) + list(bridge.U_phi.parameters())
    optimizer_ccvep = torch.optim.Adam(ccvep_params, lr=ccvep_lr)
    
    loss_history = []
    pbar_ccvep = trange(ccvep_epochs) if verbose else range(ccvep_epochs)
    
    # Set model mode
    bridge.train()  # Ensure v_theta and U_phi are in train mode

    for epoch in pbar_ccvep:
        
        # Execute one training step
        loss_dict = training_step_ccvep(
            trajectory_handler=trajectory_handler,
            v_theta_model=bridge.v_theta,
            U_phi_model=bridge.U_phi,
            optimizer=optimizer_ccvep,
            lambda_CE=lambda_CE,
            batch_size=batch_size
        )
        
        if not loss_dict: 
            continue

        # Apply Gradient Clipping (specific to v_theta, U_phi)
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(ccvep_params, max_norm=grad_clip_norm)

        # Step the optimizer
        optimizer_ccvep.step()

        loss_history.append(loss_dict)
        
        # Logging
        if verbose and isinstance(pbar_ccvep, type(trange(0))):
            total_loss = loss_dict.get('loss_total', 0.0)
            traj_loss = loss_dict.get('loss_trajectory', 0.0)
            cons_loss = loss_dict.get('loss_consistency', 0.0)
            pbar_ccvep.set_description(
                f"Loss: {total_loss:.4f} (Traj: {traj_loss:.4f}, Cons: {cons_loss:.4f})"
            )
    
    print("Stage 2 completed.")
    return loss_history