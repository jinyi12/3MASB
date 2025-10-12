#!/usr/bin/env python3
"""
Asymmetric Multi-Marginal Bridge - Affine Flow Training Script
==============================================================

A streamlined script for training affine flow models using the Asymmetric 
Multi-Marginal Bridge framework. This script demonstrates the KISS and YAGNI 
principles by providing a clean, focused implementation.

Usage:
    python train_affine_flow.py [--grf | --spiral] [--epochs EPOCHS] [--lr LR]

Examples:
    python train_affine_flow.py --grf --epochs 1000
    python train_affine_flow.py --spiral --epochs 500 --lr 1e-3
"""

import argparse
import os
import torch
import numpy as np
from typing import Dict

# Import modularized components
from models import NeuralGaussianBridge
from utilities.data_generation import generate_multiscale_grf_data, generate_spiral_distributional_data
from utilities.training import train_bridge
from utilities.simulation import generate_comparative_backward_samples
from utilities.visualization import visualize_bridge_results
from utilities.validation import validate_asymmetric_consistency, calculate_validation_metrics


def setup_experiment(args) -> Dict:
    """Setup experiment configuration based on arguments."""
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration (prefer CPU for this scale)
    device = "cpu"
    
    # Common parameters
    config = {
        'device': device,
        'T': 1.0,
        'epochs': args.epochs,
        'lr': args.lr,
        'lambda_path': 0.001,
        'sigma_reverse': 0.5,
        'n_sde_steps': 100,
        'n_viz_particles': 256,
    }
    
    if args.data_type == 'grf':
        # GRF-specific configuration
        config.update({
            'output_dir': 'output_affine_flow_grf',
            'data_type': 'grf',
            'resolution': 16,
            'data_dim': 16 * 16,
            'n_constraints': 5,
            'n_samples': 1024,
            'hidden_size': 256,
            # GRF generation parameters
            'l_domain': 1.0,
            'micro_corr_length': 0.1,
            'h_max_factor': 0.5,
            'mean_val': 10.0,
            'std_val': 2.0,
            'covariance_type': 'gaussian'
        })
    else:
        # Spiral data configuration
        config.update({
            'output_dir': 'output_affine_flow_spiral',
            'data_type': 'spiral',
            'data_dim': 3,
            'n_constraints': 5,
            'n_samples_per_marginal': 512,
            'noise_std': 0.1,
            'hidden_size': 128,
        })
    
    return config


def generate_data(config):
    """Generate marginal data based on configuration."""
    
    print("\n--- Data Generation ---")
    
    if config['data_type'] == 'grf':
        print(f"Generating multiscale GRF data (Resolution: {config['resolution']}x{config['resolution']})...")
        marginal_data, data_dim = generate_multiscale_grf_data(
            N_samples=config['n_samples'],
            T=config['T'],
            N_constraints=config['n_constraints'],
            resolution=config['resolution'],
            L_domain=config['l_domain'],
            micro_corr_length=config['micro_corr_length'],
            H_max_factor=config['h_max_factor'],
            mean_val=config['mean_val'],
            std_val=config['std_val'],
            covariance_type=config['covariance_type'],
            device=config['device']
        )
    else:
        print("Generating spiral distributional data...")
        marginal_data, _ = generate_spiral_distributional_data(
            N_constraints=config['n_constraints'],
            T=config['T'],
            data_dim=config['data_dim'],
            N_samples_per_marginal=config['n_samples_per_marginal'],
            noise_std=config['noise_std']
        )
        data_dim = config['data_dim']
    
    # Move data to device
    marginal_data = {t: samples.to(config['device']) for t, samples in marginal_data.items()}
    
    print(f"Generated {len(marginal_data)} marginal distributions with {data_dim} dimensions.")
    return marginal_data, data_dim


def create_model(config, data_dim):
    """Create and initialize the bridge model."""
    
    print("\n--- Model Initialization ---")
    
    bridge = NeuralGaussianBridge(
        data_dim=data_dim,
        hidden_size=config['hidden_size'],
        T=config['T'],
        sigma_reverse=config['sigma_reverse']
    ).to(config['device'])
    
    n_params = sum(p.numel() for p in bridge.parameters())
    print(f"Initialized NeuralGaussianBridge with {n_params:,} parameters")
    
    return bridge


def train_model(bridge, marginal_data, config):
    """Train the bridge model."""
    
    print("\n--- Training ---")
    print(f"Training for {config['epochs']} epochs with lr={config['lr']}")
    
    loss_history = train_bridge(
        bridge=bridge,
        marginal_data=marginal_data,
        epochs=config['epochs'],
        lr=config['lr'],
        lambda_path=config['lambda_path'],
        verbose=True
    )
    
    final_loss = loss_history[-1]['total']
    print(f"\nTraining completed. Final loss: {final_loss:.6f}")
    
    return loss_history


def validate_model(bridge, marginal_data, config):
    """Validate the trained model."""
    
    print("\n--- Validation ---")
    
    if config['data_type'] == 'grf':
        # Generate samples for validation
        print("Generating backward samples for validation...")
        original_data, generated_samples = generate_comparative_backward_samples(
            bridge=bridge,
            marginal_data=marginal_data,
            n_samples=min(256, list(marginal_data.values())[0].shape[0]),
            n_steps=config['n_sde_steps'],
            device=config['device']
        )
        
        # Calculate quantitative metrics (includes W2, MSE_ACF, RelF(Cov), CoV of variance)
        metrics = calculate_validation_metrics(marginal_data, generated_samples)
        
        # Print validation summary
        if metrics['w2_distances']:
            avg_w2 = np.nanmean(metrics['w2_distances'])
            avg_mse_acf = np.nanmean(metrics['mse_acf'])
            avg_rel_f = np.nanmean(metrics['rel_fro_cov'])
            
            print(f"Average Wasserstein-2 distance: {avg_w2:.6f}")
            print(f"Average MSE ACF: {avg_mse_acf:.6e}")
            print(f"Average Relative Frobenius (Cov): {avg_rel_f:.6f}")
            # Note: CoV(variance) removed to avoid ambiguous comparisons.
            
            # Simple validation criteria
            if avg_w2 < 0.5 and (np.isnan(avg_mse_acf) or avg_mse_acf < 1e-3):
                print("✓ Validation PASSED: Model successfully captures distributional dynamics")
            else:
                print("⚠ Validation WARNING: Metrics exceed recommended thresholds")
        else:
            print("⚠ Validation failed to compute metrics")
    
    else:
        # Spiral data validation
        print("Performing asymmetric consistency validation...")
        validation_results = validate_asymmetric_consistency(
            bridge=bridge,
            T=config['T'],
            n_particles=512,
            n_steps=100,
            n_validation_times=5,
            device=config['device']
        )
        
        if validation_results and validation_results['mean_errors']:
            mean_errors = np.nanmean(validation_results['mean_errors'])
            cov_errors = np.nanmean(validation_results['cov_errors'])
            w2_distances = np.nanmean(validation_results['wasserstein_distances'])

            print(f"Mean error: {mean_errors:.6f}")
            print(f"Covariance error: {cov_errors:.6f}")
            print(f"Wasserstein-2 distance: {w2_distances:.6f}")
            
            if mean_errors < 0.05 and cov_errors < 0.1:
                print("✓ Validation PASSED: Asymmetric consistency maintained")
            else:
                print("⚠ Validation WARNING: Consistency metrics exceed thresholds")
        else:
            print("⚠ Validation failed to run")


def visualize_results(bridge, marginal_data, config):
    """Generate visualizations of the results."""
    
    print("\n--- Visualization ---")
    print(f"Generating visualizations in '{config['output_dir']}'...")
    
    visualize_bridge_results(
        bridge=bridge,
        marginal_data=marginal_data,
        T=config['T'],
        output_dir=config['output_dir'],
        is_grf=(config['data_type'] == 'grf'),
        n_viz_particles=config['n_viz_particles'],
        n_sde_steps=config['n_sde_steps']
    )
    
    print("✓ Visualizations completed")


def main():
    """Main training script."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Asymmetric Multi-Marginal Bridge with Affine Flow")
    parser.add_argument('--grf', action='store_true', help='Use GRF data (default: spiral)')
    parser.add_argument('--spiral', action='store_true', help='Use spiral data')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    
    args = parser.parse_args()
    
    # Determine data type
    if args.grf:
        args.data_type = 'grf'
    else:
        args.data_type = 'spiral'  # default
    
    # Setup experiment
    config = setup_experiment(args)
    os.makedirs(config['output_dir'], exist_ok=True)
    
    print("=" * 80)
    print("ASYMMETRIC MULTI-MARGINAL BRIDGE - AFFINE FLOW")
    print(f"Data Type: {config['data_type'].upper()}")
    print(f"Device: {config['device']}")
    print(f"Output: {config['output_dir']}")
    print("=" * 80)
    
    try:
        # 1. Generate data
        marginal_data, data_dim = generate_data(config)
        
        # 2. Create model
        bridge = create_model(config, data_dim)
        
        # 3. Train model
        train_model(bridge, marginal_data, config)
        
        # 4. Validate model
        validate_model(bridge, marginal_data, config)
        
        # 5. Generate visualizations
        visualize_results(bridge, marginal_data, config)
        
        print("\n" + "=" * 80)
        print("✓ TRAINING COMPLETED SUCCESSFULLY")
        print(f"Results saved to: {config['output_dir']}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())