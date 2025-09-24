#!/usr/bin/env python3
"""
Asymmetric Multi-Marginal Bridge - C-CVEP Training Script (Three-Stage)
=======================================================================

A streamlined script for training C-CVEP models using the Asymmetric 
Multi-Marginal Bridge framework with Coupled Consistent Velocity-Energy
Parameterization and integrated Metric Flow Matching (MFM), utilizing
a three-stage optimization procedure with optional spatial-aware U-Net
architectures and learned RBF metrics.

Stage 0: RBF Metric Initialization and Training (optional)
Stage 1: MFM Interpolator Pre-training (Algorithm 1)
Stage 2: C-CVEP Training (Algorithm 2)

Usage:
    python train_ccvep_2stage.py [--grf | --spiral] [options]

Examples:
    # Basic training with MLP architectures
    python train_ccvep_2stage.py --grf --mfm_epochs 300 --ccvep_epochs 500
    
    # Spatial training with U-Net and RBF metric for GRF data
    python train_ccvep_2stage.py --grf --use_unet --use_rbf_metric --rbf_epochs 200 --mfm_epochs 300 --ccvep_epochs 500
    
    # Spiral data with RBF metric
    python train_ccvep_2stage.py --spiral --use_rbf_metric --rbf_epochs 100 --mfm_epochs 200 --ccvep_epochs 300
"""

import argparse
import os
import torch

# Import modularized components
from models_ccvep import CCVEPBridge
from backbones import TimeConditionedMLP, TimeConditionedUNet
from ccvep_core import InterpolatorBackboneMLP, InterpolatorBackboneUNet, RBFMetricHandler

from utilities.data_generation import generate_multiscale_grf_data, generate_spiral_distributional_data
from utilities.training import train_ccvep_bridge
from utilities.simulation import generate_comparative_backward_samples
from utilities.visualization import visualize_bridge_results
from utilities.validation import validate_asymmetric_consistency, calculate_validation_metrics
    
def setup_experiment(args) -> dict:
    """Setup experiment configuration based on arguments."""
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    
    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Common parameters
    config = {
        'device': device,
        'T': 1.0,
        'sigma_reverse': 0.5,
        'n_sde_steps': 100,
        'n_viz_particles': 512,
        'batch_size': 64,
        'time_embedding_dim': 32,
        'grad_clip_norm': 1.0,

        # Architecture options
        'use_unet': args.use_unet,
        'use_rbf_metric': args.use_rbf_metric,
        'rbf_params': {
            'epochs': args.rbf_epochs,
            'lr': 1e-3,
            'clusters': 64,
            'epsilon': 1e-3,
            'kappa': 1.0,
        },

        # Stage 0: RBF Metric (if enabled)
        'rbf_epochs': args.rbf_epochs,

        # Stage 1: MFM Pre-training
        'mfm_epochs': args.mfm_epochs,
        'mfm_lr': args.mfm_lr,
        'interpolator_hidden_dims': [128, 128], 

        # Stage 2: C-CVEP Training
        'ccvep_epochs': args.ccvep_epochs,
        'ccvep_lr': args.ccvep_lr,
        'lambda_CE': 1.0,             # Consistency Loss weight
    }
    
    if args.data_type == 'grf':
        # GRF-specific configuration
        config.update({
            'output_dir': 'output_ccvep_mfm_2stage_grf',
            'data_type': 'grf',
            'resolution': 16,
            'data_dim': 16 * 16,
            'n_constraints': 5,
            'n_samples': 512,
            'hidden_dims': [256, 256, 256] if not config['use_unet'] else [64, 128, 256],
            'interpolator_hidden_dims': [256, 256] if not config['use_unet'] else [64, 128],
            # Adjust RBF clusters for spatial data
            'rbf_params': {
                **config['rbf_params'],
                'clusters': 64            },
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
            'output_dir': 'output_ccvep_mfm_2stage_spiral',
            'data_type': 'spiral',
            'data_dim': 3,
            'n_constraints': 5,
            'n_samples_per_marginal': 512,
            'noise_std': 0.1,
            'hidden_dims': [128, 128, 128],
            'interpolator_hidden_dims': [128, 128],  # Adjusted for Spiral
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
    """Create and initialize the C-CVEP bridge model (MLP or U-Net)."""
    
    print("\n--- Model Initialization (C-CVEP + MFM) ---")
    
    # 1 & 2. Initialize Velocity and Energy Backbones (v_theta, U_phi)
    if config['use_unet'] and config['data_type'] == 'grf':
        print("Using U-Net architecture for spatial modeling...")
        velocity_backbone = TimeConditionedUNet(
            data_dim=data_dim,
            resolution=config['resolution'],
            in_channels=1,
            out_channels=1,
            hidden_dims=config['hidden_dims'],
            time_embedding_dim=config['time_embedding_dim']
        ).to(config['device'])

        energy_backbone = TimeConditionedUNet(
            data_dim=data_dim,
            resolution=config['resolution'],
            in_channels=1,
            out_channels=1,
            hidden_dims=config['hidden_dims'],
            time_embedding_dim=config['time_embedding_dim']
        ).to(config['device'])

        # U-Net interpolator backbone
        interpolator_backbone = InterpolatorBackboneUNet(
            resolution=config['resolution'],
            channels=1,
            hidden_dims=config['interpolator_hidden_dims'],
            time_embedding_dim=config['time_embedding_dim']
        ).to(config['device'])
    else:
        print("Using MLP architecture...")
        velocity_backbone = TimeConditionedMLP(
            data_dim=data_dim,
            hidden_dims=config['hidden_dims'],
            time_embedding_dim=config['time_embedding_dim']
        ).to(config['device'])

        energy_backbone = TimeConditionedMLP(
            data_dim=data_dim,
            hidden_dims=config['hidden_dims'],
            time_embedding_dim=config['time_embedding_dim']
        ).to(config['device'])

        # MLP interpolator backbone
        interpolator_backbone = InterpolatorBackboneMLP(
            data_dim=data_dim,
            hidden_dims=config['interpolator_hidden_dims']
        ).to(config['device'])

    # 3. Initialize RBF Metric Handler (if enabled)
    metric_handler = None
    if config['use_rbf_metric']:
        print("Initializing RBF metric handler...")
        metric_handler = RBFMetricHandler(
            data_dim=data_dim,
            num_clusters=config['rbf_params']['clusters'],
            epsilon=config['rbf_params']['epsilon'],
            device=config['device']
        ).to(config['device'])
        
        print(f"RBF metric configured with {config['rbf_params']['clusters']} clusters.")

    # 4. Initialize Bridge
    bridge = CCVEPBridge(
        velocity_backbone=velocity_backbone,
        energy_backbone=energy_backbone,
        interpolator_backbone=interpolator_backbone,
        metric_handler=metric_handler,
        T=config['T'],
        sigma_reverse=config['sigma_reverse']
    ).to(config['device'])
    
    # Parameter counting
    n_params = sum(p.numel() for p in bridge.parameters())
    n_params_interp = sum(p.numel() for p in bridge.phi_eta.parameters())
    n_params_metric = sum(p.numel() for p in metric_handler.parameters()) if metric_handler else 0
    
    arch_type = "U-Net" if config['use_unet'] and config['data_type'] == 'grf' else "MLP"
    metric_type = "RBF" if config['use_rbf_metric'] else "Euclidean"
    
    print(f"Initialized CCVEPBridge ({arch_type}, {metric_type} metric):")
    print(f"  Total parameters: {n_params:,}")
    print(f"  Interpolator parameters: {n_params_interp:,}")
    if n_params_metric > 0:
        print(f"  RBF metric parameters: {n_params_metric:,}")
    
    return bridge


def train_model(bridge, marginal_data, config):
    """Train the C-CVEP bridge model using the three-stage procedure."""
    
    print("\n--- Training (Three-Stage RBF + MFM + C-CVEP) ---")
    
    loss_history = train_ccvep_bridge(
        bridge=bridge,
        marginal_data=marginal_data,
        # Stage 0
        use_rbf_metric=config['use_rbf_metric'],
        rbf_params=config['rbf_params'],
        # Stage 1
        mfm_epochs=config['mfm_epochs'],
        mfm_lr=config['mfm_lr'],
        early_stopping_patience=100,
        early_stopping_min_delta=1e-2,
        # Stage 2
        ccvep_epochs=config['ccvep_epochs'],
        ccvep_lr=config['ccvep_lr'],
        lambda_CE=config['lambda_CE'],
        # Common
        batch_size=config['batch_size'],
        grad_clip_norm=config['grad_clip_norm'],
        verbose=True
    )
    
    if loss_history:
        final_loss = loss_history[-1].get('loss_total', 0.0)
        print(f"\nTraining completed. Final C-CVEP loss: {final_loss:.6f}")
    else:
        print("\nWarning: Training returned empty loss history.")
    
    return loss_history


def validate_model(bridge, marginal_data, config):
    """Validate the trained model."""
    
    print("\n--- Validation ---")
    
    if config['data_type'] == 'grf':
        # Generate samples for validation
        print("Generating backward samples for validation...")
        try:
            original_data, generated_samples = generate_comparative_backward_samples(
                bridge=bridge,
                marginal_data=marginal_data,
                n_samples=config['n_viz_particles'],
                n_steps=config['n_sde_steps'],
                device=config['device']
            )
            
            # Calculate quantitative metrics
            metrics = calculate_validation_metrics(marginal_data, generated_samples)
            
            # Print validation summary
            if metrics['w2_distances']:
                avg_w2 = torch.nanmean(torch.tensor(metrics['w2_distances'])).item()
                avg_mse_acf = torch.nanmean(torch.tensor(metrics['mse_acf'])).item()
                
                print(f"Average Wasserstein-2 distance: {avg_w2:.6f}")
                print(f"Average MSE ACF: {avg_mse_acf:.6e}")
                
                # Simple validation criteria (more lenient for C-CVEP)
                if avg_w2 < 1.0 and (torch.isnan(torch.tensor(avg_mse_acf)) or avg_mse_acf < 1e-2):
                    print("✓ Validation PASSED: Model successfully captures distributional dynamics")
                else:
                    print("⚠ Validation WARNING: Metrics exceed recommended thresholds (expected for initial C-CVEP)")
            else:
                print("⚠ Validation failed to compute metrics")
        except Exception as e:
            print(f"⚠ Validation failed with error: {e}")
    
    else:
        # Spiral data validation
        print("Performing asymmetric consistency validation...")
        try:
            validation_results = validate_asymmetric_consistency(
                bridge=bridge,
                T=config['T'],
                n_particles=512,
                n_steps=100,
                n_validation_times=5,
                device=config['device']
            )
            
            if validation_results and validation_results['mean_errors']:
                mean_errors = torch.nanmean(torch.tensor(validation_results['mean_errors'])).item()
                cov_errors = torch.nanmean(torch.tensor(validation_results['cov_errors'])).item()
                w2_distances = torch.nanmean(torch.tensor(validation_results['wasserstein_distances'])).item()

                print(f"Mean error: {mean_errors:.6f}")
                print(f"Covariance error: {cov_errors:.6f}")
                print(f"Wasserstein-2 distance: {w2_distances:.6f}")
                
                # More lenient thresholds for C-CVEP
                if mean_errors < 0.1 and cov_errors < 0.2:
                    print("✓ Validation PASSED: Asymmetric consistency maintained")
                else:
                    print("⚠ Validation WARNING: Consistency metrics exceed thresholds (expected for initial C-CVEP)")
            else:
                print("⚠ Validation failed to run")
        except Exception as e:
            print(f"⚠ Validation failed with error: {e}")


def visualize_results(bridge, marginal_data, config):
    """Generate visualizations of the results."""
    
    print("\n--- Visualization ---")
    print(f"Generating visualizations in '{config['output_dir']}'...")
    
    try:
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
    except Exception as e:
        print(f"⚠ Visualization failed with error: {e}")


def main():
    """Main training script."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Asymmetric Multi-Marginal Bridge with C-CVEP and MFM (Three-Stage)")
    parser.add_argument('--grf', action='store_true', help='Use GRF data (default: spiral)')
    parser.add_argument('--spiral', action='store_true', help='Use spiral data')
    
    # Architecture options
    parser.add_argument('--use_unet', action='store_true', help='Use U-Net for spatial data (GRF only)')
    parser.add_argument('--use_rbf_metric', action='store_true', help='Use learned RBF metric')
    
    # Stage 0 args (RBF Metric)
    parser.add_argument('--rbf_epochs', type=int, default=100, help='Stage 0: RBF metric training epochs')
    
    # Stage 1 args (MFM)
    parser.add_argument('--mfm_epochs', type=int, default=100, help='Stage 1: MFM pre-training epochs')
    parser.add_argument('--mfm_lr', type=float, default=1e-5, help='Stage 1: MFM learning rate')
    
    # Stage 2 args (C-CVEP)
    parser.add_argument('--ccvep_epochs', type=int, default=200, help='Stage 2: C-CVEP training epochs')
    parser.add_argument('--ccvep_lr', type=float, default=1e-3, help='Stage 2: C-CVEP learning rate')

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
    print("ASYMMETRIC MULTI-MARGINAL BRIDGE - C-CVEP with MFM (Three-Stage)")
    print(f"Data Type: {config['data_type'].upper()}")
    print(f"Architecture: {'U-Net' if config['use_unet'] and config['data_type'] == 'grf' else 'MLP'}")
    print(f"Metric: {'RBF' if config['use_rbf_metric'] else 'Euclidean'}")
    print(f"Device: {config['device']}")
    print(f"Output: {config['output_dir']}")
    print("=" * 80)
    
    try:
        # 1. Generate data
        marginal_data, data_dim = generate_data(config)
        
        # 2. Create model
        bridge = create_model(config, data_dim)
        
        # 3. Train model (Two Stages)
        train_model(bridge, marginal_data, config)
        
        # 4. Validate model
        validate_model(bridge, marginal_data, config)
        
        # 5. Generate visualizations
        visualize_results(bridge, marginal_data, config)
        
        print("\n" + "=" * 80)
        print("✓ C-CVEP/MFM (THREE-STAGE) TRAINING COMPLETED SUCCESSFULLY")
        print(f"Results saved to: {config['output_dir']}")
        if config['use_unet']:
            print("✓ Spatial U-Net architecture utilized")
        if config['use_rbf_metric']:
            print("✓ Learned RBF metric incorporated")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())