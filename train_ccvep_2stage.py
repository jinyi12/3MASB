#!/usr/bin/env python3
"""
Asymmetric Multi-Marginal Bridge - C-CVEP Training Script (Three-Stage)
=======================================================================

A streamlined script for training C-CVEP models using the Asymmetric
Multi-Marginal Bridge framework. This script is adapted for systematic
experimentation, mirroring the structure of `train_glow_flow.py`.

Key Features:
- **Three-Stage Training**:
  - Stage 0: RBF Metric Initialization (optional)
  - Stage 1: MFM Interpolator Pre-training
  - Stage 2: C-CVEP Training
- **Configurable Architecture**: Supports MLP and U-Net backbones.
- **Scriptable Experiments**: Exposes a detailed CLI for parameter sweeps.
- **Artifact Generation**: Saves models, metadata, and validation metrics.

Usage:
    python train_ccvep_2stage.py --grf --use_unet --use_rbf_metric [options]

Examples:
    # Basic GRF training with U-Net and RBF metric
    python train_ccvep_2stage.py --grf --use_unet --use_rbf_metric \
        --rbf_epochs 200 --mfm_epochs 300 --ccvep_epochs 500

    # Run a specific experiment, saving artifacts
    python train_ccvep_2stage.py --grf --use_unet --use_rbf_metric \
        --experiment_name "rbf_test_01" --output_dir "experiments/ccvep" \
        --save_metrics
"""

import argparse
import os
import json
import time
from pathlib import Path
import torch
import numpy as np

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
    """Setup experiment configuration based on arguments (flat structure)."""

    torch.manual_seed(42)
    np.random.seed(42)

    config = {
        # Core settings
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_type": args.data_type,
        "output_dir": args.output_dir if args.output_dir else f"output_ccvep_{args.data_type}",
        "experiment_name": getattr(args, 'experiment_name', None),

        # Architecture
        "use_unet": getattr(args, 'use_unet', False),
        "use_rbf_metric": getattr(args, 'use_rbf_metric', False),
        "hidden_dims": getattr(args, 'hidden_dims', [256, 256, 256]),
        "interpolator_hidden_dims": getattr(args, 'interpolator_hidden_dims', [256, 256]),
        "time_embedding_dim": getattr(args, 'time_embedding_dim', 32),

        # Training Stages
        "rbf_epochs": getattr(args, 'rbf_epochs', 100), "rbf_lr": getattr(args, 'rbf_lr', 1e-3),
        "mfm_epochs": getattr(args, 'mfm_epochs', 100), "mfm_lr": getattr(args, 'mfm_lr', 1e-5),
        "ccvep_epochs": getattr(args, 'ccvep_epochs', 200), "ccvep_lr": getattr(args, 'ccvep_lr', 1e-3),
        "lambda_CE": getattr(args, 'lambda_CE', 1.0),

        # RBF Metric params
        "rbf_clusters": getattr(args, 'rbf_clusters', 64), "rbf_epsilon": getattr(args, 'rbf_epsilon', 1e-3), "rbf_kappa": getattr(args, 'rbf_kappa', 1.0),

        # Common training params
        "batch_size": getattr(args, 'batch_size', 64),
        "grad_clip_norm": getattr(args, 'grad_clip_norm', 1.0),

        # Data Generation
        "n_samples": getattr(args, 'n_samples', 1024),
        "n_constraints": getattr(args, 'n_constraints', 5),
        "resolution": getattr(args, 'resolution', 16),
        "micro_corr_length": getattr(args, 'micro_corr_length', 0.1),
        "covariance_type": getattr(args, 'covariance_type', 'gaussian'),
        "l_domain": getattr(args, 'l_domain', 1.0),
        "h_max_factor": getattr(args, 'h_max_factor', 0.5),
        "mean_val": getattr(args, 'mean_val', 10.0),
        "std_val": getattr(args, 'std_val', 2.0),
        "n_samples_per_marginal": getattr(args, 'n_samples_per_marginal', 512),
        "noise_std": getattr(args, 'noise_std', 0.1),

        # Bridge Dynamics
        "T": getattr(args, 'T', 1.0),
        "sigma_reverse": getattr(args, 'sigma_reverse', 0.5),

        # Evaluation
        "n_sde_steps": getattr(args, 'n_sde_steps', 100),
        "n_viz_particles": getattr(args, 'n_viz_particles', 256),
        "n_validation_trajectories": getattr(args, 'n_validation_trajectories', 512),
        "save_metrics": getattr(args, 'save_metrics', False),
        "enable_covariance_analysis": getattr(args, 'enable_covariance_analysis', False),
    }

    if config["data_type"] == "grf":
        config["data_dim"] = config["resolution"] ** 2
        if config['use_unet']:
            config['hidden_dims'] = getattr(args, 'hidden_dims', [64, 128, 256])
            config['interpolator_hidden_dims'] = getattr(args, 'interpolator_hidden_dims', [64, 128])
    else:
        config["data_dim"] = 3
        config['use_unet'] = False

    return config


def generate_data(config):
    """Generate marginal data based on configuration."""
    print("\n--- Data Generation ---")
    if config['data_type'] == 'grf':
        print(f"Generating multiscale GRF data (Resolution: {config['resolution']}x{config['resolution']})...")
        marginal_data, data_dim = generate_multiscale_grf_data(
            N_samples=config['n_samples'], T=config['T'], N_constraints=config['n_constraints'],
            resolution=config['resolution'], L_domain=config['l_domain'],
            micro_corr_length=config['micro_corr_length'], H_max_factor=config['h_max_factor'],
            mean_val=config['mean_val'], std_val=config['std_val'],
            covariance_type=config['covariance_type'], device=config['device']
        )
    else:
        print("Generating spiral distributional data...")
        marginal_data, data_dim = generate_spiral_distributional_data(
            N_constraints=config['n_constraints'], T=config['T'], data_dim=config['data_dim'],
            N_samples_per_marginal=config['n_samples_per_marginal'], noise_std=config['noise_std']
        )
    marginal_data = {t: samples.to(config['device']) for t, samples in marginal_data.items()}
    print(f"Generated {len(marginal_data)} marginal distributions with {data_dim} dimensions.")
    return marginal_data, data_dim


def create_model(config, data_dim):
    """Create and initialize the C-CVEP bridge model."""
    print("\n--- Model Initialization (C-CVEP + MFM) ---")
    if config['use_unet']:
        print("Using U-Net architecture for spatial modeling...")
        velocity_backbone = TimeConditionedUNet(
            data_dim=data_dim, resolution=config['resolution'], hidden_dims=config['hidden_dims'], time_embedding_dim=config['time_embedding_dim']
        ).to(config['device'])
        energy_backbone = TimeConditionedUNet(
            data_dim=data_dim, resolution=config['resolution'], hidden_dims=config['hidden_dims'], time_embedding_dim=config['time_embedding_dim']
        ).to(config['device'])
        interpolator_backbone = InterpolatorBackboneUNet(
            resolution=config['resolution'], channels=1, hidden_dims=config['interpolator_hidden_dims'], time_embedding_dim=config['time_embedding_dim']
        ).to(config['device'])
    else:
        print("Using MLP architecture...")
        velocity_backbone = TimeConditionedMLP(
            data_dim=data_dim, hidden_dims=config['hidden_dims'], time_embedding_dim=config['time_embedding_dim']
        ).to(config['device'])
        energy_backbone = TimeConditionedMLP(
            data_dim=data_dim, hidden_dims=config['hidden_dims'], time_embedding_dim=config['time_embedding_dim']
        ).to(config['device'])
        interpolator_backbone = InterpolatorBackboneMLP(
            data_dim=data_dim, hidden_dims=config['interpolator_hidden_dims']
        ).to(config['device'])

    metric_handler = None
    if config['use_rbf_metric']:
        print(f"Initializing RBF metric handler with {config['rbf_clusters']} clusters...")
        metric_handler = RBFMetricHandler(
            data_dim=data_dim, num_clusters=config['rbf_clusters'], epsilon=config['rbf_epsilon'], device=config['device']
        ).to(config['device'])

    bridge = CCVEPBridge(
        velocity_backbone=velocity_backbone, energy_backbone=energy_backbone,
        interpolator_backbone=interpolator_backbone, metric_handler=metric_handler,
        T=config['T'], sigma_reverse=config['sigma_reverse']
    ).to(config['device'])
    n_params = sum(p.numel() for p in bridge.parameters())
    print(f"Initialized CCVEPBridge ({'U-Net' if config['use_unet'] else 'MLP'}, {'RBF' if config['use_rbf_metric'] else 'Euclidean'}) with {n_params:,} parameters.")
    return bridge


def train_model(bridge, marginal_data, config):
    """Train the C-CVEP bridge model using the three-stage procedure."""
    print("\n--- Training (Three-Stage RBF + MFM + C-CVEP) ---")
    rbf_params = {
        'epochs': config['rbf_epochs'], 'lr': config['rbf_lr'], 'clusters': config['rbf_clusters'],
        'epsilon': config['rbf_epsilon'], 'kappa': config['rbf_kappa'],
    }
    loss_history = train_ccvep_bridge(
        bridge=bridge, marginal_data=marginal_data,
        use_rbf_metric=config['use_rbf_metric'], rbf_params=rbf_params,
        mfm_epochs=config['mfm_epochs'], mfm_lr=config['mfm_lr'],
        ccvep_epochs=config['ccvep_epochs'], ccvep_lr=config['ccvep_lr'],
        lambda_CE=config['lambda_CE'], batch_size=config['batch_size'],
        grad_clip_norm=config['grad_clip_norm'], verbose=True
    )
    if loss_history:
        print(f"\nTraining completed. Final C-CVEP loss: {loss_history[-1].get('loss_total', 0.0):.6f}")
    return loss_history


def validate_model(bridge, marginal_data, config):
    """Validate the trained model and save metrics."""
    print("\n--- Validation ---")
    metrics = {}
    try:
        if config['data_type'] == 'grf':
            print(f"Generating {config['n_validation_trajectories']} backward samples...")
            _, generated_samples = generate_comparative_backward_samples(
                bridge, marginal_data, config['n_validation_trajectories'], config['n_sde_steps'], config['device']
            )
            metrics = calculate_validation_metrics(marginal_data, generated_samples)
            if metrics.get('w2_distances'):
                print(f"Avg W2: {np.nanmean(metrics['w2_distances']):.6f}, Avg MSE ACF: {np.nanmean(metrics['mse_acf']):.6e}")
        else:
            print("Performing asymmetric consistency validation...")
            metrics = validate_asymmetric_consistency(
                bridge, config['T'], 512, 100, 5, config['device']
            )
            if metrics.get('mean_errors'):
                print(f"Mean consistency error: {np.nanmean(metrics['mean_errors']):.6f}")
    except Exception as e:
        print(f"⚠ Validation failed: {e}")

    if config.get("save_metrics"):
        outdir = Path(config["output_dir"])
        outdir.mkdir(parents=True, exist_ok=True)
        metrics_file = outdir / "validation_metrics.json"
        try:
            serializable_metrics = {k: v.tolist() if isinstance(v, (torch.Tensor, np.ndarray)) else v for k, v in metrics.items()}
            with open(metrics_file, "w") as f:
                json.dump(serializable_metrics, f, indent=2)
            print(f"Saved validation metrics to {metrics_file}")
        except Exception as e:
            print(f"Warning: failed to save metrics: {e}")


def visualize_results(bridge, marginal_data, config):
    """Generate and save visualizations."""
    print("\n--- Visualization ---")
    try:
        visualize_bridge_results(
            bridge, marginal_data, config['T'], config['output_dir'],
            is_grf=(config['data_type'] == 'grf'), n_viz_particles=config['n_viz_particles'],
            n_sde_steps=config['n_sde_steps'], enable_covariance_analysis=config.get('enable_covariance_analysis', False)
        )
        print("✓ Visualizations completed")
    except Exception as e:
        print(f"⚠ Visualization failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train Asymmetric Multi-Marginal Bridge with C-CVEP (Experiment-Ready)")
    # Core settings
    parser.add_argument('--grf', action='store_true', help='Use GRF data')
    parser.add_argument('--spiral', action='store_true', help='Use spiral data')
    parser.add_argument('--output_dir', type=str, help='Output directory for artifacts')
    parser.add_argument('--experiment_name', type=str, help='Identifier for the experiment')
    # Architecture
    parser.add_argument('--use_unet', action='store_true', help='Use U-Net for GRF data')
    parser.add_argument('--use_rbf_metric', action='store_true', help='Use learned RBF metric')
    # Data generation
    parser.add_argument('--n_samples', type=int, help='Number of training samples')
    parser.add_argument('--micro_corr_length', type=float, help='GRF correlation length')
    parser.add_argument('--covariance_type', type=str, help='GRF kernel type (exponential or gaussian)')
    parser.add_argument('--l_domain', type=float, help='GRF domain length')
    parser.add_argument('--h_max_factor', type=float, help='GRF H_max factor')
    parser.add_argument('--mean_val', type=float, help='GRF mean value')
    parser.add_argument('--std_val', type=float, help='GRF standard deviation')
    # Evaluation
    parser.add_argument('--n_validation_trajectories', type=int, help='Number of trajectories for validation')
    parser.add_argument('--save_metrics', action='store_true', help='Save validation metrics to a JSON file')
    parser.add_argument('--enable_covariance_analysis', action='store_true', help='Enable detailed covariance visualizations')
    parser.add_argument('--force_retrain', action='store_true', help='Force retraining even if a saved model exists')

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Warning: ignoring unknown arguments: {unknown}")

    args.data_type = 'grf' if args.grf or not args.spiral else 'spiral'
    config = setup_experiment(args)
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"C-CVEP EXPERIMENT: {config.get('experiment_name', 'default')}")
    print(f"Config: {config['data_type'].upper()}, {'U-Net' if config['use_unet'] else 'MLP'}, {'RBF' if config['use_rbf_metric'] else 'Euclidean'}")
    print(f"Output: {config['output_dir']}")
    print("=" * 80)

    try:
        marginal_data, data_dim = generate_data(config)
        bridge = create_model(config, data_dim)

        model_path = Path(config['output_dir']) / (f"model_{config['experiment_name']}.pt" if config['experiment_name'] else "model.pt")
        meta_path = Path(config['output_dir']) / 'model_metadata.json'

        if model_path.exists() and not args.force_retrain:
            print(f"Found existing model: {model_path}. Loading and skipping training.")
            bridge.load_state_dict(torch.load(model_path, map_location=config['device']))
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    print(f"Loaded metadata: {json.load(f)}")
        else:
            loss_history = train_model(bridge, marginal_data, config)
            torch.save(bridge.state_dict(), model_path)
            print(f"Saved trained model to {model_path}")
            model_meta = {
                'experiment_name': config['experiment_name'],
                'final_loss': loss_history[-1]['loss_total'] if loss_history else None,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            }
            with open(meta_path, 'w') as f:
                json.dump(model_meta, f, indent=2)
            print(f"Saved model metadata to {meta_path}")

        validate_model(bridge, marginal_data, config)
        visualize_results(bridge, marginal_data, config)

        print("\n" + "=" * 80)
        print("✓ C-CVEP/MFM TRAINING & EVALUATION COMPLETED SUCCESSFULLY")
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