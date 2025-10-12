#!/usr/bin/env python3
"""
Parameter Experimentation for GLOW Flow
==============================================

Systematic analysis of GLOW bridge performance across different parameter spaces.
Uses LCL training with optimized parameters for systematic experimentation.

METHODOLOGICAL IMPROVEMENTS (following the recommended actions):
- Clear parameter hierarchy: ground_truth → training → validation → visualization
- Separation of concerns: validation metrics computed independently of visualization  
- Consistent trajectory counts: all validation metrics use same sample size
- Explicit naming: removes ambiguity between "samples" and "trajectories"

Key Features:
- **Kernel Analysis**: Different GRF smoothness (exponential vs gaussian kernels)
- **Correlation Length Analysis**: GRF spatial correlation (0.05 to 0.5)  
- **Training Trajectory Analysis**: LCL training subset sizes (128 to 1024)
- **Nested Experiments**: Kernel -> (Correlation + Trajectory) structure
- Fixed LCL parameters: λ_lcl=1000, epochs=10, hidden_size=64
- Methodologically rigorous parameter separation:
  * n_ground_truth_trajectories: Total simulated trajectories
  * n_training_trajectories: Subset used for training
  * n_validation_trajectories: Consistent validation metrics
  * n_viz_particles: Visualization only (separate from validation)
- Automated result collection and comparison
- Simple, scriptable parameter sweeping

Experiment Structure:
    --kernels: Test exponential vs gaussian GRF kernels
    --comprehensive: For each kernel, run full parameter analysis
    --correlation: Test correlation lengths (single kernel)
    --samples: Test trajectory counts (single kernel) 
    --both: Run correlation + trajectory analysis (single kernel)

Usage:
    python glow_experiments.py --comprehensive  # Full kernel study
    python glow_experiments.py --kernels        # Just kernel comparison
    python glow_experiments.py --both           # Standard parameter sweep
"""

import sys
import json
import time
from pathlib import Path
from train_glow_flow_decoupled import main as train_main
import numpy as np


class GLOWExperimentRunner:
    """
    Simple experiment runner following KISS principle.
    Manages parameter sweeps and result collection for GLOW flow analysis.
    
    Parameter Structure (following methodological rigor):
    - n_ground_truth_trajectories: Total trajectories from ground truth simulation
    - n_training_trajectories: Subset used for model training  
    - n_validation_trajectories: Consistent number for validation metrics
    - n_viz_particles: For visualization only (separate from validation)
    
    This ensures validation metrics are computed consistently and independently
    from visualization parameters, preventing methodological confusion.
    """
    
    def __init__(self, base_output_dir="experiments_glow_refined", save_metrics: bool = True, enable_covariance_analysis: bool = True):
        """Initialize with base output directory for all experiments."""
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Fixed experimental parameters for decoupled training
        self.fixed_params = {
            "grf": True,
            "density_epochs": 500,
            "dynamics_epochs": 500,
            "hidden_size": 64,
            "density_lr": 1e-3,
            "dynamics_lr": 1e-3,
            "resolution": 16,  # Standard GRF resolution
            "lambda_path": 0.0,  # Moderate path regularization
            "T": 1.0,  # Time horizon
            "n_ground_truth_trajectories": 1024,  # Total trajectories from ground truth simulation
            "n_blocks_flow": 2,  # Number of GLOW flow blocks
            "weight_decay": 1e-4,  # Weight decay for optimizer
            "grad_clip_norm": 1.0,  # Gradient clipping
            "n_training_trajectories": 512,  # Subset used for training (formerly n_trajectories)
            "n_validation_trajectories": 1024,  # Dedicated trajectories for validation metrics
            "training_noise_std": 0.01,  # Enable variational dequantization (was 0.0)
            "sigma_reverse": 0.5,  # Reverse SDE noise level
            "covariance_type": "gaussian",  # Default GRF kernel type
            # Evaluation parameters (methodologically rigorous separation)
            "n_viz_particles": 256,  # For visualization only (not used in validation)
            "n_sde_steps": 100,      # SDE integration steps
            # Simple toggles (booleans only)
            "save_metrics": bool(save_metrics),
            "enable_covariance_analysis": bool(enable_covariance_analysis),
            "validation_batch_size": 512,  # Batch size for validation (controls memory usage)
        }
    
    def run_single_experiment(self, experiment_name: str, param_overrides: dict) -> dict:
        """
        Run a single experiment with parameter overrides.
        
        Args:
            experiment_name: Unique name for this experiment
            param_overrides: Dictionary of parameters to override
            
        Returns:
            Dictionary with experiment results and metadata
        """
        print(f"\n{'='*60}")
        print(f"Running: {experiment_name}")
        print(f"{'='*60}")
        
        # Create experiment directory
        exp_dir = self.base_output_dir / experiment_name
        exp_dir.mkdir(exist_ok=True)
        
        # Build command line arguments
        args = []
        
        # Add fixed parameters
        if self.fixed_params.get("grf"):
            args.append("--grf")
            
        # Add all parameters (with backward compatibility mapping)
        all_params = {**self.fixed_params, **param_overrides}
        for key, value in all_params.items():
            if key in ["grf", "save_metrics", "enable_covariance_analysis"]:  # Skip boolean flags (handled separately)
                continue
            # Map new parameter names to train_glow_flow.py expected names
            if key == "n_ground_truth_trajectories":
                args.extend(["--n_samples", str(value)])  # Maps to N_samples in data generation
            elif key == "n_training_trajectories":
                args.extend(["--n_trajectories", str(value)])  # Maps to training subset size
            elif key == "n_validation_trajectories":
                args.extend(["--n_validation_trajectories", str(value)])  # New parameter for validation
            else:
                args.extend([f"--{key}", str(value)])
        
        # Add output directory
        args.extend(["--output_dir", str(exp_dir)])
        # Provide experiment name to trainer so model files can be named consistently
        args.extend(["--experiment_name", experiment_name])
        # Forward simple toggles to training script
        if all_params.get("save_metrics"):
            args.append("--save_metrics")
        if all_params.get("enable_covariance_analysis"):
            args.append("--enable_covariance_analysis")
        
        # Print experiment configuration
        print("Parameters:")
        for key, value in sorted(all_params.items()):
            print(f"  {key}: {value}")
        
        # Run experiment
        original_argv = sys.argv.copy()
        sys.argv = ["train_glow_flow_decoupled.py"] + args
        
        start_time = time.time()
        try:
            result = train_main()
            elapsed_time = time.time() - start_time
            
            status = "success" if result == 0 else "failed"
            print(f"✓ Experiment completed: {status} ({elapsed_time:.1f}s)")
            
            # Attempt to load per-experiment validation metrics produced by the trainer
            validation_metrics = None
            metrics_file = exp_dir / "validation_metrics.json"
            if metrics_file.exists():
                try:
                    with open(metrics_file, "r") as mf:
                        validation_metrics = json.load(mf)
                except Exception as e:
                    print(f"Warning: failed to load validation metrics from {metrics_file}: {e}")

            # Attempt to detect saved model file in experiment directory
            model_file = None
            candidate_model_1 = exp_dir / f"model_{experiment_name}.pt"
            candidate_model_2 = exp_dir / "model.pt"
            if candidate_model_1.exists():
                model_file = str(candidate_model_1)
            elif candidate_model_2.exists():
                model_file = str(candidate_model_2)

            # Save experiment metadata
            metadata = {
                "experiment_name": experiment_name,
                "parameters": all_params,
                "status": status,
                "elapsed_time": elapsed_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "validation_metrics_file": str(metrics_file) if metrics_file.exists() else None,
                "model_file": model_file,
            }

            # Embed compact validation summary when metrics are present
            if validation_metrics is not None:
                metadata["validation_metrics"] = validation_metrics
                # Per-timepoint metrics are expected as lists in validation_metrics
                w2_list = validation_metrics.get("w2_distances", [])
                relf_list = validation_metrics.get("rel_fro_cov", [])
                # Compute simple averages safely
                try:
                    metadata["validation_summary"] = {
                        "avg_w2": float(np.nanmean(w2_list)) if len(w2_list) > 0 else None,
                        "avg_rel_fro_cov": float(np.nanmean(relf_list)) if len(relf_list) > 0 else None,
                        "per_time_w2": [float(x) if x is not None else None for x in w2_list],
                        "per_time_rel_fro_cov": [float(x) if x is not None else None for x in relf_list],
                    }
                except Exception as e:
                    print(f"Warning: failed to compute validation summary: {e}")

            with open(exp_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            return metadata
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"✗ Experiment failed: {e} ({elapsed_time:.1f}s)")
            
            metadata = {
                "experiment_name": experiment_name,
                "parameters": all_params,
                "status": "error",
                "error": str(e),
                "elapsed_time": elapsed_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(exp_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
                
            return metadata
        
        finally:
            sys.argv = original_argv
    
    def correlation_length_analysis(self):
        """
        Systematic analysis of GRF correlation length effects.
        Tests how micro_corr_length affects model performance and training dynamics.
        """
        print("\n" + "="*80)
        print("CORRELATION LENGTH ANALYSIS")
        print("="*80)
        print("Testing GRF correlation length effects on GLOW bridge performance")
        print("Fixed: Decoupled training, epochs=1000+1000, n_ground_truth_trajectories=1024")
        
        # Correlation lengths to test (logarithmic spacing for good coverage)
        correlation_lengths = [0.05, 0.1, 0.2, 0.3, 0.5]
        
        results = []
        # We'll collect per-timepoint metric arrays for aggregation
        per_time_w2 = []
        per_time_relf = []
        
        for corr_length in correlation_lengths:
            experiment_name = f"corr_length_{corr_length:.3f}"
            param_overrides = {
                "micro_corr_length": corr_length,
                "n_ground_truth_trajectories": 1024,  # Fixed ground truth size
            }
            
            result = self.run_single_experiment(experiment_name, param_overrides)
            results.append(result)
            # If metrics present, store for aggregation
            vm = result.get("validation_metrics")
            if vm:
                w2 = vm.get("w2_distances", [])
                relf = vm.get("rel_fro_cov", [])
                per_time_w2.append(w2)
                per_time_relf.append(relf)
        
        # Compute simple per-timepoint averages if available
        aggregated = None
        try:
            if per_time_w2:
                # Convert to numpy arrays (pad to same length if necessary)
                max_len = max(len(x) for x in per_time_w2)
                w2_stack = np.array([np.pad(np.array(x, dtype=float), (0, max_len - len(x)), constant_values=np.nan) for x in per_time_w2])
                relf_stack = np.array([np.pad(np.array(x, dtype=float), (0, max_len - len(x)), constant_values=np.nan) for x in per_time_relf])
                aggregated = {
                    "per_time_avg_w2": np.nanmean(w2_stack, axis=0).tolist(),
                    "per_time_avg_rel_fro_cov": np.nanmean(relf_stack, axis=0).tolist(),
                }
        except Exception as e:
            print(f"Warning: failed to aggregate per-timepoint metrics: {e}")

        # Save summary
        summary = {
            "analysis_type": "correlation_length",
            "parameter_range": correlation_lengths,
            "fixed_parameters": self.fixed_params,
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "aggregated_metrics": aggregated,
        }
        
        summary_file = self.base_output_dir / "correlation_analysis_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("CORRELATION LENGTH ANALYSIS SUMMARY")
        print("="*80)
        
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] != "success"]
        
        print(f"Total experiments: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            avg_time = sum(r["elapsed_time"] for r in successful) / len(successful)
            print(f"Average runtime: {avg_time:.1f}s")
        
        print(f"\nDetailed results saved to: {summary_file}")
        print("Individual experiment outputs in subdirectories")
        
        return results
    
    def sample_size_analysis(self):
        """
        Systematic analysis of training trajectory count effects.
        Tests how n_trajectories affects LCL training convergence and final performance.
        """
        print("\n" + "="*80)
        print("TRAINING TRAJECTORY COUNT ANALYSIS") 
        print("="*80)
        print("Testing training trajectory count effects on GLOW bridge convergence")
        print("Fixed: Decoupled training, epochs=1000+1000, corr_length=0.1, n_ground_truth_trajectories=1024")
        
        # Training trajectory counts to test (powers of 2 for systematic scaling)
        training_trajectory_counts = [128, 256, 512, 1024]
        
        results = []
        per_time_w2 = []
        per_time_relf = []
        
        for n_train_traj in training_trajectory_counts:
            experiment_name = f"training_trajectories_{n_train_traj}"
            param_overrides = {
                "n_training_trajectories": n_train_traj,
                "micro_corr_length": 0.1,  # Fixed correlation length
                "n_ground_truth_trajectories": max(n_train_traj, 1024),  # Ensure enough ground truth data
            }
            
            result = self.run_single_experiment(experiment_name, param_overrides)
            results.append(result)
            vm = result.get("validation_metrics")
            if vm:
                per_time_w2.append(vm.get('w2_distances', []))
                per_time_relf.append(vm.get('rel_fro_cov', []))
        
        aggregated = None
        try:
            if per_time_w2:
                max_len = max(len(x) for x in per_time_w2)
                w2_stack = np.array([np.pad(np.array(x, dtype=float), (0, max_len - len(x)), constant_values=np.nan) for x in per_time_w2])
                relf_stack = np.array([np.pad(np.array(x, dtype=float), (0, max_len - len(x)), constant_values=np.nan) for x in per_time_relf])
                aggregated = {
                    "per_time_avg_w2": np.nanmean(w2_stack, axis=0).tolist(),
                    "per_time_avg_rel_fro_cov": np.nanmean(relf_stack, axis=0).tolist(),
                }
        except Exception as e:
            print(f"Warning: failed to aggregate per-timepoint metrics: {e}")

        # Save summary
        summary = {
            "analysis_type": "training_trajectory_count",
            "parameter_range": training_trajectory_counts,
            "fixed_parameters": self.fixed_params,
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "aggregated_metrics": aggregated,
         }
        
        summary_file = self.base_output_dir / "trajectory_count_analysis_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("TRAJECTORY COUNT ANALYSIS SUMMARY")
        print("="*80)
        
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] != "success"]
        
        print(f"Total experiments: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            avg_time = sum(r["elapsed_time"] for r in successful) / len(successful)
            print(f"Average runtime: {avg_time:.1f}s")
            
            # Show scaling behavior
            print("\nScaling analysis:")
            for result in successful:
                n_traj = result["parameters"]["n_training_trajectories"]
                runtime = result["elapsed_time"]
                print(f"  {n_traj:4d} training trajectories: {runtime:5.1f}s")
        
        print(f"\nDetailed results saved to: {summary_file}")
        print("Individual experiment outputs in subdirectories")
        
        return results
    
    def kernel_analysis(self):
        """
        Systematic analysis of different GRF kernel types.
        Tests how covariance_type (kernel smoothness) affects model performance.
        For each kernel, runs both correlation length and trajectory count analyses.
        """
        print("\n" + "="*80)
        print("KERNEL SMOOTHNESS ANALYSIS")
        print("="*80)
        print("Testing different GRF kernel types on GLOW bridge performance")
        print("Each kernel runs full correlation + trajectory analysis")
        
        # Available kernel types (KISS: just the two supported ones)
        kernel_types = ["exponential", "gaussian"]
        
        all_results = {}
        
        for kernel_type in kernel_types:
            print(f"\n{'='*60}")
            print(f"KERNEL: {kernel_type.upper()}")
            print(f"{'='*60}")
            
            # Create kernel-specific subdirectory
            kernel_dir = self.base_output_dir / f"kernel_{kernel_type}"
            kernel_dir.mkdir(exist_ok=True)
            
            # Temporarily override base output dir and covariance_type
            original_base_dir = self.base_output_dir
            original_cov_type = self.fixed_params["covariance_type"]
            
            self.base_output_dir = kernel_dir
            self.fixed_params["covariance_type"] = kernel_type
            
            try:
                # Run full parameter analysis for this kernel
                corr_results = self.correlation_length_analysis()
                sample_results = self.sample_size_analysis()
                
                all_results[kernel_type] = {
                    "correlation_results": corr_results,
                    "trajectory_count_results": sample_results,
                    "total_experiments": len(corr_results) + len(sample_results)
                }
                
            finally:
                # Restore original settings
                self.base_output_dir = original_base_dir
                self.fixed_params["covariance_type"] = original_cov_type
        
        # Save kernel analysis summary
        summary = {
            "analysis_type": "kernel_smoothness",
            "kernels_tested": kernel_types,
            "results_by_kernel": all_results,
            "total_experiments": sum(r["total_experiments"] for r in all_results.values()),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        summary_file = self.base_output_dir / "kernel_analysis_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("KERNEL SMOOTHNESS ANALYSIS SUMMARY")
        print("="*80)
        
        total_experiments = 0
        for kernel_type, results in all_results.items():
            n_exp = results["total_experiments"]
            total_experiments += n_exp
            print(f"  {kernel_type:12s}: {n_exp:3d} experiments")
        
        print(f"\nTotal experiments: {total_experiments}")
        print(f"Results directory: {self.base_output_dir}")
        print(f"Kernel summary: {summary_file}")
        
        return all_results
    
    def combined_analysis(self):
        """
        Run both correlation length and sample size analyses.
        Provides comprehensive parameter space exploration.
        """
        print("\n" + "="*80)
        print("COMBINED PARAMETER ANALYSIS")
        print("="*80)
        print("Running both correlation length and sample size analyses")
        
        # Run correlation analysis
        corr_results = self.correlation_length_analysis()
        
        # Run trajectory count analysis  
        sample_results = self.sample_size_analysis()
        
        # Combined summary
        combined_summary = {
            "analysis_type": "combined",
            "correlation_results": corr_results,
            "trajectory_count_results": sample_results,
            "total_experiments": len(corr_results) + len(sample_results),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        summary_file = self.base_output_dir / "combined_analysis_summary.json"
        with open(summary_file, "w") as f:
            json.dump(combined_summary, f, indent=2)
        
        print("\n" + "="*80)
        print("COMBINED ANALYSIS COMPLETED")
        print("="*80)
        print(f"Total experiments run: {len(corr_results) + len(sample_results)}")
        print(f"Results directory: {self.base_output_dir}")
        print(f"Combined summary: {summary_file}")
        
        return combined_summary
    
    def comprehensive_kernel_analysis(self):
        """
        Complete kernel analysis: for each kernel type, run full parameter exploration.
        This is the comprehensive experiment following the structure:
        Kernel -> (Correlation Analysis + Trajectory Analysis)
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE KERNEL ANALYSIS")
        print("="*80)
        print("Complete parameter exploration for each GRF kernel type")
        print("Structure: Kernel -> (Correlation + Trajectory) experiments")
        
        return self.kernel_analysis()

    # --- Helpers for aggregating metrics across experiments ---
    def aggregate_all_metrics(self, out_csv: str = None):
        """Aggregate validation metrics (per-experiment) into a single CSV for downstream plotting.

        The CSV will include columns: experiment_name, param_key(s), time_index, time_value, w2, rel_fro_cov, mse_acf
        """
        import csv

        rows = []
        for exp_dir in sorted(self.base_output_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            meta_file = exp_dir / "metadata.json"
            metrics_file = exp_dir / "validation_metrics.json"
            if not meta_file.exists() or not metrics_file.exists():
                continue
            try:
                with open(meta_file, 'r') as mf:
                    _ = json.load(mf)
                with open(metrics_file, 'r') as vf:
                    metrics = json.load(vf)
            except Exception as e:
                print(f"Warning: failed to read JSON in {exp_dir}: {e}")
                continue

            times = metrics.get('times', [])
            w2s = metrics.get('w2_distances', [])
            relfs = metrics.get('rel_fro_cov', [])
            mse_acf = metrics.get('mse_acf', [])

            for idx, t in enumerate(times):
                rows.append({
                    'experiment': exp_dir.name,
                    'time_index': idx,
                    'time': t,
                    'w2': w2s[idx] if idx < len(w2s) else None,
                    'rel_fro_cov': relfs[idx] if idx < len(relfs) else None,
                    'mse_acf': mse_acf[idx] if idx < len(mse_acf) else None,
                })

        if out_csv is None:
            out_csv = str(self.base_output_dir / 'aggregated_metrics.csv')

        # Write CSV
        if rows:
            fieldnames = ['experiment', 'time_index', 'time', 'w2', 'rel_fro_cov', 'mse_acf']
            with open(out_csv, 'w', newline='') as csvf:
                writer = csv.DictWriter(csvf, fieldnames=fieldnames)
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
            print(f"Aggregated metrics written to {out_csv}")
        else:
            print("No metrics found to aggregate.")

    def list_available_models(self):
        """List trained model files in experiment subdirectories with standardized naming.

        Returns a dict mapping experiment_name -> model_file_path (if exists) or None.
        """
        models = {}
        for exp_dir in sorted(self.base_output_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            # Standard model filename convention: model_<experiment_name>.pt
            # Also accept model.pt
            candidates = [exp_dir / f"model_{exp_dir.name}.pt", exp_dir / "model.pt"]
            found = None
            for c in candidates:
                if c.exists():
                    found = str(c)
                    break
            models[exp_dir.name] = found
        return models


def main():
    """Main experiment runner with command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Refined GLOW Flow Parameter Experimentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python glow_experiments.py --correlation
    python glow_experiments.py --samples
    python glow_experiments.py --both
    python glow_experiments.py --kernels
    python glow_experiments.py --comprehensive
    python glow_experiments.py --comprehensive --output experiments_kernel_study
        """
    )
    
    parser.add_argument(
        "--correlation", 
        action="store_true",
        help="Run correlation length analysis (0.05 to 0.5)"
    )
    
    parser.add_argument(
        "--samples",
        action="store_true", 
        help="Run training trajectory count analysis (128 to 1024)"
    )
    
    parser.add_argument(
        "--both",
        action="store_true",
        help="Run both correlation and trajectory count analyses"
    )
    
    parser.add_argument(
        "--kernels",
        action="store_true",
        help="Run kernel analysis (exponential vs gaussian kernels)"
    )
    
    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Run comprehensive kernel analysis (all kernels x all parameters)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="experiments_glow_refined",
        help="Base output directory for experiments"
    )
    parser.add_argument(
        "--save_metrics",
        action="store_true",
        help="Enable saving of per-experiment validation metrics (W2, MSE_ACF, Rel-F Frobenius)",
    )
    parser.add_argument(
        "--enable_covariance_analysis",
        action="store_true",
        help="Enable covariance visual analyses during experiments (may be heavy)",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (args.correlation or args.samples or args.both or args.kernels or args.comprehensive):
        print("Error: Must specify at least one analysis type")
        print("Use --correlation, --samples, --both, --kernels, or --comprehensive")
        parser.print_help()
        return 1
    
    # Initialize experiment runner
    runner = GLOWExperimentRunner(args.output, save_metrics=getattr(args, 'save_metrics', True), enable_covariance_analysis=getattr(args, 'enable_covariance_analysis', True))
    
    print("="*80)
    print("REFINED GLOW FLOW EXPERIMENTATION (DECOUPLED)")
    print("="*80)
    print("Systematic parameter analysis with decoupled architecture")
    print("Methodologically rigorous parameter separation:")
    print("  • n_ground_truth_trajectories: Total simulated dataset size")
    print("  • n_training_trajectories: Subset used for model training")
    print("  • n_validation_trajectories: Consistent validation metrics")
    print("  • n_viz_particles: Visualization only (separate from validation)")
    print(f"Output directory: {runner.base_output_dir}")
    print("Fixed parameters:")
    for key, value in sorted(runner.fixed_params.items()):
        print(f"  {key}: {value}")
    print("="*80)
    
    try:
        if args.comprehensive:
            runner.comprehensive_kernel_analysis()
        elif args.kernels:
            runner.kernel_analysis()
        elif args.both:
            runner.combined_analysis()
        else:
            if args.correlation:
                runner.correlation_length_analysis()
            if args.samples:
                runner.sample_size_analysis()
        
        print("\n✓ All experiments completed successfully!")
        print(f"📁 Results saved to: {runner.base_output_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠ Experiments interrupted by user")
        return 1
    except Exception as e:
        print(f"\n✗ Experiment runner failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())