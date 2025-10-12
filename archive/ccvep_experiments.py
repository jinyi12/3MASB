#!/usr/bin/env python3
"""
Parameter Experimentation for C-CVEP Bridge
===========================================

Systematic analysis of C-CVEP bridge performance across different parameter spaces.
Uses three-stage training (RBF + MFM + C-CVEP) with optimized parameters for systematic experimentation.

METHODOLOGICAL IMPROVEMENTS (following the recommended actions):
- Clear parameter hierarchy: ground_truth → training → validation → visualization
- Separation of concerns: validation metrics computed independently of visualization  
- Consistent trajectory counts: all validation metrics use same sample size
- Explicit naming: removes ambiguity between "samples" and "trajectories"

Key Features:
- **Kernel Analysis**: Different GRF smoothness (exponential vs gaussian kernels)
- **Correlation Length Analysis**: GRF spatial correlation (0.05 to 0.5)  
- **Training Trajectory Analysis**: Training dataset sizes (128 to 1024)
- **Nested Experiments**: Kernel -> (Correlation + Trajectory) structure
- Fixed C-CVEP parameters: RBF+U-Net architecture, three-stage training
- Methodologically rigorous parameter separation:
  * n_samples: Total simulated trajectories (ground truth)
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
    python ccvep_experiments.py --comprehensive  # Full kernel study
    python ccvep_experiments.py --kernels        # Just kernel comparison
    python ccvep_experiments.py --both           # Standard parameter sweep
"""

import sys
import json
import time
import argparse
from pathlib import Path
from train_ccvep_2stage import main as train_main
import numpy as np


class CCVEPExperimentRunner:
    """
    Manages parameter sweeps and result collection for C-CVEP analysis.
    Adopts the same rigorous parameter structure as the GLOW runner.
    """

    def __init__(self, base_output_dir="experiments_ccvep_refined", save_metrics: bool = True, enable_covariance_analysis: bool = True):
        """Initialize with a base output directory."""
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)

        # Fixed experimental parameters for C-CVEP training
        self.fixed_params = {
            "grf": True,
            "use_unet": False,      # Default to MLP (not U-Net)
            "use_rbf_metric": True, # Default to Euclidean metric (not RBF)

            # Training stage epochs (with early stopping, can set higher limits)
            "rbf_epochs": 30000,
            # "mfm_epochs": 1000,  # Increased with early stopping
            # "ccvep_epochs": 500,  # Increased with early stopping
            "mfm_epochs": 10000,  # Increased with early stopping
            "ccvep_epochs": 10000,  # Increased with early stopping

            # Learning rates
            "rbf_lr": 1e-3,         # RBF learning rate
            # "mfm_lr": 1e-4,
            # "ccvep_lr": 1e-4,
            "mfm_lr": 1e-3,
            "ccvep_lr": 1e-3,
            "lambda_CE": 1.0,       # C-CVEP loss weight
            
            # Architecture parameters
            "hidden_dims": [256, 256, 256], # MLP hidden dimensions
            "interpolator_hidden_dims": [256, 256],  # Interpolator dimensions
            "time_embedding_dim": 32,
            "rbf_clusters": 64,     # Number of RBF clusters
            "rbf_epsilon": 1e-3,    # RBF epsilon
            "rbf_alpha": 8.0,       # RBF alpha (metric steepness)
            "rbf_kappa": 1.0,   # RBF kappa
            
            # Training hyperparameters
            "batch_size": 64,
            "grad_clip_norm": 1.0,
            
            # Early stopping parameters (enabled by default for best model selection)
            "early_stopping": True,
            "early_stopping_patience": 1000,  # More aggressive for MFM
            "early_stopping_min_delta": 1e-4,
            "ccvep_early_stopping": True,
            "ccvep_early_stopping_patience": 1000,  # More aggressive for CCVEP
            "ccvep_early_stopping_min_delta": 1e-5,

            # Data and dynamics parameters
            "resolution": 16,       # Standard GRF resolution
            "n_constraints": 5,     # Number of time marginals
            "T": 1.0,               # Time horizon
            "n_samples": 1024,      # Total trajectories from ground truth simulation
            "n_validation_trajectories": 1024,  # Dedicated trajectories for validation metrics
            "covariance_type": "gaussian",  # Default GRF kernel type
            "l_domain": 1.0,
            "h_max_factor": 0.5,
            "mean_val": 10.0,
            "std_val": 2.0,
            "sigma_reverse": 0.5,   # Reverse SDE noise level

            # Evaluation and visualization
            "n_viz_particles": 256,
            "n_sde_steps": 100,
            "save_metrics": bool(save_metrics),
            "enable_covariance_analysis": bool(enable_covariance_analysis),
            "no_normalize": False,  # Default to using normalization
        }

    def run_single_experiment(self, experiment_name: str, param_overrides: dict) -> dict:
        """
        Run a single C-CVEP experiment with given parameter overrides.
        """
        print(f"\n{'='*60}")
        print(f"Running C-CVEP Experiment: {experiment_name}")
        print(f"{'='*60}")

        exp_dir = self.base_output_dir / experiment_name
        exp_dir.mkdir(exist_ok=True)

        args = []
        all_params = {**self.fixed_params, **param_overrides}

        # Handle boolean flags first
        if all_params.get("grf"): 
            args.append("--grf")
        if all_params.get("use_unet"): 
            args.append("--use_unet")
        if all_params.get("use_rbf_metric"): 
            args.append("--use_rbf_metric")
        if all_params.get("save_metrics"): 
            args.append("--save_metrics")
        if all_params.get("enable_covariance_analysis"): 
            args.append("--enable_covariance_analysis")
        if all_params.get("no_normalize"): 
            args.append("--no_normalize")
        # Early stopping is enabled by default, only add flag to disable it
        if not all_params.get("early_stopping", True) or not all_params.get("ccvep_early_stopping", True):
            args.append("--disable_early_stopping")

        # Add all other parameters (with parameter mapping)
        for key, value in all_params.items():
            # Skip boolean flags (handled separately) and early stopping flags (handled via --disable_early_stopping)
            if key in ["grf", "use_unet", "use_rbf_metric", "save_metrics", "enable_covariance_analysis", "no_normalize", "early_stopping", "ccvep_early_stopping"]:  
                continue
            if key == "hidden_dims" or key == "interpolator_hidden_dims":
                # Handle list parameters
                if isinstance(value, list):
                    args.extend([f"--{key}"] + [str(v) for v in value])
                continue
            args.extend([f"--{key}", str(value)])

        args.extend(["--output_dir", str(exp_dir)])
        args.extend(["--experiment_name", experiment_name])

        print("Parameters:")
        for key, value in sorted(all_params.items()):
            print(f"  {key}: {value}")

        original_argv = sys.argv.copy()
        sys.argv = ["train_ccvep_2stage.py"] + args

        start_time = time.time()
        try:
            result = train_main()
            elapsed_time = time.time() - start_time
            status = "success" if result == 0 else "failed"
            print(f"✓ Experiment '{experiment_name}' completed: {status} ({elapsed_time:.1f}s)")

            # Load metrics and create metadata
            metrics_file = exp_dir / "validation_metrics.json"
            validation_metrics = None
            if metrics_file.exists():
                try:
                    with open(metrics_file, "r") as f:
                        validation_metrics = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {metrics_file}")


            model_file = exp_dir / f"model_{experiment_name}.pt"
            metadata = {
                "experiment_name": experiment_name,
                "parameters": all_params,
                "status": status,
                "elapsed_time": elapsed_time,
                "model_file": str(model_file) if model_file.exists() else None,
                "validation_metrics": validation_metrics,
            }
            with open(exp_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            return metadata

        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"✗ Experiment '{experiment_name}' failed: {e} ({elapsed_time:.1f}s)")
            metadata = {"experiment_name": experiment_name, "parameters": all_params, "status": "error", "error": str(e)}
            with open(exp_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            return metadata
        finally:
            sys.argv = original_argv

    def correlation_length_analysis(self):
        """
        Systematic analysis of GRF correlation length effects.
        Tests how micro_corr_length affects C-CVEP model performance and training dynamics.
        """
        print("\n" + "="*80)
        print("C-CVEP: CORRELATION LENGTH ANALYSIS")
        print("="*80)
        print("Testing GRF correlation length effects on C-CVEP bridge performance")
        print("Fixed: Three-stage training (RBF+MFM+C-CVEP), MLP+RBF architecture, early stopping enabled")        # Correlation lengths to test (logarithmic spacing for good coverage)
        correlation_lengths = [0.05, 0.1, 0.2, 0.3, 0.5]
        
        results = []
        # We'll collect per-timepoint metric arrays for aggregation
        per_time_w2 = []
        per_time_relf = []
        
        for corr_length in correlation_lengths:
            experiment_name = f"corr_length_{corr_length:.3f}"
            param_overrides = {
                "micro_corr_length": corr_length,
                "n_samples": 1024,  # Fixed ground truth size
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
        print("C-CVEP: CORRELATION LENGTH ANALYSIS SUMMARY")
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
        Systematic analysis of training sample size effects.
        Tests how n_samples affects C-CVEP training convergence and final performance.
        """
        print("\n" + "="*80)
        print("C-CVEP: TRAINING SAMPLE SIZE ANALYSIS") 
        print("="*80)
        print("Testing training sample size effects on C-CVEP bridge convergence")
        print("Fixed: Three-stage training, corr_length=0.1, MLP+RBF architecture, early stopping enabled")        # Training sample sizes to test (powers of 2 for systematic scaling)
        sample_sizes = [128, 256, 512, 1024]
        
        results = []
        per_time_w2 = []
        per_time_relf = []
        
        for n_samples in sample_sizes:
            experiment_name = f"sample_size_{n_samples}"
            param_overrides = {
                "n_samples": n_samples,
                "micro_corr_length": 0.1,  # Fixed correlation length
                "n_validation_trajectories": min(n_samples, 1024),  # Use available samples for validation
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
            "analysis_type": "sample_size",
            "parameter_range": sample_sizes,
            "fixed_parameters": self.fixed_params,
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "aggregated_metrics": aggregated,
         }
        
        summary_file = self.base_output_dir / "sample_size_analysis_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("C-CVEP: SAMPLE SIZE ANALYSIS SUMMARY")
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
                n_samples = result["parameters"]["n_samples"]
                runtime = result["elapsed_time"]
                print(f"  {n_samples:4d} samples: {runtime:5.1f}s")
        
        print(f"\nDetailed results saved to: {summary_file}")
        print("Individual experiment outputs in subdirectories")
        
        return results
    
    def kernel_analysis(self):
        """
        Systematic analysis of different GRF kernel types.
        Tests how covariance_type (kernel smoothness) affects C-CVEP model performance.
        For each kernel, runs both correlation length and sample size analyses.
        """
        print("\n" + "="*80)
        print("C-CVEP: KERNEL SMOOTHNESS ANALYSIS")
        print("="*80)
        print("Testing different GRF kernel types on C-CVEP bridge performance")
        print("Each kernel runs full correlation + sample size analysis with early stopping")        # Available kernel types (KISS: just the two supported ones)
        kernel_types = ["exponential", "gaussian"]
        
        all_results = {}
        
        for kernel_type in kernel_types:
            print(f"\n{'='*60}")
            print(f"C-CVEP KERNEL: {kernel_type.upper()}")
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
                    "sample_size_results": sample_results,
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
        print("C-CVEP: KERNEL SMOOTHNESS ANALYSIS SUMMARY")
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
    
    def normalization_analysis(self):
        """
        Systematic analysis of normalization effects on C-CVEP performance.
        Tests both normalized and non-normalized training on the same data.
        """
        print("\n" + "="*80)
        print("C-CVEP: NORMALIZATION ANALYSIS")
        print("="*80)
        print("Testing normalized vs non-normalized training")
        print("Fixed: correlation_length=0.1, n_samples=1024, early stopping enabled")
        
        normalization_settings = [False, True]  # False = normalized (default), True = no normalization
        
        results = []
        per_time_w2 = []
        per_time_relf = []
        
        for no_normalize in normalization_settings:
            norm_status = "denormalized" if no_normalize else "normalized"
            experiment_name = f"norm_{norm_status}"
            param_overrides = {
                "no_normalize": no_normalize,
                "micro_corr_length": 0.1,  # Fixed correlation length
                "n_samples": 1024,  # Fixed sample size
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
            "analysis_type": "normalization",
            "parameter_range": normalization_settings,
            "fixed_parameters": self.fixed_params,
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "aggregated_metrics": aggregated,
        }
        
        summary_file = self.base_output_dir / "normalization_analysis_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("C-CVEP: NORMALIZATION ANALYSIS SUMMARY")
        print("="*80)
        
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] != "success"]
        
        print(f"Total experiments: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            avg_time = sum(r["elapsed_time"] for r in successful) / len(successful)
            print(f"Average runtime: {avg_time:.1f}s")
            
            # Show normalization comparison
            print("\nNormalization comparison:")
            for result in successful:
                no_normalize = result["parameters"]["no_normalize"]
                norm_status = "denormalized" if no_normalize else "normalized"
                runtime = result["elapsed_time"]
                print(f"  {norm_status:12s}: {runtime:5.1f}s")
        
        print(f"\nDetailed results saved to: {summary_file}")
        print("Individual experiment outputs in subdirectories")
        
        return results
    
    def combined_analysis(self):
        """
        Run both correlation length and sample size analyses.
        Provides comprehensive parameter space exploration.
        """
        print("\n" + "="*80)
        print("C-CVEP: COMBINED PARAMETER ANALYSIS")
        print("="*80)
        print("Running both correlation length and sample size analyses with early stopping")
        
        # Run correlation analysis
        corr_results = self.correlation_length_analysis()
        
        # Run sample size analysis  
        sample_results = self.sample_size_analysis()
        
        # Combined summary
        combined_summary = {
            "analysis_type": "combined",
            "correlation_results": corr_results,
            "sample_size_results": sample_results,
            "total_experiments": len(corr_results) + len(sample_results),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        summary_file = self.base_output_dir / "combined_analysis_summary.json"
        with open(summary_file, "w") as f:
            json.dump(combined_summary, f, indent=2)
        
        print("\n" + "="*80)
        print("C-CVEP: COMBINED ANALYSIS COMPLETED")
        print("="*80)
        print(f"Total experiments run: {len(corr_results) + len(sample_results)}")
        print(f"Results directory: {self.base_output_dir}")
        print(f"Combined summary: {summary_file}")
        
        return combined_summary
    
    def comprehensive_kernel_analysis(self):
        """
        Complete kernel analysis: for each kernel type, run full parameter exploration.
        This is the comprehensive experiment following the structure:
        Kernel -> (Correlation Analysis + Sample Size Analysis)
        """
        print("\n" + "="*80)
        print("C-CVEP: COMPREHENSIVE KERNEL ANALYSIS")
        print("="*80)
        print("Complete parameter exploration for each GRF kernel type")
        print("Structure: Kernel -> (Correlation + Sample Size) experiments with early stopping")
        
        return self.kernel_analysis()


def main():
    """Main experiment runner with command line interface."""
    parser = argparse.ArgumentParser(
        description="Refined C-CVEP Bridge Parameter Experimentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python ccvep_experiments.py --correlation
    python ccvep_experiments.py --samples
    python ccvep_experiments.py --both
    python ccvep_experiments.py --kernels
    python ccvep_experiments.py --comprehensive
    python ccvep_experiments.py --normalization
    python ccvep_experiments.py --comprehensive --output experiments_ccvep_study
    python ccvep_experiments.py --correlation --no_normalize
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
        help="Run training sample size analysis (128 to 1024)"
    )
    
    parser.add_argument(
        "--both",
        action="store_true",
        help="Run both correlation and sample size analyses"
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
        "--normalization",
        action="store_true",
        help="Run normalization analysis (normalized vs non-normalized training)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="experiments_ccvep_refined",
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
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Disable data normalization for all experiments - train and evaluate on original data scale",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (args.correlation or args.samples or args.both or args.kernels or args.comprehensive or args.normalization):
        print("Error: Must specify at least one analysis type")
        print("Use --correlation, --samples, --both, --kernels, --comprehensive, or --normalization")
        parser.print_help()
        return 1
    
    # Initialize experiment runner
    runner = CCVEPExperimentRunner(args.output, save_metrics=getattr(args, 'save_metrics', True), enable_covariance_analysis=getattr(args, 'enable_covariance_analysis', True))
    
    # Override normalization setting if specified
    if getattr(args, 'no_normalize', False):
        runner.fixed_params["no_normalize"] = True
        print(">>> NORMALIZATION DISABLED: All experiments will train on original data scale <<<")
    
    print("="*80)
    print("REFINED C-CVEP BRIDGE EXPERIMENTATION")
    print("="*80)
    print("Systematic parameter analysis with three-stage training (RBF+MFM+C-CVEP)")
    print("Methodologically rigorous parameter separation:")
    print("  • n_samples: Total simulated dataset size")
    print("  • n_validation_trajectories: Consistent validation metrics")
    print("  • n_viz_particles: Visualization only (separate from validation)")
    print("  • no_normalize: Toggle data normalization (default: normalized)")
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
        elif args.normalization:
            runner.normalization_analysis()
        else:
            if args.correlation:
                runner.correlation_length_analysis()
            if args.samples:
                runner.sample_size_analysis()
        
        print("\n✓ All C-CVEP experiments completed successfully!")
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