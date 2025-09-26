#!/usr/bin/env python3
"""
Parameter Experimentation for C-CVEP Bridge
===========================================

Systematic analysis of the C-CVEP bridge performance across different
parameter spaces, mirroring the experimental setup for GLOW. This script
utilizes the three-stage training process (RBF, MFM, C-CVEP) and is
designed for rigorous, scriptable experimentation.

METHODOLOGICAL IMPROVEMENTS:
- Follows the same rigorous parameter hierarchy as `glow_experiments.py`.
- Separates validation metrics from visualization parameters.
- Uses consistent trajectory counts for all validation metrics.
- Employs explicit naming to avoid ambiguity (e.g., "trajectories").

Key Features:
- **Kernel Analysis**: Compares GRF smoothness (exponential vs. gaussian).
- **Correlation Length Analysis**: Studies GRF spatial correlation effects.
- **Training Sample Analysis**: Assesses impact of training dataset size.
- **Automated Execution**: Manages parameter sweeps and collects results.
- **Default C-CVEP Config**: Uses U-Net and RBF metrics for GRF data by default.

Experiment Structure:
    --kernels: Test exponential vs gaussian GRF kernels.
    --comprehensive: Run full parameter analysis for each kernel.
    --correlation: Test correlation lengths for a single kernel.
    --samples: Test training sample sizes.
    --both: Run both correlation and sample size analyses.

Usage:
    python ccvep_experiments.py --comprehensive
    python ccvep_experiments.py --correlation --samples
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
            "use_unet": True,       # Default to U-Net for GRF data
            "use_rbf_metric": True, # Default to RBF metric

            # Training stage epochs
            "rbf_epochs": 200,
            "mfm_epochs": 300,
            "ccvep_epochs": 500,

            # Learning rates
            "mfm_lr": 1e-4,
            "ccvep_lr": 1e-4,

            # Model and data parameters
            "resolution": 16,
            "T": 1.0,
            "n_samples": 1024, # This is the ground truth trajectory count
            "n_validation_trajectories": 1024,
            "covariance_type": "gaussian",
            "sigma_reverse": 0.5,

            # Evaluation and visualization
            "n_viz_particles": 256,
            "n_sde_steps": 100,
            "save_metrics": bool(save_metrics),
            "enable_covariance_analysis": bool(enable_covariance_analysis),
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
        if all_params.get("grf"): args.append("--grf")
        if all_params.get("use_unet"): args.append("--use_unet")
        if all_params.get("use_rbf_metric"): args.append("--use_rbf_metric")
        if all_params.get("save_metrics"): args.append("--save_metrics")
        if all_params.get("enable_covariance_analysis"): args.append("--enable_covariance_analysis")

        # Add all other parameters
        for key, value in all_params.items():
            if key in ["grf", "use_unet", "use_rbf_metric", "save_metrics", "enable_covariance_analysis"] or value is None:
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
        """Systematic analysis of GRF correlation length effects."""
        print("\n" + "="*80)
        print("C-CVEP: CORRELATION LENGTH ANALYSIS")
        print("="*80)
        correlation_lengths = [0.05, 0.1, 0.2, 0.3, 0.5]
        results = []
        for corr_length in correlation_lengths:
            exp_name = f"corr_length_{corr_length:.3f}"
            overrides = {"micro_corr_length": corr_length}
            result = self.run_single_experiment(exp_name, overrides)
            results.append(result)
        self._save_summary("correlation_analysis", results)
        return results

    def sample_size_analysis(self):
        """Systematic analysis of training sample size effects."""
        print("\n" + "="*80)
        print("C-CVEP: TRAINING SAMPLE SIZE ANALYSIS")
        print("="*80)
        sample_sizes = [128, 256, 512, 1024]
        results = []
        for n_samples in sample_sizes:
            exp_name = f"sample_size_{n_samples}"
            # This corresponds to the total number of trajectories generated.
            overrides = {"n_samples": n_samples}
            result = self.run_single_experiment(exp_name, overrides)
            results.append(result)
        self._save_summary("sample_size_analysis", results)
        return results

    def kernel_analysis(self):
        """Systematic analysis of different GRF kernel types."""
        print("\n" + "="*80)
        print("C-CVEP: KERNEL SMOOTHNESS ANALYSIS")
        print("="*80)
        kernel_types = ["exponential", "gaussian"]
        all_results = {}
        original_base_dir = self.base_output_dir
        for kernel in kernel_types:
            print(f"\n--- KERNEL: {kernel.upper()} ---")
            self.base_output_dir = original_base_dir / f"kernel_{kernel}"
            self.fixed_params["covariance_type"] = kernel
            corr_res = self.correlation_length_analysis()
            traj_res = self.sample_size_analysis()
            all_results[kernel] = {"correlation": corr_res, "trajectory": traj_res}
        self.base_output_dir = original_base_dir
        self._save_summary("kernel_analysis", all_results)
        return all_results

    def _save_summary(self, analysis_name, results):
        summary_file = self.base_output_dir / f"{analysis_name}_summary.json"
        summary = {
            "analysis_type": analysis_name,
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved {analysis_name} summary to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="C-CVEP Bridge Parameter Experimentation")
    parser.add_argument("--correlation", action="store_true", help="Run correlation length analysis.")
    parser.add_argument("--samples", action="store_true", help="Run training sample size analysis.")
    parser.add_argument("--kernels", action="store_true", help="Run kernel analysis (implies correlation and samples).")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive analysis (same as --kernels).")
    parser.add_argument("--both", action="store_true", help="Run both correlation and sample size analyses.")
    parser.add_argument("--output", type=str, default="experiments_ccvep_refined", help="Base output directory.")
    parser.add_argument("--save_metrics", action="store_true", help="Enable saving of validation metrics.")
    parser.add_argument("--enable_covariance_analysis", action="store_true", help="Enable covariance visualizations.")
    args = parser.parse_args()

    if not any([args.correlation, args.samples, args.kernels, args.comprehensive, args.both]):
        parser.error("No analysis specified. Use --correlation, --samples, --kernels, --comprehensive, or --both.")

    runner = CCVEPExperimentRunner(
        base_output_dir=args.output,
        save_metrics=args.save_metrics,
        enable_covariance_analysis=args.enable_covariance_analysis
    )

    print("="*80)
    print("C-CVEP EXPERIMENTATION RUNNER")
    print(f"Output Directory: {runner.base_output_dir}")
    print("="*80)

    try:
        if args.comprehensive or args.kernels:
            runner.kernel_analysis()

        if args.both:
            runner.correlation_length_analysis()
            runner.sample_size_analysis()
        else:
            if args.correlation:
                runner.correlation_length_analysis()
            if args.samples:
                runner.sample_size_analysis()

        print("\n✓ All experiments completed successfully!")
        print(f"📁 Results saved to: {runner.base_output_dir}")
        return 0

    except Exception as e:
        print(f"\n✗ Experiment runner failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())