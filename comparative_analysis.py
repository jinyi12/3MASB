#!/usr/bin/env python3
"""
Comprehensive comparative analysis and visualization of CCVEP vs GLOW Flow experiments.
This script aggregates results from both frameworks and creates detailed comparative visualizations.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import argparse

# Try to import seaborn, use fallback if not available
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available, some advanced visualizations will be skipped")

class ComparativeAnalyzer:
    """Analyzes and visualizes comparative metrics between CCVEP and GLOW Flow experiments."""
    
    def __init__(self, ccvep_path: str, glow_path: str, output_dir: str = "comparative_analysis"):
        self.ccvep_path = Path(ccvep_path)
        self.glow_path = Path(glow_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load summary data
        self.ccvep_data = self._load_json(self.ccvep_path / "kernel_analysis_summary.json")
        self.glow_data = self._load_json(self.glow_path / "kernel_analysis_summary.json")
        
        # Aggregated metrics
        self.aggregated_metrics = {}
        
    def _load_json(self, path: Path) -> Dict:
        """Load JSON data from file."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Could not find {path}")
            return {}
    
    def aggregate_metrics(self) -> Dict[str, Any]:
        """Aggregate metrics from both frameworks for comparison."""
        aggregated = {
            'correlation_analysis': {},
            'sample_size_analysis': {},
            'framework_comparison': {}
        }
        
        # Process both kernels
        for kernel in ['exponential', 'gaussian']:
            if kernel in self.ccvep_data.get('results_by_kernel', {}):
                ccvep_kernel = self.ccvep_data['results_by_kernel'][kernel]
                glow_kernel = self.glow_data['results_by_kernel'][kernel]
                
                # Correlation length analysis
                aggregated['correlation_analysis'][kernel] = self._process_correlation_results(
                    ccvep_kernel.get('correlation_results', []),
                    glow_kernel.get('correlation_results', [])
                )
                
                # Sample size analysis (CCVEP) vs Trajectory count analysis (GLOW)
                aggregated['sample_size_analysis'][kernel] = self._process_sample_size_results(
                    ccvep_kernel.get('sample_size_results', []),
                    glow_kernel.get('trajectory_count_results', [])
                )
        
        # Overall framework comparison
        aggregated['framework_comparison'] = self._compute_framework_comparison()
        
        self.aggregated_metrics = aggregated
        return aggregated
    
    def _process_correlation_results(self, ccvep_results: List, glow_results: List) -> Dict:
        """Process correlation length analysis results."""
        ccvep_data = {}
        glow_data = {}
        
        # Extract CCVEP results
        for result in ccvep_results:
            corr_length = result['parameters']['micro_corr_length']
            metrics = result['validation_metrics']
            ccvep_data[corr_length] = {
                'times': metrics['times'],
                'w2_distances': metrics['w2_distances'],
                'mse_acf': metrics['mse_acf'],
                'rel_fro_cov': metrics['rel_fro_cov'],
                'avg_w2': np.nanmean(metrics['w2_distances'][:-1]),  # Exclude final 0, ignore NaN
                'avg_mse_acf': np.nanmean(metrics['mse_acf'][:-1]),
                'avg_rel_fro_cov': np.nanmean(metrics['rel_fro_cov'][:-1]),
                'final_w2': metrics['w2_distances'][-2] if len(metrics['w2_distances']) > 1 and np.isfinite(metrics['w2_distances'][-2]) else np.nan,
                'elapsed_time': result['elapsed_time']
            }
        
        # Extract GLOW results
        for result in glow_results:
            corr_length = result['parameters']['micro_corr_length']
            metrics = result['validation_metrics']
            glow_data[corr_length] = {
                'times': metrics['times'],
                'w2_distances': metrics['w2_distances'],
                'mse_acf': metrics['mse_acf'],
                'rel_fro_cov': metrics['rel_fro_cov'],
                'avg_w2': np.nanmean(metrics['w2_distances'][:-1]),  # Ignore NaN
                'avg_mse_acf': np.nanmean(metrics['mse_acf'][:-1]),
                'avg_rel_fro_cov': np.nanmean(metrics['rel_fro_cov'][:-1]),
                'final_w2': metrics['w2_distances'][-2] if len(metrics['w2_distances']) > 1 and np.isfinite(metrics['w2_distances'][-2]) else np.nan,
                'elapsed_time': result['elapsed_time']
            }
        
        return {'ccvep': ccvep_data, 'glow': glow_data}
    
    def _process_sample_size_results(self, ccvep_results: List, glow_results: List) -> Dict:
        """Process sample size analysis results."""
        ccvep_data = {}
        glow_data = {}
        
        # Extract CCVEP sample size results
        for result in ccvep_results:
            n_samples = result['parameters']['n_samples']
            metrics = result['validation_metrics']
            ccvep_data[n_samples] = {
                'times': metrics['times'],
                'w2_distances': metrics['w2_distances'],
                'mse_acf': metrics['mse_acf'],
                'rel_fro_cov': metrics['rel_fro_cov'],
                'avg_w2': np.nanmean(metrics['w2_distances'][:-1]),  # Ignore NaN
                'avg_mse_acf': np.nanmean(metrics['mse_acf'][:-1]),
                'avg_rel_fro_cov': np.nanmean(metrics['rel_fro_cov'][:-1]),
                'elapsed_time': result['elapsed_time']
            }
        
        # Extract GLOW trajectory count results
        for result in glow_results:
            n_trajectories = result['parameters']['n_training_trajectories']
            metrics = result['validation_metrics']
            glow_data[n_trajectories] = {
                'times': metrics['times'],
                'w2_distances': metrics['w2_distances'],
                'mse_acf': metrics['mse_acf'],
                'rel_fro_cov': metrics['rel_fro_cov'],
                'avg_w2': np.nanmean(metrics['w2_distances'][:-1]),  # Ignore NaN
                'avg_mse_acf': np.nanmean(metrics['mse_acf'][:-1]),
                'avg_rel_fro_cov': np.nanmean(metrics['rel_fro_cov'][:-1]),
                'elapsed_time': result['elapsed_time']
            }
        
        return {'ccvep': ccvep_data, 'glow': glow_data}
    
    def _compute_framework_comparison(self) -> Dict:
        """Compute overall framework comparison metrics."""
        ccvep_all_w2 = []
        ccvep_all_cov = []
        ccvep_times = []
        
        glow_all_w2 = []
        glow_all_cov = []
        glow_times = []
        
        # Collect all metrics from both frameworks
        for kernel in ['exponential', 'gaussian']:
            if kernel in self.ccvep_data.get('results_by_kernel', {}):
                ccvep_kernel = self.ccvep_data['results_by_kernel'][kernel]
                
                # CCVEP metrics
                for result_type in ['correlation_results', 'sample_size_results']:
                    results = ccvep_kernel.get(result_type, [])
                    print(f"  CCVEP {kernel} {result_type}: {len(results)} experiments")
                    for result in results:
                        try:
                            metrics = result.get('validation_metrics', {})
                            w2_list = metrics.get('w2_distances', [])
                            cov_list = metrics.get('rel_fro_cov', [])
                            
                            # Filter out NaN values and extend
                            if w2_list and len(w2_list) > 1:
                                valid_w2 = [x for x in w2_list[:-1] if np.isfinite(x)]
                                if valid_w2:
                                    ccvep_all_w2.extend(valid_w2)
                                elif not valid_w2 and len(w2_list) > 1:
                                    print(f"    Warning: All NaN values in {result.get('experiment_name', 'unknown')} W2")
                            
                            if cov_list and len(cov_list) > 1:
                                valid_cov = [x for x in cov_list[:-1] if np.isfinite(x)]
                                if valid_cov:
                                    ccvep_all_cov.extend(valid_cov)
                                elif not valid_cov and len(cov_list) > 1:
                                    print(f"    Warning: All NaN values in {result.get('experiment_name', 'unknown')} Cov")
                            
                            if 'elapsed_time' in result:
                                ccvep_times.append(result['elapsed_time'])
                        except Exception as e:
                            print(f"    Warning: Error processing CCVEP {kernel} {result_type}: {e}")
            
            if kernel in self.glow_data.get('results_by_kernel', {}):
                glow_kernel = self.glow_data['results_by_kernel'][kernel]
                
                # GLOW metrics
                for result_type in ['correlation_results', 'trajectory_count_results']:
                    results = glow_kernel.get(result_type, [])
                    print(f"  GLOW {kernel} {result_type}: {len(results)} experiments")
                    for result in results:
                        try:
                            metrics = result.get('validation_metrics', {})
                            w2_list = metrics.get('w2_distances', [])
                            cov_list = metrics.get('rel_fro_cov', [])
                            
                            # Filter out NaN values and extend
                            if w2_list and len(w2_list) > 1:
                                valid_w2 = [x for x in w2_list[:-1] if np.isfinite(x)]
                                if valid_w2:
                                    glow_all_w2.extend(valid_w2)
                                elif not valid_w2 and len(w2_list) > 1:
                                    print(f"    Warning: All NaN values in {result.get('experiment_name', 'unknown')} W2")
                            
                            if cov_list and len(cov_list) > 1:
                                valid_cov = [x for x in cov_list[:-1] if np.isfinite(x)]
                                if valid_cov:
                                    glow_all_cov.extend(valid_cov)
                                elif not valid_cov and len(cov_list) > 1:
                                    print(f"    Warning: All NaN values in {result.get('experiment_name', 'unknown')} Cov")
                            
                            if 'elapsed_time' in result:
                                glow_times.append(result['elapsed_time'])
                        except Exception as e:
                            print(f"    Warning: Error processing GLOW {kernel} {result_type}: {e}")
        
        print("\nCollected metrics:")
        print(f"  CCVEP: {len(ccvep_all_w2)} W2 values, {len(ccvep_all_cov)} cov values, {len(ccvep_times)} time values")
        print(f"  GLOW: {len(glow_all_w2)} W2 values, {len(glow_all_cov)} cov values, {len(glow_times)} time values")
        
        # Use safe mean to handle empty lists
        return {
            'ccvep': {
                'mean_w2': self._safe_mean(ccvep_all_w2),
                'std_w2': float(np.std(ccvep_all_w2)) if ccvep_all_w2 else np.nan,
                'mean_rel_fro_cov': self._safe_mean(ccvep_all_cov),
                'std_rel_fro_cov': float(np.std(ccvep_all_cov)) if ccvep_all_cov else np.nan,
                'mean_time': self._safe_mean(ccvep_times),
                'std_time': float(np.std(ccvep_times)) if ccvep_times else np.nan
            },
            'glow': {
                'mean_w2': self._safe_mean(glow_all_w2),
                'std_w2': float(np.std(glow_all_w2)) if glow_all_w2 else np.nan,
                'mean_rel_fro_cov': self._safe_mean(glow_all_cov),
                'std_rel_fro_cov': float(np.std(glow_all_cov)) if glow_all_cov else np.nan,
                'mean_time': self._safe_mean(glow_times),
                'std_time': float(np.std(glow_times)) if glow_times else np.nan
            }
        }
    
    def _safe_mean(self, values):
        """Return mean of values or np.nan if empty."""
        try:
            if not values:
                return np.nan
            return float(np.mean(values))
        except Exception:
            return np.nan

    def _get_framework_stats(self, framework_key: str) -> Dict[str, float]:
        """Get framework stats (mean/std) robustly from aggregated_metrics. If framework_comparison has valid numbers use them; otherwise compute from per-experiment aggregated data."""
        comp = self.aggregated_metrics.get('framework_comparison', {})
        # Try direct values first
        if framework_key in comp:
            fw = comp[framework_key]
            # Check for finite numbers
            if all(np.isfinite([fw.get('mean_w2', np.nan), fw.get('mean_rel_fro_cov', np.nan), fw.get('mean_time', np.nan)])):
                return {
                    'mean_w2': fw.get('mean_w2', np.nan),
                    'std_w2': fw.get('std_w2', np.nan),
                    'mean_rel_fro_cov': fw.get('mean_rel_fro_cov', np.nan),
                    'std_rel_fro_cov': fw.get('std_rel_fro_cov', np.nan),
                    'mean_time': fw.get('mean_time', np.nan),
                    'std_time': fw.get('std_time', np.nan)
                }
        # Fallback: compute from aggregated time-series stored under correlation/sample_size analyses
        w2_vals = []
        cov_vals = []
        times = []
        for kernel in ['exponential', 'gaussian']:
            # correlation analysis
            corr = self.aggregated_metrics.get('correlation_analysis', {}).get(kernel, {})
            for result in corr.get(framework_key, {}).values():
                if 'avg_w2' in result:
                    try:
                        w2_vals.append(float(result['avg_w2']))
                    except Exception:
                        pass
                if 'avg_rel_fro_cov' in result:
                    try:
                        cov_vals.append(float(result['avg_rel_fro_cov']))
                    except Exception:
                        pass
                if 'elapsed_time' in result:
                    try:
                        times.append(float(result['elapsed_time']))
                    except Exception:
                        pass
            # sample size analysis
            sample = self.aggregated_metrics.get('sample_size_analysis', {}).get(kernel, {})
            for result in sample.get(framework_key, {}).values():
                if 'avg_w2' in result:
                    try:
                        w2_vals.append(float(result['avg_w2']))
                    except Exception:
                        pass
                if 'avg_rel_fro_cov' in result:
                    try:
                        cov_vals.append(float(result['avg_rel_fro_cov']))
                    except Exception:
                        pass
                if 'elapsed_time' in result:
                    try:
                        times.append(float(result['elapsed_time']))
                    except Exception:
                        pass
        return {
            'mean_w2': self._safe_mean(w2_vals),
            'std_w2': float(np.std(w2_vals)) if w2_vals else np.nan,
            'mean_rel_fro_cov': self._safe_mean(cov_vals),
            'std_rel_fro_cov': float(np.std(cov_vals)) if cov_vals else np.nan,
            'mean_time': self._safe_mean(times),
            'std_time': float(np.std(times)) if times else np.nan
        }

    def create_correlation_comparison_plots(self):
        """Create correlation length comparison plots showing metrics over time."""
        for kernel in ['exponential', 'gaussian']:
            if kernel not in self.aggregated_metrics['correlation_analysis']:
                continue
                
            corr_data = self.aggregated_metrics['correlation_analysis'][kernel]
            ccvep_data = corr_data['ccvep']
            glow_data = corr_data['glow']
            
            # Get common correlation lengths
            common_lengths = sorted(set(ccvep_data.keys()) & set(glow_data.keys()))
            
            # Create figure with subplots for each metric
            fig, axes = plt.subplots(3, len(common_lengths), figsize=(5*len(common_lengths), 12))
            if len(common_lengths) == 1:
                axes = axes.reshape(-1, 1)
            
            fig.suptitle(f'Correlation Length Analysis: CCVEP vs GLOW Flow ({kernel.capitalize()} Kernel)', 
                        fontsize=16, fontweight='bold')
            
            metrics = ['w2_distances', 'mse_acf', 'rel_fro_cov']
            metric_labels = ['W2 Distance', 'MSE ACF', 'Relative Frobenius Covariance']
            
            for i, corr_length in enumerate(common_lengths):
                ccvep_metrics = ccvep_data[corr_length]
                glow_metrics = glow_data[corr_length]
                times = ccvep_metrics['times']
                
                for j, (metric, label) in enumerate(zip(metrics, metric_labels)):
                    ax = axes[j, i]
                    
                    ccvep_values = ccvep_metrics[metric]
                    glow_values = glow_metrics[metric]
                    
                    ax.plot(times, ccvep_values, 'o-', label='CCVEP', linewidth=2, 
                           markersize=8, color='blue', alpha=0.7)
                    ax.plot(times, glow_values, 's-', label='GLOW Flow', linewidth=2, 
                           markersize=8, color='red', alpha=0.7)
                    
                    ax.set_xlabel('Time', fontsize=10)
                    ax.set_ylabel(label, fontsize=10)
                    ax.set_title(f'Corr Length = {corr_length:.3f}', fontsize=11)
                    ax.legend(loc='best', fontsize=9)
                    ax.grid(True, alpha=0.3)
                    
                    # Add annotations for initial and final values
                    if len(ccvep_values) > 1:
                        ax.annotate(f'{ccvep_values[0]:.3f}', 
                                   xy=(times[0], ccvep_values[0]),
                                   xytext=(5, 5), textcoords='offset points', 
                                   fontsize=7, color='blue')
                        ax.annotate(f'{glow_values[0]:.3f}', 
                                   xy=(times[0], glow_values[0]),
                                   xytext=(5, -10), textcoords='offset points', 
                                   fontsize=7, color='red')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{kernel}_correlation_length_comparison_timeseries.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Created {kernel} correlation length time-series comparison plot")
    
    def create_sample_size_comparison_plots(self):
        """Create sample size comparison plots showing metrics over time."""
        for kernel in ['exponential', 'gaussian']:
            if kernel not in self.aggregated_metrics['sample_size_analysis']:
                continue
                
            sample_data = self.aggregated_metrics['sample_size_analysis'][kernel]
            ccvep_data = sample_data['ccvep']
            glow_data = sample_data['glow']
            
            # Get common sample sizes
            common_sizes = sorted(set(ccvep_data.keys()) & set(glow_data.keys()))
            
            # Create figure with subplots for each metric
            fig, axes = plt.subplots(3, len(common_sizes), figsize=(5*len(common_sizes), 12))
            if len(common_sizes) == 1:
                axes = axes.reshape(-1, 1)
            
            fig.suptitle(f'Sample Size Analysis: CCVEP vs GLOW Flow ({kernel.capitalize()} Kernel)', 
                        fontsize=16, fontweight='bold')
            
            metrics = ['w2_distances', 'mse_acf', 'rel_fro_cov']
            metric_labels = ['W2 Distance', 'MSE ACF', 'Relative Frobenius Covariance']
            
            for i, sample_size in enumerate(common_sizes):
                ccvep_metrics = ccvep_data[sample_size]
                glow_metrics = glow_data[sample_size]
                times = ccvep_metrics['times']
                
                for j, (metric, label) in enumerate(zip(metrics, metric_labels)):
                    ax = axes[j, i]
                    
                    ccvep_values = ccvep_metrics[metric]
                    glow_values = glow_metrics[metric]
                    
                    ax.plot(times, ccvep_values, 'o-', label='CCVEP', linewidth=2, 
                           markersize=8, color='blue', alpha=0.7)
                    ax.plot(times, glow_values, 's-', label='GLOW Flow', linewidth=2, 
                           markersize=8, color='red', alpha=0.7)
                    
                    ax.set_xlabel('Time', fontsize=10)
                    ax.set_ylabel(label, fontsize=10)
                    ax.set_title(f'Sample Size = {sample_size}', fontsize=11)
                    ax.legend(loc='best', fontsize=9)
                    ax.grid(True, alpha=0.3)
                    
                    # Add annotations for initial and final values
                    if len(ccvep_values) > 1:
                        ax.annotate(f'{ccvep_values[0]:.3f}', 
                                   xy=(times[0], ccvep_values[0]),
                                   xytext=(5, 5), textcoords='offset points', 
                                   fontsize=7, color='blue')
                        ax.annotate(f'{glow_values[0]:.3f}', 
                                   xy=(times[0], glow_values[0]),
                                   xytext=(5, -10), textcoords='offset points', 
                                   fontsize=7, color='red')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{kernel}_sample_size_comparison_timeseries.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Created {kernel} sample size time-series comparison plot")
    
    def create_framework_summary_plot(self):
        """Create overall framework comparison summary."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Framework Performance Summary: CCVEP vs GLOW Flow', fontsize=16, fontweight='bold')

        # Retrieve robust stats for both frameworks
        ccvep_stats = self._get_framework_stats('ccvep')
        glow_stats = self._get_framework_stats('glow')

        frameworks = ['CCVEP', 'GLOW Flow']
        colors = ['skyblue', 'lightcoral']

        # W2 Distance comparison
        ax1 = axes[0, 0]
        w2_means = [ccvep_stats['mean_w2'], glow_stats['mean_w2']]
        w2_stds = [ccvep_stats['std_w2'], glow_stats['std_w2']]
        # Replace nan with zeros for plotting but keep original values for annotations
        w2_plot_values = [0.0 if not np.isfinite(v) else v for v in w2_means]
        w2_plot_err = [0.0 if not np.isfinite(v) else v for v in w2_stds]
        bars = ax1.bar(frameworks, w2_plot_values, yerr=w2_plot_err, capsize=5, color=colors, alpha=0.8)
        ax1.set_ylabel('W2 Distance')
        ax1.set_title('Average W2 Distance Performance')
        ax1.grid(True, alpha=0.3)
        for i, (bar, mean, std) in enumerate(zip(bars, w2_means, w2_stds)):
            ann = 'n/a' if not np.isfinite(mean) else f'{mean:.3f}±{std:.3f}'
            ax1.annotate(ann,
                        xy=(bar.get_x() + bar.get_width() / 2, (0.0 if not np.isfinite(mean) else mean)),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

        # Relative Frobenius Covariance comparison
        ax2 = axes[0, 1]
        cov_means = [ccvep_stats['mean_rel_fro_cov'], glow_stats['mean_rel_fro_cov']]
        cov_stds = [ccvep_stats['std_rel_fro_cov'], glow_stats['std_rel_fro_cov']]
        cov_plot_values = [0.0 if not np.isfinite(v) else v for v in cov_means]
        cov_plot_err = [0.0 if not np.isfinite(v) else v for v in cov_stds]
        bars = ax2.bar(frameworks, cov_plot_values, yerr=cov_plot_err, capsize=5, color=colors, alpha=0.8)
        ax2.set_ylabel('Relative Frobenius Covariance')
        ax2.set_title('Average Covariance Error Performance')
        ax2.grid(True, alpha=0.3)
        for i, (bar, mean, std) in enumerate(zip(bars, cov_means, cov_stds)):
            ann = 'n/a' if not np.isfinite(mean) else f'{mean:.3f}±{std:.3f}'
            ax2.annotate(ann,
                        xy=(bar.get_x() + bar.get_width() / 2, (0.0 if not np.isfinite(mean) else mean)),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

        # Execution time comparison
        ax3 = axes[1, 0]
        time_means = [ccvep_stats['mean_time'], glow_stats['mean_time']]
        time_stds = [ccvep_stats['std_time'], glow_stats['std_time']]
        time_plot_values = [0.0 if not np.isfinite(v) else v for v in time_means]
        time_plot_err = [0.0 if not np.isfinite(v) else v for v in time_stds]
        bars = ax3.bar(frameworks, time_plot_values, yerr=time_plot_err, capsize=5, color=colors, alpha=0.8)
        ax3.set_ylabel('Execution Time (seconds)')
        ax3.set_title('Average Execution Time')
        ax3.grid(True, alpha=0.3)
        for i, (bar, mean, std) in enumerate(zip(bars, time_means, time_stds)):
            ann = 'n/a' if not np.isfinite(mean) else f'{mean:.1f}±{std:.1f}'
            ax3.annotate(ann,
                        xy=(bar.get_x() + bar.get_width() / 2, (0.0 if not np.isfinite(mean) else mean)),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

        # Performance ratio analysis (safe division)
        ax4 = axes[1, 1]
        def safe_div(a, b):
            try:
                if not np.isfinite(b) or b == 0:
                    return np.nan
                return a / b
            except Exception:
                return np.nan

        w2_ratio = safe_div(ccvep_stats['mean_w2'], glow_stats['mean_w2'])
        cov_ratio = safe_div(ccvep_stats['mean_rel_fro_cov'], glow_stats['mean_rel_fro_cov'])
        time_ratio = safe_div(ccvep_stats['mean_time'], glow_stats['mean_time'])

        ratios = [w2_ratio, cov_ratio, time_ratio]
        ratio_labels = ['W2 Distance\nRatio', 'Covariance Error\nRatio', 'Time\nRatio']
        colors_ratio = ['green' if (np.isfinite(r) and r < 1) else 'red' for r in ratios]

        bars = ax4.bar(ratio_labels, [0.0 if not np.isfinite(r) else r for r in ratios], color=colors_ratio, alpha=0.7)
        ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax4.set_ylabel('CCVEP / GLOW Flow Ratio')
        ax4.set_title('Performance Ratios (< 1 = CCVEP Better)')
        ax4.grid(True, alpha=0.3)
        for bar, ratio in zip(bars, ratios):
            ann = 'n/a' if not np.isfinite(ratio) else f'{ratio:.2f}'
            y = 0.0 if not np.isfinite(ratio) else ratio
            ax4.annotate(ann,
                        xy=(bar.get_x() + bar.get_width() / 2, y),
                        xytext=(0, 5 if (np.isfinite(ratio) and ratio > 1) else -15), textcoords="offset points",
                        ha='center', va='bottom' if (np.isfinite(ratio) and ratio > 1) else 'top', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'framework_summary_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_detailed_metrics_heatmap(self):
        """Create detailed heatmap of all metrics across experiments."""
        if not HAS_SEABORN:
            print("Skipping detailed heatmap - seaborn not available")
            return
            
        # Collect all data for heatmap
        heatmap_data = []
        
        for kernel in ['exponential', 'gaussian']:
            if kernel not in self.aggregated_metrics['correlation_analysis']:
                continue
                
            # Correlation analysis
            corr_data = self.aggregated_metrics['correlation_analysis'][kernel]
            for framework in ['ccvep', 'glow']:
                for corr_length, metrics in corr_data[framework].items():
                    heatmap_data.append({
                        'Kernel': kernel.capitalize(),
                        'Framework': framework.upper(),
                        'Experiment': f'Corr_{corr_length}',
                        'W2_Distance': metrics['avg_w2'],
                        'Rel_Fro_Cov': metrics['avg_rel_fro_cov'],
                        'Time': metrics['elapsed_time']
                    })
            
            # Sample size analysis
            sample_data = self.aggregated_metrics['sample_size_analysis'][kernel]
            for framework in ['ccvep', 'glow']:
                for sample_size, metrics in sample_data[framework].items():
                    heatmap_data.append({
                        'Kernel': kernel.capitalize(),
                        'Framework': framework.upper(),
                        'Experiment': f'Size_{sample_size}',
                        'W2_Distance': metrics['avg_w2'],
                        'Rel_Fro_Cov': metrics['avg_rel_fro_cov'],
                        'Time': metrics['elapsed_time']
                    })
        
        df = pd.DataFrame(heatmap_data)
        
        # Create separate heatmaps for each metric
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        fig.suptitle('Detailed Metrics Heatmap: All Experiments', fontsize=16, fontweight='bold')
        
        metrics = ['W2_Distance', 'Rel_Fro_Cov', 'Time']
        titles = ['W2 Distance', 'Relative Frobenius Covariance', 'Execution Time']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            pivot_df = df.pivot_table(values=metric, index=['Kernel', 'Framework'], columns='Experiment')
            
            sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='viridis', ax=axes[i])
            axes[i].set_title(title)
            axes[i].set_xlabel('Experiment')
            axes[i].set_ylabel('Kernel / Framework')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'detailed_metrics_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_comprehensive_timeseries_comparison(self):
        """Create comprehensive time-series comparison showing all experiments together."""
        for kernel in ['exponential', 'gaussian']:
            if kernel not in self.aggregated_metrics['correlation_analysis']:
                continue
            
            # Create figure with 3 rows (one for each metric) and 2 columns (correlation and sample size)
            fig, axes = plt.subplots(3, 2, figsize=(16, 14))
            fig.suptitle(f'Comprehensive Time-Series Comparison: {kernel.capitalize()} Kernel', 
                        fontsize=16, fontweight='bold')
            
            metrics = ['w2_distances', 'mse_acf', 'rel_fro_cov']
            metric_labels = ['W2 Distance', 'MSE ACF', 'Relative Frobenius Covariance']
            
            # Correlation length analysis (left column)
            corr_data = self.aggregated_metrics['correlation_analysis'][kernel]
            ccvep_corr = corr_data['ccvep']
            glow_corr = corr_data['glow']
            common_corr_lengths = sorted(set(ccvep_corr.keys()) & set(glow_corr.keys()))
            
            # Sample size analysis (right column)
            sample_data = self.aggregated_metrics['sample_size_analysis'][kernel]
            ccvep_sample = sample_data['ccvep']
            glow_sample = sample_data['glow']
            common_sample_sizes = sorted(set(ccvep_sample.keys()) & set(glow_sample.keys()))
            
            # Use colormap for different experiments
            colors_ccvep = plt.cm.Blues(np.linspace(0.4, 0.9, len(common_corr_lengths)))
            colors_glow = plt.cm.Reds(np.linspace(0.4, 0.9, len(common_corr_lengths)))
            
            for metric_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
                # Correlation length plots (left column)
                ax_corr = axes[metric_idx, 0]
                for i, corr_length in enumerate(common_corr_lengths):
                    ccvep_metrics = ccvep_corr[corr_length]
                    glow_metrics = glow_corr[corr_length]
                    times = ccvep_metrics['times']
                    
                    ax_corr.plot(times, ccvep_metrics[metric], 'o-', 
                               label=f'CCVEP {corr_length:.3f}',
                               linewidth=2, markersize=6, color=colors_ccvep[i], alpha=0.7)
                    ax_corr.plot(times, glow_metrics[metric], 's--', 
                               label=f'GLOW {corr_length:.3f}',
                               linewidth=2, markersize=6, color=colors_glow[i], alpha=0.7)
                
                ax_corr.set_xlabel('Time', fontsize=11)
                ax_corr.set_ylabel(label, fontsize=11)
                ax_corr.set_title(f'{label} - Correlation Length Variation', fontsize=12, fontweight='bold')
                ax_corr.legend(loc='best', fontsize=8, ncol=2)
                ax_corr.grid(True, alpha=0.3)
                
                # Sample size plots (right column)
                ax_sample = axes[metric_idx, 1]
                colors_ccvep_sample = plt.cm.Blues(np.linspace(0.4, 0.9, len(common_sample_sizes)))
                colors_glow_sample = plt.cm.Reds(np.linspace(0.4, 0.9, len(common_sample_sizes)))
                
                for i, sample_size in enumerate(common_sample_sizes):
                    ccvep_metrics = ccvep_sample[sample_size]
                    glow_metrics = glow_sample[sample_size]
                    times = ccvep_metrics['times']
                    
                    ax_sample.plot(times, ccvep_metrics[metric], 'o-', 
                                 label=f'CCVEP {sample_size}',
                                 linewidth=2, markersize=6, color=colors_ccvep_sample[i], alpha=0.7)
                    ax_sample.plot(times, glow_metrics[metric], 's--', 
                                 label=f'GLOW {sample_size}',
                                 linewidth=2, markersize=6, color=colors_glow_sample[i], alpha=0.7)
                
                ax_sample.set_xlabel('Time', fontsize=11)
                ax_sample.set_ylabel(label, fontsize=11)
                ax_sample.set_title(f'{label} - Sample Size Variation', fontsize=12, fontweight='bold')
                ax_sample.legend(loc='best', fontsize=8, ncol=2)
                ax_sample.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{kernel}_comprehensive_timeseries_comparison.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Created {kernel} comprehensive time-series comparison plot")
    
    def generate_summary_report(self):
        """Generate a comprehensive text summary report."""
        report_path = self.output_dir / 'comparative_analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("COMPARATIVE ANALYSIS REPORT: CCVEP vs GLOW Flow\n")
            f.write("=" * 60 + "\n\n")
            
            # Overview
            f.write("OVERVIEW:\n")
            f.write("-" * 20 + "\n")
            ccvep_exp_count = sum(len(kernel_data.get('correlation_results', [])) + len(kernel_data.get('sample_size_results', [])) 
                                 for kernel_data in self.ccvep_data.get('results_by_kernel', {}).values())
            glow_exp_count = sum(len(kernel_data.get('correlation_results', [])) + len(kernel_data.get('trajectory_count_results', [])) 
                                for kernel_data in self.glow_data.get('results_by_kernel', {}).values())
            
            f.write(f"CCVEP Experiments: {ccvep_exp_count}\n")
            f.write(f"GLOW Flow Experiments: {glow_exp_count}\n")
            f.write(f"Kernels Analyzed: {', '.join(self.aggregated_metrics['correlation_analysis'].keys())}\n\n")
            
            # Framework comparison summary
            f.write("FRAMEWORK PERFORMANCE SUMMARY:\n")
            f.write("-" * 35 + "\n")
            comparison = self.aggregated_metrics['framework_comparison']
            
            f.write(f"{'Metric':<25} {'CCVEP':<15} {'GLOW Flow':<15} {'Ratio':<10} {'Winner':<10}\n")
            f.write("-" * 80 + "\n")
            
            ccvep_w2 = comparison['ccvep']['mean_w2']
            glow_w2 = comparison['glow']['mean_w2']
            w2_ratio = ccvep_w2 / glow_w2
            w2_winner = "CCVEP" if w2_ratio < 1 else "GLOW Flow"
            f.write(f"{'W2 Distance':<25} {ccvep_w2:<15.4f} {glow_w2:<15.4f} {w2_ratio:<10.3f} {w2_winner:<10}\n")
            
            ccvep_cov = comparison['ccvep']['mean_rel_fro_cov']
            glow_cov = comparison['glow']['mean_rel_fro_cov']
            cov_ratio = ccvep_cov / glow_cov
            cov_winner = "CCVEP" if cov_ratio < 1 else "GLOW Flow"
            f.write(f"{'Covariance Error':<25} {ccvep_cov:<15.4f} {glow_cov:<15.4f} {cov_ratio:<10.3f} {cov_winner:<10}\n")
            
            ccvep_time = comparison['ccvep']['mean_time']
            glow_time = comparison['glow']['mean_time']
            time_ratio = ccvep_time / glow_time
            time_winner = "CCVEP" if time_ratio < 1 else "GLOW Flow"
            f.write(f"{'Execution Time':<25} {ccvep_time:<15.2f} {glow_time:<15.2f} {time_ratio:<10.3f} {time_winner:<10}\n\n")
            
            # Detailed analysis by kernel
            for kernel in ['exponential', 'gaussian']:
                if kernel not in self.aggregated_metrics['correlation_analysis']:
                    continue
                    
                f.write(f"{kernel.upper()} KERNEL ANALYSIS:\n")
                f.write("-" * 30 + "\n")
                
                # Correlation length analysis
                f.write("Correlation Length Performance (Initial vs Final):\n")
                corr_data = self.aggregated_metrics['correlation_analysis'][kernel]
                ccvep_corr = corr_data['ccvep']
                glow_corr = corr_data['glow']
                
                common_lengths = sorted(set(ccvep_corr.keys()) & set(glow_corr.keys()))
                f.write(f"{'Length':<10} {'Framework':<12} {'Initial W2':<12} {'Final W2':<12} {'Initial Cov':<12} {'Final Cov':<12}\n")
                f.write("-" * 80 + "\n")
                for length in common_lengths:
                    ccvep_metrics = ccvep_corr[length]
                    glow_metrics = glow_corr[length]
                    
                    # CCVEP row
                    f.write(f"{length:<10.3f} {'CCVEP':<12} "
                           f"{ccvep_metrics['w2_distances'][0]:<12.4f} "
                           f"{ccvep_metrics['w2_distances'][-2]:<12.4f} "
                           f"{ccvep_metrics['rel_fro_cov'][0]:<12.4f} "
                           f"{ccvep_metrics['rel_fro_cov'][-2]:<12.4f}\n")
                    
                    # GLOW row
                    f.write(f"{'':10} {'GLOW':<12} "
                           f"{glow_metrics['w2_distances'][0]:<12.4f} "
                           f"{glow_metrics['w2_distances'][-2]:<12.4f} "
                           f"{glow_metrics['rel_fro_cov'][0]:<12.4f} "
                           f"{glow_metrics['rel_fro_cov'][-2]:<12.4f}\n")
                    f.write("-" * 80 + "\n")
                f.write("\n")
                
                # Sample size analysis
                f.write("Sample Size Performance (Initial vs Final):\n")
                sample_data = self.aggregated_metrics['sample_size_analysis'][kernel]
                ccvep_sample = sample_data['ccvep']
                glow_sample = sample_data['glow']
                
                common_sizes = sorted(set(ccvep_sample.keys()) & set(glow_sample.keys()))
                f.write(f"{'Size':<10} {'Framework':<12} {'Initial W2':<12} {'Final W2':<12} {'Initial Cov':<12} {'Final Cov':<12}\n")
                f.write("-" * 80 + "\n")
                for size in common_sizes:
                    ccvep_metrics = ccvep_sample[size]
                    glow_metrics = glow_sample[size]
                    
                    # CCVEP row
                    f.write(f"{size:<10} {'CCVEP':<12} "
                           f"{ccvep_metrics['w2_distances'][0]:<12.4f} "
                           f"{ccvep_metrics['w2_distances'][-2]:<12.4f} "
                           f"{ccvep_metrics['rel_fro_cov'][0]:<12.4f} "
                           f"{ccvep_metrics['rel_fro_cov'][-2]:<12.4f}\n")
                    
                    # GLOW row
                    f.write(f"{'':10} {'GLOW':<12} "
                           f"{glow_metrics['w2_distances'][0]:<12.4f} "
                           f"{glow_metrics['w2_distances'][-2]:<12.4f} "
                           f"{glow_metrics['rel_fro_cov'][0]:<12.4f} "
                           f"{glow_metrics['rel_fro_cov'][-2]:<12.4f}\n")
                    f.write("-" * 80 + "\n")
                f.write("\n")
            
            # Key findings
            f.write("KEY FINDINGS:\n")
            f.write("-" * 15 + "\n")
            f.write(f"1. W2 Distance: {'CCVEP performs better' if w2_ratio < 1 else 'GLOW Flow performs better'} "
                   f"({abs(1-w2_ratio)*100:.1f}% {'lower' if w2_ratio < 1 else 'higher'})\n")
            f.write(f"2. Covariance Error: {'CCVEP performs better' if cov_ratio < 1 else 'GLOW Flow performs better'} "
                   f"({abs(1-cov_ratio)*100:.1f}% {'lower' if cov_ratio < 1 else 'higher'})\n")
            f.write(f"3. Execution Time: {'CCVEP is faster' if time_ratio < 1 else 'GLOW Flow is faster'} "
                   f"({abs(1-time_ratio)*100:.1f}% {'faster' if time_ratio < 1 else 'slower'})\n")
        
        return report_path
    
    def export_timeseries_to_csv(self):
        """Export time-series data to CSV files for detailed analysis."""
        # Export correlation length time-series
        for kernel in ['exponential', 'gaussian']:
            if kernel not in self.aggregated_metrics['correlation_analysis']:
                continue
            
            corr_data = self.aggregated_metrics['correlation_analysis'][kernel]
            
            # Create DataFrame for correlation analysis
            corr_rows = []
            for framework in ['ccvep', 'glow']:
                for corr_length, metrics in corr_data[framework].items():
                    times = metrics['times']
                    for i, time in enumerate(times):
                        corr_rows.append({
                            'kernel': kernel,
                            'framework': framework.upper(),
                            'corr_length': corr_length,
                            'time': time,
                            'w2_distance': metrics['w2_distances'][i],
                            'mse_acf': metrics['mse_acf'][i],
                            'rel_fro_cov': metrics['rel_fro_cov'][i]
                        })
            
            corr_df = pd.DataFrame(corr_rows)
            corr_csv_path = self.output_dir / f'{kernel}_correlation_comparison.csv'
            corr_df.to_csv(corr_csv_path, index=False)
            print(f"Exported {kernel} correlation time-series to {corr_csv_path}")
            
            # Create DataFrame for sample size analysis
            sample_data = self.aggregated_metrics['sample_size_analysis'][kernel]
            sample_rows = []
            for framework in ['ccvep', 'glow']:
                for sample_size, metrics in sample_data[framework].items():
                    times = metrics['times']
                    for i, time in enumerate(times):
                        sample_rows.append({
                            'kernel': kernel,
                            'framework': framework.upper(),
                            'sample_size': sample_size,
                            'time': time,
                            'w2_distance': metrics['w2_distances'][i],
                            'mse_acf': metrics['mse_acf'][i],
                            'rel_fro_cov': metrics['rel_fro_cov'][i]
                        })
            
            sample_df = pd.DataFrame(sample_rows)
            sample_csv_path = self.output_dir / f'{kernel}_sample_size_comparison.csv'
            sample_df.to_csv(sample_csv_path, index=False)
            print(f"Exported {kernel} sample size time-series to {sample_csv_path}")
        
        # Export framework summary
        comparison = self.aggregated_metrics['framework_comparison']
        summary_data = []
        for framework in ['ccvep', 'glow']:
            summary_data.append({
                'framework': framework.upper(),
                'mean_w2': comparison[framework]['mean_w2'],
                'std_w2': comparison[framework]['std_w2'],
                'mean_rel_fro_cov': comparison[framework]['mean_rel_fro_cov'],
                'std_rel_fro_cov': comparison[framework]['std_rel_fro_cov'],
                'mean_time': comparison[framework]['mean_time'],
                'std_time': comparison[framework]['std_time']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = self.output_dir / 'framework_summary_comparison.csv'
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Exported framework summary to {summary_csv_path}")
    
    def run_full_analysis(self):
        """Run the complete comparative analysis."""
        print("Starting comparative analysis...")
        
        # Aggregate metrics
        print("Aggregating metrics...")
        self.aggregate_metrics()
        
        # Create visualizations
        print("Creating correlation length comparison plots (time-series)...")
        self.create_correlation_comparison_plots()
        
        print("Creating sample size comparison plots (time-series)...")
        self.create_sample_size_comparison_plots()
        
        print("Creating comprehensive time-series comparison plots...")
        self.create_comprehensive_timeseries_comparison()
        
        print("Creating framework summary plot...")
        self.create_framework_summary_plot()
        
        print("Creating detailed metrics heatmap...")
        self.create_detailed_metrics_heatmap()
        
        # Generate report
        print("Generating summary report...")
        report_path = self.generate_summary_report()
        
        # Export time-series data to CSV
        print("Exporting time-series data to CSV files...")
        self.export_timeseries_to_csv()
        
        # Save aggregated data
        aggregated_path = self.output_dir / 'aggregated_metrics.json'
        with open(aggregated_path, 'w') as f:
            json.dump(self.aggregated_metrics, f, indent=2)
        
        print(f"Analysis complete! Results saved to {self.output_dir}")
        print(f"Summary report: {report_path}")
        print(f"Aggregated data: {aggregated_path}")
        
        return self.aggregated_metrics


def main():
    parser = argparse.ArgumentParser(description="Comparative analysis of CCVEP vs GLOW Flow experiments")
    parser.add_argument("--ccvep-path", default="ccvep_experiments", 
                       help="Path to CCVEP experiments directory")
    parser.add_argument("--glow-path", default="glow_flow_experiments", 
                       help="Path to GLOW Flow experiments directory")
    parser.add_argument("--output-dir", default="comparative_analysis", 
                       help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    analyzer = ComparativeAnalyzer(args.ccvep_path, args.glow_path, args.output_dir)
    results = analyzer.run_full_analysis()
    
    return results


if __name__ == "__main__":
    main()