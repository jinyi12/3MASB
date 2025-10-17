#!/usr/bin/env python3
"""
Generate additional comprehensive tables and summary for CCVEP vs GLOW Flow comparison.
This version includes time-series analysis.
"""

import json
import pandas as pd
from pathlib import Path

def create_comprehensive_tables():
    """Create comprehensive comparison tables including time-series data."""
    
    # Load aggregated metrics
    with open('comparative_analysis/aggregated_metrics.json', 'r') as f:
        data = json.load(f)
    
    output_dir = Path('comparative_analysis')
    
    # Create correlation length comparison table
    print("=== CORRELATION LENGTH ANALYSIS (TIME-SERIES) ===\n")
    
    for kernel in ['exponential', 'gaussian']:
        print(f"{kernel.upper()} KERNEL:")
        print("-" * 80)
        
        corr_data = data['correlation_analysis'][kernel]
        
        # Create time-series DataFrame
        time_series_rows = []
        for corr_length in sorted(corr_data['ccvep'].keys(), key=float):
            ccvep = corr_data['ccvep'][corr_length]
            glow = corr_data['glow'][corr_length]
            
            times = ccvep['times']
            for i, time in enumerate(times[:-1]):  # Exclude final 0 value
                time_series_rows.append({
                    'Corr_Length': float(corr_length),
                    'Time': time,
                    'CCVEP_W2': ccvep['w2_distances'][i],
                    'GLOW_W2': glow['w2_distances'][i],
                    'W2_Diff': ccvep['w2_distances'][i] - glow['w2_distances'][i],
                    'CCVEP_MSE_ACF': ccvep['mse_acf'][i],
                    'GLOW_MSE_ACF': glow['mse_acf'][i],
                    'CCVEP_Cov': ccvep['rel_fro_cov'][i],
                    'GLOW_Cov': glow['rel_fro_cov'][i],
                    'Cov_Diff': ccvep['rel_fro_cov'][i] - glow['rel_fro_cov'][i]
                })
        
        ts_df = pd.DataFrame(time_series_rows)
        
        # Summary statistics per correlation length
        summary_rows = []
        for corr_length in sorted(corr_data['ccvep'].keys(), key=float):
            ccvep = corr_data['ccvep'][corr_length]
            glow = corr_data['glow'][corr_length]
            
            # Calculate improvement rates (from initial to final non-zero time)
            ccvep_w2_improvement = (ccvep['w2_distances'][0] - ccvep['w2_distances'][-2]) / ccvep['w2_distances'][0] * 100
            glow_w2_improvement = (glow['w2_distances'][0] - glow['w2_distances'][-2]) / glow['w2_distances'][0] * 100
            
            ccvep_cov_improvement = (ccvep['rel_fro_cov'][0] - ccvep['rel_fro_cov'][-2]) / ccvep['rel_fro_cov'][0] * 100
            glow_cov_improvement = (glow['rel_fro_cov'][0] - glow['rel_fro_cov'][-2]) / glow['rel_fro_cov'][0] * 100
            
            summary_rows.append({
                'Corr_Length': float(corr_length),
                'CCVEP_Initial_W2': ccvep['w2_distances'][0],
                'CCVEP_Final_W2': ccvep['w2_distances'][-2],
                'CCVEP_W2_Improvement_%': ccvep_w2_improvement,
                'GLOW_Initial_W2': glow['w2_distances'][0],
                'GLOW_Final_W2': glow['w2_distances'][-2],
                'GLOW_W2_Improvement_%': glow_w2_improvement,
                'CCVEP_Initial_Cov': ccvep['rel_fro_cov'][0],
                'CCVEP_Final_Cov': ccvep['rel_fro_cov'][-2],
                'CCVEP_Cov_Improvement_%': ccvep_cov_improvement,
                'GLOW_Initial_Cov': glow['rel_fro_cov'][0],
                'GLOW_Final_Cov': glow['rel_fro_cov'][-2],
                'GLOW_Cov_Improvement_%': glow_cov_improvement,
                'CCVEP_Time': ccvep['elapsed_time'],
                'GLOW_Time': glow['elapsed_time']
            })
        
        summary_df = pd.DataFrame(summary_rows)
        
        # Display formatted summary table
        print(f"{'Length':<8} {'Framework':<10} {'Initial W2':<12} {'Final W2':<12} {'W2 Improv%':<12} {'Initial Cov':<12} {'Final Cov':<12} {'Cov Improv%':<12}")
        print("-" * 110)
        for _, row in summary_df.iterrows():
            print(f"{row['Corr_Length']:<8.3f} {'CCVEP':<10} {row['CCVEP_Initial_W2']:<12.4f} {row['CCVEP_Final_W2']:<12.4f} "
                  f"{row['CCVEP_W2_Improvement_%']:<12.1f} {row['CCVEP_Initial_Cov']:<12.4f} {row['CCVEP_Final_Cov']:<12.4f} {row['CCVEP_Cov_Improvement_%']:<12.1f}")
            print(f"{'':8} {'GLOW':<10} {row['GLOW_Initial_W2']:<12.4f} {row['GLOW_Final_W2']:<12.4f} "
                  f"{row['GLOW_W2_Improvement_%']:<12.1f} {row['GLOW_Initial_Cov']:<12.4f} {row['GLOW_Final_Cov']:<12.4f} {row['GLOW_Cov_Improvement_%']:<12.1f}")
            print("-" * 110)
        
        # Save to CSV
        ts_csv_path = output_dir / f'{kernel}_correlation_timeseries.csv'
        ts_df.to_csv(ts_csv_path, index=False, float_format='%.6f')
        
        summary_csv_path = output_dir / f'{kernel}_correlation_summary.csv'
        summary_df.to_csv(summary_csv_path, index=False, float_format='%.6f')
        
        print(f"\nSaved time-series data to {ts_csv_path}")
        print(f"Saved summary data to {summary_csv_path}")
        print()
    
    print("\n=== SAMPLE SIZE ANALYSIS (TIME-SERIES) ===\n")
    
    for kernel in ['exponential', 'gaussian']:
        print(f"{kernel.upper()} KERNEL:")
        print("-" * 80)
        
        sample_data = data['sample_size_analysis'][kernel]
        
        # Create time-series DataFrame
        time_series_rows = []
        for sample_size in sorted(sample_data['ccvep'].keys(), key=int):
            ccvep = sample_data['ccvep'][sample_size]
            glow = sample_data['glow'][sample_size]
            
            times = ccvep['times']
            for i, time in enumerate(times[:-1]):  # Exclude final 0 value
                time_series_rows.append({
                    'Sample_Size': int(sample_size),
                    'Time': time,
                    'CCVEP_W2': ccvep['w2_distances'][i],
                    'GLOW_W2': glow['w2_distances'][i],
                    'W2_Diff': ccvep['w2_distances'][i] - glow['w2_distances'][i],
                    'CCVEP_MSE_ACF': ccvep['mse_acf'][i],
                    'GLOW_MSE_ACF': glow['mse_acf'][i],
                    'CCVEP_Cov': ccvep['rel_fro_cov'][i],
                    'GLOW_Cov': glow['rel_fro_cov'][i],
                    'Cov_Diff': ccvep['rel_fro_cov'][i] - glow['rel_fro_cov'][i]
                })
        
        ts_df = pd.DataFrame(time_series_rows)
        
        # Summary statistics per sample size
        summary_rows = []
        for sample_size in sorted(sample_data['ccvep'].keys(), key=int):
            ccvep = sample_data['ccvep'][sample_size]
            glow = sample_data['glow'][sample_size]
            
            # Calculate improvement rates
            ccvep_w2_improvement = (ccvep['w2_distances'][0] - ccvep['w2_distances'][-2]) / ccvep['w2_distances'][0] * 100
            glow_w2_improvement = (glow['w2_distances'][0] - glow['w2_distances'][-2]) / glow['w2_distances'][0] * 100
            
            ccvep_cov_improvement = (ccvep['rel_fro_cov'][0] - ccvep['rel_fro_cov'][-2]) / ccvep['rel_fro_cov'][0] * 100
            glow_cov_improvement = (glow['rel_fro_cov'][0] - glow['rel_fro_cov'][-2]) / glow['rel_fro_cov'][0] * 100
            
            summary_rows.append({
                'Sample_Size': int(sample_size),
                'CCVEP_Initial_W2': ccvep['w2_distances'][0],
                'CCVEP_Final_W2': ccvep['w2_distances'][-2],
                'CCVEP_W2_Improvement_%': ccvep_w2_improvement,
                'GLOW_Initial_W2': glow['w2_distances'][0],
                'GLOW_Final_W2': glow['w2_distances'][-2],
                'GLOW_W2_Improvement_%': glow_w2_improvement,
                'CCVEP_Initial_Cov': ccvep['rel_fro_cov'][0],
                'CCVEP_Final_Cov': ccvep['rel_fro_cov'][-2],
                'CCVEP_Cov_Improvement_%': ccvep_cov_improvement,
                'GLOW_Initial_Cov': glow['rel_fro_cov'][0],
                'GLOW_Final_Cov': glow['rel_fro_cov'][-2],
                'GLOW_Cov_Improvement_%': glow_cov_improvement,
                'CCVEP_Time': ccvep['elapsed_time'],
                'GLOW_Time': glow['elapsed_time']
            })
        
        summary_df = pd.DataFrame(summary_rows)
        
        # Display formatted summary table
        print(f"{'Size':<8} {'Framework':<10} {'Initial W2':<12} {'Final W2':<12} {'W2 Improv%':<12} {'Initial Cov':<12} {'Final Cov':<12} {'Cov Improv%':<12}")
        print("-" * 110)
        for _, row in summary_df.iterrows():
            print(f"{row['Sample_Size']:<8} {'CCVEP':<10} {row['CCVEP_Initial_W2']:<12.4f} {row['CCVEP_Final_W2']:<12.4f} "
                  f"{row['CCVEP_W2_Improvement_%']:<12.1f} {row['CCVEP_Initial_Cov']:<12.4f} {row['CCVEP_Final_Cov']:<12.4f} {row['CCVEP_Cov_Improvement_%']:<12.1f}")
            print(f"{'':8} {'GLOW':<10} {row['GLOW_Initial_W2']:<12.4f} {row['GLOW_Final_W2']:<12.4f} "
                  f"{row['GLOW_W2_Improvement_%']:<12.1f} {row['GLOW_Initial_Cov']:<12.4f} {row['GLOW_Final_Cov']:<12.4f} {row['GLOW_Cov_Improvement_%']:<12.1f}")
            print("-" * 110)
        
        # Save to CSV
        ts_csv_path = output_dir / f'{kernel}_sample_size_timeseries.csv'
        ts_df.to_csv(ts_csv_path, index=False, float_format='%.6f')
        
        summary_csv_path = output_dir / f'{kernel}_sample_size_summary.csv'
        summary_df.to_csv(summary_csv_path, index=False, float_format='%.6f')
        
        print(f"\nSaved time-series data to {ts_csv_path}")
        print(f"Saved summary data to {summary_csv_path}")
        print()
    
    # Overall framework summary
    framework_comp = data['framework_comparison']
    print("\n=== OVERALL FRAMEWORK COMPARISON ===\n")
    
    print(f"{'Metric':<30} {'CCVEP':<20} {'GLOW Flow':<20} {'Ratio':<10} {'Winner':<15}")
    print("-" * 100)
    
    # W2 Distance
    ccvep_w2 = framework_comp['ccvep']['mean_w2']
    glow_w2 = framework_comp['glow']['mean_w2']
    w2_ratio = ccvep_w2 / glow_w2
    w2_winner = "CCVEP" if w2_ratio < 1 else "GLOW Flow"
    print(f"{'W2 Distance (lower better)':<30} {f'{ccvep_w2:.4f} ± {framework_comp['ccvep']['std_w2']:.4f}':<20} "
          f"{f'{glow_w2:.4f} ± {framework_comp['glow']['std_w2']:.4f}':<20} {w2_ratio:<10.3f} {w2_winner:<15}")
    
    # Covariance Error
    ccvep_cov = framework_comp['ccvep']['mean_rel_fro_cov']
    glow_cov = framework_comp['glow']['mean_rel_fro_cov']
    cov_ratio = ccvep_cov / glow_cov
    cov_winner = "CCVEP" if cov_ratio < 1 else "GLOW Flow"
    print(f"{'Covariance Error (lower better)':<30} {f'{ccvep_cov:.4f} ± {framework_comp['ccvep']['std_rel_fro_cov']:.4f}':<20} "
          f"{f'{glow_cov:.4f} ± {framework_comp['glow']['std_rel_fro_cov']:.4f}':<20} {cov_ratio:<10.3f} {cov_winner:<15}")
    
    # Execution Time
    ccvep_time = framework_comp['ccvep']['mean_time']
    glow_time = framework_comp['glow']['mean_time']
    time_ratio = ccvep_time / glow_time
    time_winner = "CCVEP" if time_ratio < 1 else "GLOW Flow"
    print(f"{'Execution Time (lower better)':<30} {f'{ccvep_time:.2f} ± {framework_comp['ccvep']['std_time']:.2f}':<20} "
          f"{f'{glow_time:.2f} ± {framework_comp['glow']['std_time']:.2f}':<20} {time_ratio:<10.3f} {time_winner:<15}")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame([
        {
            'Metric': 'W2_Distance',
            'CCVEP_Mean': ccvep_w2,
            'CCVEP_Std': framework_comp['ccvep']['std_w2'],
            'GLOW_Mean': glow_w2,
            'GLOW_Std': framework_comp['glow']['std_w2'],
            'Ratio_CCVEP_vs_GLOW': w2_ratio,
            'Winner': w2_winner,
            'Performance_Difference_Percent': abs(1-w2_ratio) * 100
        },
        {
            'Metric': 'Covariance_Error',
            'CCVEP_Mean': ccvep_cov,
            'CCVEP_Std': framework_comp['ccvep']['std_rel_fro_cov'],
            'GLOW_Mean': glow_cov,
            'GLOW_Std': framework_comp['glow']['std_rel_fro_cov'],
            'Ratio_CCVEP_vs_GLOW': cov_ratio,
            'Winner': cov_winner,
            'Performance_Difference_Percent': abs(1-cov_ratio) * 100
        },
        {
            'Metric': 'Execution_Time',
            'CCVEP_Mean': ccvep_time,
            'CCVEP_Std': framework_comp['ccvep']['std_time'],
            'GLOW_Mean': glow_time,
            'GLOW_Std': framework_comp['glow']['std_time'],
            'Ratio_CCVEP_vs_GLOW': time_ratio,
            'Winner': time_winner,
            'Performance_Difference_Percent': abs(1-time_ratio) * 100
        }
    ])
    
    summary_path = output_dir / 'framework_summary_comparison.csv'
    summary_df.to_csv(summary_path, index=False, float_format='%.6f')
    print(f"\nFramework summary saved to {summary_path}")
    
    print("\n=== KEY INSIGHTS ===")
    if w2_ratio < 1:
        print(f"1. CCVEP achieves {(1-w2_ratio)*100:.1f}% better W2 distances on average")
    else:
        print(f"1. GLOW Flow achieves {(w2_ratio-1)*100:.1f}% better W2 distances on average")
    
    if cov_ratio < 1:
        print(f"2. CCVEP achieves {(1-cov_ratio)*100:.1f}% better covariance preservation on average")
    else:
        print(f"2. GLOW Flow achieves {(cov_ratio-1)*100:.1f}% better covariance preservation on average")
    
    if time_ratio < 1:
        print(f"3. CCVEP is {(1-time_ratio)*100:.1f}% faster in execution time on average")
    else:
        print(f"3. GLOW Flow is {(time_ratio-1)*100:.1f}% faster in execution time on average")
    
    print("4. Time-series analysis shows the evolution of metrics over time for both frameworks")
    print("5. CSV files include detailed improvement rates from initial to final states")

if __name__ == "__main__":
    create_comprehensive_tables()