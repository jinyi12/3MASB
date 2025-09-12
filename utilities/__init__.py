"""
Utilities Package for Asymmetric Bridge Models
==============================================

This package contains utility modules for the asymmetric bridge framework:
- common: Automatic differentiation utilities (jvp, t_dir)
- simulation: SDE solvers and sample generation
- visualization: Plotting and visualization functions
- data_generation: Data generation utilities for GRF and spiral data
- training: Training utilities for bridge models
"""

from .common import jvp, t_dir
from .simulation import (
    solve_gaussian_bridge_reverse_sde,
    solve_backward_sde_euler_maruyama,
    generate_backward_samples
)
from .visualization import (
    visualize_bridge_results,
    plot_confidence_ellipse
)
from .data_generation import (
    generate_multiscale_grf_data,
    generate_spiral_distributional_data,
    RandomFieldGenerator2D
)
from .training import train_bridge

__all__ = [
    'jvp', 't_dir',
    'solve_gaussian_bridge_reverse_sde', 'solve_backward_sde_euler_maruyama', 'generate_backward_samples',
    'visualize_bridge_results', 'plot_confidence_ellipse',
    'generate_multiscale_grf_data', 'generate_spiral_distributional_data', 'RandomFieldGenerator2D',
    'train_bridge'
]