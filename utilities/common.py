"""
Common Utilities
================

This module contains common utility functions used across different bridge
implementations, such as automatic differentiation helpers.
"""

from typing import Tuple, Callable, Any
import torch
from torch import Tensor

def jvp(f: Callable[[Tensor], Any], x: Tensor, v: Tensor) -> Tuple[Tensor, ...]:
    """Compute Jacobian-vector product. Used for time derivatives."""
    # Ensures gradients can flow back through the JVP calculation if needed
    return torch.autograd.functional.jvp(
        f, x, v,
        create_graph=torch.is_grad_enabled()
    )

def t_dir(f: Callable[[Tensor], Any], t: Tensor) -> Tuple[Tensor, ...]:
    """Compute the time derivative of f(t) by using jvp with v=1."""
    return jvp(f, t, torch.ones_like(t))
