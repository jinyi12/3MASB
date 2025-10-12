"""
Common Utilities
================

This module contains common utility functions used across different bridge
implementations, such as automatic differentiation helpers.
"""

from typing import Tuple, Callable, Any, Optional
import torch
from torch import Tensor

def jvp(
    f: Callable[[Tensor], Any],
    x: Tensor,
    v: Tensor,
    *,
    create_graph: Optional[bool] = None,
) -> Tuple[Tensor, ...]:
    """Compute Jacobian-vector product. Used for time derivatives."""
    # Ensures gradients can flow back through the JVP calculation if needed
    if create_graph is None:
        create_graph = torch.is_grad_enabled()
    return torch.autograd.functional.jvp(
        f,
        x,
        v,
        create_graph=create_graph,
    )

def t_dir(
    f: Callable[[Tensor], Any],
    t: Tensor,
    *,
    create_graph: Optional[bool] = None,
) -> Tuple[Tensor, ...]:
    """Compute the time derivative of f(t) by using jvp with v=1."""
    return jvp(f, t, torch.ones_like(t), create_graph=create_graph)
