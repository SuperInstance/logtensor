"""
LOG-Tensor: Geometric Tensor Transformers
==========================================

A tensor research engine implementing transformers as guidance systems,
inspired by missile navigation theory (proportional navigation, Kalman filtering,
and optimal control).

Core Innovation:
    Treating transformers as guidance systems, not inference engines.

Key Modules:
    transforms  — HGT, UGT, Permutation, Rubiks tensor transformers
    decomp      — CP, Tucker, TT, SVD, QR, EIG, Schur decomposition
    research    — Multi-model simulators and batch processors
    reports     — PDF/report generators
    utils       — Tile libraries and permutation tools
"""

__version__ = "0.1.0"
__author__ = "LOG-Tensor Research Team"
__license__ = "MIT"

__all__ = [
    "transforms",
    "decomp",
    "research",
    "reports",
    "utils",
]
