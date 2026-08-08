"""Tensor and matrix decomposition tools."""

from .cp import CPResult, khatri_rao_product, cp_als
from .svd import SVDResult, stable_svd
from .eig import EigResult, stable_eig
from .qr import QRResult, stable_qr
from .tucker import TuckerResult, mode_n_product, hosvd
from .tensor_train import TTResult, tt_svd
from .schur import SchurResult, schur_decomposition

__all__ = [
    "CPResult", "khatri_rao_product", "cp_als",
    "SVDResult", "stable_svd",
    "EigResult", "stable_eig",
    "QRResult", "stable_qr",
    "TuckerResult", "mode_n_product", "hosvd",
    "TTResult", "tt_svd",
    "SchurResult", "schur_decomposition",
]
