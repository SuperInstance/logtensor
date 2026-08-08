"""Tests for logtensor package."""
import numpy as np
import pytest


def test_import():
    """Test that the package imports cleanly."""
    import logtensor
    assert logtensor.__version__ == "0.1.0"


def test_decp_imports():
    """Test decomposition module imports."""
    from logtensor.decomp import stable_svd, stable_eig, stable_qr
    assert stable_svd is not None
    assert stable_eig is not None
    assert stable_qr is not None


def test_svd():
    """Test SVD decomposition."""
    from logtensor.decomp import stable_svd
    np.random.seed(42)
    A = np.random.randn(10, 8)
    result = stable_svd(A)
    assert result.U.shape == (10, 10)
    assert result.S.shape == (8,)
    assert result.Vt.shape == (8, 8)
    assert result.rank > 0
    # Reconstruction
    reconstructed = result.U[:, :8] @ np.diag(result.S) @ result.Vt
    assert np.allclose(reconstructed, A, atol=1e-10)


def test_eig():
    """Test eigenvalue decomposition."""
    from logtensor.decomp import stable_eig
    np.random.seed(42)
    A = np.random.randn(6, 6)
    A = A + A.T  # Make symmetric
    result = stable_eig(A, assume_hermitian=True)
    assert result.values.shape == (6,)
    assert result.is_hermitian
    # Reconstruction
    reconstructed = result.vectors @ np.diag(result.values) @ result.vectors.T
    assert np.allclose(reconstructed, A, atol=1e-10)


def test_qr():
    """Test QR decomposition."""
    from logtensor.decomp import stable_qr
    np.random.seed(42)
    A = np.random.randn(8, 5)
    result = stable_qr(A)
    assert result.Q.shape[0] == 8
    assert result.R.shape[1] == 5
    # Q should be orthogonal
    assert np.allclose(result.Q.T @ result.Q, np.eye(result.Q.shape[1]), atol=1e-10)


def test_cp_decomposition():
    """Test CP decomposition."""
    from logtensor.decomp.cp import khatri_rao_product, cp_als
    A = np.random.randn(4, 3)
    B = np.random.randn(5, 3)
    KR = khatri_rao_product(A, B)
    assert KR.shape == (20, 3)


def test_tucker():
    """Test Tucker decomposition."""
    from logtensor.decomp.tucker import mode_n_product
    tensor = np.random.randn(4, 5, 6)
    matrix = np.random.randn(3, 5)
    result = mode_n_product(tensor, matrix, mode=1)
    assert result.shape == (4, 3, 6)


def test_geometric_product():
    """Test geometric algebra operations."""
    from logtensor.transforms.ugt import GeometricProduct
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    scalar = GeometricProduct.inner(a, b)
    bivector = GeometricProduct.outer(a, b)
    assert scalar == 0.0  # Orthogonal
    assert np.allclose(bivector, [0, 0, 1])  # z-axis cross product


def test_ugt_config():
    """Test UGT configuration."""
    from logtensor.transforms.ugt import UGTConfig
    config = UGTConfig(dim=128, n_heads=4)
    assert config.dim == 128
    assert config.n_heads == 4
    assert config.geometric_dim == 3
