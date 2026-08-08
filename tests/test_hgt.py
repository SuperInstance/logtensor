"""Tests for HGT (Homing Geometric Transformer) module."""

import numpy as np
import pytest

from logtensor.transforms.hgt import (
    GeometricProduct,
    LineOfSight,
)


class TestGeometricProductExtended:
    """Extended tests for geometric algebra operations."""

    def test_inner_product_parallel_vectors(self):
        """Parallel vectors should have maximum inner product."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert GeometricProduct.inner(a, b) == pytest.approx(1.0)

    def test_inner_product_anti_parallel(self):
        """Anti-parallel vectors should have negative inner product."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        assert GeometricProduct.inner(a, b) == pytest.approx(-1.0)

    def test_inner_product_arbitrary(self):
        """Inner product should match numpy dot."""
        np.random.seed(42)
        a = np.random.randn(3)
        b = np.random.randn(3)
        assert GeometricProduct.inner(a, b) == pytest.approx(np.dot(a, b))

    def test_outer_product_parallel_is_zero(self):
        """Parallel vectors should produce zero bivector."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([2.0, 0.0, 0.0])
        result = GeometricProduct.outer(a, b)
        assert np.allclose(result, 0.0)

    def test_outer_product_orthogonal(self):
        """Orthogonal unit vectors should produce unit bivector."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        result = GeometricProduct.outer(a, b)
        assert np.allclose(result, [0.0, 0.0, 1.0])

    def test_outer_product_antisymmetric(self):
        """Outer product should be antisymmetric: a∧b = -(b∧a)."""
        np.random.seed(123)
        a = np.random.randn(3)
        b = np.random.randn(3)
        ab = GeometricProduct.outer(a, b)
        ba = GeometricProduct.outer(b, a)
        assert np.allclose(ab, -ba)


class TestLineOfSight:
    """Tests for the semantic Line-of-Sight computation."""

    def test_los_computation(self):
        """LOS should be the difference vector."""
        los = LineOfSight()
        current = np.array([1.0, 0.0, 0.0])
        target = np.array([0.0, 1.0, 0.0])
        result = los.compute(current, target)
        assert np.allclose(result, [-1.0, 1.0, 0.0])

    def test_los_zero_when_aligned(self):
        """LOS should be zero when current equals target."""
        los = LineOfSight()
        v = np.array([1.0, 2.0, 3.0])
        result = los.compute(v, v)
        assert np.allclose(result, 0.0)

    def test_los_rate(self):
        """LOS rate should compute without error."""
        los = LineOfSight()
        los_vec = np.array([1.0, 0.0, 0.0])
        velocity = np.array([0.0, 1.0, 0.0])
        rate = los.rate(los_vec, velocity)
        assert isinstance(rate, (int, float, np.floating))


class TestHGTImports:
    """Test that HGT module imports are clean."""

    def test_hgt_module_imports(self):
        """All key HGT classes should import without error."""
        from logtensor.transforms import hgt
        assert hasattr(hgt, 'GeometricProduct')
        assert hasattr(hgt, 'LineOfSight')

    def test_hgt_geometric_product_type(self):
        """GeometricProduct should be a class."""
        from logtensor.transforms.hgt import GeometricProduct
        assert isinstance(GeometricProduct, type)
