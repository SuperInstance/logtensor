"""
Tests for the Unified Geometric Transformer (UGT) module.

Covers GeometricProduct, BivectorConnection, UGTConfig, UnifiedAttention,
and the UnifiedGeometricTransformer wrapper.
"""

import numpy as np
import pytest
from logtensor.transforms.ugt import (
    GeometricProduct,
    BivectorConnection,
    UGTConfig,
    UnifiedAttention,
    UnifiedGeometricTransformer,
    SymplecticMomentum,
    ChernSimonsRegularizer,
    QDeformation,
    RGFlowScheduler,
)


# ── GeometricProduct ───────────────────────────────────────────


class TestGeometricProduct:
    def test_inner_product_parallel_vectors(self):
        """Inner product of parallel unit vectors is 1."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert GeometricProduct.inner(a, b) == pytest.approx(1.0)

    def test_inner_product_anti_parallel(self):
        """Inner product of anti-parallel vectors is -1."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        assert GeometricProduct.inner(a, b) == pytest.approx(-1.0)

    def test_inner_product_orthogonal_is_zero(self):
        """Inner product of orthogonal vectors is 0."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert GeometricProduct.inner(a, b) == pytest.approx(0.0)

    def test_outer_product_parallel_is_zero(self):
        """Outer product of parallel vectors is the zero bivector."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([2.0, 0.0, 0.0])
        result = GeometricProduct.outer(a, b)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-10)

    def test_outer_product_orthogonal(self):
        """Outer product of x̂ and ŷ is ẑ (via cross product)."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        result = GeometricProduct.outer(a, b)
        np.testing.assert_allclose(result, [0.0, 0.0, 1.0], atol=1e-10)

    def test_outer_product_antisymmetric(self):
        """a∧b = -(b∧a)."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        lhs = GeometricProduct.outer(a, b)
        rhs = GeometricProduct.outer(b, a)
        np.testing.assert_allclose(lhs, -rhs, atol=1e-10)

    def test_geometric_product_returns_tuple(self):
        """Full geometric product returns (scalar, bivector)."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 1.0, 0.0])
        scalar, bivec = GeometricProduct.geometric(a, b)
        assert scalar == pytest.approx(1.0)
        assert bivec.shape == (3,)

    def test_inner_product_symmetric(self):
        """Inner product is symmetric: a·b = b·a."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        assert GeometricProduct.inner(a, b) == pytest.approx(GeometricProduct.inner(b, a))

    def test_inner_product_scaling(self):
        """Scaling one vector scales the inner product."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([3.0, 0.0, 0.0])
        assert GeometricProduct.inner(a, b) == pytest.approx(3.0)


# ── BivectorConnection ────────────────────────────────────────


class TestBivectorConnection:
    def test_omega_initialization(self):
        """Bivector connection initializes omega vector of correct dimension."""
        bc = BivectorConnection(dim=3, init_scale=0.01)
        assert bc.omega.shape == (3,)

    def test_omega_small_at_init(self):
        """At small init_scale, omega values are close to zero."""
        bc = BivectorConnection(dim=3, init_scale=0.001)
        assert np.all(np.abs(bc.omega) < 0.1)

    def test_call_returns_scalar(self):
        """Calling the connection on a bivector returns a scalar."""
        bc = BivectorConnection(dim=3, init_scale=0.01)
        bivec = np.array([1.0, 0.0, 0.0])
        result = bc(bivec)
        assert isinstance(result, (float, np.floating))

    def test_update_modifies_omega(self):
        """Update method modifies omega values."""
        bc = BivectorConnection(dim=3, init_scale=0.0)
        original = bc.omega.copy()
        bc.update(np.array([0.1, 0.2, 0.3]))
        assert not np.allclose(original, bc.omega)


# ── UGTConfig ─────────────────────────────────────────────────


class TestUGTConfig:
    def test_defaults(self):
        """Default config has expected values."""
        cfg = UGTConfig()
        assert cfg.dim == 256
        assert cfg.n_heads == 8
        assert cfg.n_layers == 6
        assert cfg.dropout == 0.1

    def test_custom_config(self):
        """Can create custom configs."""
        cfg = UGTConfig(dim=128, n_heads=4, n_layers=3)
        assert cfg.dim == 128
        assert cfg.n_heads == 4
        assert cfg.n_layers == 3

    def test_geometric_params(self):
        """Geometric parameters are configurable."""
        cfg = UGTConfig(geometric_dim=4, omega_init_scale=0.05)
        assert cfg.geometric_dim == 4
        assert cfg.omega_init_scale == 0.05

    def test_optional_regularizers_default_off(self):
        """Regularizers default to off."""
        cfg = UGTConfig()
        assert cfg.use_chern_simons is False
        assert cfg.use_rg_flow is False
        assert cfg.use_q_deformation is False


# ── UnifiedAttention ──────────────────────────────────────────


class TestUnifiedAttention:
    def test_construction(self):
        """Can construct UnifiedAttention with dimension and heads."""
        attn = UnifiedAttention(dim=64, n_heads=4)
        assert attn.dim == 64
        assert attn.n_heads == 4
        assert attn.head_dim == 16

    def test_weight_shapes(self):
        """Weight matrices have correct shapes."""
        attn = UnifiedAttention(dim=64, n_heads=4)
        assert attn.W_q.shape == (64, 64)
        assert attn.W_k.shape == (64, 64)
        assert attn.W_v.shape == (64, 64)
        assert attn.W_o.shape == (64, 64)

    def test_forward_returns_dict(self):
        """Forward pass returns a dictionary with expected keys."""
        attn = UnifiedAttention(dim=32, n_heads=4)
        x = np.random.randn(8, 32).astype(np.float32)
        result = attn.forward(x)
        assert isinstance(result, dict)
        assert 'output' in result
        assert 'attention' in result

    def test_forward_output_shape(self):
        """Output has the same shape as input (seq_len, dim)."""
        attn = UnifiedAttention(dim=32, n_heads=4)
        x = np.random.randn(8, 32).astype(np.float32)
        result = attn.forward(x)
        assert result['output'].shape == (8, 32)

    def test_attention_shape(self):
        """Attention weights have shape (seq_len, seq_len, n_heads)."""
        attn = UnifiedAttention(dim=32, n_heads=4)
        x = np.random.randn(8, 32).astype(np.float32)
        result = attn.forward(x)
        assert result['attention'].shape == (8, 8, 4)

    def test_attention_rows_sum_to_one(self):
        """Attention weights are normalized (softmax) — rows sum to ~1."""
        attn = UnifiedAttention(dim=32, n_heads=4)
        x = np.random.randn(8, 32).astype(np.float32)
        result = attn.forward(x)
        for h in range(4):
            for i in range(8):
                assert result['attention'][i, :, h].sum() == pytest.approx(1.0, abs=1e-5)

    def test_without_bivector(self):
        """Can disable bivector coupling."""
        attn = UnifiedAttention(dim=32, n_heads=4, use_bivector=False)
        assert attn.use_bivector is False
        x = np.random.randn(4, 32).astype(np.float32)
        result = attn.forward(x)
        assert result['output'].shape == (4, 32)

    def test_single_token(self):
        """Works with a single token (seq_len=1)."""
        attn = UnifiedAttention(dim=32, n_heads=4)
        x = np.random.randn(1, 32).astype(np.float32)
        result = attn.forward(x)
        assert result['output'].shape == (1, 32)


# ── UnifiedGeometricTransformer ───────────────────────────────


class TestUnifiedGeometricTransformer:
    def test_construction(self):
        """Can construct the full transformer from config."""
        cfg = UGTConfig(dim=64, n_heads=4, n_layers=2, max_seq_len=128)
        model = UnifiedGeometricTransformer(cfg)
        assert model is not None

    def test_forward_single_sequence(self):
        """Forward pass on a single sequence (no batch dim)."""
        cfg = UGTConfig(dim=32, n_heads=4, n_layers=2, max_seq_len=64)
        model = UnifiedGeometricTransformer(cfg)
        x = np.random.randn(8, 32).astype(np.float32)
        result = model.forward(x)
        assert isinstance(result, dict)

    def test_config_attributes_preserved(self):
        """Model preserves config attributes."""
        cfg = UGTConfig(dim=48, n_heads=4, n_layers=3)
        model = UnifiedGeometricTransformer(cfg)
        assert model.config.dim == 48


# ── Auxiliary Modules ─────────────────────────────────────────


class TestSymplecticMomentum:
    def test_construction(self):
        """SymplecticMomentum constructs with default params."""
        sm = SymplecticMomentum()
        assert sm is not None


class TestChernSimonsRegularizer:
    def test_construction(self):
        """ChernSimonsRegularizer constructs."""
        cs = ChernSimonsRegularizer()
        assert cs is not None


class TestQDeformation:
    def test_construction(self):
        """QDeformation constructs with a q value."""
        qd = QDeformation(q=1.5)
        assert qd is not None


class TestRGFlowScheduler:
    def test_construction(self):
        """RGFlowScheduler constructs with required params."""
        rg = RGFlowScheduler(omega_0=np.array([0.01, 0.02, 0.03]), n_layers=6)
        assert rg is not None
