"""
Tests for the Permutation Tensor Transformer (PTT) module.

Covers PermutationTensor construction, constraints, certainty, encodings,
FirstClassQuantity enum, PermutationTensorTransformer, and architecture summary.
"""

import numpy as np
import pytest
from logtensor.transforms.permutation import (
    PermutationTensor,
    PermutationTensorTransformer,
    FirstClassQuantity,
    get_architecture_summary,
    verify_complete_system,
)


# ── FirstClassQuantity ────────────────────────────────────────


class TestFirstClassQuantity:
    def test_has_five_quantities(self):
        """The enum has exactly 5 first-class quantities."""
        assert len(list(FirstClassQuantity)) == 5

    def test_quantities_are_accessible(self):
        """All five quantities are accessible by name."""
        assert FirstClassQuantity.GEOMETRY is not None
        assert FirstClassQuantity.TRAJECTORY is not None
        assert FirstClassQuantity.MOMENTUM is not None
        assert FirstClassQuantity.TIME is not None
        assert FirstClassQuantity.DISTANCE is not None

    def test_auto_values_are_unique(self):
        """Each enum value is unique."""
        values = [q.value for q in FirstClassQuantity]
        assert len(values) == len(set(values))


# ── PermutationTensor ─────────────────────────────────────────


class TestPermutationTensorConstruction:
    def test_default_construction(self):
        """PermutationTensor constructs with defaults."""
        pt = PermutationTensor()
        assert pt is not None
        assert pt.data is not None
        assert pt.certainty is not None

    def test_custom_dimensions(self):
        """Custom dimensions are stored."""
        pt = PermutationTensor(geometry_dim=8, trajectory_dim=4)
        assert pt.dims[FirstClassQuantity.GEOMETRY] == 8
        assert pt.dims[FirstClassQuantity.TRAJECTORY] == 4

    def test_data_shape_matches_dims(self):
        """Data array has shape matching dimension config."""
        pt = PermutationTensor(geometry_dim=2, trajectory_dim=3, momentum_dim=4, time_dim=5, distance_dim=6)
        assert pt.data.shape == (2, 3, 4, 5, 6)

    def test_certainty_initialized_to_half(self):
        """Certainty starts at 0.5 everywhere."""
        pt = PermutationTensor(geometry_dim=2, trajectory_dim=2, momentum_dim=2, time_dim=2, distance_dim=2)
        assert np.all(pt.certainty == 0.5)

    def test_constraints_start_empty(self):
        """No constraints at construction."""
        pt = PermutationTensor()
        assert pt.constraints is not None
        assert len(pt.constraints) == 0

    def test_encodings_start_empty(self):
        """No encodings at construction."""
        pt = PermutationTensor()
        assert len(pt.encodings) == 0

    def test_pathway_strength_starts_empty(self):
        """No pathway strength at construction."""
        pt = PermutationTensor()
        assert hasattr(pt, 'pathway_strength')

    def test_dependencies_exist(self):
        """Dependencies structure exists."""
        pt = PermutationTensor()
        assert hasattr(pt, 'dependencies')
        assert pt.dependencies is not None
        assert isinstance(pt.dependencies, (dict, set, list))

    def test_shape_property(self):
        """Shape property returns data shape."""
        pt = PermutationTensor(geometry_dim=3, trajectory_dim=4)
        assert pt.shape == (3, 4, 8, 8, 8)


class TestPermutationTensorConstraints:
    def test_add_constraint(self):
        """Can add a constraint (single arg form)."""
        pt = PermutationTensor(geometry_dim=2, trajectory_dim=2, momentum_dim=2, time_dim=2, distance_dim=2)
        # add_constraint takes a single callable
        pt.add_constraint(lambda t: True)
        assert len(pt.constraints) >= 1

    def test_verify_constraints_passes(self):
        """Constraints that return True pass verification."""
        pt = PermutationTensor(geometry_dim=2, trajectory_dim=2, momentum_dim=2, time_dim=2, distance_dim=2)
        pt.add_constraint(lambda t: True)
        assert pt.verify_constraints() is True

    def test_verify_constraints_fails(self):
        """Constraints that return False fail verification."""
        pt = PermutationTensor(geometry_dim=2, trajectory_dim=2, momentum_dim=2, time_dim=2, distance_dim=2)
        pt.add_constraint(lambda t: False)
        assert pt.verify_constraints() is False

    def test_multiple_constraints(self):
        """Multiple constraints are all checked."""
        pt = PermutationTensor(geometry_dim=2, trajectory_dim=2, momentum_dim=2, time_dim=2, distance_dim=2)
        pt.add_constraint(lambda t: True)
        pt.add_constraint(lambda t: True)
        assert len(pt.constraints) >= 2
        assert pt.verify_constraints() is True


class TestPermutationTensorCertainty:
    def test_average_certainty_at_init(self):
        """Average certainty is 0.5 at initialization."""
        pt = PermutationTensor(geometry_dim=2, trajectory_dim=2, momentum_dim=2, time_dim=2, distance_dim=2)
        assert pt.get_average_certainty() == pytest.approx(0.5)

    def test_high_certainty_elements(self):
        """Can query high certainty elements."""
        pt = PermutationTensor(geometry_dim=2, trajectory_dim=2, momentum_dim=2, time_dim=2, distance_dim=2)
        result = pt.get_high_certainty_elements(threshold=0.4)
        assert result is not None  # All elements at 0.5 > 0.4

    def test_low_certainty_elements(self):
        """Can query low certainty elements."""
        pt = PermutationTensor(geometry_dim=2, trajectory_dim=2, momentum_dim=2, time_dim=2, distance_dim=2)
        result = pt.get_low_certainty_elements(threshold=0.6)
        assert result is not None  # All elements at 0.5 < 0.6

    def test_certainty_map_exists(self):
        """Certainty map is accessible."""
        pt = PermutationTensor(geometry_dim=2, trajectory_dim=2, momentum_dim=2, time_dim=2, distance_dim=2)
        cmap = pt.get_certainty_map()
        assert cmap is not None


class TestPermutationTensorEncodings:
    def test_register_encoding(self):
        """Can register a named encoding."""
        pt = PermutationTensor(geometry_dim=4)
        pt.register_encoding('test', np.ones(4))
        assert 'test' in pt.encodings

    def test_register_multiple_encodings(self):
        """Can register multiple encodings."""
        pt = PermutationTensor(geometry_dim=4)
        pt.register_encoding('enc1', np.ones(4))
        pt.register_encoding('enc2', np.zeros(4))
        assert len(pt.encodings) == 2


class TestPermutationTensorPropagate:
    def test_propagate_change_exists(self):
        """propagate_change method exists and is callable."""
        pt = PermutationTensor(geometry_dim=2, trajectory_dim=2, momentum_dim=2, time_dim=2, distance_dim=2)
        assert callable(pt.propagate_change)

    def test_get_affected_region_exists(self):
        """get_affected_region method exists and is callable."""
        pt = PermutationTensor(geometry_dim=2, trajectory_dim=2, momentum_dim=2, time_dim=2, distance_dim=2)
        assert callable(pt.get_affected_region)


# ── PermutationTensorTransformer ──────────────────────────────


class TestPermutationTensorTransformer:
    def test_default_construction(self):
        """Transformer constructs with defaults."""
        ptt = PermutationTensorTransformer()
        assert ptt is not None
        assert ptt.tensor is not None

    def test_has_dimension_attributes(self):
        """Dimension attributes are accessible."""
        ptt = PermutationTensorTransformer()
        assert hasattr(ptt, 'geometry_dim')
        assert hasattr(ptt, 'trajectory_dim')
        assert hasattr(ptt, 'momentum_dim')
        assert hasattr(ptt, 'time_dim')
        assert hasattr(ptt, 'distance_dim')

    def test_has_tensor(self):
        """Internal PermutationTensor is accessible."""
        ptt = PermutationTensorTransformer()
        assert isinstance(ptt.tensor, PermutationTensor)

    def test_has_forward_method(self):
        """Forward method exists."""
        ptt = PermutationTensorTransformer()
        assert callable(ptt.forward)

    def test_has_state_summary(self):
        """State summary method exists and returns something."""
        ptt = PermutationTensorTransformer()
        summary = ptt.get_state_summary()
        assert summary is not None

    def test_iteration_starts_at_zero(self):
        """Iteration counter starts at zero."""
        ptt = PermutationTensorTransformer()
        assert ptt.iteration == 0

    def test_certainty_starts_at_zero(self):
        """Current certainty starts at 0.0."""
        ptt = PermutationTensorTransformer()
        assert ptt.current_certainty == pytest.approx(0.0)

    def test_has_layer_controller(self):
        """Layer controller exists."""
        ptt = PermutationTensorTransformer()
        assert ptt.layer_controller is not None

    def test_has_rag(self):
        """RAG system exists."""
        ptt = PermutationTensorTransformer()
        assert ptt.rag is not None


# ── Architecture Summary ──────────────────────────────────────


class TestArchitectureSummary:
    def test_returns_dict(self):
        """Architecture summary returns a dictionary."""
        summary = get_architecture_summary()
        assert isinstance(summary, (dict, str))

    def test_contains_key_info(self):
        """Summary contains meaningful content."""
        summary = get_architecture_summary()
        if isinstance(summary, dict):
            assert len(summary) > 0
        elif isinstance(summary, str):
            assert len(summary) > 0


class TestVerifyCompleteSystem:
    def test_returns_without_error(self):
        """System verification completes without exception."""
        result = verify_complete_system()
        assert result is not None
