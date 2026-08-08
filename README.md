# LOG-Tensor

## Geometric Tensor Transformers — Guidance-System-Inspired Attention Mechanics

LOG-Tensor is a tensor research engine that reconceives transformers as **guidance systems** rather than inference engines. Drawing on missile navigation theory — proportional navigation, Kalman filtering, and optimal control — it implements attention mechanisms that *home in on* target meaning the way a guided missile intercepts its target.

### The Core Paradigm Shift

Standard transformers process input through layered attention and feed-forward networks, treating inference as a sequence of transformations. LOG-Tensor reframes this:

| Standard Transformer | LOG-Tensor Guidance View |
|---|---|
| Prompt as input | Prompt as **target** to home in on |
| Tokens as data | Tokens as **sensor observations** (noisy, ambiguous) |
| Attention as weighting | Attention as **guidance commands** navigating semantic space |
| Fixed depth | Reasoning depth **decreases** as certainty increases |
| Static payload processing | Real-time **feed processing** (like a tracking system) |

### Key Equation

The Homing Geometric Transformer (HGT) extends the Unified Geometric Transformer (UGT) with a proportional navigation term:

```
Attention = softmax(⟨Q, K⟩ + ω·(Q ∧ K) + N·Vc·λ̇)
```

Where:
- `⟨Q, K⟩` — geometric inner product (rotation invariant)
- `ω·(Q ∧ K)` — bivector coupling (equivariance from Clifford algebra)
- `N·Vc·λ̇` — proportional navigation term (homing guidance)

---

## Architecture

```
logtensor/
├── transforms/           # Core transformer architectures
│   ├── ugt.py            # Unified Geometric Transformer (Clifford algebra Cl(3,0))
│   ├── hgt.py            # Homing Geometric Transformer (production)
│   ├── hgt_research.py   # HGT research version (6DOF semantic manifold)
│   ├── permutation.py    # Permutation Tensor Transformer (Rubik's cube paradigm)
│   ├── rubiks.py         # Rubik's Tensor Transformer (production, PyTorch)
│   ├── rubiks_foundation.py  # Bedrock permutation group mathematics
│   ├── synthesis.py      # Unified architecture synthesis
│   ├── advanced_foundations.py  # Information geometry, spin geometry, category theory
│   ├── physics_tensor.py # Physics-grounded tensor design (quaternions, Hamiltonians)
│   └── sacred_geometry.py # Sacred geometry architecture (tetrahedral, icosahedral)
│
├── decomp/               # Tensor & matrix decomposition
│   ├── cp.py             # CP (CANDECOMP/PARAFAC) decomposition
│   ├── tucker.py         # Tucker decomposition (higher-order SVD)
│   ├── tensor_train.py   # Tensor Train (Matrix Product State)
│   ├── svd.py            # SVD with diagnostics
│   ├── eig.py            # Eigenvalue decomposition (stable)
│   ├── qr.py             # QR decomposition (with pivoting)
│   └── schur.py          # Schur decomposition
│
├── research/             # Multi-model research simulators
│   ├── deepseek_engine.py     # DeepSeek API simulation engine
│   ├── multi_api_simulator.py # Multi-API (DeepSeek, Kimi, DeepInfra) framework
│   ├── batch_processor.py     # Batch execution with progress saving
│   ├── multi_model_simulator.py
│   ├── multi_model_engine.py
│   ├── simulation_framework.py
│   ├── cross_domain.py        # Cross-domain synergy (SE(3), Lie groups, Wigner-D)
│   ├── novel_schemas.py       # Novel simulation schemas
│   ├── schema_simulations.py
│   ├── novel_simulations.py
│   ├── sequential_processor.py
│   ├── optimized_processor.py
│   ├── iterative_refinement.py
│   └── multi_round.py
│
├── reports/              # Report generators
│   ├── comprehensive.py
│   ├── novel_findings.py
│   ├── novel_report.py
│   ├── qgt_report.py
│   ├── se3_report.py
│   ├── simulation_report.py
│   ├── spin_report.py
│   └── synergy_report.py
│
└── utils/                # Tile libraries and tools
    ├── tile_library.py       # Composable mathematical tiles (Tier 0-2)
    ├── tile_library_v2.py
    ├── tile_generator.py
    ├── permutation_research.py
    ├── symmetric_tiles.py
    └── sensor_tiles.py
```

---

## Installation

```bash
pip install -e .

# With optional dependencies:
pip install -e ".[torch,scipy]"    # For PyTorch models and scientific computing
pip install -e ".[full]"           # Everything
pip install -e ".[dev]"            # Development tools
```

**Requirements:** Python ≥3.9, NumPy ≥1.21

---

## Quick Start

### Unified Geometric Transformer (UGT)

The foundational equation — all geometric transformer variants reduce to:

```python
from logtensor.transforms.ugt import GeometricProduct, BivectorConnection

# Geometric product: ab = a·b + a∧b
scalar, bivector = GeometricProduct.geometric(query, key)
```

### Homing Geometric Transformer (HGT)

Treats attention as proportional navigation:

```python
from logtensor.transforms.hgt import LineOfSight, HomingGeometricTransformer

los = LineOfSight()
drift = los.rate(los.compute(current_state, target_meaning), velocity)
# Acceleration command = N × V_closing × LOS_rate
```

### Tensor Decomposition

```python
from logtensor.decomp import stable_svd, cp_decompose, tucker_decompose

result = stable_svd(matrix)
# result.U, result.S, result.Vt, result.rank, result.condition_number
```

### Rubik's Tensor Transformer

Permutation-equivariant transformer with certainty-encoded layer removal:

```python
from logtensor.transforms.rubiks import CertainTensor
# Layers are REMOVED as certainty increases:
# L(c) = ⌊L_max · (1 - mean(c))²⌋
```

---

## Key Concepts

### 1. Transformers as Guidance Systems

The Homing Geometric Transformer applies three layers from missile guidance:

- **Perception (Kalman Filter):** Filters noisy token embeddings, estimates true semantic state, quantifies uncertainty
- **Strategy (Proportional Navigation):** Computes semantic Line-of-Sight, calculates drift rate, generates attention commands
- **Execution (Control Theory):** Adjusts attention weights, manages reasoning depth, achieves "intercept" with target meaning

### 2. Geometric Algebra Foundation

Built on Clifford algebra Cl(3,0). The geometric product unifies the inner product (rotation-invariant scalar) and exterior product (bivector encoding rotation plane) into a single operation.

### 3. Permutation Structure

Inspired by the Rubik's cube: 43 quintillion states, all connected through constrained permutations. The Permutation Tensor Transformer makes tensor elements *dependent* rather than independent, creating meaningful structure through constraints (parity, orientation).

### 4. Certainty-Encoded Depth

Unlike standard transformers with fixed depth, LOG-Tensor **removes layers** as certainty increases. Early layers explore possibilities; later layers merely confirm. Each removed layer saves computation.

### 5. Decomposition Toolkit

Seven decomposition methods (CP, Tucker, TT, SVD, EIG, QR, Schur) with numerical stability focus, regularization, and iterative refinement.

---

## Research Tools

The `research` module includes multi-model simulation frameworks for large-scale experiments:

- **DeepSeek Engine:** Async batch queries with iterative deepening
- **Multi-API Simulator:** Orchestrate multiple LLM APIs as "ghost tiles"
- **Cross-Domain Synergy:** Unifies SE(3) equivariance, Lie group neural networks, Wigner-D harmonics, and quaternion networks
- **Novel Schemas:** Discrete rotation group discovery, quaternion Wigner-D decomposition, conjugacy class attention

Set `DEEPSEEK_API_KEY` (and optionally `KIMI_API_KEY`, `DEEPINFRA_API_KEY`) as environment variables to use the research tools.

---

## Ecosystem Context

LOG-Tensor originated within the POLLN research collective, which explores novel AI architectures through multi-model collaborative research. This standalone package extracts the core tensor transformer implementations for broader use. The research artifacts (JSON results, PDFs, multi-language tiles in Haskell/F#/Julia/Rust) remain in the original POLLN repository.

---

## License

MIT
