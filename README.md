# GAP-TQ: Geometric Algebra Post-Training Quantization

A research experiment exploring whether **geometric algebra rotors** can improve post-training quantization of neural networks, inspired by [QuaRot](https://github.com/spcl/QuaRot).

## Core Idea

**QuaRot** applies random orthogonal rotations (via Hadamard matrices) to weight matrices before quantization, making weight distributions more uniform and reducing quantization error.

**GAP-TQ** replaces the fixed Hadamard rotations with **learnable rotors** from geometric algebra (Clifford algebra). Rotors naturally represent rotations and are parameterized by **differentiable angles** that can be optimized to **minimize quantization error** on calibration data.

### Key Advantages Over Hadamard

| Feature | Hadamard (QuaRot) | Rotors (GAP-TQ) |
|---------|-------------------|-----------------|
| Parameterization | Fixed matrix | Differentiable angles |
| Optimization | Not possible | Gradient-based optimization |
| Block structure | Power-of-2 blocks | Arbitrary 2D planes |

## Project Structure

```
gaptq/
├── __init__.py          # Package exports
├── ga.py                # Geometric algebra primitives
│   ├── MultiVector      # Multivector in Cl(n,0)
│   ├── rotor_from_angle # Create rotor for 2D plane
│   ├── apply_rotor_to_vector  # R * v * R†
│   ├── rotor_matrix_2d  # 2x2 rotation matrix from rotor
│   ├── block_diag_rotor_matrix  # Compose rotors into block-diagonal matrix
│   └── RandomRotorTransform  # Trainable rotor transform
├── quantization.py      # Quantization utilities
│   ├── UniformQuantizer # Configurable uniform quantizer
│   ├── quantize         # Convenience function
│   ├── quantization_error  # NMSE computation
│   └── quantize_weight_matrix  # Rotated quantization
├── rotor_quant.py       # Rotor-based QuaRot implementation
│   ├── optimize_rotor_angles  # Gradient-based optimization
│   ├── RotorQuaRot      # Quantization module with learnable rotors
│   └── quantize_with_rotors  # One-shot quantization
└── experiment.py        # Comparison experiment
    ├── run_quantization_experiment  # Single matrix comparison
    ├── run_batch_experiment  # Multi-matrix aggregation
    └── summarize_results  # Print summary statistics

tests/
└── test_ga.py           # 20 tests for GA primitives
```

## Quick Start

```bash
pip install torch numpy matplotlib scipy pytest

# Run tests
python -m pytest tests/ -v

# Run experiment
python -m gaptq.experiment
```

## Methods Compared

1. **RTN** (Round-to-Nearest): Direct quantization, no rotation
2. **Orthogonal**: Random orthogonal matrix (QuaRot-style Hadamard or QR fallback)
3. **Random Rotors**: Random rotor-based block-diagonal rotation
4. **Optimized Rotors**: Gradient-optimized rotor angles
5. **Best-of-N Random**: Best of N random rotor configurations

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Matplotlib (optional, for plots)
- SciPy (optional)

## References

- [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456)
- [Geometric Algebra for Physicists](https://www.cambridge.org/core/books/geometric-algebra-for-physicists/) - Doran & Lasenby
