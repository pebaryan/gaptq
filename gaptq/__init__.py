"""gaptq: Geometric Algebra Post-Training Quantization.

Research experiment exploring the use of geometric algebra rotors
for post-training quantization of neural networks, inspired by QuaRot.
"""

from .ga import (
    MultiVector,
    rotor_from_angle,
    rotor_matrix_2d,
    block_diag_rotor_matrix,
    RandomRotorTransform,
)
from .quantization import (
    UniformQuantizer,
    quantize,
    quantization_error,
)
from .rotor_quant import (
    optimize_rotor_angles,
    RotorQuaRot,
    quantize_with_rotors,
)

__version__ = "0.1.0"
