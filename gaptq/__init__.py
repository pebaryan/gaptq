"""Core PTQ-facing exports for GAP-TQ.

The package keeps the public surface focused on the main learned-rotation
quantization path. Experimental branches live under gaptq.experimental.
"""

from .ga import rotor_from_angle, rotor_matrix_2d, block_diag_rotor_matrix
from .quantization import UniformQuantizer, quantize, quantization_error
from .rotor_quant import optimize_rotor_angles, RotorQuaRot, quantize_with_rotors

__version__ = "0.1.0"
