"""Quantization utilities for post-training quantization experiments.

Provides uniform (linear) quantization with optional scaling,
symmetric and asymmetric modes, and evaluation metrics.
"""

from typing import Optional, Tuple, Union

import torch


class UniformQuantizer:
    """Uniform (linear) quantizer with configurable bit-width and mode.

    Supports both symmetric (zero-centered) and asymmetric (min-max) quantization.
    """

    def __init__(
        self,
        n_bits: int = 4,
        symmetric: bool = True,
        per_channel: bool = False,
    ):
        if n_bits < 1:
            raise ValueError("n_bits must be >= 1")
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.per_channel = per_channel

        if symmetric:
            n_levels = 2 ** (n_bits - 1)
            self.q_min = -n_levels
            self.q_max = n_levels - 1
        else:
            self.q_min = 0
            self.q_max = 2**n_bits - 1

    def _compute_scale_zero_point(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scale and zero-point for quantization."""
        if self.per_channel and x.dim() > 1:
            # Per-output-channel quantization (along dim=0)
            if self.symmetric:
                max_abs = x.abs().reshape(x.shape[0], -1).max(dim=1).values
                scale = max_abs / self.q_max
                zero_point = torch.zeros_like(scale)
            else:
                x_2d = x.reshape(x.shape[0], -1)
                x_min = x_2d.min(dim=1).values
                x_max = x_2d.max(dim=1).values
                scale = (x_max - x_min) / (self.q_max - self.q_min)
                zero_point = torch.round(-x_min / scale).to(torch.int32)
                zero_point = torch.clamp(zero_point, self.q_min, self.q_max)
        else:
            # Tensor-wise quantization
            if self.symmetric:
                max_abs = x.abs().max()
                scale = max_abs / self.q_max if max_abs > 0 else torch.tensor(1.0)
                zero_point = torch.tensor(0, dtype=torch.int32)
            else:
                x_min, x_max = x.min(), x.max()
                if x_max == x_min:
                    scale = torch.tensor(1.0)
                    zero_point = torch.tensor(0, dtype=torch.int32)
                else:
                    scale = (x_max - x_min) / (self.q_max - self.q_min)
                    zero_point = torch.round(-x_min / scale).to(torch.int32)
                    zero_point = torch.clamp(zero_point, self.q_min, self.q_max)

        return scale, zero_point

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize a tensor to integers.

        Args:
            x: Input tensor.

        Returns:
            Tuple of (quantized_ints, scale, zero_point).
        """
        scale, zero_point = self._compute_scale_zero_point(x)
        # Need to handle broadcasting for per-channel
        if self.per_channel and x.dim() > 1:
            scale_expanded = scale.reshape(-1, *([1] * (x.dim() - 1)))
            zp_expanded = zero_point.reshape(-1, *([1] * (x.dim() - 1)))
        else:
            scale_expanded = scale
            zp_expanded = zero_point

        q_ints = torch.round(x / scale_expanded) + zp_expanded
        q_ints = torch.clamp(q_ints, self.q_min, self.q_max).to(torch.int32)
        return q_ints, scale, zero_point

    def dequantize(
        self, q_ints: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
    ) -> torch.Tensor:
        """Dequantize integer values back to floating point.

        Args:
            q_ints: Quantized integer values.
            scale: Scale factor(s).
            zero_point: Zero point(s).

        Returns:
            Dequantized float tensor.
        """
        if self.per_channel and q_ints.dim() > 1:
            scale_expanded = scale.reshape(-1, *([1] * (q_ints.dim() - 1)))
            zp_expanded = zero_point.reshape(-1, *([1] * (q_ints.dim() - 1)))
        else:
            scale_expanded = scale
            zp_expanded = zero_point
        return (q_ints.float() - zp_expanded) * scale_expanded

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize then dequantize (simulated quantization).

        Uses Straight-Through Estimator (STE): during forward pass,
        applies quantization; during backward pass, gradients pass
        through as if quantization was identity.
        """
        q_ints, scale, zp = self.quantize(x)
        x_q = self.dequantize(q_ints, scale, zp)
        # STE: detach quantized path, add residual with gradient flow
        return x + (x_q - x).detach()

    def __repr__(self) -> str:
        mode = "sym" if self.symmetric else "asym"
        pc = ", per-channel" if self.per_channel else ""
        return f"UniformQuantizer({self.n_bits}b, {mode}{pc})"


def quantize(
    x: torch.Tensor,
    n_bits: int = 4,
    symmetric: bool = True,
    per_channel: bool = False,
) -> torch.Tensor:
    """Convenience function: quantize and dequantize in one step."""
    q = UniformQuantizer(n_bits, symmetric, per_channel)
    return q(x)


def quantization_error(
    x: torch.Tensor, x_q: torch.Tensor, relative: bool = True
) -> torch.Tensor:
    """Compute MSE quantization error.

    Args:
        x: Original tensor.
        x_q: Quantized (and dequantized) tensor.
        relative: If True, return relative MSE (NMSE).

    Returns:
        (N)MSE value.
    """
    mse = torch.nn.functional.mse_loss(x, x_q)
    if relative:
        return mse / (x.pow(2).mean() + 1e-10)
    return mse


def quantize_weight_matrix(
    W: torch.Tensor,
    n_bits: int = 4,
    symmetric: bool = True,
    per_channel: bool = False,
    rotation_matrix: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Quantize a weight matrix with optional pre-rotation.

    If rotation_matrix is provided:
        W_q = Q(W @ R^T) @ R
    where Q() is quantization+dequantization and R is orthogonal.
    The rotation is applied to the input (column) space of W.

    Args:
        W: Weight matrix of shape (out_dim, in_dim).
        n_bits: Quantization bit-width.
        symmetric: Use symmetric quantization.
        per_channel: Quantize per output channel.
        rotation_matrix: Optional orthogonal matrix of shape (in_dim, in_dim).

    Returns:
        Quantized weight matrix.
    """
    W_ = W.clone()

    if rotation_matrix is not None:
        # Rotate columns of W (input space): W' = W @ R^T
        W_ = W_ @ rotation_matrix.T

    q = UniformQuantizer(n_bits, symmetric, per_channel)
    W_q = q(W_)

    if rotation_matrix is not None:
        # Undo rotation: W_q @ R
        W_q = W_q @ rotation_matrix

    return W_q
