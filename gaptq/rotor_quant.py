"""Rotor-based QuaRot: Post-training quantization using optimized rotors.

The key idea:
1. Initialize a set of 2D rotors (block-diagonal) OR full bivector exponentials
2. Optimize rotor parameters to minimize quantization error on a calibration set
3. Apply the optimal rotation before quantization, invert after

Supports two rotation modes:
  - 'block': Block-diagonal 2D rotors (efficient, QuaRot-style)
  - 'full': Full matrix exponential of bivectors (from gamuon, more expressive)

This provides a *learnable* alternative to the fixed Hadamard rotations in QuaRot.
"""

from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F

from .ga import (
    CliffordAlgebra,
    RandomReflectionTransform,
    RandomRotorTransform,
    block_diag_rotor_matrix,
    bivector_exp,
    householder_reflection_matrix,
    grade_decompose,
)
from .quantization import UniformQuantizer, quantization_error


def _expand_pair_scale(scale: Optional[torch.Tensor], dim: int) -> Optional[torch.Tensor]:
    """Expand pairwise scale coefficients to per-dimension scale factors."""
    if scale is None:
        return None
    if scale.numel() == dim:
        return scale
    n_pairs = dim // 2
    if scale.numel() == n_pairs:
        expanded = scale.repeat_interleave(2)
        if dim % 2 == 1:
            expanded = torch.cat(
                [expanded, torch.ones(1, device=scale.device, dtype=scale.dtype)]
            )
        return expanded
    raise ValueError(f"Unsupported scale length {scale.numel()} for dim={dim}")


def optimize_rotor_angles(
    W: torch.Tensor,
    n_bits: int = 4,
    n_iterations: int = 200,
    lr: float = 0.05,
    verbose: bool = False,
    n_restarts: int = 3,
    mode: Literal["block", "full"] = "block",
) -> torch.Tensor:
    """Optimize rotor parameters to minimize quantization error.

    Uses gradient-based optimization to find rotor parameters that minimize:
        loss = ||W - R^T * Q(R * W)||_F^2

    where Q is quantization+dequantization and R is a rotation matrix
    parameterized either as block-diagonal 2D rotors or full bivector exp.

    Args:
        W: Weight matrix of shape (out_dim, in_dim).
        n_bits: Quantization bit-width.
        n_iterations: Number of optimization steps.
        lr: Learning rate for optimization.
        verbose: Print loss during optimization.
        n_restarts: Number of random restarts.
        mode: 'block' for block-diagonal 2D rotors, 'full' for bivector matrix exp.

    Returns:
        Optimized parameters (angles for 'block', bivector matrix for 'full').
    """
    device = W.device
    dim = W.shape[1]

    if mode == "full":
        return _optimize_full_rotor(W, n_bits, n_iterations, lr, verbose, n_restarts)
    else:
        return _optimize_block_rotor(W, n_bits, n_iterations, lr, verbose, n_restarts)


def optimize_reflection_vector(
    W: torch.Tensor,
    n_bits: int = 4,
    n_iterations: int = 200,
    lr: float = 0.05,
    verbose: bool = False,
    n_restarts: int = 3,
    calibration_inputs: Optional[torch.Tensor] = None,
    calibration_targets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Optimize a Householder reflection direction for quantization.

    Uses gradient descent over a single direction vector u defining
    H = I - 2uu^T. If calibration data is provided, minimizes output MSE on
    the calibration set; otherwise minimizes weight reconstruction MSE.
    """
    device = W.device
    dim = W.shape[1]

    if dim <= 1:
        return torch.randn(dim, device=device)

    best_vector = None
    best_loss = float("inf")
    n_trials = n_restarts if dim > 1 else 1

    for trial in range(n_trials):
        transform = RandomReflectionTransform(dim, requires_grad=True).to(device)
        optimizer = torch.optim.Adam([transform.vector], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_iterations
        )

        for step in range(n_iterations):
            optimizer.zero_grad()

            H = householder_reflection_matrix(transform.vector)
            W_reflected = W @ H.T
            W_q = UniformQuantizer(n_bits, symmetric=True, per_channel=True)(W_reflected)
            W_deq = W_q @ H

            if calibration_inputs is not None and calibration_targets is not None:
                pred = calibration_inputs @ W_deq.T
                loss = F.mse_loss(pred, calibration_targets)
            else:
                loss = F.mse_loss(W_deq, W)

            loss.backward()
            optimizer.step()
            scheduler.step()

            if verbose and (step + 1) % 50 == 0:
                print(f"  [reflect] Trial {trial+1}, Step {step+1}: loss = {loss.item():.6f}")

        with torch.no_grad():
            H = householder_reflection_matrix(transform.vector)
            W_reflected = W @ H.T
            W_q = UniformQuantizer(n_bits, symmetric=True, per_channel=True)(W_reflected)
            W_deq = W_q @ H
            if calibration_inputs is not None and calibration_targets is not None:
                final_loss = F.mse_loss(calibration_inputs @ W_deq.T, calibration_targets).item()
            else:
                final_loss = F.mse_loss(W_deq, W).item()

        if final_loss < best_loss:
            best_loss = final_loss
            best_vector = transform.vector.detach().clone()
            if verbose:
                print(f"  [reflect] -> New best: loss = {final_loss:.6f}")

    if verbose:
        print(f"Reflection optimization complete. Best loss: {best_loss:.6f}")

    return best_vector


def get_optimal_reflection(
    W: torch.Tensor,
    n_bits: int = 4,
    n_optimization_steps: int = 200,
    verbose: bool = False,
    calibration_inputs: Optional[torch.Tensor] = None,
    calibration_targets: Optional[torch.Tensor] = None,
    n_restarts: int = 3,
) -> torch.Tensor:
    """Get the optimal Householder reflection matrix for quantization."""
    dim = W.shape[1]
    params = optimize_reflection_vector(
        W,
        n_bits=n_bits,
        n_iterations=n_optimization_steps,
        verbose=verbose,
        n_restarts=n_restarts,
        calibration_inputs=calibration_inputs,
        calibration_targets=calibration_targets,
    )
    return householder_reflection_matrix(params)


def _optimize_block_rotor(
    W: torch.Tensor,
    n_bits: int,
    n_iterations: int,
    lr: float,
    verbose: bool,
    n_restarts: int,
) -> torch.Tensor:
    """Optimize block-diagonal 2D rotor angles."""
    device = W.device
    dim = W.shape[1]
    n_rotors = dim // 2

    best_angles = None
    best_loss = float("inf")

    for trial in range(n_restarts if n_rotors > 1 else 1):
        transform = RandomRotorTransform(dim, requires_grad=True).to(device)
        optimizer = torch.optim.Adam([transform.angles], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_iterations
        )

        for step in range(n_iterations):
            optimizer.zero_grad()

            R = block_diag_rotor_matrix(transform.angles, dim)
            W_rotated = W @ R.T
            W_q = UniformQuantizer(n_bits, symmetric=True, per_channel=True)(W_rotated)
            W_deq = W_q @ R

            loss = F.mse_loss(W_deq, W)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if verbose and (step + 1) % 50 == 0:
                print(f"  Trial {trial+1}, Step {step+1}: loss = {loss.item():.6f}")

        with torch.no_grad():
            R = block_diag_rotor_matrix(transform.angles, dim)
            W_rotated = W @ R.T
            W_q = UniformQuantizer(n_bits, symmetric=True, per_channel=True)(W_rotated)
            W_deq = W_q @ R
            final_loss = F.mse_loss(W_deq, W).item()

        if final_loss < best_loss:
            best_loss = final_loss
            best_angles = transform.angles.detach().clone()
            if verbose:
                print(f"  -> New best: loss = {final_loss:.6f}")

    if verbose:
        print(f"Optimization complete. Best loss: {best_loss:.6f}")
    return best_angles


def _optimize_full_rotor(
    W: torch.Tensor,
    n_bits: int,
    n_iterations: int,
    lr: float,
    verbose: bool,
    n_restarts: int,
) -> torch.Tensor:
    """Optimize a full bivector matrix (via matrix exponential) for rotation.

    Uses grade_decompose + bivector_exp from gamuon to parameterize the rotation
    as exp(B) where B is a learnable bivector (antisymmetric matrix).
    This is more expressive than block-diagonal rotors but more expensive.
    """
    device = W.device
    dim = W.shape[1]

    best_biv = None
    best_loss = float("inf")

    for trial in range(n_restarts):
        # Initialize random bivector (antisymmetric matrix)
        B_init = torch.randn(dim, dim, device=device) * 0.1
        B_init = (B_init - B_init.T) / 2  # Make antisymmetric
        bivector = torch.nn.Parameter(B_init, requires_grad=True)

        optimizer = torch.optim.Adam([bivector], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_iterations
        )

        for step in range(n_iterations):
            optimizer.zero_grad()

            # Ensure antisymmetry and compute rotor via matrix exp
            B = (bivector - bivector.T) / 2
            R = bivector_exp(B)

            # Rotate columns, quantize, undo rotation
            W_rotated = W @ R.T
            W_q = UniformQuantizer(n_bits, symmetric=True, per_channel=True)(W_rotated)
            W_deq = W_q @ R

            loss = F.mse_loss(W_deq, W)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if verbose and (step + 1) % 50 == 0:
                print(f"  [full] Trial {trial+1}, Step {step+1}: loss = {loss.item():.6f}")

        with torch.no_grad():
            B = (bivector - bivector.T) / 2
            R = bivector_exp(B)
            W_rotated = W @ R.T
            W_q = UniformQuantizer(n_bits, symmetric=True, per_channel=True)(W_rotated)
            W_deq = W_q @ R
            final_loss = F.mse_loss(W_deq, W).item()

        if final_loss < best_loss:
            best_loss = final_loss
            best_biv = bivector.detach().clone()
            if verbose:
                print(f"  [full] -> New best: loss = {final_loss:.6f}")

    if verbose:
        print(f"Full rotor optimization complete. Best loss: {best_loss:.6f}")
    return best_biv


def analyze_quantization_by_grade(
    W: torch.Tensor,
    W_q: torch.Tensor,
    verbose: bool = True,
) -> dict:
    """Analyze quantization error by geometric grade components.

    Decomposes the weight matrix and its quantization error into
    scalar, bivector, and strain grades (from gamuon) to understand
    which geometric components contribute most to quantization loss.

    Args:
        W: Original weight matrix (out_dim, in_dim).
            If non-square, only the square part is analyzed.
        W_q: Quantized weight matrix.
        verbose: Print analysis.

    Returns:
        Dictionary of grade -> MSE contribution.
    """
    min_dim = min(W.shape[0], W.shape[1])
    W_sq = W[:min_dim, :min_dim]
    Wq_sq = W_q[:min_dim, :min_dim]

    error = W_sq - Wq_sq
    scalar_e, bivector_e, strain_e = grade_decompose(error)
    scalar_w, bivector_w, strain_w = grade_decompose(W_sq)

    results = {
        "error_scalar": scalar_e.pow(2).mean().item(),
        "error_bivector": bivector_e.pow(2).mean().item(),
        "error_strain": strain_e.pow(2).mean().item(),
        "magnitude_scalar": scalar_w.pow(2).mean().item(),
        "magnitude_bivector": bivector_w.pow(2).mean().item(),
        "magnitude_strain": strain_w.pow(2).mean().item(),
    }

    if verbose:
        print("\nQuantization Error by Geometric Grade:")
        for grade in ["scalar", "bivector", "strain"]:
            err = results[f"error_{grade}"]
            mag = results[f"magnitude_{grade}"]
            rel = err / (mag + 1e-10)
            print(f"  {grade:10s}: error={err:.6e}, rel={rel:.4f}")

    return results


def get_optimal_rotation(
    W: torch.Tensor,
    mode: Literal["block", "full"] = "block",
    n_bits: int = 4,
    n_optimization_steps: int = 200,
    verbose: bool = False,
) -> torch.Tensor:
    """Get the optimal rotation matrix for quantizing weight matrix W.

    Uses the optimized rotor parameters to construct the rotation matrix.

    Args:
        W: Weight matrix of shape (out_dim, in_dim).
        mode: 'block' for block-diagonal, 'full' for bivector exp.
        n_bits: Quantization bit-width.
        n_optimization_steps: Optimization steps.
        verbose: Print progress.

    Returns:
        Rotation matrix of shape (in_dim, in_dim).
    """
    dim = W.shape[1]
    params = optimize_rotor_angles(
        W, n_bits, n_optimization_steps, verbose=verbose, mode=mode
    )

    if mode == "full":
        B = (params - params.T) / 2
        return bivector_exp(B)
    else:
        return block_diag_rotor_matrix(params, dim)


def apply_rotor_scale(
    W: torch.Tensor,
    R: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply optional input-space scaling followed by a rotor."""
    scale = _expand_pair_scale(scale, W.shape[1])
    if scale is None:
        return W @ R.T
    return (W * scale.unsqueeze(0)) @ R.T


def finalize_rotor_scale(
    W_q: torch.Tensor,
    R: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Undo the rotor and optional scaling after quantization."""
    scale = _expand_pair_scale(scale, W_q.shape[1])
    W_out = W_q @ R
    if scale is None:
        return W_out
    inv_scale = (1.0 / scale.clamp(min=1e-4)).unsqueeze(0)
    return W_out * inv_scale


def optimize_rotor_scale(
    W: torch.Tensor,
    n_bits: int = 4,
    n_optimization_steps: int = 200,
    lr: float = 0.05,
    verbose: bool = False,
    n_restarts: int = 2,
    calibration_inputs: Optional[torch.Tensor] = None,
    calibration_targets: Optional[torch.Tensor] = None,
    learn_scale: bool = True,
    scale_reg: float = 5e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optimize block-diagonal rotor angles plus diagonal scaling.

    If calibration data is provided, minimize output MSE on the calibration
    set; otherwise minimize weight reconstruction MSE.
    """
    device = W.device
    dim = W.shape[1]
    n_rotors = dim // 2
    n_trials = n_restarts if n_rotors > 1 else 1
    has_scale = learn_scale and dim > 1
    n_scale = dim // 2

    best_angles = None
    best_scale = None
    best_loss = float("inf")

    for trial in range(n_trials):
        transform = RandomRotorTransform(dim, requires_grad=True).to(device)
        params = [transform.angles]
        log_scale = None
        if has_scale:
            log_scale = torch.nn.Parameter(torch.zeros(n_scale, device=device))
            params.append(log_scale)

        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_optimization_steps
        )

        for step in range(n_optimization_steps):
            optimizer.zero_grad()

            R = block_diag_rotor_matrix(transform.angles, dim)
            scale = None
            if log_scale is not None:
                pair_scale = torch.exp(log_scale - log_scale.mean()).clamp(0.75, 1.33)
                scale = _expand_pair_scale(pair_scale, dim)

            W_rotated = apply_rotor_scale(W, R, scale)
            q = UniformQuantizer(n_bits, symmetric=True, per_channel=True)
            W_q = q(W_rotated)
            W_final = finalize_rotor_scale(W_q, R, scale)

            if calibration_inputs is not None and calibration_targets is not None:
                pred = calibration_inputs @ W_final.T
                loss = F.mse_loss(pred, calibration_targets)
            else:
                loss = F.mse_loss(W_final, W)

            if log_scale is not None:
                loss = loss + scale_reg * (log_scale ** 2).mean()

            loss.backward()
            optimizer.step()
            scheduler.step()

            if verbose and (step + 1) % 50 == 0:
                print(f"  Trial {trial+1}, Step {step+1}: loss = {loss.item():.6f}")

        with torch.no_grad():
            R = block_diag_rotor_matrix(transform.angles, dim)
            scale = None
            if log_scale is not None:
                pair_scale = torch.exp(log_scale - log_scale.mean()).clamp(0.75, 1.33)
                scale = _expand_pair_scale(pair_scale, dim)

            q = UniformQuantizer(n_bits, symmetric=True, per_channel=True)
            W_rotated = apply_rotor_scale(W, R, scale)
            W_q = q(W_rotated)
            W_final = finalize_rotor_scale(W_q, R, scale)

            if calibration_inputs is not None and calibration_targets is not None:
                pred = calibration_inputs @ W_final.T
                final_loss = F.mse_loss(pred, calibration_targets).item()
            else:
                final_loss = F.mse_loss(W_final, W).item()

        if final_loss < best_loss:
            best_loss = final_loss
            best_angles = transform.angles.detach().clone()
            best_scale = (
                _expand_pair_scale(
                    torch.exp(log_scale - log_scale.mean()).detach().clone().clamp(0.75, 1.33),
                    dim,
                )
                if log_scale is not None
                else torch.ones(dim, device=device)
            )
            if verbose:
                print(f"  -> New best: loss = {final_loss:.6f}")

    if best_angles is None:
        best_angles = torch.zeros(n_rotors, device=device)
    if best_scale is None:
        best_scale = torch.ones(dim, device=device)

    if verbose:
        print(f"Optimization complete. Best loss: {best_loss:.6f}")
    return best_angles, best_scale


class RotorQuaRot(torch.nn.Module):
    """Rotor-based QuaRot quantization module.

    Applies a learnable block-diagonal rotation (composed of 2D rotors)
    to a weight matrix before quantization, improving quantization quality
    by making weight distributions more uniform.

    Usage:
        transform = RotorQuaRot(dim, n_bits=4)
        W_q = transform(W)  # quantized with rotation

    The rotation can be:
    - Random (as in QuaRot with Hadamard)
    - Optimized on calibration data
    """

    def __init__(
        self,
        dim: int,
        n_bits: int = 4,
        symmetric: bool = True,
        per_channel: bool = True,
        angles: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.dim = dim
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.quantizer = UniformQuantizer(n_bits, symmetric, per_channel)

        if angles is not None:
            self.register_buffer("angles", angles.clone())
        else:
            # Random initialization (like QuaRot's random Hadamard)
            n_rotors = dim // 2
            self.register_buffer(
                "angles", torch.rand(n_rotors) * 2 * torch.pi
            )

    def get_rotation_matrix(self) -> torch.Tensor:
        return block_diag_rotor_matrix(self.angles, self.dim)

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        """Apply quantize-then-dequantize with optimal rotation.

        W_q = Q(W @ R^T) @ R
        """
        R = self.get_rotation_matrix()
        W_rotated = W @ R.T
        W_q = self.quantizer(W_rotated)
        W_deq = W_q @ R
        return W_deq

    def quantize_forward(self, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """Quantized forward pass: y = (x · R^T) · Q(W @ R^T)

        For inference, the input rotation is absorbed into the quantized weights.
        """
        R = self.get_rotation_matrix()
        W_rotated = W @ R.T
        q_ints, scale, zp = self.quantizer.quantize(W_rotated)
        return x, (q_ints, scale, zp), R

    def __repr__(self) -> str:
        return (
            f"RotorQuaRot(dim={self.dim}, n_bits={self.n_bits}, "
            f"{'optimized' if self.angles is not None else 'random'})"
        )


def quantize_with_rotors(
    W: torch.Tensor,
    n_bits: int = 4,
    optimize: bool = False,
    n_optimization_steps: int = 200,
    verbose: bool = False,
) -> torch.Tensor:
    """Quantize a weight matrix using rotor-based rotation.

    Args:
        W: Weight matrix of shape (out_dim, in_dim).
        n_bits: Quantization bit-width.
        optimize: If True, optimize rotor angles; otherwise use random angles.
        n_optimization_steps: Number of optimization steps.
        verbose: Print progress.

    Returns:
        Quantized weight matrix (dequantized floats).
    """
    dim = W.shape[1]

    if optimize:
        angles = optimize_rotor_angles(
            W, n_bits, n_optimization_steps, verbose=verbose
        )
    else:
        # Random angles
        n_rotors = dim // 2
        angles = torch.rand(n_rotors) * 2 * torch.pi

    transform = RotorQuaRot(dim, n_bits, angles=angles.to(W.device))
    return transform(W)
