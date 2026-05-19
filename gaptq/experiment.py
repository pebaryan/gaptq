"""
Experiment script comparing rotor-based QuaRot against baseline quantization methods.

Methods compared:
1. RTN (Round-to-Nearest): No rotation, direct quantization
2. Hadamard: Random Hadamard/orthogonal rotation (QuaRot baseline)
3. Random Rotors: Random block-diagonal rotor rotation
4. Optimized Rotors (block): Optimized block-diagonal 2D rotors
5. Optimized Rotors (full): Optimized full bivector matrix exp (from gamuon)
6. Best-of-N Random: Best among N random block-diagonal rotors

Also includes grade-decomposition analysis (from gamuon) to study
quantization error by geometric grade.
"""

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .ga import (
    CliffordAlgebra,
    block_diag_rotor_matrix,
    bivector_exp,
    grade_decompose,
)
from .quantization import UniformQuantizer, quantization_error, quantize_weight_matrix
from .rotor_quant import (
    RotorQuaRot,
    analyze_quantization_by_grade,
    get_optimal_rotation,
    optimize_rotor_angles,
)


def _hadamard_or_fallback(n: int) -> torch.Tensor:
    """Generate a Hadamard matrix (if n is power of 2) or random orthogonal matrix.

    In QuaRot, random orthogonal matrices are used (approximated via Hadamard).
    For non-power-of-2 dimensions, we use a random orthogonal matrix from QR.
    """
    if n == 1:
        return torch.tensor([[1.0]])

    # Check if n is a power of 2
    if n & (n - 1) == 0:
        k = int(math.log2(n))
        H = torch.tensor([[1.0]])
        for _ in range(k):
            H = torch.cat([
                torch.cat([H, H], dim=1),
                torch.cat([H, -H], dim=1),
            ], dim=0)
        return H / math.sqrt(n)
    else:
        # Random orthogonal matrix via QR decomposition
        A = torch.randn(n, n)
        Q, R = torch.linalg.qr(A)
        return Q @ torch.diag(torch.sign(torch.diag(R)))


def run_quantization_experiment(
    W: torch.Tensor,
    n_bits: int = 4,
    n_optimization_steps: int = 200,
    verbose: bool = True,
    run_full_rotor: bool = False,
    analyze_grades: bool = False,
) -> Dict[str, float]:
    """Run a full quantization comparison experiment on a single weight matrix.

    Args:
        W: Weight matrix of shape (out_dim, in_dim).
        n_bits: Quantization bit-width.
        n_optimization_steps: Steps for rotor optimization.
        verbose: Print results.
        run_full_rotor: If True, also run the full bivector exp rotor (gamuon-style).
        analyze_grades: If True, run grade-decomposition analysis.

    Returns:
        Dictionary of method -> NMSE (normalized MSE).
    """
    device = W.device
    in_dim = W.shape[1]
    results = {}

    q = UniformQuantizer(n_bits, symmetric=True, per_channel=True)

    # 1. RTN (no rotation)
    W_rtn = q(W)
    loss_rtn = quantization_error(W, W_rtn).item()
    results["RTN"] = loss_rtn
    if verbose:
        print(f"  RTN:                  NMSE = {loss_rtn:.6f}")

    # 2. Orthogonal rotation (QuaRot-style: random orthogonal matrix)
    try:
        H = _hadamard_or_fallback(in_dim).to(device)
        W_orth = W @ H.T
        W_orth_q = q(W_orth)
        W_orth_deq = W_orth_q @ H
        loss_orth = quantization_error(W, W_orth_deq).item()
        results["Orthogonal"] = loss_orth
        if verbose:
            print(f"  Orthogonal (QuaRot):  NMSE = {loss_orth:.6f}")
    except Exception as e:
        results["Orthogonal"] = float("nan")
        if verbose:
            print(f"  Orthogonal: skipped ({e})")

    # 3. Random block-diagonal rotor rotation
    n_rotors = max(in_dim // 2, 1)
    random_angles = torch.rand(n_rotors, device=device) * 2 * math.pi
    R_random = block_diag_rotor_matrix(random_angles, in_dim).to(device)
    W_random_rot = W @ R_random.T
    W_random_q = q(W_random_rot)
    W_random_deq = W_random_q @ R_random
    loss_random = quantization_error(W, W_random_deq).item()
    results["RandomRotors"] = loss_random
    if verbose:
        print(f"  Random Rotors (block): NMSE = {loss_random:.6f}")

    # 4. Optimized block-diagonal rotor rotation
    t0 = time.time()
    opt_angles = optimize_rotor_angles(
        W, n_bits, n_iterations=n_optimization_steps, verbose=False, mode="block"
    ).to(device)
    R_opt = block_diag_rotor_matrix(opt_angles, in_dim).to(device)
    W_opt_rot = W @ R_opt.T
    W_opt_q = q(W_opt_rot)
    W_opt_deq = W_opt_q @ R_opt
    loss_optimized = quantization_error(W, W_opt_deq).item()
    results["OptimizedRotors"] = loss_optimized
    opt_time = time.time() - t0
    if verbose:
        print(f"  Optimized Rotors:     NMSE = {loss_optimized:.6f} (in {opt_time:.1f}s)")

    # 5. Optimized full rotor (via bivector matrix exp, from gamuon)
    if run_full_rotor:
        try:
            t0 = time.time()
            R_full = get_optimal_rotation(
                W, mode="full", n_bits=n_bits,
                n_optimization_steps=n_optimization_steps // 2, verbose=False
            )
            W_full_rot = W @ R_full.T
            W_full_q = q(W_full_rot)
            W_full_deq = W_full_q @ R_full
            loss_full = quantization_error(W, W_full_deq).item()
            results["FullRotor"] = loss_full
            full_time = time.time() - t0
            if verbose:
                print(f"  Full Rotor (gamuon):  NMSE = {loss_full:.6f} (in {full_time:.1f}s)")
        except Exception as e:
            results["FullRotor"] = float("nan")
            if verbose:
                print(f"  Full Rotor: skipped ({e})")

    # 6. Multiple random restarts (best of N random)
    n_restarts = 10
    best_random_loss = float("inf")
    for _ in range(n_restarts):
        angles = torch.rand(n_rotors, device=device) * 2 * math.pi
        R = block_diag_rotor_matrix(angles, in_dim).to(device)
        W_rot = W @ R.T
        W_rot_q = q(W_rot)
        W_rot_deq = W_rot_q @ R
        loss = quantization_error(W, W_rot_deq).item()
        if loss < best_random_loss:
            best_random_loss = loss
    results["BestOfRandom"] = best_random_loss
    if verbose:
        print(f"  Best-of-{n_restarts} Random: NMSE = {best_random_loss:.6f}")

    # 7. Grade decomposition analysis (from gamuon)
    if analyze_grades:
        W_q_rtn = q(W)
        grade_analysis = analyze_quantization_by_grade(W, W_q_rtn, verbose=verbose)
        results.update(grade_analysis)

    return results


def run_batch_experiment(
    n_matrices: int = 10,
    dim_range: Tuple[int, int] = (64, 256),
    n_bits: int = 4,
    n_optimization_steps: int = 100,
    verbose: bool = True,
    run_full_rotor: bool = False,
) -> Dict[str, List[float]]:
    """Run experiment across multiple random matrices and aggregate results.

    Args:
        n_matrices: Number of random matrices to test.
        dim_range: (min_dim, max_dim) for matrix dimensions.
        n_bits: Quantization bit-width.
        n_optimization_steps: Steps for rotor optimization.
        run_full_rotor: If True, also run full bivector exp rotor.

    Returns:
        Dictionary of method -> list of NMSE values.
    """
    all_results: Dict[str, List[float]] = {}

    for i in range(n_matrices):
        in_dim = int(np.random.randint(*dim_range))
        out_dim = int(np.random.randint(*dim_range))
        in_dim = max(in_dim, 2)
        out_dim = max(out_dim, 2)

        # Generate realistic structured weight matrix (low-rank + noise)
        rank = max(min(in_dim, out_dim) // 4, 1)
        A = torch.randn(out_dim, rank)
        B = torch.randn(rank, in_dim)
        W = (A @ B) * 0.5 + torch.randn(out_dim, in_dim) * 0.1

        if verbose:
            print(f"\nMatrix {i+1}/{n_matrices}: {out_dim}x{in_dim}")

        results = run_quantization_experiment(
            W, n_bits, n_optimization_steps, verbose=False,
            run_full_rotor=run_full_rotor
        )

        for method, loss in results.items():
            if method not in all_results:
                all_results[method] = []
            all_results[method].append(loss)

        if verbose:
            for method, loss in results.items():
                print(f"  {method:20s}: NMSE = {loss:.6f}")

    return all_results


def summarize_results(results: Dict[str, List[float]]) -> None:
    """Print summary statistics across all experiment runs."""
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    methods = sorted(results.keys())

    # Find best method for each run
    n_runs = len(next(iter(results.values())))
    wins = {m: 0 for m in methods}

    for i in range(n_runs):
        best_loss = min(results[m][i] for m in methods)
        for m in methods:
            if results[m][i] == best_loss:
                wins[m] += 1

    print(f"\n{'Method':<20s} {'Mean NMSE':<12s} {'Std NMSE':<12s} {'Wins':<8s}")
    print("-" * 52)
    for m in methods:
        mean_val = np.mean(results[m])
        std_val = np.std(results[m])
        print(f"{m:<20s} {mean_val:.6f}     {std_val:.6f}     {wins[m]:<8d}")

    print(f"\nTotal runs: {n_runs}")


def analyze_rotor_angles(angles: torch.Tensor) -> None:
    """Analyze the distribution of optimized rotor angles."""
    angles_np = angles.detach().cpu().numpy()
    print(f"\nRotor Angle Analysis:")
    print(f"  Min: {angles_np.min():.4f} rad")
    print(f"  Max: {angles_np.max():.4f} rad")
    print(f"  Mean: {angles_np.mean():.4f} rad")
    print(f"  Std: {angles_np.std():.4f} rad")

    angles_mod = angles_np % (2 * math.pi)
    print(f"  Near identity (theta ~ 0 mod pi): {(angles_mod % math.pi < 0.1).mean():.1%}")
    print(f"  Near 90 deg mod pi: {((angles_mod % math.pi - math.pi/2).abs() < 0.1).mean():.1%}")


def demo_clifford_algebra() -> None:
    """Demonstrate the CliffordAlgebra engine from gattrlm."""
    print("\n--- CliffordAlgebra Engine Demo ---")
    alg = CliffordAlgebra(dim=3)

    # Create two vectors as multivectors
    v1 = torch.zeros(8)
    v1[1] = 1.0  # e1
    v2 = torch.zeros(8)
    v2[2] = 1.0  # e2

    # Geometric product: e1 * e2 = e1^e2 (bivector)
    gp = alg.geometric_product(v1, v2)
    assert gp[3] == 1.0  # e1^e2 at index 3
    print(f"  e1 * e2 = bivector at index 3: coeff={gp[3]:.0f} (expected 1)")

    # Sandwich product: R * v * R~
    angle = math.pi / 4
    R = torch.zeros(8)
    R[0] = math.cos(angle / 2)
    R[3] = -math.sin(angle / 2)  # e1^e2 component
    R_rev = R * alg._rev_sign  # reverse (both are float32)

    v = torch.zeros(8)
    v[1] = 1.0  # e1

    rotated = alg.sandwich_product(R, v, R_rev)
    vec_e1 = rotated[1].item()
    vec_e2 = rotated[2].item()
    print(f"  Rotated e1 by 45 deg in e1^e2 plane: [{vec_e1:.4f}, {vec_e2:.4f}]")
    print(f"  Expected: [{math.cos(angle):.4f}, {math.sin(angle):.4f}]")


def demo_grade_decomposition() -> None:
    """Demonstrate grade_decompose from gamuon."""
    print("\n--- Grade Decomposition Demo (from gamuon) ---")
    W = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    scalar, bivector, strain = grade_decompose(W)

    print(f"  W = {W.numpy().tolist()}")
    print(f"  Scalar:   {scalar.numpy().tolist()}")
    print(f"  Bivector: {bivector.numpy().tolist()}")
    print(f"  Strain:   {strain.numpy().tolist()}")
    print(f"  W = scalar + bivector + strain: {torch.allclose(W, scalar + bivector + strain)}")


if __name__ == "__main__":
    print("Geometric Algebra Post-Training Quantization (GAP-TQ)")
    print("=" * 60)

    # Demo new features
    demo_clifford_algebra()
    demo_grade_decomposition()

    # Quick single-matrix demo
    print("\n--- Single Matrix Demo ---")
    torch.manual_seed(42)
    W = torch.randn(128, 128) * 0.5 + torch.randn(128, 64) @ torch.randn(64, 128) * 0.3
    results = run_quantization_experiment(W, n_bits=4, n_optimization_steps=100)

    # Run full batch experiment
    print("\n--- Batch Experiment ---")
    batch_results = run_batch_experiment(
        n_matrices=5, dim_range=(64, 128), n_bits=4, n_optimization_steps=100
    )
    summarize_results(batch_results)
