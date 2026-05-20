"""Per-grade quantization: decompose weight matrices into geometric grades
(scalar, bivector, strain) and assign different bit-widths to each grade.

Uses grade_decompose from ..ga to split a (square) matrix into its
geometric components, then quantizes each component independently.

For non-square matrices, we decompose the matrix in square blocks along the
long axis. This matches GPT-2 Conv1D weights well, since they are built from
repeated square sub-blocks (for example, `c_attn` is 3*h by h and `c_proj`
is h by 4*h). Any leftover tail that does not form a full square block falls
back to RTN.

Key idea from GA:
  W = <W>_0 + <W>_2 + <W>_+
    Scalar:   tr(W)/n * I           (isotropic dilation, low entropy)
    Bivector: (W - W^T)/2           (antisymmetric, rotation-like)
    Strain:   (W + W^T)/2 - <W>_0   (symmetric traceless, most complex)

Each grade has different quantization properties:
  - Scalar: very low entropy (all values equal) → can use very few bits
  - Bivector: zero-centered, symmetric distribution → good for low-bit quantization
  - Strain: most variation, hardest to quantize → needs higher bits
"""

from typing import Dict, List, Optional, Tuple, Union

import torch

from ..ga import grade_decompose
from ..quantization import UniformQuantizer, quantization_error


# ── Shape utilities for rectangular matrices ───────────────────────


def _pad_to_square(W: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad a non-square matrix to square with zeros.

    Args:
        W: Matrix of shape (m, n).

    Returns:
        (padded_matrix of shape (max(m,n), max(m,n)), original_shape).
    """
    m, n = W.shape
    if m == n:
        return W, (m, n)
    max_dim = max(m, n)
    padded = torch.zeros(max_dim, max_dim, device=W.device, dtype=W.dtype)
    padded[:m, :n] = W
    return padded, (m, n)


def _extract_shape(W_padded: torch.Tensor, original_shape: Tuple[int, int]) -> torch.Tensor:
    """Extract original shape from a padded square matrix."""
    m, n = original_shape
    return W_padded[:m, :n]


def _quantize_square_grade_block(
    W: torch.Tensor,
    bit_map: Dict[Union[int, str], int],
    symmetric: bool,
    per_channel: bool,
) -> Tuple[torch.Tensor, Dict]:
    """Quantize a square block with per-grade decomposition."""
    scalar, bivector, strain = grade_decompose(W)

    bits_0 = bit_map.get(0, 16)
    bits_2 = bit_map.get(2, 4)
    bits_strain = bit_map.get("strain", 8)

    q_scalar = UniformQuantizer(bits_0, symmetric, per_channel)(scalar)
    q_bivector = UniformQuantizer(bits_2, symmetric, per_channel)(bivector)
    q_strain = UniformQuantizer(bits_strain, symmetric, per_channel)(strain)

    W_q = q_scalar + q_bivector + q_strain

    power_scalar = scalar.pow(2).sum().item()
    power_bivector = bivector.pow(2).sum().item()
    power_strain = strain.pow(2).sum().item()
    total_power = W.pow(2).sum().item() + 1e-10
    info = {
        "mode": "per_grade",
        "nmse_scalar": quantization_error(scalar, q_scalar).item(),
        "nmse_bivector": quantization_error(bivector, q_bivector).item(),
        "nmse_strain": quantization_error(strain, q_strain).item(),
        "power_scalar": power_scalar,
        "power_bivector": power_bivector,
        "power_strain": power_strain,
        "frac_scalar": power_scalar / total_power,
        "frac_bivector": power_bivector / total_power,
        "frac_strain": power_strain / total_power,
    }
    return W_q, info


# ── Per-grade quantizer ────────────────────────────────────────────


def quantize_per_grade(
    W: torch.Tensor,
    bit_map: Optional[Dict[Union[int, str], int]] = None,
    symmetric: bool = True,
    per_channel: bool = True,
    verbose: bool = False,
) -> Tuple[torch.Tensor, Dict]:
    """Quantize a weight matrix using per-grade bit allocation.

    Decomposes the matrix into geometric grades via grade_decompose:
      - Scalar (grade 0):  isotropic dilation, tr(W)/n * I
      - Bivector (grade 2): antisymmetric part, (W - W^T)/2
      - Strain:             symmetric traceless, (W + W^T)/2 - <W>_0

    Each grade is quantized independently with its own bit-width, then
    summed to form the quantized matrix.

    For non-square matrices, the matrix is split into square blocks along the
    long axis. Each full square block is grade-decomposed and quantized
    independently; any leftover tail falls back to RTN.

    Args:
        W: Weight matrix of shape (out_dim, in_dim).
        bit_map: Mapping grade → bit-width. Keys can be:
            - 0: scalar grade
            - 2: bivector grade
            - 'strain': strain (symmetric traceless) grade
            If None, defaults to {0: 16, 2: 4, 'strain': 8}.
        symmetric: Use symmetric quantization.
        per_channel: Use per-channel quantization.
        verbose: Print per-grade stats.

    Returns:
        (quantized_weight, info_dict) where info_dict contains:
          - mode: 'per_grade'
          - bit_map: the bit allocation used
          - nmse_scalar, nmse_bivector, nmse_strain: per-grade NMSE
          - nmse_rotor: combined NMSE
          - nmse_rtn: uniform 4-bit RTN NMSE for reference
          - improvement_pct: improvement over uniform 4-bit RTN
          - frac_scalar, frac_bivector, frac_strain: grade power fractions
    """
    if bit_map is None:
        bit_map = {0: 16, 2: 4, 'strain': 8}

    m, n = W.shape
    square_size = min(m, n)
    q_rtn = UniformQuantizer(4, symmetric, per_channel)
    total_power = W.pow(2).sum().item() + 1e-10

    def _grade_block_info(block_info: Dict) -> Dict:
        return {
            "nmse_scalar": block_info.get("nmse_scalar", 0.0),
            "nmse_bivector": block_info.get("nmse_bivector", 0.0),
            "nmse_strain": block_info.get("nmse_strain", 0.0),
            "frac_scalar": block_info.get("frac_scalar", 0.0),
            "frac_bivector": block_info.get("frac_bivector", 0.0),
            "frac_strain": block_info.get("frac_strain", 0.0),
        }

    # Exact per-grade decomposition for square weights.
    if m == n:
        W_q, block_info = _quantize_square_grade_block(W, bit_map, symmetric, per_channel)
        nmse_total = quantization_error(W, W_q).item()
        nmse_rtn = quantization_error(W, q_rtn(W)).item()
        improvement = (nmse_rtn - nmse_total) / nmse_rtn * 100 if nmse_rtn > 0 else 0.0

        bits_0 = bit_map.get(0, 16)
        bits_2 = bit_map.get(2, 4)
        bits_strain = bit_map.get("strain", 8)
        info = {
            "mode": "per_grade",
            "in_dim": n,
            "out_dim": m,
            "bit_map": {str(k): v for k, v in bit_map.items()},
            "nmse_scalar": block_info["nmse_scalar"],
            "nmse_bivector": block_info["nmse_bivector"],
            "nmse_strain": block_info["nmse_strain"],
            "nmse_rotor": nmse_total,
            "nmse_rtn": nmse_rtn,
            "improvement_pct": improvement,
            "frac_scalar": block_info["frac_scalar"],
            "frac_bivector": block_info["frac_bivector"],
            "frac_strain": block_info["frac_strain"],
            "block_size": square_size,
            "n_blocks": 1,
            "n_grade_blocks": 1,
            "n_rtn_blocks": 0,
        }

        if verbose:
            avg_bits = _avg_bit_width(square_size, bit_map)
            print(f"  Per-Grade ({bits_0}b scalar, {bits_2}b bivector, {bits_strain}b strain, "
                  f"~{avg_bits:.1f}b avg):")
            print(f"    Scalar:   NMSE={info['nmse_scalar']:.6e}  ({info['frac_scalar']:.1%} power)")
            print(f"    Bivector: NMSE={info['nmse_bivector']:.6e}  ({info['frac_bivector']:.1%} power)")
            print(f"    Strain:   NMSE={info['nmse_strain']:.6e}  ({info['frac_strain']:.1%} power)")
            print(f"    Combined: NMSE={nmse_total:.6f}  (vs RTN 4b: {nmse_rtn:.6f}, {improvement:+.1f}%)")

        return W_q, info

    # Rectangular matrix: split into square blocks along the long axis.
    block_infos: List[Dict] = []
    q_blocks: List[torch.Tensor] = []

    if m > n:
        # Vertical stacking of n x n blocks.
        for start in range(0, m, n):
            end = min(start + n, m)
            block = W[start:end, :]
            if block.shape[0] == n:
                q_block, binfo = _quantize_square_grade_block(block, bit_map, symmetric, per_channel)
                q_blocks.append(q_block)
                binfo["mode"] = "per_grade"
            else:
                q_block = q_rtn(block)
                binfo = {
                    "mode": "rtn_fallback",
                    "nmse_rotor": quantization_error(block, q_block).item(),
                    "nmse_rtn": quantization_error(block, q_block).item(),
                    "improvement_pct": 0.0,
                }
                q_blocks.append(q_block)
            binfo["shape"] = tuple(block.shape)
            block_infos.append(binfo)
        W_q = torch.cat(q_blocks, dim=0)
    else:
        # Horizontal stacking of m x m blocks.
        for start in range(0, n, m):
            end = min(start + m, n)
            block = W[:, start:end]
            if block.shape[1] == m:
                q_block, binfo = _quantize_square_grade_block(block, bit_map, symmetric, per_channel)
                q_blocks.append(q_block)
                binfo["mode"] = "per_grade"
            else:
                q_block = q_rtn(block)
                binfo = {
                    "mode": "rtn_fallback",
                    "nmse_rotor": quantization_error(block, q_block).item(),
                    "nmse_rtn": quantization_error(block, q_block).item(),
                    "improvement_pct": 0.0,
                }
                q_blocks.append(q_block)
            binfo["shape"] = tuple(block.shape)
            block_infos.append(binfo)
        W_q = torch.cat(q_blocks, dim=1)

    nmse_total = quantization_error(W, W_q).item()
    nmse_rtn = quantization_error(W, q_rtn(W)).item()
    improvement = (nmse_rtn - nmse_total) / nmse_rtn * 100 if nmse_rtn > 0 else 0.0

    grade_blocks = [b for b in block_infos if b["mode"] == "per_grade"]
    if grade_blocks:
        scalar_power = sum(b.get("power_scalar", 0.0) for b in grade_blocks)
        bivector_power = sum(b.get("power_bivector", 0.0) for b in grade_blocks)
        strain_power = sum(b.get("power_strain", 0.0) for b in grade_blocks)
        nmse_scalar = (
            sum(b["nmse_scalar"] * b.get("power_scalar", 0.0) for b in grade_blocks)
            / (scalar_power + 1e-10)
        )
        nmse_bivector = (
            sum(b["nmse_bivector"] * b.get("power_bivector", 0.0) for b in grade_blocks)
            / (bivector_power + 1e-10)
        )
        nmse_strain = (
            sum(b["nmse_strain"] * b.get("power_strain", 0.0) for b in grade_blocks)
            / (strain_power + 1e-10)
        )
        frac_scalar = scalar_power / total_power
        frac_bivector = bivector_power / total_power
        frac_strain = strain_power / total_power
    else:
        nmse_scalar = nmse_bivector = nmse_strain = 0.0
        frac_scalar = frac_bivector = frac_strain = 0.0

    info = {
        "mode": "per_grade_rectangular",
        "in_dim": n,
        "out_dim": m,
        "bit_map": {str(k): v for k, v in bit_map.items()},
        "nmse_scalar": nmse_scalar,
        "nmse_bivector": nmse_bivector,
        "nmse_strain": nmse_strain,
        "nmse_rotor": nmse_total,
        "nmse_rtn": nmse_rtn,
        "improvement_pct": improvement,
        "frac_scalar": frac_scalar,
        "frac_bivector": frac_bivector,
        "frac_strain": frac_strain,
        "block_size": square_size,
        "n_blocks": len(block_infos),
        "n_grade_blocks": sum(1 for b in block_infos if b["mode"] == "per_grade"),
        "n_rtn_blocks": sum(1 for b in block_infos if b["mode"] == "rtn_fallback"),
    }

    if verbose:
        bits_0 = bit_map.get(0, 16)
        bits_2 = bit_map.get(2, 4)
        bits_strain = bit_map.get("strain", 8)
        avg_bits = _avg_bit_width(square_size, bit_map)
        print(f"  Per-Grade Rectangular ({bits_0}b scalar, {bits_2}b bivector, {bits_strain}b strain, "
              f"~{avg_bits:.1f}b avg per square block):")
        print(f"    Blocks: {info['n_grade_blocks']} graded, {info['n_rtn_blocks']} RTN fallback")
        print(f"    Scalar:   NMSE={nmse_scalar:.6e}  ({frac_scalar:.1%} power)")
        print(f"    Bivector: NMSE={nmse_bivector:.6e}  ({frac_bivector:.1%} power)")
        print(f"    Strain:   NMSE={nmse_strain:.6e}  ({frac_strain:.1%} power)")
        print(f"    Combined: NMSE={nmse_total:.6f}  (vs RTN 4b: {nmse_rtn:.6f}, {improvement:+.1f}%)")

    return W_q, info


# ── Equivalent bit-width computation ───────────────────────────────


def _avg_bit_width(dim: int, bit_map: Dict[Union[int, str], int]) -> float:
    """Compute the average bit-width across grades for a matrix of given dim.

    Scalar: n params (diagonal of I scaled by trace/n)
    Bivector: n*(n-1)/2 params (strictly upper triangle of antisymmetric)
    Strain: n*(n-1)/2 params (off-diagonal symmetric) + (n-1) params for traceless diag
    
    Actually, total params in each grade:
    - Scalar: 1 (single value on diagonal = tr/n, repeated n times on the diag of I,
              but I is the identity matrix, so it's n params but all equal)
    - Bivector: n*(n-1)/2 independent values (antisymmetric off-diagonal)
    - Strain: n*(n+1)/2 - 1 independent values (symmetric, traceless)
    
    For total bits: 
    bits_total = bits_0 * 1 + bits_2 * n*(n-1)/2 + bits_strain * (n*(n+1)/2 - 1)
    
    Per-element average:
    avg = bits_total / n^2
    """
    n = dim
    # Independent parameters per grade component
    n0 = 1       # scalar: 1 value (repeated n times on diagonal)
    n2 = n * (n - 1) // 2 if n >= 2 else 0   # bivector: antisymmetric off-diagonal
    n_strain = n * (n + 1) // 2 - 1 if n >= 1 else 0  # strain: symmetric traceless
    
    b0 = bit_map.get(0, 16)
    b2 = bit_map.get(2, 4)
    bs = bit_map.get('strain', 8)
    
    total_bits = b0 * n0 + b2 * n2 + bs * n_strain
    avg = total_bits / (n * n)
    return avg


# ── Batch comparison utility ───────────────────────────────────────


def compare_per_grade_vs_rtn(
    W: torch.Tensor,
    bit_maps: Optional[list] = None,
    verbose: bool = True,
) -> Dict[str, Tuple[float, float, Dict]]:
    """Compare multiple per-grade bit allocations against RTN.

    Args:
        W: Weight matrix.
        bit_maps: List of bit_map dicts to test. If None, uses defaults.
        verbose: Print results.

    Returns:
        Dict mapping label -> (nmse, improvement_pct, info).
    """
    if bit_maps is None:
        bit_maps = [
            {0: 16, 2: 4, 'strain': 8},    # low-bit bivector only
            {0: 8,  2: 3, 'strain': 6},     # aggressive mixed precision
            {0: 8,  2: 4, 'strain': 4},     # symmetric: bivector = strain
            {0: 4,  2: 4, 'strain': 4},     # uniform 4b all grades
        ]

    results = {}
    dim = W.shape[1]

    for bit_map in bit_maps:
        label = f"PG({bit_map.get(0,16)}/{bit_map.get(2,4)}/{bit_map.get('strain',8)})"
        W_q, info = quantize_per_grade(W, bit_map, verbose=False)
        avg = _avg_bit_width(dim, bit_map)
        nmse = info['nmse_rotor']
        impro = info['improvement_pct']
        results[label] = (nmse, impro, info)
        if verbose:
            print(f"  {label:30s} ~{avg:.1f}b avg   NMSE={nmse:.6f}  ({impro:+.1f}% vs RTN)")

    return results


def _estimate_avg_bits(W: torch.Tensor, bit_map: Dict) -> float:
    """Estimate the effective average bit-width for a per-grade quantization."""
    return _avg_bit_width(W.shape[1], bit_map)
