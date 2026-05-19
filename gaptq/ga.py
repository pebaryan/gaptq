"""
Geometric Algebra (Clifford Algebra) primitives for rotor-based quantization.

Integrates three approaches:
  - gattrlm-style CliffordAlgebra engine with sparse Cayley tensor operations
    for general-purpose GA (geometric product, exp_bivector, sandwich product)
  - gamuon-style grade decomposition and rotor ops for matrix-level optimization
  - Custom block-diagonal rotors for the efficient quantization path (QuaRot-style)

Provides both general-purpose GA operations and quantization-specific utilities.
Supports Euclidean Cl(n,0) by default with metric Cl(p,q,r) available.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════
#  PART I: Blade / Index Utilities  (shared foundation)
# ═══════════════════════════════════════════════════════════════════════


def _blade_to_index(blade_indices: Tuple[int, ...], max_dim: int) -> int:
    """Map a sorted tuple of basis vector indices to a unique blade index.

    Uses bit representation: each bit corresponds to a basis vector.
    Example: (0, 2) -> bits 0 and 2 set -> 101_2 = 5
    """
    idx = 0
    for i in blade_indices:
        idx |= 1 << i
    return idx


def _index_to_blade(idx: int, max_dim: int) -> Tuple[int, ...]:
    """Convert a blade index back to a sorted tuple of basis vector indices."""
    return tuple(i for i in range(max_dim) if idx & (1 << i))


def _grade(idx: int) -> int:
    """Grade of the blade: number of basis vectors (bits set)."""
    return idx.bit_count()


# ═══════════════════════════════════════════════════════════════════════
#  PART II: Geometric Product of Basis Blades  (from gattrlm)
#  Supports Cl(p,q,r) — handles metric signs.
# ═══════════════════════════════════════════════════════════════════════


def _gp_blade(a: int, b: int, p: int, q: int, r: int, n: int) -> Tuple[int, int]:
    """Geometric product of two basis blades in Cl(p,q,r).

    Args:
        a, b: Bitmask indices of the two basis blades.
        p, q, r: Metric signature (p positive, q negative, r null dimensions).
        n: Total dimension (p + q + r).

    Returns:
        (result_bitmask, sign). Returns (0, 0) if the product vanishes (null).
    """
    result = a
    sign = 1
    for i in range(n):
        if b & (1 << i):
            # Count swaps needed to bring e_i past elements to its right
            swaps = bin(result >> (i + 1)).count("1")
            sign *= -1 if swaps & 1 else 1
            if result & (1 << i):
                # Contraction: e_i * e_i
                result &= ~(1 << i)
                if i < p:
                    pass  # e_i^2 = 1
                elif i < p + q:
                    sign *= -1  # e_i^2 = -1
                else:
                    return 0, 0  # e_i^2 = 0 (null)
            else:
                # Wedge: e_i ^ ... (no contraction)
                result |= (1 << i)
    return result, sign


def _gp_blade_euclidean(a: int, b: int, dim: int) -> Tuple[int, int]:
    """Geometric product of two basis blades in Cl(dim, 0) (Euclidean)."""
    return _gp_blade(a, b, dim, 0, 0, dim)


def build_gp_table(p: int, q: int, r: int = 0) -> torch.Tensor:
    """Build geometric product Cayley table for Cl(p,q,r).

    Returns T[dim, dim, dim] where T[i,j,k] = coeff of blade k in gp(basis_i, basis_j).

    NOTE: The table dimension is 2^(p+q+r), so this is only feasible for
    small total dimensions (n <= 12, giving 2^12 = 4096 blades).
    """
    n = p + q + r
    dim = 1 << n
    if n > 12:
        raise ValueError(
            f"Cannot build GP table for n={n} (dim={dim}): "
            "table would be too large. Maximum supported n is 12."
        )
    table = torch.zeros(dim, dim, dim, dtype=torch.float32)
    for i in range(dim):
        for j in range(dim):
            result, sign = _gp_blade(i, j, p, q, r, n)
            if result != 0 or sign != 0:
                table[i, j, result] = float(sign)
    return table


def _extract_sparse_gp(
    table: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract sparse (i, j, k, sign) triples from a dense GP Cayley table.

    Returns:
        i_idx, j_idx, k_idx, signs — each [nnz].
    """
    nnz_mask = table.abs() > 0
    indices = torch.where(nnz_mask)
    i_idx, j_idx, k_idx = indices
    signs = table[i_idx, j_idx, k_idx]
    return i_idx, j_idx, k_idx, signs


# ═══════════════════════════════════════════════════════════════════════
#  PART III: CliffordAlgebra Engine  (from gattrlm, adapted)
#  Provides batched GA operations for tensors of shape [..., 2^n].
#  NOTE: Only works for small dimensions (n <= 12) due to table size.
#  For large dimensions (n=64-256), use block_diag_rotor_matrix instead.
# ═══════════════════════════════════════════════════════════════════════


class CliffordAlgebra(nn.Module):
    """Geometric algebra engine for Cl(n, 0) (Euclidean) with sparse tensor ops.

    Precomputes Cayley tables and sparse index buffers for fast batched
    geometric, outer (wedge), and inner products.

    Multivectors are represented as tensors of shape [..., 2^n], where
    each basis blade corresponds to a bit in the index (dimension n).

    **NOTE**: This class builds a dense GP table of shape [2^n, 2^n, 2^n].
    For n > 12 the table becomes infeasibly large. This engine is intended
    for small-algebra operations (analysis, demos). For weight matrix
    quantization with dim=64-256, use `block_diag_rotor_matrix` instead.

    Args:
        dim: The dimension n of the Euclidean space Cl(n, 0). Must be <= 12.
        use_sparse: If True, use sparse index_select + index_add_ for
            efficiency. Otherwise use dense einsum.
    """

    def __init__(self, dim: int, use_sparse: bool = False):
        super().__init__()
        if dim > 12:
            raise ValueError(
                f"CliffordAlgebra does not support dim={dim} (max is 12). "
                "For large dimensions, use block_diag_rotor_matrix instead."
            )
        self.dim = dim
        self.num_blades = 1 << dim
        self.use_sparse = use_sparse

        # Precompute the GP table for Cl(dim, 0)
        gp_table = build_gp_table(dim, 0, 0)  # [2^n, 2^n, 2^n]
        self.register_buffer("_gp_table", gp_table)

        # Precompute grade masks
        grades = torch.zeros(self.num_blades, dtype=torch.long)
        for i in range(self.num_blades):
            grades[i] = _grade(i)
        self.register_buffer("_grades", grades)

        # Precompute sparse indices for GP
        i_idx, j_idx, k_idx, signs = _extract_sparse_gp(gp_table)
        self.register_buffer("_gp_i", i_idx)
        self.register_buffer("_gp_j", j_idx)
        self.register_buffer("_gp_k", k_idx)
        self.register_buffer("_gp_sign", signs)

        # Precompute outer (wedge) product sparse indices
        grades_flat = self._grades
        op_filter = (grades_flat[i_idx] + grades_flat[j_idx]) == grades_flat[k_idx]
        self.register_buffer("_op_i", i_idx[op_filter])
        self.register_buffer("_op_j", j_idx[op_filter])
        self.register_buffer("_op_k", k_idx[op_filter])
        self.register_buffer("_op_sign", signs[op_filter])

        # Precompute inner product (Hestenes) sparse indices
        ip_filter = (grades_flat[i_idx] - grades_flat[j_idx]).abs() == grades_flat[k_idx]
        self.register_buffer("_ip_i", i_idx[ip_filter])
        self.register_buffer("_ip_j", j_idx[ip_filter])
        self.register_buffer("_ip_k", k_idx[ip_filter])
        self.register_buffer("_ip_sign", signs[ip_filter])

        # Precompute reverse signs: (-1)^(k*(k-1)/2)
        rev_sign = torch.ones(self.num_blades)
        for i in range(self.num_blades):
            g = _grade(i)
            rev_sign[i] = (-1) ** (g * (g - 1) // 2)
        self.register_buffer("_rev_sign", rev_sign)

        # Build dense OP and IP tables from the GP table
        # (used by the dense path instead of filtering at runtime)
        gi = grades[:, None, None]     # [dim, 1, 1]
        gj = grades[None, :, None]     # [1, dim, 1]
        gk = grades[None, None, :]     # [1, 1, dim]

        op_mask = gk == (gi + gj)       # [dim, dim, dim]
        ip_mask = gk == (gi - gj).abs()  # [dim, dim, dim]

        self.register_buffer("_op_table", gp_table * op_mask)
        self.register_buffer("_ip_table", gp_table * ip_mask)

    def _sparse_contract(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        i_idx: torch.Tensor,
        j_idx: torch.Tensor,
        k_idx: torch.Tensor,
        signs: torch.Tensor,
    ) -> torch.Tensor:
        """Generic sparse contraction: out[k] += x[i] * y[j] * sign."""
        i_idx = i_idx.to(device=x.device)
        j_idx = j_idx.to(device=x.device)
        k_idx = k_idx.to(device=x.device)
        signs = signs.to(device=x.device, dtype=x.dtype)

        x_gathered = x.index_select(-1, i_idx)
        y_gathered = y.index_select(-1, j_idx)
        products = x_gathered * y_gathered * signs

        out = products.new_zeros(*products.shape[:-1], self.num_blades)
        return out.index_add_(-1, k_idx, products)

    def geometric_product(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Geometric product: x * y. Supports arbitrary batch dims.

        Args:
            x, y: Tensors of shape [..., 2^n] (multivector coefficients).

        Returns:
            Tensor of shape [..., 2^n] (multivector coefficients).
        """
        if self.use_sparse:
            return self._sparse_contract(
                x, y, self._gp_i, self._gp_j, self._gp_k, self._gp_sign
            )
        gp = self._gp_table.to(device=x.device, dtype=x.dtype)
        return torch.einsum("...i,...j,ijk->...k", x, y, gp)

    def outer_product(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Outer (wedge) product: grade-raising part of the geometric product."""
        if self.use_sparse:
            return self._sparse_contract(
                x, y, self._op_i, self._op_j, self._op_k, self._op_sign
            )
        op = self._op_table.to(device=x.device, dtype=x.dtype)
        return torch.einsum("...i,...j,ijk->...k", x, y, op)

    def inner_product(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Hestenes inner product: grade-lowering part of the geometric product."""
        if self.use_sparse:
            return self._sparse_contract(
                x, y, self._ip_i, self._ip_j, self._ip_k, self._ip_sign
            )
        ip = self._ip_table.to(device=x.device, dtype=x.dtype)
        return torch.einsum("...i,...j,ijk->...k", x, y, ip)

    def sandwich_product(
        self, left: torch.Tensor, x: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        """Sandwich product: left * x * right (typically R * v * R~)."""
        temp = self.geometric_product(left, x)
        return self.geometric_product(temp, right)

    def reverse(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse (dagger) operation on multivector coefficients.

        For a grade-k blade, reverse introduces (-1)^(k*(k-1)/2).
        """
        rev = self._rev_sign.to(device=x.device, dtype=x.dtype)
        return x * rev

    def grade_projection(self, x: torch.Tensor, g: int) -> torch.Tensor:
        """Extract the grade-g part of a multivector."""
        grades = self._grades.to(device=x.device)
        mask = (grades == g).float()
        return x * mask

    def scalar_part(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the scalar (grade-0) part. Keeps the [..., 2^n] shape."""
        return self.grade_projection(x, 0)

    def norm_sq(self, x: torch.Tensor) -> torch.Tensor:
        """Squared Euclidean norm of a multivector (sum of squared coeffs)."""
        return (x ** 2).sum(dim=-1, keepdim=True)

    def exp_bivector(self, biv: torch.Tensor) -> torch.Tensor:
        """Exponential of a bivector in Cl(n,0), producing a rotor.

        For B^2 < 0 (elliptic, Euclidean case): exp(B) = cos|B| + (B/|B|) sin|B|

        Uses Taylor expansion for small angles to avoid numerical instability
        and maintain gradient flow.

        Args:
            biv: Tensor of shape [..., 2^n] (only grade-2 components non-zero).

        Returns:
            Rotor: [..., 2^n] (scalar + bivector components).
        """
        B2 = self.geometric_product(biv, biv)
        scalar_B2 = self.scalar_part(B2)  # [..., 2^n], only index 0 non-zero

        theta_sq = scalar_B2.abs()
        theta_sq_safe = theta_sq.clamp(min=1e-16)
        theta_scalar = theta_sq_safe.sqrt()
        theta_val = theta_scalar[..., 0:1]  # [..., 1]

        # Euclidean signature: B^2 < 0 -> elliptic
        one = torch.zeros_like(biv)
        one[..., 0:1] = 1.0

        # Use Taylor expansion for small theta to avoid 1/theta blowup
        small_angle = 1e-4
        is_small = theta_val < small_angle

        # --- Standard formula ---
        theta_clamped = theta_val.clamp(min=small_angle)
        cos_std = theta_clamped.cos()
        sin_over_theta_std = theta_clamped.sin() / theta_clamped

        # --- Taylor expansion for small theta ---
        cos_taylor = 1.0 - 0.5 * theta_val.square()
        sin_over_theta_taylor = 1.0 - theta_val.square() / 6.0

        cos_val = torch.where(is_small, cos_taylor, cos_std)
        sin_over_theta = torch.where(is_small, sin_over_theta_taylor, sin_over_theta_std)

        return cos_val * one + sin_over_theta * biv


# ═══════════════════════════════════════════════════════════════════════
#  PART IV: MultiVector  (educational/debugging class)
#  Simple wrapper around coefficient tensors.
# ═══════════════════════════════════════════════════════════════════════


class MultiVector:
    """A multivector in geometric algebra Cl(n, 0) (Euclidean).

    Coefficients are stored as a 1D tensor of length 2^n, where each entry
    corresponds to a basis blade indexed by its bit representation.

    For Cl(3,0):
        index 0: scalar (1)
        index 1: e1
        index 2: e2
        index 3: e1^e2 (bivector)
        index 4: e3
        index 5: e1^e3
        index 6: e2^e3
        index 7: e1^e2^e3 (trivector/pseudoscalar)
    """

    def __init__(self, coeffs: Union[torch.Tensor, dict, list], dim: int = 3):
        self.dim = dim
        self.num_blades = 1 << dim

        if isinstance(coeffs, dict):
            self.coeffs = torch.zeros(self.num_blades)
            for idx, val in coeffs.items():
                self.coeffs[idx] = val
        elif isinstance(coeffs, torch.Tensor):
            if coeffs.shape[-1] != self.num_blades:
                raise ValueError(
                    f"Expected last dim {self.num_blades}, got {coeffs.shape[-1]}"
                )
            self.coeffs = coeffs.squeeze()
        else:
            self.coeffs = torch.tensor(list(coeffs), dtype=torch.float32)

    @staticmethod
    def scalar(val: float, dim: int = 3) -> "MultiVector":
        return MultiVector({0: val}, dim)

    @staticmethod
    def vector(vals: torch.Tensor, dim: int = 3) -> "MultiVector":
        """Create a grade-1 multivector (vector) from a tensor."""
        blade_indices = {1 << i: vals[i] for i in range(min(len(vals), dim))}
        return MultiVector(blade_indices, dim)

    def grade(self, g: int) -> "MultiVector":
        """Extract the grade-g part of this multivector."""
        result = torch.zeros_like(self.coeffs)
        for idx in range(self.num_blades):
            if _grade(idx) == g:
                result[idx] = self.coeffs[idx]
        return MultiVector(result, self.dim)

    def reverse(self) -> "MultiVector":
        """Reverse (dagger) operation: reverse the order of basis vectors."""
        result = self.coeffs.clone()
        for i in range(self.num_blades):
            g = _grade(i)
            result[i] *= (-1) ** (g * (g - 1) // 2)
        return MultiVector(result, self.dim)

    def __mul__(self, other: "MultiVector") -> "MultiVector":
        """Geometric product using the efficient _gp_blade approach."""
        result = torch.zeros(self.num_blades)
        for i in range(self.num_blades):
            si = self.coeffs[i]
            if si == 0:
                continue
            for j in range(self.num_blades):
                sj = other.coeffs[j]
                if sj == 0:
                    continue
                result_idx, sign = _gp_blade_euclidean(i, j, self.dim)
                result[result_idx] += sign * si * sj
        return MultiVector(result, self.dim)

    def __rmul__(self, scalar: float) -> "MultiVector":
        return MultiVector(self.coeffs * scalar, self.dim)

    def __add__(self, other: "MultiVector") -> "MultiVector":
        return MultiVector(self.coeffs + other.coeffs, self.dim)

    def __sub__(self, other: "MultiVector") -> "MultiVector":
        return MultiVector(self.coeffs - other.coeffs, self.dim)

    def __repr__(self) -> str:
        return f"MultiVector(dim={self.dim}, coeffs={self.coeffs})"


# ═══════════════════════════════════════════════════════════════════════
#  PART V: Matrix-Level Grade Operations  (from gamuon, adapted)
#  Decompose matrices into geometric grades.
# ═══════════════════════════════════════════════════════════════════════


def grade_decompose(M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose a square matrix into its geometric grades in Cl(n,0).

    In GA, a matrix M decomposes into:
      - Scalar grade    <M>_0 = tr(M)/n * I          (isotropic dilation)
      - Bivector grade  <M>_2 = (M - M^T) / 2        (antisymmetric: rotation)
      - Strain grade    <M>_+ = (M + M^T)/2 - <M>_0  (symmetric traceless: stretch)

    Args:
        M: Square matrix of shape (n, n).

    Returns:
        Tuple of (scalar_part, bivector_part, strain_part), each (n, n).
    """
    n = M.shape[-1]
    I = torch.eye(n, device=M.device, dtype=M.dtype)

    if M.dim() == 2:
        scalar = (M.trace() / n) * I
    else:
        trace = M.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        scalar = (trace / n) * I

    bivector = (M - M.transpose(-2, -1)) / 2
    symmetric = (M + M.transpose(-2, -1)) / 2
    strain = symmetric - scalar
    return scalar, bivector, strain


# ═══════════════════════════════════════════════════════════════════════
#  PART VI: Rotor Functions  (block-diagonal + full matrix)
# ═══════════════════════════════════════════════════════════════════════


def rotor_from_angle(
    angle: float, plane: Tuple[int, int], dim: int = 3
) -> MultiVector:
    """Create a rotor for rotation in a 2D plane in Cl(n, 0).

    R = cos(theta/2) - sin(theta/2) * (e_i ^ e_j)
    """
    i, j = plane
    biv_idx = _blade_to_index((min(i, j), max(i, j)), dim)

    coeffs = torch.zeros(1 << dim)
    coeffs[0] = math.cos(angle / 2)  # scalar part
    coeffs[biv_idx] = -math.sin(angle / 2)  # bivector part (e_i ^ e_j)

    return MultiVector(coeffs, dim)


def apply_rotor_to_vector(
    rotor: MultiVector, vector: torch.Tensor, dim: int
) -> torch.Tensor:
    """Apply a rotor to a vector: v' = R * v * Rdag

    Args:
        rotor: A rotor (MultiVector).
        vector: 1D tensor of length `dim`.
        dim: Dimension of the space.

    Returns:
        Rotated vector (1D tensor of length `dim`).
    """
    v_mv = MultiVector.vector(vector, dim)
    rotated = rotor * v_mv * rotor.reverse()
    vec_part = rotated.grade(1)
    return torch.stack([vec_part.coeffs[1 << i] for i in range(dim)])


def rotor_matrix_2d(angle: float) -> torch.Tensor:
    """Get the 2x2 rotation matrix for a rotor with the given angle.

    For rotor R = cos(theta/2) - sin(theta/2)*e1^e2 in Cl(2,0),
    the matrix representation is:
        [[cos(theta), -sin(theta)],
         [sin(theta),  cos(theta)]]
    """
    c = math.cos(angle)
    s = math.sin(angle)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float32)


def block_diag_rotor_matrix(
    angles: torch.Tensor, dim: int
) -> torch.Tensor:
    """Build a block-diagonal rotation matrix from 2D rotor angles.

    Each rotor angle theta generates a 2x2 rotation block.
    Uses torch operations to maintain the computational graph.

    Args:
        angles: 1D tensor of angles, length = dim // 2.
        dim: Total dimension. If odd, the last dimension is left as-is.

    Returns:
        A (dim x dim) block-diagonal rotation matrix (with gradient flow).
    """
    if dim < 2:
        return torch.eye(dim, dtype=angles.dtype, device=angles.device)

    blocks = []
    for i in range(dim // 2):
        angle = angles[i] if i < len(angles) else torch.tensor(0.0, device=angles.device)
        c = torch.cos(angle).unsqueeze(0)
        s = torch.sin(angle).unsqueeze(0)
        row1 = torch.cat([c, -s], dim=0).unsqueeze(0)
        row2 = torch.cat([s, c], dim=0).unsqueeze(0)
        blocks.append(torch.cat([row1, row2], dim=0))

    if dim % 2 == 1:
        one = torch.tensor([[1.0]], dtype=angles.dtype, device=angles.device)
        blocks.append(one)

    return torch.block_diag(*blocks)


# ─── Full matrix rotor operations (from gamuon, adapted) ──────────────


def bivector_exp(B: torch.Tensor) -> torch.Tensor:
    """Compute exponential of a bivector (antisymmetric matrix) to get a rotor.

    The matrix exponential of an antisymmetric matrix B produces an
    orthogonal matrix R = exp(B) in SO(n), which acts as a rotor.

    For n=2: closed-form Rodrigues formula
    For n=3: closed-form Rodrigues formula
    For n>3: falls back to torch.matrix_exp

    Args:
        B: Antisymmetric matrix of shape (n, n) (bivector in Cl(n,0)).

    Returns:
        Orthogonal matrix R of shape (n, n) (rotor in Spin(n)).
    """
    n = B.shape[-1]

    if n == 1:
        return torch.eye(1, device=B.device, dtype=B.dtype)

    if n == 2:
        return _bivector_exp_2d(B)
    elif n == 3:
        return _bivector_exp_3d(B)
    else:
        return torch.matrix_exp(B)


def _bivector_exp_2d(B: torch.Tensor) -> torch.Tensor:
    """Closed-form 2D rotation matrix from a bivector.

    For B = [[0, theta], [-theta, 0]], exp(B) = [[cos(theta), -sin(theta)],
                                                   [sin(theta),  cos(theta)]]
    """
    theta = B[0, 1]  # The only independent component
    c = torch.cos(theta)
    s = torch.sin(theta)
    row1 = torch.cat([c.unsqueeze(0), (-s).unsqueeze(0)], dim=0)
    row2 = torch.cat([s.unsqueeze(0), c.unsqueeze(0)], dim=0)
    return torch.stack([row1, row2], dim=0)


def _bivector_exp_3d(B: torch.Tensor) -> torch.Tensor:
    """Closed-form 3D rotation matrix from a bivector (Rodrigues formula).

    For B = [[0, -omega_z, omega_y],
             [omega_z, 0, -omega_x],
             [-omega_y, omega_x, 0]]

    exp(B) = I + (sin(theta)/theta) * B + ((1-cos(theta))/theta^2) * B^2

    Uses Taylor expansion for numerical stability when theta is small.
    """
    omega_x = B[2, 1]
    omega_y = B[0, 2]
    omega_z = B[1, 0]
    theta_sq = omega_x**2 + omega_y**2 + omega_z**2
    theta = theta_sq.sqrt()

    I = torch.eye(3, device=B.device, dtype=B.dtype)

    # Taylor expansion for small angles to avoid 0/0
    small_angle = 1e-8
    is_zero = theta < small_angle

    # Standard formula
    theta_safe = theta.clamp(min=small_angle)
    sin_theta_over_theta = theta.sin() / theta_safe
    one_minus_cos_over_theta_sq = (1.0 - theta.cos()) / theta_sq.clamp(min=small_angle**2)

    # Taylor for small theta: sin(theta)/theta ~ 1 - theta^2/6,
    # (1-cos(theta))/theta^2 ~ 1/2 - theta^2/24
    sin_taylor = 1.0 - theta_sq / 6.0
    cos_taylor = 0.5 - theta_sq / 24.0

    sin_term = torch.where(is_zero, sin_taylor, sin_theta_over_theta)
    cos_term = torch.where(is_zero, cos_taylor, one_minus_cos_over_theta_sq)

    return I + sin_term * B + cos_term * (B @ B)


def rotor_apply(R: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Apply a rotor to a matrix via versor sandwich product.

    In Cl(n,0), a rotor R acts on a multivector W via conjugation:
        W' = R @ W @ R^T

    This preserves the geometric structure (grade) of W.

    Args:
        R: Orthogonal matrix of shape (n, n) (rotor).
        W: Matrix of shape (n, n).

    Returns:
        Rotated matrix of shape (n, n).
    """
    return R @ W @ R.transpose(-2, -1)


# ═══════════════════════════════════════════════════════════════════════
#  PART VII: Random Rotor Transform (QuaRot-style)
# ═══════════════════════════════════════════════════════════════════════


class RandomRotorTransform(nn.Module):
    """A random orthogonal transform composed of 2D rotors.

    This is the GA equivalent of the random Hadamard/Householder transforms
    used in QuaRot. Instead of fixed Hadamard matrices, we use rotors with
    randomly initialized angles (optionally optimizable).

    For non-square matrices, the rotation is applied to the input (column) space.
    """

    def __init__(self, dim: int, requires_grad: bool = False):
        super().__init__()
        self.dim = dim
        n_rotors = max(dim // 2, 1)
        angles = torch.rand(n_rotors) * 2 * math.pi
        self.angles = nn.Parameter(angles, requires_grad=requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the rotation to a tensor.

        For 1D (vector): R @ x
        For 2D (matrix): x @ R.T (rotate input/column space)
        """
        R = block_diag_rotor_matrix(self.angles, self.dim)
        if x.ndim == 1:
            return R @ x
        return x @ R.T

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the inverse rotation (R^T)."""
        R = block_diag_rotor_matrix(self.angles, self.dim)
        if x.ndim == 1:
            return R.T @ x
        return x @ R

    def get_matrix(self) -> torch.Tensor:
        """Get the full block-diagonal rotation matrix."""
        return block_diag_rotor_matrix(self.angles, self.dim)
