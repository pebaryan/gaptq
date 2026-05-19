"""Tests for geometric algebra primitives — covering gattrlm's CliffordAlgebra
engine and gamuon's grade_decompose/bivector_exp/rotor_apply."""

import math
import torch
import pytest

from gaptq.ga import (
    MultiVector,
    CliffordAlgebra,
    _blade_to_index,
    _index_to_blade,
    _grade,
    _gp_blade,
    _gp_blade_euclidean,
    build_gp_table,
    grade_decompose,
    rotor_from_angle,
    apply_rotor_to_vector,
    rotor_matrix_2d,
    block_diag_rotor_matrix,
    bivector_exp,
    _bivector_exp_2d,
    _bivector_exp_3d,
    rotor_apply,
    RandomRotorTransform,
)


# ═══════════════════════════════════════════════════════════════════════
#  Blade/Index Utilities
# ═══════════════════════════════════════════════════════════════════════


class TestBladeUtilities:
    def test_blade_to_index(self):
        assert _blade_to_index((0,), 3) == 1
        assert _blade_to_index((1,), 3) == 2
        assert _blade_to_index((0, 1), 3) == 3
        assert _blade_to_index((2,), 3) == 4
        assert _blade_to_index((0, 2), 3) == 5
        assert _blade_to_index((1, 2), 3) == 6
        assert _blade_to_index((0, 1, 2), 3) == 7
        assert _blade_to_index((), 3) == 0

    def test_index_to_blade(self):
        assert _index_to_blade(1, 3) == (0,)
        assert _index_to_blade(2, 3) == (1,)
        assert _index_to_blade(3, 3) == (0, 1)
        assert _index_to_blade(4, 3) == (2,)
        assert _index_to_blade(7, 3) == (0, 1, 2)
        assert _index_to_blade(0, 3) == ()

    def test_grade(self):
        assert _grade(0) == 0  # scalar
        assert _grade(1) == 1  # e0
        assert _grade(3) == 2  # e0^e1
        assert _grade(7) == 3  # e0^e1^e2


# ═══════════════════════════════════════════════════════════════════════
#  Geometric Product (from gattrlm's _gp_blade)
# ═══════════════════════════════════════════════════════════════════════


class TestGeometricProductBlade:
    def test_vector_square_euclidean(self):
        """In Cl(3,0), e_i^2 = 1 for all i."""
        for i in range(3):
            result_idx, sign = _gp_blade_euclidean(1 << i, 1 << i, 3)
            assert result_idx == 0  # scalar
            assert sign == 1  # e_i^2 = 1

    def test_anticommute(self):
        """e_i * e_j = -e_j * e_i for i != j."""
        # e0 * e1 = e0^e1 (sign +1)
        result_01, sign_01 = _gp_blade_euclidean(1 << 0, 1 << 1, 3)
        assert sign_01 == 1
        assert result_01 == _blade_to_index((0, 1), 3)

        # e1 * e0 = -e0^e1 (sign -1)
        result_10, sign_10 = _gp_blade_euclidean(1 << 1, 1 << 0, 3)
        assert sign_10 == -1
        assert result_10 == _blade_to_index((0, 1), 3)

    def test_triple_product(self):
        """e0 * e1 * e2 = e0^e1^e2 (sign +1)."""
        # e0 * (e1^e2)
        b12 = _blade_to_index((1, 2), 3)
        result_idx, sign = _gp_blade_euclidean(1 << 0, b12, 3)
        assert result_idx == _blade_to_index((0, 1, 2), 3)
        assert sign == 1

    def test_scalar_product(self):
        """1 * e_i = e_i."""
        for i in range(3):
            result_idx, sign = _gp_blade_euclidean(0, 1 << i, 3)
            assert result_idx == 1 << i
            assert sign == 1

    def test_ei_ei_squared(self):
        """e_i^2 = 1 in Cl(3,0)."""
        for i in range(3):
            result_idx, sign = _gp_blade_euclidean(1 << i, 1 << i, 3)
            assert result_idx == 0
            assert sign == 1


# ═══════════════════════════════════════════════════════════════════════
#  MultiVector class
# ═══════════════════════════════════════════════════════════════════════


class TestMultiVector:
    def test_scalar_creation(self):
        s = MultiVector.scalar(3.0, dim=3)
        assert s.coeffs[0] == 3.0
        assert s.coeffs[1:].sum() == 0

    def test_vector_creation(self):
        v = MultiVector.vector(torch.tensor([1.0, 2.0, 3.0]), dim=3)
        assert v.coeffs[1] == 1.0  # e1 component
        assert v.coeffs[2] == 2.0  # e2 component
        assert v.coeffs[4] == 3.0  # e3 component

    def test_grade_extraction(self):
        mv = MultiVector({3: 1.0, 5: 2.0}, dim=3)
        g0 = mv.grade(0)
        assert g0.coeffs[0] == 0
        g2 = mv.grade(2)
        assert g2.coeffs[3] == 1.0
        assert g2.coeffs[5] == 2.0

    def test_reverse_vector(self):
        """Reverse of a vector is itself (grade-1)."""
        v = MultiVector.vector(torch.tensor([1.0, 2.0, 3.0]), dim=3)
        r = v.reverse()
        assert torch.allclose(r.coeffs, v.coeffs)

    def test_reverse_bivector(self):
        """Reverse of a bivector: (-1)^(2*1/2) = -1."""
        bv = MultiVector({3: 1.0}, dim=3)
        r = bv.reverse()
        assert r.coeffs[3] == -1.0

    def test_geometric_product_vector_vector(self):
        """v * v = |v|^2 (scalar)."""
        v = MultiVector.vector(torch.tensor([1.0, 2.0, 3.0]), dim=3)
        p = v * v
        assert torch.allclose(p.coeffs[0], torch.tensor(14.0))
        assert p.coeffs[1:].sum() == 0

    def test_geometric_product_via_gp_blade(self):
        """Use _gp_blade_euclidean directly for geometric product."""
        # e1 * e2 = e1^e2
        idx, sign = _gp_blade_euclidean(1, 2, 3)
        assert idx == 3  # e1^e2
        assert sign == 1

        # e2 * e1 = -e1^e2
        idx, sign = _gp_blade_euclidean(2, 1, 3)
        assert idx == 3
        assert sign == -1


# ═══════════════════════════════════════════════════════════════════════
#  CliffordAlgebra Engine (from gattrlm)
# ═══════════════════════════════════════════════════════════════════════


class TestCliffordAlgebra:
    def test_initialization(self):
        """CliffordAlgebra for Cl(3,0) has 2^3 = 8 blades."""
        alg = CliffordAlgebra(dim=3)
        assert alg.num_blades == 8
        assert alg._grades.shape == (8,)

    def test_geometric_product(self):
        """e1 * e2 = e1^e2 via the engine."""
        alg = CliffordAlgebra(dim=3)
        v1 = torch.zeros(8)
        v1[1] = 1.0  # e1
        v2 = torch.zeros(8)
        v2[2] = 1.0  # e2
        gp = alg.geometric_product(v1, v2)
        assert gp[3] == 1.0  # e1^e2

    def test_geometric_product_square(self):
        """e1 * e1 = 1 (scalar)."""
        alg = CliffordAlgebra(dim=3)
        v = torch.zeros(8)
        v[1] = 1.0  # e1
        gp = alg.geometric_product(v, v)
        assert gp[0] == 1.0  # scalar

    def test_outer_product(self):
        """e1 ^ e2 = e1^e2 (grade 2)."""
        alg = CliffordAlgebra(dim=3)
        v1 = torch.zeros(8)
        v1[1] = 1.0
        v2 = torch.zeros(8)
        v2[2] = 1.0
        op = alg.outer_product(v1, v2)
        assert op[3] == pytest.approx(1.0)
        assert op[0] == 0.0  # No scalar contribution

    def test_inner_product(self):
        """e1 . e1 = 1 (scalar)."""
        alg = CliffordAlgebra(dim=3)
        v = torch.zeros(8)
        v[1] = 1.0
        ip = alg.inner_product(v, v)
        assert ip[0] == 1.0  # scalar

    def test_scalar_part(self):
        """scalar_part extracts grade-0 components."""
        alg = CliffordAlgebra(dim=3)
        mv = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        s = alg.scalar_part(mv)
        assert s[0] == 1.0
        assert s[1:].sum() == 0

    def test_grade_projection(self):
        """Grade projection for grade 1."""
        alg = CliffordAlgebra(dim=3)
        mv = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        g1 = alg.grade_projection(mv, 1)
        assert g1[1] == 2.0  # e1
        assert g1[2] == 3.0  # e2
        assert g1[4] == 5.0  # e3
        assert g1[0] == 0.0  # No scalar

    def test_sandwich_product(self):
        """R * v * R~ rotates a vector."""
        alg = CliffordAlgebra(dim=3)
        angle = math.pi / 4
        scalar = math.cos(angle / 2)
        biv = -math.sin(angle / 2)

        R = torch.zeros(8)
        R[0] = scalar
        R[3] = biv  # e1^e2

        # Reverse of R
        rev_sign = (-1) ** (2 * 1 // 2)  # grade 2 -> -1
        R_rev = R.clone()
        R_rev[3] *= rev_sign

        v = torch.zeros(8)
        v[1] = 1.0  # e1

        rotated = alg.sandwich_product(R, v, R_rev)
        vec_e1 = rotated[1].item()
        vec_e2 = rotated[2].item()
        expected_cos = math.cos(angle)
        expected_sin = math.sin(angle)
        assert vec_e1 == pytest.approx(expected_cos, abs=1e-6)
        assert vec_e2 == pytest.approx(expected_sin, abs=1e-6)

    def test_exp_bivector(self):
        """exp(bivector) produces a normalized rotor (scalar^2 + biv^2 = 1)."""
        alg = CliffordAlgebra(dim=3)
        biv = torch.zeros(8)
        biv[3] = 0.5  # bivector in e1^e2 plane
        rotor = alg.exp_bivector(biv)
        # Rotor norm: scalar^2 + ||biv||^2 = 1
        norm_sq = rotor[0]**2 + rotor[3]**2
        assert norm_sq == pytest.approx(1.0, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════════
#  Rotors (original + block-diagonal)
# ═══════════════════════════════════════════════════════════════════════


class TestRotors:
    def test_rotor_creation(self):
        """Basic rotor: R = cos(theta/2) - sin(theta/2)*e1^e2."""
        angle = math.pi / 2
        R = rotor_from_angle(angle, (0, 1), dim=3)
        assert torch.allclose(R.coeffs[0], torch.tensor(math.cos(angle / 2)))
        assert torch.allclose(R.coeffs[3], torch.tensor(-math.sin(angle / 2)))

    def test_rotor_preserves_norm(self):
        """|R v R~| = |v|."""
        angle = 0.7
        R = rotor_from_angle(angle, (0, 1), dim=3)
        v = torch.tensor([3.0, 4.0, 0.0])
        v_rotated = apply_rotor_to_vector(R, v, 3)
        assert torch.allclose(v_rotated.norm(), v.norm())

    def test_rotor_2d_rotation(self):
        """2D rotor matrix matches standard rotation."""
        angle = math.pi / 4
        R_mat = rotor_matrix_2d(angle)
        expected = torch.tensor([
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ])
        assert torch.allclose(R_mat, expected)

    def test_rotor_rotation_accuracy(self):
        """Rotate by 90 deg in xy-plane."""
        angle = math.pi / 2
        R = rotor_from_angle(angle, (0, 1), dim=3)
        v = torch.tensor([1.0, 0.0, 0.0])
        v_rotated = apply_rotor_to_vector(R, v, 3)
        expected = torch.tensor([0.0, 1.0, 0.0])
        assert torch.allclose(v_rotated, expected, atol=1e-6)

    def test_rotor_composition(self):
        """Composing rotors adds their angles."""
        angle1 = 0.3
        angle2 = 0.5
        R1 = rotor_from_angle(angle1, (0, 1), dim=3)
        R2 = rotor_from_angle(angle2, (0, 1), dim=3)
        R_composed = R2 * R1  # R2 then R1

        v = torch.tensor([1.0, 0.0, 0.0])
        v_composed = apply_rotor_to_vector(R_composed, v, 3)
        R_sum = rotor_from_angle(angle1 + angle2, (0, 1), dim=3)
        v_sum = apply_rotor_to_vector(R_sum, v, 3)
        assert torch.allclose(v_composed, v_sum, atol=1e-6)

    def test_block_diag_rotor_matrix(self):
        """Block diagonal matrix should be orthogonal."""
        angles = torch.tensor([0.3, 0.7, 1.2])
        R = block_diag_rotor_matrix(angles, 6)
        assert torch.allclose(R.T @ R, torch.eye(6), atol=1e-6)
        assert torch.allclose(R @ R.T, torch.eye(6), atol=1e-6)

    def test_random_rotor_transform(self):
        """RandomRotorTransform produces orthogonal matrices."""
        dim = 8
        transform = RandomRotorTransform(dim)
        R = transform.get_matrix()
        assert torch.allclose(R.T @ R, torch.eye(dim), atol=1e-6)

        x = torch.randn(dim)
        x_rot = transform(x)
        x_back = transform.inverse(x_rot)
        assert torch.allclose(x, x_back, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════
#  Grade Decompose (from gamuon)
# ═══════════════════════════════════════════════════════════════════════


class TestGradeDecompose:
    def test_decompose_symmetric(self):
        """A symmetric matrix has zero bivector component."""
        M = torch.tensor([[1.0, 2.0], [2.0, 3.0]])
        scalar, bivector, strain = grade_decompose(M)
        assert torch.allclose(bivector, torch.zeros(2, 2))

    def test_decompose_antisymmetric(self):
        """An antisymmetric matrix has zero scalar and strain."""
        M = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
        scalar, bivector, strain = grade_decompose(M)
        assert torch.allclose(scalar, torch.zeros(2, 2))
        assert torch.allclose(strain, torch.zeros(2, 2))
        assert torch.allclose(bivector, M)

    def test_decompose_reconstruction(self):
        """scalar + bivector + strain = M."""
        M = torch.randn(4, 4)
        scalar, bivector, strain = grade_decompose(M)
        reconstructed = scalar + bivector + strain
        assert torch.allclose(reconstructed, M, atol=1e-6)

    def test_scalar_is_diag(self):
        """Scalar component is proportional to identity."""
        M = torch.randn(5, 5)
        scalar, _, _ = grade_decompose(M)
        n = M.shape[0]
        expected_diag = M.trace() / n
        assert torch.allclose(scalar, torch.eye(n) * expected_diag)

    def test_bivector_is_antisymmetric(self):
        """Bivector component is antisymmetric."""
        M = torch.randn(6, 6)
        _, bivector, _ = grade_decompose(M)
        assert torch.allclose(bivector, -bivector.T)


# ═══════════════════════════════════════════════════════════════════════
#  Bivector Exponential (from gamuon)
# ═══════════════════════════════════════════════════════════════════════


class TestBivectorExp:
    def test_bivector_exp_2d(self):
        """bivector_exp for 2D produces a proper rotation matrix."""
        theta = 0.7
        B = torch.tensor([[0.0, theta], [-theta, 0.0]])
        R = bivector_exp(B)
        expected = torch.tensor([
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)],
        ])
        assert torch.allclose(R, expected, atol=1e-6)

    def test_bivector_exp_3d(self):
        """bivector_exp for 3D rotation about z-axis."""
        theta = 0.5
        B = torch.tensor([
            [0.0, -theta, 0.0],
            [theta, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        R = bivector_exp(B)
        expected = torch.tensor([
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ])
        assert torch.allclose(R, expected, atol=1e-6)

    def test_bivector_exp_orthogonal(self):
        """bivector_exp produces orthogonal matrix: R^T @ R = I."""
        B = torch.randn(4, 4)
        B = (B - B.T) / 2  # Make antisymmetric
        R = bivector_exp(B)
        assert torch.allclose(R.T @ R, torch.eye(4), atol=1e-5)

    def test_bivector_exp_3d_identity_at_zero(self):
        """B = 0 -> exp(0) = I."""
        B = torch.zeros(3, 3)
        R = bivector_exp(B)
        assert torch.allclose(R, torch.eye(3), atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════
#  Rotor Apply (from gamuon)
# ═══════════════════════════════════════════════════════════════════════


class TestRotorApply:
    def test_rotor_apply_identity(self):
        """R = I -> W stays the same."""
        W = torch.randn(4, 4)
        R = torch.eye(4)
        W_rotated = rotor_apply(R, W)
        assert torch.allclose(W_rotated, W)

    def test_rotor_apply_preserves_trace(self):
        """Sandwich product preserves the trace (scalar grade)."""
        W = torch.randn(4, 4)
        B = torch.randn(4, 4)
        B = (B - B.T) / 2
        R = bivector_exp(B)
        W_rotated = rotor_apply(R, W)
        assert torch.allclose(W.trace(), W_rotated.trace(), atol=1e-5)

    def test_rotor_apply_with_block_diag(self):
        """block_diag_rotor_matrix followed by rotor_apply should match."""
        angles = torch.tensor([0.3, 0.7])
        R = block_diag_rotor_matrix(angles, 4)
        W = torch.randn(4, 4)

        # Direct: R @ W @ R^T
        direct = R @ W @ R.T
        # rotor_apply
        applied = rotor_apply(R, W)
        assert torch.allclose(direct, applied)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
