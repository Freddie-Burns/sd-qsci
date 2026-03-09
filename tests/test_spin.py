"""
Tests for spin operators and expectation values in sd_qsci.spin.
"""
import numpy as np
import pytest
from sd_qsci.spin import total_spin_S2, expectation


@pytest.fixture(scope="module")
def S2_one_orbital():
    """S^2 operator for a single spatial orbital (2 spin-orbitals, 4x4 matrix)."""
    return total_spin_S2(1)


def test_s2_vacuum(S2_one_orbital):
    """Vacuum state |00> has <S^2> = 0."""
    psi = np.array([1, 0, 0, 0], dtype=float)
    assert np.isclose(expectation(S2_one_orbital, psi).real, 0.0)


def test_s2_alpha_electron(S2_one_orbital):
    """Single alpha electron |10> (bit-1 set) has <S^2> = 3/4."""
    psi = np.array([0, 0, 1, 0], dtype=float)
    assert np.isclose(expectation(S2_one_orbital, psi).real, 0.75)


def test_s2_beta_electron(S2_one_orbital):
    """Single beta electron |01> (bit-0 set) has <S^2> = 3/4."""
    psi = np.array([0, 1, 0, 0], dtype=float)
    assert np.isclose(expectation(S2_one_orbital, psi).real, 0.75)


def test_s2_singlet_pair(S2_one_orbital):
    """Doubly-occupied orbital |11> is a singlet with <S^2> = 0."""
    psi = np.array([0, 0, 0, 1], dtype=float)
    assert np.isclose(expectation(S2_one_orbital, psi).real, 0.0)


def test_expectation_zero_norm_raises(S2_one_orbital):
    """expectation() must raise for a zero-norm state."""
    psi = np.zeros(4)
    with pytest.raises(ValueError, match="zero norm"):
        expectation(S2_one_orbital, psi)
