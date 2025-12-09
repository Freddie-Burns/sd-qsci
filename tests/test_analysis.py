import sys
import types
import importlib
import numpy as np
import pytest


@pytest.fixture
def _insert_stub_modules(monkeypatch):
    """
    Scoped stub modules for pyscf and qiskit using pytest.monkeypatch.
    This prevents polluting sys.modules across the full test session.
    """
    # Stub pyscf.gto and pyscf.fci
    pyscf_mod = types.ModuleType('pyscf')
    gto_mod = types.ModuleType('pyscf.gto')
    fci_mod = types.ModuleType('pyscf.fci')

    class Mole:
        def __init__(self):
            # Minimal attributes used in tests (set by caller)
            self.nao = None
            self.nelec = None

    gto_mod.Mole = Mole

    class FCI:
        def __init__(self, rhf):
            pass

        def kernel(self):
            # Return a dummy energy and vector; tests that call real FCI should not rely on this
            return 0.0, np.array([1.0])

    # cistring.make_strings used by fci_to_fock_space; provide minimal behaviour
    cistring = types.SimpleNamespace(make_strings=lambda rng, n: [0])

    fci_mod.FCI = FCI
    fci_mod.cistring = cistring

    pyscf_mod.gto = gto_mod
    pyscf_mod.fci = fci_mod

    monkeypatch.setitem(sys.modules, 'pyscf', pyscf_mod)
    monkeypatch.setitem(sys.modules, 'pyscf.gto', gto_mod)
    monkeypatch.setitem(sys.modules, 'pyscf.fci', fci_mod)

    # Stub qiskit.quantum_info.Statevector
    qiskit_mod = types.ModuleType('qiskit')
    qi_mod = types.ModuleType('qiskit.quantum_info')

    class Statevector:
        def __init__(self, data):
            self.data = np.array(data, dtype=complex)

        @classmethod
        def from_label(cls, label):
            # Not used in tests
            return cls(np.array([1.0]))

    qi_mod.Statevector = Statevector
    qiskit_mod.quantum_info = qi_mod

    monkeypatch.setitem(sys.modules, 'qiskit', qiskit_mod)
    monkeypatch.setitem(sys.modules, 'qiskit.quantum_info', qi_mod)


def test_calc_qsci_energy_with_size_small_matrix(_insert_stub_modules):
    """Test calc_qsci_energy_with_size against direct diagonalization of submatrix."""

    # Import the module after stubbing
    analysis = importlib.import_module('sd_qsci.analysis')
    importlib.reload(analysis)

    # Create a small Hermitian Hamiltonian (4x4)
    np.random.seed(0)
    A = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
    H = (A + A.conj().T) / 2

    # Create a statevector with varying amplitudes
    sv = analysis.Statevector(np.array([0.1, 0.4, 0.2, 0.05], dtype=complex))

    # Compute QSCI energy using top 2 configurations
    E0, psi_full, idx = analysis.calc_qsci_energy_with_size(H, sv, 2, return_vector=True)

    # Manually compute expected energy: restrict H to top-2 indices
    top_idx = np.argsort(np.abs(sv.data))[-2:]
    H_sub = H[np.ix_(top_idx, top_idx)]
    w, v = np.linalg.eigh(H_sub)
    expected_E0 = float(np.min(w))

    assert np.isclose(E0, expected_E0), f"E0 {E0} != expected {expected_E0}"
    # psi_full should have non-zero entries only at top indices
    nz = np.nonzero(psi_full)[0]
    assert set(nz) == set(top_idx)


def test_calc_fci_subspace_energy_small_matrix(_insert_stub_modules):
    """Test calc_fci_subspace_energy against direct diagonalization of submatrix."""
    analysis = importlib.import_module('sd_qsci.analysis')
    importlib.reload(analysis)

    # 5x5 Hamiltonian
    np.random.seed(1)
    A = np.random.randn(5, 5)
    H = (A + A.T) / 2

    # Fake FCI vector amplitudes
    fci_vec = np.array([0.01, 0.2, 0.5, 0.1, 0.05])

    # Choose top 3 amplitudes
    E0 = analysis.calc_fci_subspace_energy(H, fci_vec, 3)

    idx = np.argsort(np.abs(fci_vec))[-3:]
    H_sub = H[np.ix_(idx, idx)]
    w = np.linalg.eigvalsh(H_sub)
    expected_E0 = float(np.min(w))

    assert np.isclose(E0, expected_E0), f"E0 {E0} != expected {expected_E0}"

