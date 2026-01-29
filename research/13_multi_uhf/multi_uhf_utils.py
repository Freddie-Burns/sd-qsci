import numpy as np
from matplotlib import pyplot as plt
from pyscf import scf

def create_spin_pattern_dm(mol, spin_pattern):
    """
    Create initial density matrix with specified spin pattern.

    Args:
        mol: PySCF molecule object
        spin_pattern: list of +1 (alpha) or -1 (beta) for each atom
                     e.g., [1, 1, -1, -1] for "up up down down"
    """
    # Get initial guess (this gives us reasonable atomic orbitals)
    # mf_init = scf.UHF(mol)
    # dm_init = mf_init.get_init_guess()

    # Get atomic orbital information
    # For each atom, find which AOs belong to it
    ao_labels = mol.ao_labels()

    # Initialize alpha and beta density matrices
    nao = mol.nao
    dm_alpha = np.zeros((nao, nao))
    dm_beta = np.zeros((nao, nao))

    # Build density matrix based on spin pattern
    for atom_idx, spin in enumerate(spin_pattern):
        # Count AOs for this atom
        atom_aos = []
        for i, label in enumerate(ao_labels):
            if label.startswith(f'{atom_idx} '):
                atom_aos.append(i)

        # Assign electron to alpha or beta based on spin
        for ao in atom_aos:
            if spin > 0:  # Alpha electron
                dm_alpha[ao, ao] = 0.5
            else:  # Beta electron
                dm_beta[ao, ao] = 0.5

    # Stack alpha and beta to make UHF density matrix
    dm = np.array([dm_alpha, dm_beta])
    return dm


def solve_with_spin_pattern(mol, spin_pattern, label=""):
    """Solve UHF with a specific spin pattern as initial guess."""
    print(f"\n{'=' * 60}")
    print(f"Solving with spin pattern: {label}")
    print(f"Pattern: {spin_pattern}")
    print(f"{'=' * 60}")

    # Create initial density matrix
    dm_init = create_spin_pattern_dm(mol, spin_pattern)

    # Run UHF calculation
    mf = scf.UHF(mol)
    mf.kernel(dm_init)

    print(f"Converged: {mf.converged}")
    print(f"Energy: {mf.e_tot:.8f} Ha")
    print(f"<S^2>: {mf.spin_square()[0]:.6f}")

    # This prints Mulliken populations including spin
    mf.analyze()

    return mf


