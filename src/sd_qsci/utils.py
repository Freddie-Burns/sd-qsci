import numpy as np
from pyscf import scf
from pyscf.gto import Mole
from pyscf.scf import RHF, UHF, stability


def uhf_from_rhf(mol: Mole, rhf: RHF) -> UHF:
    """
    Run a UHF calculation initialised from an RHF reference using PySCF
    stability analysis (RHF -> UHF external instability) to generate a
    spin-broken starting guess when appropriate.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecular system.
    rhf : pyscf.scf.hf.RHF
        A (preferably converged) RHF mean-field object.

    Returns
    -------
    uhf : pyscf.scf.uhf.UHF
        The resulting UHF mean-field object after SCF.
    """
    # Find MO coefficient matrices to seed UHF calculation
    # external refers to finding instability from RHF -> UHF
    try:
        mo_alpha, mo_beta = stability.rhf_external(rhf)
    except Exception as err:
        # RHF is likely stable -> fall back to direct RHF orbitals
        print(f"[info] RHF appears stable, starting UHF from RHF orbitals ({err}).")
        mo_alpha = mo_beta = rhf.mo_coeff

    # Build occupations for alpha and beta spins from electron counts
    n_alpha, n_beta = mol.nelec
    occ_a = np.array([1] * n_alpha + [0] * (mol.nao - n_alpha))
    occ_b = np.array([1] * n_beta + [0] * (mol.nao - n_beta))

    # Create UHF object and density from the (possibly) spin-broken MOs
    uhf = UHF(mol)
    dm0 = uhf.make_rdm1((mo_alpha, mo_beta), (occ_a, occ_b))

    # Run UHF starting from this density
    uhf.kernel(dm0=dm0)
    return uhf


def uhf_to_rhf_unitaries(mol: Mole, rhf: RHF, uhf: UHF) -> list:
    """
    Compute the orbital-space unitaries transforming RHF molecular orbitals
    into the corresponding α and β UHF molecular orbitals.

    This function determines the overlap-based transformation matrices that
    relate the restricted Hartree–Fock (RHF) molecular orbital (MO) coefficients
    to the unrestricted Hartree–Fock (UHF) α and β spin-orbital coefficients.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        PySCF molecule object providing the atomic basis and overlap integrals.
    rhf : pyscf.scf.hf.RHF
        A converged RHF mean-field object containing the restricted MO coefficients.
    uhf : pyscf.scf.uhf.UHF
        A converged UHF mean-field object containing separate α and β MO coefficients.

    Returns
    -------
    tuple of numpy.ndarray
        A pair of unitary transformation matrices ``(Ua, Ub)`` that map the
        RHF coefficient matrix onto the UHF α and β coefficient matrices,
        respectively:

        - ``Ua`` : array of shape (n_orb, n_orb)
            Transformation for α spin orbitals.
        - ``Ub`` : array of shape (n_orb, n_orb)
            Transformation for β spin orbitals.

    The final row of each matrix is flipped in sign if ``det(U) < 0`` to enforce
    a positive determinant (consistent orientation).

    Examples
    --------
    >>> mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    >>> rhf = scf.RHF(mol).run()
    >>> uhf = scf.UHF(mol).run()
    >>> Ua, Ub = uhf_to_rhf_unitaries(mol, rhf, uhf)
    >>> np.allclose(uhf.mo_coeff[0] @ Ua, rhf.mo_coeff)
    True
    """
    # RHF and UHF coefficient matrices
    C_rhf = rhf.mo_coeff
    Ca_uhf, Cb_uhf = uhf.mo_coeff  # α and β coefficients

    # Orbital-space unitaries that map UHF MOs -> RHF MOs (α and β)
    S = mol.intor("int1e_ovlp")
    Ua = Ca_uhf.conj().T @ S @ C_rhf
    Ub = Cb_uhf.conj().T @ S @ C_rhf

    # Todo: why is the row negative not the column?
    if np.linalg.det(Ua) < 0:
        Ua[0] *= -1
    if np.linalg.det(Ub) < 0:
        Ub[0] *= -1

    return Ua, Ub


def find_spin_symmetric_configs(n_bits, idx):
    """
    Identifies which configurations have/don't have their spin-flipped counterparts in the sample.

    Args:
        n_bits: Total number of bits in the configuration
        idx: Indices of sampled configurations

    Returns:
        sampled_configs: List of sampled configurations
        symm_configs: Set of all configurations and their spin-flipped counterparts
    """
    sampled_configs = [f"{i:>0{n_bits}b}" for i in idx]
    symm_configs = set()
    for config in sampled_configs:
        half = len(config) // 2
        spin_swapped = config[half:] + config[:half]
        symm_configs.add(config)
        symm_configs.add(spin_swapped)

    # Order the bitstrings for consistency
    sampled_configs = sorted(sampled_configs, key=lambda x: int(x, 2))
    symm_configs = sorted(symm_configs, key=lambda x: int(x, 2))
    return sampled_configs, symm_configs


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


