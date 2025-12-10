import itertools

import numpy as np
from scipy.sparse import csr_matrix, kron, identity
from scipy.sparse import csc_matrix


# ----- Pauli / ladder matrices (sparse) -----
I2  = csr_matrix(np.eye(2))
Z   = csr_matrix(np.array([[1, 0], [0,-1]], dtype=complex))
Sp  = csr_matrix(np.array([[0, 1], [0, 0]], dtype=complex))   # sigma^+
Sm  = csr_matrix(np.array([[0, 0], [1, 0]], dtype=complex))   # sigma^-


def jw_creation(n_modes: int, j: int) -> csc_matrix:
    """
    Construct the Jordan-Wigner creation operator for a given mode.

    Builds the fermionic creation operator a_j^dagger as a sparse matrix
    using the Jordan-Wigner transformation, which maps fermionic operators
    to qubit operators via the string of Pauli Z operators.

    Parameters
    ----------
    n_modes : int
        Total number of fermionic modes (qubits).
    j : int
        Mode index for the creation operator, in range [0, n_modes-1].
        Mode 0 corresponds to the leftmost qubit.

    Returns
    -------
    csc_matrix
        The creation operator as a 2^n_modes x 2^n_modes sparse CSC matrix.

    Notes
    -----
    The Jordan-Wigner transformation uses:
    a_j^dagger = (Z_0 ⊗ Z_1 ⊗ ... ⊗ Z_{j-1} ⊗ σ^+_j ⊗ I_{j+1} ⊗ ... ⊗ I_{n-1})
    where σ^+ is the Pauli raising operator.
    """
    op = None
    for k in range(n_modes):
        fac = Z if k < j else (Sp if k == j else I2)
        op = fac if op is None else kron(op, fac, format="csr")
    return op.tocsc()

def jw_annihilation(n_modes: int, j: int) -> csc_matrix:
    """
    Construct the Jordan-Wigner annihilation operator for a given mode.

    Builds the fermionic annihilation operator a_j as a sparse matrix
    using the Jordan-Wigner transformation, which maps fermionic operators
    to qubit operators via the string of Pauli Z operators.

    Parameters
    ----------
    n_modes : int
        Total number of fermionic modes (qubits).
    j : int
        Mode index for the annihilation operator, in range [0, n_modes-1].
        Mode 0 corresponds to the leftmost qubit.

    Returns
    -------
    csc_matrix
        The annihilation operator as a 2^n_modes x 2^n_modes sparse CSC matrix.

    Notes
    -----
    The Jordan-Wigner transformation uses:
    a_j = (Z_0 ⊗ Z_1 ⊗ ... ⊗ Z_{j-1} ⊗ σ^-_j ⊗ I_{j+1} ⊗ ... ⊗ I_{n-1})
    where σ^- is the Pauli lowering operator.
    """
    op = None
    for k in range(n_modes):
        fac = Z if k < j else (Sm if k == j else I2)
        op = fac if op is None else kron(op, fac, format="csr")
    return op.tocsc()


def number_op(n_modes: int, j: int) -> csc_matrix:
    """
    Construct the number operator for a given mode.

    The number operator n_j = a_j^dagger a_j counts the occupation
    of fermionic mode j. Its eigenvalues are 0 (unoccupied) or 1 (occupied).

    Parameters
    ----------
    n_modes : int
        Total number of fermionic modes (qubits).
    j : int
        Mode index for the number operator, in range [0, n_modes-1].

    Returns
    -------
    csc_matrix
        The number operator as a 2^n_modes x 2^n_modes sparse CSC matrix.

    Notes
    -----
    This operator is diagonal in the occupation number basis, with
    eigenvalues 0 or 1 corresponding to the absence or presence of
    a fermion in mode j.
    """
    a_dag = jw_creation(n_modes, j)
    a     = jw_annihilation(n_modes, j)
    return (a_dag @ a).tocsc()


def spin_symmetric_configs(config):
    """
    Generate all spin-symmetric configurations from a given bitstring.

    Takes a bitstring representing a fermionic occupation configuration in
    RHF ordering (alpha orbitals followed by beta orbitals) and generates
    all spin-symmetric configurations by permuting the open-shell spins
    while preserving the occupation pattern of each spatial orbital.

    Parameters
    ----------
    config : str
        A bitstring of even length representing the occupation configuration.
        The first half represents alpha spin orbitals, the second half
        represents beta spin orbitals. For example, "10100101" means
        alpha orbitals [1,0,1,0] and beta orbitals [0,1,0,1].

    Returns
    -------
    list of str
        A list of bitstrings representing all spin-symmetric configurations.
        Each configuration has the same spatial orbital occupation pattern
        but with different arrangements of open-shell spins.

    Raises
    ------
    ValueError
        If the input bitstring has odd length.

    Notes
    -----
    The function identifies closed-shell orbitals (doubly occupied or empty)
    and open-shell orbitals (singly occupied with either alpha or beta spin).
    It then generates all unique permutations of the open-shell spin
    assignments while maintaining the overall occupation pattern.

    For example, if orbital i is doubly occupied (alpha=1, beta=1), it
    remains doubly occupied in all configurations. If orbital j has only
    alpha spin (alpha=1, beta=0), it may exchange with other open-shell
    orbitals in the generated configurations.

    Examples
    --------
    >>> spin_symmetric_configs("10100101")
    # Returns configurations with the same spatial occupation but
    # different open-shell spin arrangements
    """
    n_bits = len(config)
    if n_bits % 2 != 0:
        raise ValueError("Bitstring must have even length.")

    # Alpha, beta, and occupation number vectors
    alpha, beta = config[:n_bits//2], config[n_bits//2:]
    occ_vec = [int(alpha[i]) + int(beta[i]) for i in range(n_bits//2)]

    # 1 for alpha spins, -1 for beta spins, 0 for closed shell
    open_shells = [int(alpha[i]) - int(beta[i]) for i in range(n_bits//2)]

    # Generate all unique permutations of open-shell spins
    open_spins = [x for x in open_shells if x]
    perms = list(itertools.permutations(open_spins))

    # Prepare a mask for the open shell indices in occ_vec
    occ_vec = np.array(occ_vec)
    open_mask = [True if x else False for x in open_shells]

    # Generate bitstrings from open shell permutations
    bitstrings = []
    for perm in perms:
        occ_vec[open_mask] = perm
        alpha = [0] * (n_bits//2)
        beta  = [0] * (n_bits//2)
        for i, n in enumerate(occ_vec):
            if n == 2:
                alpha[i] = beta[i] = 1
            elif n == 1:
                alpha[i] = 1
            elif n == -1:
                beta[i] = 1
        bits = alpha + beta
        bitstring = ''.join(map(str, bits))
        bitstrings.append(bitstring)

    return bitstrings


# ---------- Total spin operators ----------
def spin_ops(n_spatial_orbs: int):
    """
    Build the total spin operators S_z, S_+, and S_-.

    Constructs the spin operators for a system with RHF (restricted
    Hartree-Fock) orbital ordering, where alpha spin orbitals occupy
    indices [0, M-1] and beta spin orbitals occupy indices [M, 2M-1].

    Parameters
    ----------
    n_spatial_orbs : int
        Number of spatial orbitals (M). The total number of spin
        orbitals (modes) is 2*M.

    Returns
    -------
    Sz : csc_matrix
        The z-component of the total spin operator S_z = (1/2) * sum_p (n_pα - n_pβ).
        This is a 2^(2M) x 2^(2M) sparse CSC matrix.
    Splus : csc_matrix
        The spin raising operator S_+ = sum_p a^dagger_pα a_pβ.
        This is a 2^(2M) x 2^(2M) sparse CSC matrix.
    Sminus : csc_matrix
        The spin lowering operator S_- = sum_p a^dagger_pβ a_pα.
        This is a 2^(2M) x 2^(2M) sparse CSC matrix.

    Notes
    -----
    These operators satisfy the angular momentum commutation relations:
    [S_+, S_-] = 2*S_z
    [S_z, S_±] = ±S_±
    
    They can be combined to form S_x and S_y via:
    S_x = (S_+ + S_-) / 2
    S_y = (S_+ - S_-) / (2i)
    """
    M = n_spatial_orbs
    n_modes = 2*M
    dim = 1 << n_modes

    # Pre-build number operators for efficiency
    n_alpha = [number_op(n_modes, p)       for p in range(M)]
    n_beta  = [number_op(n_modes, M + p)   for p in range(M)]

    # S_z
    Sz = sum((n_alpha[p] - n_beta[p]) for p in range(M)) * 0.5

    # S_+ = sum_p a^\dagger_{pα} a_{pβ}
    # S_- = sum_p a^\dagger_{pβ} a_{pα}
    Splus_terms  = []
    Sminus_terms = []
    for p in range(M):
        a_alpha_dag = jw_creation(n_modes, p)
        a_beta      = jw_annihilation(n_modes, M + p)
        a_beta_dag  = jw_creation(n_modes, M + p)
        a_alpha     = jw_annihilation(n_modes, p)

        Splus_terms.append((a_alpha_dag @ a_beta).tocsc())
        Sminus_terms.append((a_beta_dag @ a_alpha).tocsc())

    Splus  = sum(Splus_terms)
    Sminus = sum(Sminus_terms)

    return Sz.tocsc(), Splus.tocsc(), Sminus.tocsc()


def total_spin_S2(n_spatial_orbs: int) -> csc_matrix:
    """
    Construct the total spin operator S^2.

    Builds the total spin squared operator using the relation:
    S^2 = S_z^2 + (1/2)(S_+ S_- + S_- S_+)

    Parameters
    ----------
    n_spatial_orbs : int
        Number of spatial orbitals (M). The total number of spin
        orbitals (modes) is 2*M.

    Returns
    -------
    csc_matrix
        The S^2 operator as a 2^(2M) x 2^(2M) sparse CSC matrix.

    Notes
    -----
    The eigenvalues of S^2 are S(S+1), where S is the total spin
    quantum number. For example:
    - Singlet (S=0): eigenvalue = 0
    - Doublet (S=1/2): eigenvalue = 0.75
    - Triplet (S=1): eigenvalue = 2
    """
    Sz, Splus, Sminus = spin_ops(n_spatial_orbs)
    return (Sz @ Sz + 0.5 * (Splus @ Sminus + Sminus @ Splus)).tocsc()


# ---------- Utilities ----------
def expectation(op: csc_matrix, psi: np.ndarray) -> complex:
    """
    Compute the expectation value of an operator with respect to a state.
    Calculates <psi|op|psi> for a given operator and state vector.
    The state vector is normalized internally before computing.

    Parameters
    ----------
    op : csc_matrix
        The operator as a sparse CSC matrix.
    psi : np.ndarray
        The state vector as a 1D numpy array.

    Returns
    -------
    complex
        The expectation value <psi|op|psi>.
    """
    psi = psi.astype(complex)
    norm = np.vdot(psi, psi)
    if norm == 0:
        raise ValueError("State vector has zero norm.")
    psi_norm = psi / np.sqrt(norm)
    return np.vdot(psi_norm, op @ psi_norm)


# ---------- Example usage ----------
if __name__ == "__main__":
    # One spatial orbital => modes: [alpha0, beta0]
    M = 1
    S2 = total_spin_S2(M)

    # Your example state: (|00> + |10>)/sqrt(2)
    # Basis order is [|00>, |01>, |10>, |11>] with alpha first, then beta.
    psi = np.array([0.707, 0.0, 0.707, 0.0], dtype=float)

    val = expectation(S2, psi)
    print(f"<S^2> = {val.real:.6f}")

    # Some sanity checks:
    # |00>  (vacuum)   -> <S^2> = 0
    # |10>  (α only)   -> <S^2> = 3/4
    # |01>  (β only)   -> <S^2> = 3/4
    # |11>  (αβ pair)  -> <S^2> = 0  (singlet on one spatial orbital)
    for label, vec in {
        "|00>": np.array([1,0,0,0], float),
        "|10>": np.array([0,0,1,0], float),
        "|01>": np.array([0,1,0,0], float),
        "|11>": np.array([0,0,0,1], float),
    }.items():
        print(label, "->", expectation(S2, vec).real)
