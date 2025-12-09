"""
Try to improve the QSCI results by adding spin recovery.
"""
import itertools

import numpy as np


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
    perms = set(itertools.permutations(open_spins))

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


if __name__ == "__main__":
    bitstrings = spin_symmetric_configs("10100101")
    print(bitstrings)
