

from pathlib import Path

import numpy as np
import seaborn
import seaborn as sns
from matplotlib import pyplot as plt
from pyscf import gto, scf

from sd_qsci import analysis, circuit
from sd_qsci.greedy import greedy_from_results
from sd_qsci.utils import uhf_from_rhf


def main():
    """
    Run H6 chain energy calculations and analyze QSCI convergence.
    """
    bond_length = 2
    n_atoms = 6
    print(f"Running H{n_atoms} chain bond length: {bond_length:.2f} Angstrom")
    run_full_analysis(bond_length, n_atoms)


def run_full_analysis(bond_length, n_atoms):
    # Save under this script's full stem name inside this script's folder
    stem = Path(__file__).stem
    data_dir = Path(__file__).parent / 'data' / stem

    # Run quantum chemistry calculations
    mol = build_h_chain(bond_length, n_atoms)
    rhf = scf.RHF(mol).run()
    qc_results = analysis.run_quantum_chemistry_calculations(mol, rhf, bond_length)

    # Calculate convergence data with spin symmetry recovery
    conv_results = analysis.calc_convergence_data(qc_results, spin_symm=True)

    # Save data to CSV
    analysis.save_convergence_data(data_dir, qc_results, conv_results)

    # Plot only FCI subspace energy and spin‑symmetric QSCI energy vs subspace size
    sns.set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    df = conv_results.df.copy()

    # Greedy energy curve (based on FCI amplitudes thresholded by 1e-5)
    k_max = int(df['subspace_size'].max())
    _, greedy_energies = greedy_from_results(qc_results, k_max=k_max, amp_thresh=1e-5)
    x_greedy = np.arange(1, len(greedy_energies) + 1)

    # Prepare data series
    x_vals = df['subspace_size']
    fci_err = df['fci_subspace_energy'] - qc_results.fci_energy
    qsci_err = df['qsci_energy'] - qc_results.fci_energy
    qsci_symm_err = df['spin_symm_energy'] - qc_results.fci_energy
    greedy_err = np.asarray(greedy_energies) - qc_results.fci_energy

    ax.plot(
        x_vals,
        fci_err,
        's-',
        label='FCI subspace',
        linewidth=2,
        markersize=4,
    )
    ax.plot(
        x_vals,
        qsci_err,
        '^-',
        label='QSCI',
        linewidth=2,
        markersize=4,
    )
    ax.plot(
        x_vals,
        qsci_symm_err,
        '^-',
        label='QSCI spin recovery',
        linewidth=2,
        markersize=4,
    )
    ax.plot(
        x_greedy,
        greedy_err,
        'o-',
        label='Greedy (FCI-guided)',
        linewidth=2,
        markersize=4,
    )

    # Use a logarithmic-like scale on Y. Energies can be negative, so use
    # symmetric log to preserve sign while providing log scaling around zero.
    ax.set_yscale('log')
    ax.set_xlabel('Subspace size (number of configurations)')
    ax.set_ylabel('Energy error (Ha)')

    # Shade region below chemical accuracy threshold (1.6e-3 Ha)
    # and ensure the y-axis minimum is 1e-4.
    ax.axhspan(1e-4, 1.6e-3, facecolor='0.5', alpha=0.15, label='Chemical accuracy')

    # Set axis limits based on data (include greedy coverage)
    x_min, x_max_val = np.nanmin(x_vals), max(np.nanmax(x_vals), np.max(x_greedy))
    ax.set_xlim(left=x_min, right=x_max_val)

    # Upper y-limit should be the highest data point across all series
    y_top = np.nanmax(np.concatenate([fci_err.values, qsci_err.values, greedy_err]))
    ax.set_ylim(bottom=1e-4, top=y_top)
    ax.set_title('Energy error vs subspace size: H6 chain at 2A bond length')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    data_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(data_dir / 'fci_vs_qsci_spin.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def print_top_configs(bond_length=2, n_atoms=6):
    """
    Printing the top amplitudes from the rotated UHF statevector.
    This is to quickly see if the spin recovery is required/working.
    """
    # Run quantum chemistry calculations
    mol = build_h_chain(bond_length, n_atoms)
    rhf = scf.RHF(mol).run()

    uhf = uhf_from_rhf(mol, rhf)
    qc = circuit.rhf_uhf_orbital_rotation_circuit(mol, rhf, uhf)
    sv = circuit.simulate(qc)
    print("sv norm:", np.linalg.norm(sv.data))

    sorted_idx = np.argsort(np.abs(sv.data))[::-1]
    max_idx = sorted_idx[:20]
    n_bits = int(np.log2(sv.data.size))

    # For the highest amplitude configurations print their amplitude, index,
    # bitstring, and occupation vector in a table format.
    print("Norm" + ' '*6 + "Int" + ' '*5 + "Bitstring" + ' '*(n_bits-4) + "Occ")
    for i in max_idx:
        bitstring = bin(i)[2:].zfill(n_bits)
        occ_vec = occupation_vector(bitstring, n_bits)
        sv_amp = np.abs(sv.data[i])
        bitstring_ab = bitstring[:n_bits//2] + ' ' + bitstring[n_bits//2:]
        print(f"{sv_amp:.4f}    {i:4d}    {bitstring_ab}    {occ_vec}")
    print('\n')

    symm_amp = analysis.spin_symm_amplitudes(sv.data)
    print("symm amp norm:", np.linalg.norm(symm_amp))

    sorted_idx = np.argsort(np.abs(symm_amp))[::-1]
    max_idx = sorted_idx[:20]
    n_bits = int(np.log2(symm_amp.size))

    # For the highest amplitude configurations after spin recovery, print their
    # amplitude, index, bitstring, and occupation vector in a table format.
    print("Norm" + ' '*6 + "Int" + ' '*5 + "Bitstring" + ' '*(n_bits-4) + "Occ")
    for i in max_idx:
        bitstring = bin(i)[2:].zfill(n_bits)
        occ_vec = occupation_vector(bitstring, n_bits)
        sv_amp = np.abs(symm_amp[i])
        bitstring_ab = bitstring[:n_bits//2] + ' ' + bitstring[n_bits//2:]
        print(f"{sv_amp:.4f}    {i:4d}    {bitstring_ab}    {occ_vec}")


def occupation_vector(bitstring, n_bits) -> str:
    """
    Given a bitstring, return the occupation vector.
    Closed shells are represented by 0 or 2 for unoccupied or occupied.
    Open shells are represented by α or β for up or down spin occupancy.

    The RHF molecular orbitals are ordered from highest to lowest energy.
    This is inline with the Qiskit qubit ordering convention.

    Parameters
    ----------
    bitstring: str
        Binary representation of the bitstring.

    Returns
    -------
    occ_vec: str
        Occupation vector representation of the bitstring.
    """
    # Ensure string format
    bitstring = str(bitstring)
    alpha, beta = bitstring[:n_bits//2], bitstring[n_bits//2:]
    occ_vec = ""
    for i in range(n_bits//2):
        if alpha[i] == "1" and beta[i] == "1":
            occ_vec += '2'
        elif alpha[i] == "1":
            occ_vec += '\u03b1'
        elif beta[i] == "1":
            occ_vec += '\u03b2'
        else:
            occ_vec += '0'
    return occ_vec


def build_h_chain(bond_length, n_atoms=6) -> gto.Mole:
    """
    Build a chain of hydrogen atoms.
    """
    coords = [(i * bond_length, 0, 0) for i in range(n_atoms)]
    geometry = '; '.join([f'H {x:.8f} {y:.8f} {z:.8f}' for x, y, z in coords])
    mol = gto.Mole()
    mol.build(
        atom=geometry,
        unit='Angstrom',
        basis='sto-3g',
        charge=0,
        spin=0,
        verbose=0,
    )
    return mol


def print_summary(
        data_dir: Path,
        qc_results: analysis.QuantumChemistryResults,
        conv_results: analysis.ConvergenceResults,
        qsci_energy_final: float,
) -> None:
    """
    Print summary of results to console.
    """
    print(f"\nReference Energies:")
    print(f"  RHF: {qc_results.rhf.e_tot:.8f} Ha")
    print(f"  UHF: {qc_results.uhf.e_tot:.8f} Ha")
    print(f"  FCI: {qc_results.fci_energy:.8f} Ha")
    print(f"  QSCI (max subspace): {qsci_energy_final:.8f} Ha")
    print(f"\nFCI Solution:")
    print(f"  Number of configurations: {qc_results.n_fci_configs}")
    print(f"\nQSCI Convergence:")
    print(f"  Max subspace size: {conv_results.max_size}")
    print(f"  Min QSCI energy: {conv_results.df['qsci_energy'].min():.8f} Ha")
    print(f"  Energy difference to FCI: {conv_results.df['qsci_energy'].min() - qc_results.fci_energy:.2e} Ha")
    print(f"\nMilestones:")
    print(f"  Configs to fall below UHF: {conv_results.n_configs_below_uhf if conv_results.n_configs_below_uhf else 'Never achieved'}")
    print(f"  Configs to reach FCI: {conv_results.n_configs_reach_fci if conv_results.n_configs_reach_fci else 'Never achieved'}")
    print(f"\nData saved to '{data_dir}' directory:")
    print("  - h6_qsci_convergence.csv (full energy data)")
    print("  - h6_summary.csv (summary statistics)")
    print("  - h6_qsci_convergence.png (plot)")
    print("  - h6_energy_vs_samples.png (energy vs mean sample number)")
    print("  - statevector_coefficients.png (top 20 coefficients bar chart)")
    print("  - statevector_coefficients_full.png (all significant coefficients)")


if __name__ == "__main__":
    main()
