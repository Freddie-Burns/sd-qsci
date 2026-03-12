"""
H4 chain simulation to compare to hardware.
"""

from pathlib import Path

from pyscf import gto, scf
from sd_qsci import plot, utils
from sd_qsci import analysis


def main():
    """
    Run H6 triangular lattice energy calculations and analyze QSCI convergence.
    """
    # Paths to save data from parent directory
    stem = Path(__file__).stem
    base_data_dir = Path(__file__).parent / 'data' / stem

    # Run quantum chemistry calculations
    mol = build_h4_chain()
    rhf = scf.RHF(mol).run()

    patterns = {
        "antiferromagnetic": [1, -1, 1, -1],
        "ferromagnetic": [1, 1, -1, -1],
    }

    import numpy as np
    results_dict = {}

    for label, pattern in patterns.items():
        print(f"\n--- Running for pattern: {label} ---")
        data_dir = base_data_dir / label
        data_dir.mkdir(parents=True, exist_ok=True)

        # Seed UHF solution
        uhf = utils.solve_with_spin_pattern(mol, pattern, label)

        qc_results = analysis.run_quantum_chemistry_calculations(mol, rhf, bond_length=2, uhf=uhf)

        # Save mol object
        with open(data_dir / "mol.pyscf", "w") as f:
            f.write(mol.dumps())

        # Calculate convergence data
        conv_results = analysis.calc_convergence_data(qc_results)
        results_dict[label] = (qc_results, conv_results)

        # Save data to CSV
        analysis.save_convergence_data(data_dir, qc_results, conv_results)

        # Create plots
        plot.energy_vs_samples(data_dir, qc_results, conv_results)
        plot.convergence_comparison(data_dir, qc_results, conv_results)

        # Compute final QSCI wavefunction and plot coefficients
        print(f"\nComputing QSCI ground state wavefunction with {conv_results.max_size} configurations...")
        qsci_energy_final, qsci_vec, qsci_indices = analysis.calc_qsci_energy_with_size(
            qc_results.H, qc_results.sv.data, conv_results.max_size, return_vector=True)
        plot.statevector_coefficients(qc_results.sv.data, qc_results.fci_vec, data_dir, n_top=20)

        # Print summary
        print_summary(data_dir, qc_results, conv_results, qsci_energy_final)

    # Combined analysis
    print(f"\n--- Running for combined_mean pattern ---")
    combined_mean_dir = base_data_dir / "combined_mean"
    combined_mean_dir.mkdir(parents=True, exist_ok=True)

    # Get statevectors
    sv_afm = results_dict["antiferromagnetic"][0].sv.data
    sv_fm = results_dict["ferromagnetic"][0].sv.data

    # Method 1: Average probability distribution
    # P_combined = 0.5 * (|c_afm|^2 + |c_fm|^2)
    combined_probs_mean = 0.5 * (np.abs(sv_afm)**2 + np.abs(sv_fm)**2)
    combined_sv_mean = np.sqrt(combined_probs_mean)

    # Run analysis for combined_mean statevector
    uhf_combined = results_dict["antiferromagnetic"][0].uhf
    qc_results_mean = analysis.run_quantum_chemistry_calculations(
        mol, rhf, bond_length=2, statevector=combined_sv_mean, uhf=uhf_combined
    )

    # Save mol object
    with open(combined_mean_dir / "mol.pyscf", "w") as f:
        f.write(mol.dumps())

    # Calculate convergence data
    conv_results_mean = analysis.calc_convergence_data(qc_results_mean)

    # Save data to CSV
    analysis.save_convergence_data(combined_mean_dir, qc_results_mean, conv_results_mean)

    # Create plots
    plot.energy_vs_samples(combined_mean_dir, qc_results_mean, conv_results_mean)
    plot.convergence_comparison(combined_mean_dir, qc_results_mean, conv_results_mean)

    # Compute final QSCI wavefunction and plot coefficients
    print(f"\nComputing QSCI ground state wavefunction with {conv_results_mean.max_size} configurations...")
    qsci_energy_final_mean, _, _ = analysis.calc_qsci_energy_with_size(
        qc_results_mean.H, qc_results_mean.sv.data, conv_results_mean.max_size, return_vector=True)
    plot.statevector_coefficients(qc_results_mean.sv.data, qc_results_mean.fci_vec, combined_mean_dir, n_top=20)

    # Print summary
    print_summary(combined_mean_dir, qc_results_mean, conv_results_mean, qsci_energy_final_mean)

    # Method 2: Combined highest amplitude
    print(f"\n--- Running for combined_highest pattern ---")
    combined_highest_dir = base_data_dir / "combined_highest"
    combined_highest_dir.mkdir(parents=True, exist_ok=True)

    # Take higher amplitude (absolute) from AFM and FM
    combined_sv_highest = np.where(np.abs(sv_afm) >= np.abs(sv_fm), sv_afm, sv_fm)
    # Normalize
    combined_sv_highest = combined_sv_highest / np.linalg.norm(combined_sv_highest)

    # Run analysis for combined_highest statevector
    qc_results_highest = analysis.run_quantum_chemistry_calculations(
        mol, rhf, bond_length=2, statevector=combined_sv_highest, uhf=uhf_combined
    )

    # Save mol object
    with open(combined_highest_dir / "mol.pyscf", "w") as f:
        f.write(mol.dumps())

    # Calculate convergence data - doubled sample number for this method
    conv_results_highest = analysis.calc_convergence_data(qc_results_highest, sample_multiplier=2.0)

    # Save data to CSV
    analysis.save_convergence_data(combined_highest_dir, qc_results_highest, conv_results_highest)

    # Create plots
    plot.energy_vs_samples(combined_highest_dir, qc_results_highest, conv_results_highest)
    plot.convergence_comparison(combined_highest_dir, qc_results_highest, conv_results_highest)

    # Compute final QSCI wavefunction and plot coefficients
    print(f"\nComputing QSCI ground state wavefunction with {conv_results_highest.max_size} configurations...")
    qsci_energy_final_highest, _, _ = analysis.calc_qsci_energy_with_size(
        qc_results_highest.H, qc_results_highest.sv.data, conv_results_highest.max_size, return_vector=True)
    plot.statevector_coefficients(qc_results_highest.sv.data, qc_results_highest.fci_vec, combined_highest_dir, n_top=20)

    # Print summary
    print_summary(combined_highest_dir, qc_results_highest, conv_results_highest, qsci_energy_final_highest)

    # Multi-run comparison plots
    print(f"\nCreating multi-run comparison plots...")
    multi_results = {
        "Antiferromagnetic": results_dict["antiferromagnetic"],
        "Ferromagnetic": results_dict["ferromagnetic"],
        "Combined Mean": (qc_results_mean, conv_results_mean),
        "Combined Highest": (qc_results_highest, conv_results_highest),
    }
    plot.multi_run_convergence_comparison(base_data_dir, multi_results, title_prefix="H4 Chain")
    plot.multi_run_energy_vs_samples(base_data_dir, multi_results, title_prefix="H4 Chain")


def build_h4_chain():
    """
    Build a triangular lattice of 6 hydrogen atoms.
    """
    coords = [
        (0, 0, 0),
        (2, 0, 0),
        (4, 0, 0),
        (6, 0, 0),
    ]
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


def print_summary(data_dir: Path, qc_results: analysis.QuantumChemistryResults,
                 conv_results: analysis.ConvergenceResults, qsci_energy_final: float):
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
    print(f"\nData saved to '{data_dir}' directory:")
    print("  - h6_qsci_convergence.csv (full energy data)")
    print("  - h6_summary.csv (summary statistics)")
    print("  - h6_qsci_convergence.png (plot)")
    print("  - h6_energy_vs_samples.png (energy vs mean sample number)")
    print("  - statevector_coefficients.png (top 20 coefficients bar chart)")
    print("  - statevector_coefficients_full.png (all significant coefficients)")


if __name__ == "__main__":
    main()
