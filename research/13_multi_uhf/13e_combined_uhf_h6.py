"""
Combine RHF basis counts from multiple UHF solutions.
Compare resulting energy calculations to FCI energy.

1) Seed two UHF solutions with different spin patterns
2) Run quantum chemistry calculations for each solution
3) Create and simulate orbital rotation circuit from UHF to RHF.
4) Combine results into single statevector that would be reconstructed
   from the likely counts.
5) Run spin recovery on this statevector.
6) Calculate energy per subspace and samples.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches, transforms
from pyscf import gto, scf

import multi_uhf_utils as utils
from sd_qsci import analysis, circuit, hamiltonian, plot
from sd_qsci.analysis import spin_closed_subspace_sizes

# Setup
bond_length = 2.0
n_atoms = 6
print(f"Running H{n_atoms} chain bond length: {bond_length:.2f} Angstrom")

# Save under this script's full stem name inside this script's folder
stem = Path(__file__).stem
data_dir = Path(
    __file__).parent / 'data' / stem

# Run quantum chemistry calculations
# Define H6 in a line
mol = gto.M(
    atom='''
    H 0 0 0
    H 0 0 2
    H 0 0 4
    H 0 0 6
    H 0 0 8
    H 0 0 10
    ''',
    basis='sto-3g',
    spin=0,
    charge=0
)

patterns = {
    "antiferromagnetic": [1, -1, 1, -1, 1, -1],  # lowest energy soln
    "ferromagnetic": [1, 1, 1, -1, -1, -1],  # higher energy soln
}

rhf = scf.RHF(mol).run()
H = hamiltonian.hamiltonian_from_pyscf(mol, rhf)
all_results = {}
statevectors = []

for label, pattern in patterns.items():
    print(f"\n--- Running for pattern: {label} ---")
    # SCF to UHF solution from seeded spin pattern.
    uhf = utils.solve_with_spin_pattern(mol, pattern, label)

    #
    qc_res = analysis.run_quantum_chemistry_calculations(
        mol,
        rhf,
        bond_length,
        uhf=uhf,
    )
    conv_res = analysis.calc_convergence_data(qc_res, spin_symm=True)
    all_results[label] = (qc_res, conv_res)

    qc = circuit.rhf_uhf_orbital_rotation_circuit(mol, rhf, uhf)
    statevectors.append(circuit.simulate(qc))

# Combine statevectors
combined_probs = np.zeros(statevectors[0].shape)
for sv in statevectors:
    probs = sv.data * np.conj(sv.data)
    combined_probs += probs.real
combined_probs /= len(statevectors)  # Average probabilities
combined_amps = np.sqrt(combined_probs)

print("\n--- Running for Combined Amplitudes ---")
# Use the combined amplitudes as the statevector
combined_qc_results = analysis.run_quantum_chemistry_calculations(
    mol,
    rhf,
    bond_length,
    statevector=combined_amps,
)
combined_conv_results = analysis.calc_convergence_data(combined_qc_results,
                                                       spin_symm=True)
all_results["Combined"] = (combined_qc_results, combined_conv_results)

# Plots
print(f"Creating plots in {data_dir}...")
data_dir.mkdir(parents=True, exist_ok=True)

# Plot coefficients for combined vs FCI
plot.statevector_coefficients(
    combined_qc_results.sv.data,
    combined_qc_results.fci_vec,
    data_dir,
    n_top=20,
    title="Combined vs FCI Amplitudes"
)

# Also plot for individual solutions
for label, (qc_res, conv_res) in all_results.items():
    if label == "Combined":
        continue
    pattern_dir = data_dir / label.replace(" ", "_")
    pattern_dir.mkdir(parents=True, exist_ok=True)
    plot.statevector_coefficients(
        qc_res.sv.data,
        qc_res.fci_vec,
        pattern_dir,
        n_top=20,
        title=f"{label} vs FCI Amplitudes"
    )


# Use comparison plots from 13c
def compare_convergence_custom(data_dir, results_dict, title_prefix=None):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))
    y_lo = 1.0e-4
    chem_acc = 1.6e-3
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.add_patch(
        patches.Rectangle((0.0, y_lo), 1.0, chem_acc - y_lo, transform=trans,
                          facecolor='#D0D0D0', alpha=0.35, zorder=0))
    ax.add_patch(
        patches.Rectangle((0.0, y_lo), 1.0, chem_acc - y_lo, transform=trans,
                          facecolor='none', edgecolor='#2ca02c', hatch='///',
                          linewidth=0.0, zorder=0))
    ax.set_ylim(bottom=y_lo)

    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
    for i, (label, (qc_res, conv_res)) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        fci_E = qc_res.fci_energy
        ax.plot(conv_res.df['subspace_size'],
                conv_res.df['qsci_energy'] - fci_E, 'o-', label=f"{label} QSCI",
                color=color)
        spin_closed_sizes = set(spin_closed_subspace_sizes(qc_res.sv.data))
        df_symm = conv_res.df[
            conv_res.df['subspace_size'].isin(spin_closed_sizes)]
        ax.plot(df_symm['subspace_size'], df_symm['spin_symm_energy'] - fci_E,
                '^-', label=f"{label} Spin-Rec", color=color, alpha=0.7)

    # Plot FCI and RHF from the last results (should be the same for all)
    ax.plot(conv_res.df['subspace_size'],
            conv_res.df['fci_subspace_energy'] - fci_E, 's-',
            label='FCI Subspace', color='black', alpha=0.5)
    ax.axhline(y=qc_res.rhf.e_tot - fci_E, color='grey', linestyle='--',
               label='RHF')

    ax.set_yscale('log')
    ax.set_xlabel('Subspace Size')
    ax.set_ylabel('Energy Error (Ha)')
    title = "Convergence Comparison"
    if title_prefix: title = f"{title_prefix}: {title}"
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(data_dir / 'compare_convergence.png', dpi=300)
    plt.close()


def compare_energy_vs_samples_custom(data_dir, results_dict, title_prefix=None):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))
    y_lo = 1.0e-4
    chem_acc = 1.6e-3
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.add_patch(
        patches.Rectangle((0.0, y_lo), 1.0, chem_acc - y_lo, transform=trans,
                          facecolor='#D0D0D0', alpha=0.35, zorder=0))
    ax.add_patch(
        patches.Rectangle((0.0, y_lo), 1.0, chem_acc - y_lo, transform=trans,
                          facecolor='none', edgecolor='#2ca02c', hatch='///',
                          linewidth=0.0, zorder=0))
    ax.set_ylim(bottom=y_lo)

    def mean_samples_for_sizes(data, sizes_list):
        order = np.argsort(np.abs(data))
        vals = []
        for size in sizes_list:
            idx = order[-int(size):]
            min_coeff = float(np.min(np.abs(data[idx]))) if len(
                idx) > 0 else 0.0
            vals.append(
                float(1.0 / (min_coeff ** 2)) if min_coeff > 0 else np.inf)
        return vals

    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
    for i, (label, (qc_res, conv_res)) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        fci_E = qc_res.fci_energy
        sizes = list(conv_res.df['subspace_size'])
        ms_raw = mean_samples_for_sizes(qc_res.sv.data, sizes)
        ax.semilogx(ms_raw, conv_res.df['qsci_energy'] - fci_E, 'o-',
                    label=f"{label} QSCI", color=color)

        spin_sizes_all = sorted(set(spin_closed_subspace_sizes(qc_res.sv.data)))
        max_size = sizes[-1] if sizes else 0
        spin_sizes = [int(s) for s in spin_sizes_all if int(s) <= max_size]
        if spin_sizes:
            ms_symm = mean_samples_for_sizes(qc_res.spin_symm_amp, spin_sizes)
            df_symm = conv_res.df[conv_res.df['subspace_size'].isin(spin_sizes)]
            ax.semilogx(ms_symm, df_symm['spin_symm_energy'] - fci_E, '^-',
                        label=f"{label} Spin-Rec", color=color, alpha=0.7)

    ax.set_yscale('log')
    ax.set_xlabel('Mean Samples')
    ax.set_ylabel('Energy Error (Ha)')
    title = "Energy vs Samples Comparison"
    if title_prefix: title = f"{title_prefix}: {title}"
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(data_dir / 'compare_energy_vs_samples.png', dpi=300)
    plt.close()


compare_convergence_custom(data_dir, all_results,
                           title_prefix=f"H{n_atoms} Combined Comparison")
compare_energy_vs_samples_custom(data_dir, all_results,
                                 title_prefix=f"H{n_atoms} Combined Comparison")