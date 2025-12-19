"""
LUCJ layers study (09c)
-----------------------

Plot log energy error to FCI for LUCJ circuits with layers 1–10 against
increasing subspace size (spin-symmetric QSCI), using a plasma colormap and a
lower bound of 1e-5 Ha on the y-axis. Also overlay the FCI subspace absolute
error curve on the same plot for comparison.

This script now saves/updates CSVs and always plots from CSV so you can replot
without recomputing. Set RECOMPUTE=False to skip the expensive calculations and
only regenerate figures from previously saved CSVs.

Outputs are saved under:
    research/09_lucj/data/09c_lucj_layers/bond_length_2.00/
Files:
  - h6_lucj_layers_convergence.csv  (long table with per-layer convergence)
  - h6_lucj_layers_summary.csv      (per-layer summary)
  - fci_subspace_energy.csv         (FCI subspace energies vs size)
  - lucj_layers_error_vs_subspace.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import patches, transforms
from pyscf import gto, scf
from pyscf.cc import CCSD
from qiskit_aer import Aer
from qiskit import transpile

from sd_qsci import analysis, circuit, hamiltonian
from sd_qsci import greedy as greedy_sel
from sd_qsci.utils import uhf_from_rhf


# User-togglable constants
RECOMPUTE: bool = True  # False skips quantum chemistry and replot from CS
BOND_LENGTH: float = 2.0
N_ATOMS: int = 6
LAYERS: list[int] = list(range(1, 11))  # which LUCJ layer counts to compute

# Plot theme
sns.set_theme()


def build_h_chain(bond_length: float, n_atoms: int = 6) -> gto.Mole:
    coords = [(i * bond_length, 0.0, 0.0) for i in range(n_atoms)]
    geometry = '; '.join([f'H {x:.8f} {y:.8f} {z:.8f}' for x, y, z in coords])
    mol = gto.Mole()
    mol.build(atom=geometry, unit='Angstrom', basis='sto-3g', charge=0, spin=0, verbose=0)
    return mol


def main():
    bond_length = BOND_LENGTH
    n_atoms = N_ATOMS

    stem = Path(__file__).stem
    out_dir = Path(__file__).parent / 'data' / stem / f"bond_length_{bond_length:.2f}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV paths
    conv_csv = out_dir / 'h6_lucj_layers_convergence.csv'
    summ_csv = out_dir / 'h6_lucj_layers_summary.csv'
    fci_csv = out_dir / 'fci_subspace_energy.csv'
    greedy_csv = out_dir / 'greedy_subspace_energy.csv'
    gate_counts_csv = out_dir / 'gate_counts.csv'

    # Optional recomputation of heavy data
    if RECOMPUTE:
        print(f"Building H{n_atoms} chain, bond length {bond_length:.2f} Å …")
        mol = build_h_chain(bond_length, n_atoms)
        rhf = scf.RHF(mol).run()
        uhf = uhf_from_rhf(mol, rhf)
        ccsd = CCSD(rhf).run()

        # Reference quantities
        fci_energy, n_fci_configs, fci_vec = analysis.calc_fci_energy(rhf)
        H = hamiltonian.hamiltonian_from_pyscf(mol, rhf)

        backend = Aer.get_backend("statevector_simulator")

        # Load existing CSVs if present (to preserve previously computed layers)
        conv_df_existing = pd.read_csv(conv_csv) if conv_csv.exists() else pd.DataFrame()
        summ_df_existing = pd.read_csv(summ_csv) if summ_csv.exists() else pd.DataFrame()
        fci_df_existing = pd.read_csv(fci_csv) if fci_csv.exists() else pd.DataFrame()
        greedy_df_existing = pd.read_csv(greedy_csv) if greedy_csv.exists() else pd.DataFrame()

        max_subspace_overall = 0
        new_conv_rows = []
        new_summ_rows = []
        new_gate_rows = []

        for n_layers in LAYERS:
            print(f"Simulating LUCJ with {n_layers} layer(s)…")
            qc = circuit.get_lucj_circuit(ccsd_obj=ccsd, backend=backend, n_reps=n_layers)
            sv = circuit.simulate(qc)
            spin_symm_amp = analysis.spin_symm_amplitudes(sv.data)

            qc_results = analysis.QuantumChemistryResults(
                mol=mol,
                rhf=rhf,
                uhf=uhf,
                sv=sv,
                H=H,
                fci_energy=fci_energy,
                n_fci_configs=n_fci_configs,
                fci_vec=fci_vec,
                bond_length=bond_length,
                spin_symm_amp=spin_symm_amp,
            )

            conv = analysis.calc_convergence_data(qc_results, spin_symm=True)

            # Create long-form rows for CSV (one row per subspace_size per layer)
            df_layer = conv.df.copy()
            df_layer['bond_length'] = bond_length
            df_layer['layer'] = n_layers
            df_layer['fci_energy'] = fci_energy
            df_layer['rhf_energy'] = rhf.e_tot
            df_layer['uhf_energy'] = uhf.e_tot
            df_layer['abs_error_to_fci'] = np.abs(df_layer['spin_symm_energy'] - fci_energy)
            # Reorder columns for readability
            cols = ['bond_length', 'layer', 'subspace_size', 'qsci_energy', 'spin_symm_energy',
                    'fci_subspace_energy', 'mean_sample_number', 'qsci_S2', 'spin_symm_S2',
                    'abs_error_to_fci', 'rhf_energy', 'uhf_energy', 'fci_energy']
            df_layer = df_layer[[c for c in cols if c in df_layer.columns]]
            new_conv_rows.append(df_layer)

            # Summary per layer
            chem_acc = 1.6e-3
            reach_mask = (df_layer['abs_error_to_fci'] <= chem_acc)
            n_configs_reach = int(df_layer.loc[reach_mask, 'subspace_size'].min()) if reach_mask.any() else None
            new_summ_rows.append({
                'bond_length': bond_length,
                'layer': n_layers,
                'max_subspace_size': int(df_layer['subspace_size'].max()),
                'min_abs_error': float(df_layer['abs_error_to_fci'].min()),
                'n_configs_to_chem_acc': n_configs_reach,
                'rhf_energy': rhf.e_tot,
                'uhf_energy': uhf.e_tot,
                'fci_energy': fci_energy,
            })

            max_subspace_overall = max(max_subspace_overall, int(df_layer['subspace_size'].max()))

            # Gate counts (transpile to basis gates, then count by arity)
            try:
                basis = [
                    'u', 'u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'sx', 'x', 'y', 'z', 'h',
                    'cx', 'cz', 'iswap', 'swap', 'rxx', 'ryy', 'rzz', 'xx', 'yy', 'zz'
                ]
                qc_t = transpile(qc, basis_gates=basis, optimization_level=0)
            except Exception:
                qc_t = qc  # fallback without transpile

            one_q = 0
            two_q = 0
            total = 0
            other = 0
            # Qiskit 1.2+: items in .data are CircuitInstruction; access attributes explicitly
            for ci in qc_t.data:
                inst = getattr(ci, 'operation', None)
                qargs = getattr(ci, 'qubits', [])
                # Skip barriers/measurements if present
                name = getattr(inst, 'name', '').lower() if inst is not None else ''
                if name in {'barrier', 'measure'}:
                    continue
                nq = len(qargs)
                total += 1
                if nq == 1:
                    one_q += 1
                elif nq == 2:
                    two_q += 1
                else:
                    other += 1

            if other:
                print(f"Note: encountered {other} multi-qubit (>2) ops for layer {n_layers}; counted only in total_gates.")

            new_gate_rows.append({
                'bond_length': bond_length,
                'layer': n_layers,
                'n_qubits': int(qc.num_qubits),
                'one_q_gates': int(one_q),
                'two_q_gates': int(two_q),
                'total_gates': int(total),
            })

        # Merge/overwrite for the layers we just computed
        if new_conv_rows:
            conv_new = pd.concat(new_conv_rows, ignore_index=True)
            if not conv_df_existing.empty:
                mask_keep = ~(
                    (conv_df_existing.get('bond_length') == bond_length) &
                    (conv_df_existing.get('layer').isin(LAYERS))
                )
                conv_df_existing = conv_df_existing[mask_keep]
                conv_combined = pd.concat([conv_df_existing, conv_new], ignore_index=True)
            else:
                conv_combined = conv_new
            conv_combined.sort_values(['bond_length', 'layer', 'subspace_size'], inplace=True)
            conv_combined.to_csv(conv_csv, index=False)

        if new_summ_rows:
            summ_new = pd.DataFrame(new_summ_rows)
            if not summ_df_existing.empty:
                mask_keep = ~(
                    (summ_df_existing.get('bond_length') == bond_length) &
                    (summ_df_existing.get('layer').isin(LAYERS))
                )
                summ_df_existing = summ_df_existing[mask_keep]
                pieces = [df for df in (summ_df_existing, summ_new) if df is not None and not df.empty]
                summ_combined = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
            else:
                summ_combined = summ_new
            summ_combined.sort_values(['bond_length', 'layer'], inplace=True)
            summ_combined.to_csv(summ_csv, index=False)

        # Merge/overwrite gate counts for the layers we just computed
        if new_gate_rows:
            gates_new = pd.DataFrame(new_gate_rows)
            if gate_counts_csv.exists():
                gates_existing = pd.read_csv(gate_counts_csv)
                mask_keep = ~(
                    (gates_existing.get('bond_length') == bond_length) &
                    (gates_existing.get('layer').isin(LAYERS))
                )
                gates_existing = gates_existing[mask_keep]
                pieces = [df for df in (gates_existing, gates_new) if df is not None and not df.empty]
                gates_combined = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
            else:
                gates_combined = gates_new
            gates_combined.sort_values(['bond_length', 'layer'], inplace=True)
            gates_combined.to_csv(gate_counts_csv, index=False)

        # Update FCI subspace energies up to the maximum subspace encountered
        if max_subspace_overall > 0:
            fci_sub_sizes = np.arange(1, max_subspace_overall + 1, dtype=int)
            fci_sub_energies = np.array([
                analysis.calc_fci_subspace_energy(H, fci_vec, int(k)) for k in fci_sub_sizes
            ], dtype=float)
            fci_new = pd.DataFrame({
                'bond_length': bond_length,
                'subspace_size': fci_sub_sizes,
                'fci_subspace_energy': fci_sub_energies,
                'fci_energy': fci_energy,
            })

            if not fci_df_existing.empty:
                # Remove any rows that collide on bond_length and subspace_size (will be replaced)
                merged = fci_df_existing.merge(
                    fci_new[['bond_length', 'subspace_size']],
                    on=['bond_length', 'subspace_size'], how='left', indicator=True
                )
                fci_keep = merged['_merge'] == 'left_only'
                fci_df_existing = fci_df_existing[fci_keep]
                pieces = [df for df in (fci_df_existing, fci_new) if df is not None and not df.empty]
                fci_combined = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
            else:
                fci_combined = fci_new

            fci_combined.sort_values(['bond_length', 'subspace_size'], inplace=True)
            fci_combined.to_csv(fci_csv, index=False)

            # Also compute/update greedy subspace energies up to the same max size
            try:
                _, greedy_energies = greedy_sel.greedy_best_subspace(H, fci_vec, int(max_subspace_overall))
                greedy_sizes = np.arange(1, len(greedy_energies) + 1, dtype=int)
                greedy_new = pd.DataFrame({
                    'bond_length': bond_length,
                    'subspace_size': greedy_sizes,
                    'greedy_energy': np.asarray(greedy_energies, dtype=float),
                    'fci_energy': fci_energy,
                })

                if not greedy_df_existing.empty:
                    merged_g = greedy_df_existing.merge(
                        greedy_new[['bond_length', 'subspace_size']],
                        on=['bond_length', 'subspace_size'], how='left', indicator=True
                    )
                    keep_mask = merged_g['_merge'] == 'left_only'
                    greedy_df_existing = greedy_df_existing[keep_mask]
                    pieces = [df for df in (greedy_df_existing, greedy_new) if df is not None and not df.empty]
                    greedy_combined = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
                else:
                    greedy_combined = greedy_new

                greedy_combined.sort_values(['bond_length', 'subspace_size'], inplace=True)
                greedy_combined.to_csv(greedy_csv, index=False)
            except Exception as e:
                print(f"Warning: Greedy computation failed or skipped: {e}")

    # At this point, plot strictly from CSVs
    if not conv_csv.exists():
        raise FileNotFoundError(f"Missing convergence CSV: {conv_csv}. Run with RECOMPUTE=True first.")
    if not fci_csv.exists():
        raise FileNotFoundError(f"Missing FCI subspace CSV: {fci_csv}. Run with RECOMPUTE=True first.")

    conv_df = pd.read_csv(conv_csv)
    fci_df = pd.read_csv(fci_csv)
    greedy_df = pd.read_csv(greedy_csv) if greedy_csv.exists() else pd.DataFrame()
    gates_df = pd.read_csv(gate_counts_csv) if gate_counts_csv.exists() else pd.DataFrame()

    # Plot: log error (spin-symmetric LUCJ-QSCI) vs subspace size, layers colored by plasma
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_yscale('log')
    ax.set_xlabel('Subspace Size (Number of Configurations)')
    ax.set_ylabel('Absolute energy error to FCI (Ha) [log]')
    ax.set_title(f'H6 Chain: LUCJ layers (1–10) error vs subspace size\nBond length = {bond_length:.2f} Å')
    ax.set_ylim(bottom=1e-5)

    # Chemical accuracy region (≤ 1.6e-3 Ha): diagonally striped green over light grey
    y0, y1 = 1.0e-5, 1.6e-3
    try:
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        # Background light grey box
        bg = patches.Rectangle(
            (0.0, y0), 1.0, y1 - y0,
            transform=trans,
            facecolor='#D0D0D0',
            edgecolor='none',
            alpha=0.35,
            zorder=0,
        )
        ax.add_patch(bg)
        # Green diagonal hatching overlay
        hat = patches.Rectangle(
            (0.0, y0), 1.0, y1 - y0,
            transform=trans,
            facecolor='none',
            edgecolor='#2ca02c',  # matplotlib "tab:green"
            hatch='///',
            linewidth=0.0,
            zorder=0,
        )
        ax.add_patch(hat)
        # Proxy artist for legend
        proxy = patches.Rectangle((0, 0), 1, 1, facecolor='#D0D0D0', edgecolor='#2ca02c', hatch='///', alpha=0.35, label='chemical accuracy')
        ax.add_artist(proxy)
    except Exception:
        # Fallback to simple shaded span if patches/hatching fail in a given backend
        ax.axhspan(y0, y1, facecolor='#B0B0B0', alpha=0.2, zorder=0, label='chemical accuracy')

    # Plot each available layer for the configured bond length
    df_bl = conv_df[conv_df['bond_length'] == bond_length]
    layers_available = sorted(int(x) for x in df_bl['layer'].unique())
    cmap = plt.get_cmap('plasma', max(layers_available) if layers_available else 10)

    x_min, x_max = None, None
    for n_layers in layers_available:
        d = df_bl[df_bl['layer'] == n_layers]
        x = d['subspace_size'].to_numpy()
        y = d['abs_error_to_fci'].to_numpy()
        if x.size == 0:
            continue
        # enforce lower bound visually
        y_plot = np.maximum(y, 1e-5)
        ax.plot(x, y_plot, marker='o', linestyle='-', linewidth=1.8, markersize=3.5,
                color=cmap(n_layers - 1), label=f'{n_layers} layer' + ('s' if n_layers > 1 else ''))
        x_min = int(min(x_min, x.min())) if x_min is not None else int(x.min())
        x_max = int(max(x_max, x.max())) if x_max is not None else int(x.max())

    # Also overlay the FCI subspace absolute error curve for the same bond length
    df_bl_fci = fci_df[fci_df['bond_length'] == bond_length]
    if not df_bl_fci.empty:
        x_fci = df_bl_fci['subspace_size'].to_numpy()
        # absolute error to full FCI energy
        if 'fci_energy' in df_bl_fci.columns:
            fci_energy_vals = df_bl_fci['fci_energy'].to_numpy()
            # fci_energy should be constant per bond length; take from column row-wise
            y_fci_err = np.abs(df_bl_fci['fci_subspace_energy'].to_numpy() - fci_energy_vals)
        else:
            # fallback: use the last known FCI energy from conv_df rows for this bond length
            fallback_fci = float(conv_df[conv_df['bond_length'] == bond_length]['fci_energy'].iloc[0])
            y_fci_err = np.abs(df_bl_fci['fci_subspace_energy'].to_numpy() - fallback_fci)
        y_fci_plot = np.maximum(y_fci_err, 1e-5)
        ax.plot(x_fci, y_fci_plot, 's-', color='black', linewidth=1.8, markersize=3.5, label='FCI subspace (error)')
        # expand x-bounds to include FCI curve
        x_min = int(min(x_min, x_fci.min())) if x_min is not None else int(x_fci.min())
        x_max = int(max(x_max, x_fci.max())) if x_max is not None else int(x_fci.max())

    # Overlay greedy-algorithm absolute error curve (if available)
    df_bl_greedy = greedy_df[greedy_df.get('bond_length') == bond_length] if not greedy_df.empty else pd.DataFrame()
    if not df_bl_greedy.empty:
        x_gr = df_bl_greedy['subspace_size'].to_numpy()
        # prefer per-row fci_energy column; fallback to conv_df constant
        if 'fci_energy' in df_bl_greedy.columns:
            fci_vals_g = df_bl_greedy['fci_energy'].to_numpy()
            y_gr_err = np.abs(df_bl_greedy['greedy_energy'].to_numpy() - fci_vals_g)
        else:
            fallback_fci = float(conv_df[conv_df['bond_length'] == bond_length]['fci_energy'].iloc[0])
            y_gr_err = np.abs(df_bl_greedy['greedy_energy'].to_numpy() - fallback_fci)
        y_gr_plot = np.maximum(y_gr_err, 1e-5)
        ax.plot(x_gr, y_gr_plot, 'd--', color='#ff7f0e', linewidth=1.8, markersize=4.0, label='Greedy (error)')
        x_min = int(min(x_min, x_gr.min())) if x_min is not None else int(x_gr.min())
        x_max = int(max(x_max, x_gr.max())) if x_max is not None else int(x_gr.max())

    # Bound x-axis to data range
    if x_min is not None and x_max is not None:
        ax.set_xlim(left=max(1, x_min), right=x_max)

    ax.legend(title='LUCJ layers + FCI + Greedy', fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'lucj_layers_error_vs_subspace.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Plot: gate counts vs layers (from CSV only)
    if gates_df.empty:
        if not RECOMPUTE:
            print(f"Warning: Missing gate counts CSV: {gate_counts_csv}. Run with RECOMPUTE=True to generate it.")
    else:
        df_g = gates_df[gates_df['bond_length'] == bond_length].copy()
        if not df_g.empty:
            # Ensure integer sorting of layers
            df_g['layer'] = df_g['layer'].astype(int)
            df_g.sort_values('layer', inplace=True)

            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.plot(df_g['layer'], df_g['one_q_gates'], 'o-', label='1-qubit gates', linewidth=2, markersize=4)
            ax2.plot(df_g['layer'], df_g['two_q_gates'], 's-', label='2-qubit gates', linewidth=2, markersize=4)
            if 'total_gates' in df_g.columns:
                ax2.plot(df_g['layer'], df_g['total_gates'], 'd--', label='total gates', linewidth=1.6, markersize=4)

            ax2.set_xlabel('LUCJ layers (n_reps)')
            ax2.set_ylabel('Gate count')
            ax2.set_title(f'H6 LUCJ: gate counts vs layers\nBond length = {bond_length:.2f} Å')
            # Bound x-axis to data range (avoid identical limits warning)
            x_lo = int(df_g['layer'].min())
            x_hi = int(df_g['layer'].max())
            if x_lo == x_hi:
                ax2.set_xlim(x_lo - 0.5, x_hi + 0.5)
            else:
                ax2.set_xlim(left=x_lo, right=x_hi)
            # Use integer ticks
            try:
                ax2.set_xticks(sorted(df_g['layer'].unique()))
            except Exception:
                pass
            ax2.grid(True, which='both', alpha=0.3)
            ax2.legend()
            plt.tight_layout()
            plt.savefig(out_dir / 'gate_counts_vs_layers.png', dpi=300, bbox_inches='tight')
            plt.close(fig2)

    print(f"Saved CSVs and plots to: {out_dir}")


if __name__ == '__main__':
    main()
