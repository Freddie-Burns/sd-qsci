"""
Heat-bath CI (HCI) study (12)
-----------------------------

Compute HCI energies across increasing variational subspace sizes and save
results to CSV, mirroring the workflow used in research/09_lucj/09c_lucj_layers.py
but replacing the LUCJ circuit with a selected CI (Heat-bath CI) calculation
performed by the qc-pyci package.

Outputs (under research/12_hci/data/12_hci/bond_length_<BL>/):
  - h6_hci_convergence.csv       (per-threshold convergence table)
  - fci_subspace_energy.csv      (FCI subspace energies vs size)

Notes:
  - This script expects qc-pyci to be installed. Since qc-pyci APIs may vary
    by version, the integration wrapper attempts a few common import patterns
    and raises a helpful error if none succeed.
  - The HCI “subspace size” reported is the number of selected determinants
    in the variational space for a given selection threshold (eps1). If your
    qc-pyci provides both variational and perturbative corrections, both are
    recorded as hci_energy_var and hci_energy_pt2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyci
from pyscf import gto, scf

from sd_qsci import analysis, hamiltonian
from sd_qsci.utils import uhf_from_rhf


# -----------------
# User configuration
# -----------------
RECOMPUTE: bool = True  # Set False to skip recomputation and only rely on CSVs
BOND_LENGTH: float = 2.0
N_ATOMS: int = 6

# HCI thresholds to scan (typical decreasing eps1 increases the variational subspace size)
HCI_EPS1_LIST: list[float] = [
    1.0e-2, 5.0e-3, 2.0e-3, 1.0e-3, 5.0e-4, 2.0e-4, 1.0e-4,
]

# Optional sweep over target variational subspace sizes. If set > 0, we will run
# a second scan that forces the HCI calculation to cap at a given number of
# determinants (via max_det, if supported by qc-pyci), recording the energy for
# each subspace size from 1 to SUBSPACE_SWEEP_MAX. We use a very small eps1 so
# the determinant growth is primarily limited by max_det.
SUBSPACE_SWEEP_MAX: int = 200
SUBSPACE_SWEEP_EPS1: float = 1.0e-8


def build_h_chain(bond_length: float, n_atoms: int = 6) -> gto.Mole:
    coords = [(i * bond_length, 0.0, 0.0) for i in range(n_atoms)]
    geometry = '; '.join([f'H {x:.8f} {y:.8f} {z:.8f}' for x, y, z in coords])
    mol = gto.Mole()
    mol.build(atom=geometry, unit='Angstrom', basis='sto-3g', charge=0, spin=0, verbose=0)
    return mol


def run_hci_from_pyscf(
    rhf: scf.hf.SCF,
    eps1: float,
    eps2: Optional[float] = None,
    max_det: Optional[int] = None,
) -> tuple[int, float, Optional[float]]:
    """Run an HCI calculation via qc-pyci, using a PySCF RHF reference.

    Returns (n_variational_dets, e_var, e_var_plus_pt2_or_None).

    Because qc-pyci's API can differ by version, this function tries multiple
    likely call patterns. If all fail, it raises a ValueError with helpful tips.
    """

    # Extract PySCF data commonly needed by selected-CI solvers
    mol = rhf.mol

    # Pattern A: Hypothetical high-level API: pyci.HCI(mol=..., rhf=..., eps1=...)
    if hasattr(pyci, 'HCI'):
        HCI = getattr(pyci, 'HCI')
        solver = HCI(mol=mol, rhf=rhf, eps1=eps1, eps2=eps2, max_det=max_det, reference='RHF')
        res = solver.run()
        # Try common attribute names
        ndet = int(getattr(res, 'n_dets', getattr(res, 'ndet', getattr(res, 'nDet', 0))))
        e_var = float(getattr(res, 'e_var', getattr(res, 'eVar', getattr(res, 'variational_energy', res.energy))))
        e_tot = getattr(res, 'e_tot', getattr(res, 'eTot', getattr(res, 'pt2_energy', None)))
        e_tot_val = float(e_tot) if e_tot is not None else None
        return ndet, e_var, e_tot_val

    for fn_name in ('hci', 'run_hci', 'shci', 'run_shci'):
        if hasattr(pyci, fn_name):
            fn = getattr(pyci, fn_name)
            res = fn(mol=mol, rhf=rhf, eps1=eps1, eps2=eps2, max_det=max_det)
            ndet = int(getattr(res, 'n_dets', getattr(res, 'ndet', getattr(res, 'nDet', 0))))
            e_var = float(getattr(res, 'e_var', getattr(res, 'eVar', getattr(res, 'variational_energy', res.energy))))
            e_tot = getattr(res, 'e_tot', getattr(res, 'eTot', getattr(res, 'pt2_energy', None)))
            e_tot_val = float(e_tot) if e_tot is not None else None
            return ndet, e_var, e_tot_val

    sub = getattr(pyci, 'hci', None)
    if sub is not None and hasattr(sub, 'run'):
        res = sub.run(mol=mol, rhf=rhf, eps1=eps1, eps2=eps2, max_det=max_det)
        ndet = int(getattr(res, 'n_dets', getattr(res, 'ndet', getattr(res, 'nDet', 0))))
        e_var = float(getattr(res, 'e_var', getattr(res, 'eVar', getattr(res, 'variational_energy', res.energy))))
        e_tot = getattr(res, 'e_tot', getattr(res, 'eTot', getattr(res, 'pt2_energy', None)))
        e_tot_val = float(e_tot) if e_tot is not None else None
        return ndet, e_var, e_tot_val


def main() -> None:
    bond_length = BOND_LENGTH
    n_atoms = N_ATOMS

    stem = Path(__file__).stem
    out_dir = Path(__file__).parent / 'data' / stem / f"bond_length_{bond_length:.2f}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV paths
    conv_csv = out_dir / 'h6_hci_convergence.csv'
    fci_csv = out_dir / 'fci_subspace_energy.csv'
    hci_sub_csv = out_dir / 'hci_subspace_energy.csv'

    if RECOMPUTE:
        print(f"Building H{n_atoms} chain, bond length {bond_length:.2f} Å …")
        mol = build_h_chain(bond_length, n_atoms)
        rhf = scf.RHF(mol).run()
        uhf = uhf_from_rhf(mol, rhf)

        # Reference quantities
        fci_energy, n_fci_configs, fci_vec = analysis.calc_fci_energy(rhf)
        H = hamiltonian.hamiltonian_from_pyscf(mol, rhf)

        # Load existing CSVs if present
        conv_df_existing = pd.read_csv(conv_csv) if conv_csv.exists() else pd.DataFrame()
        fci_df_existing = pd.read_csv(fci_csv) if fci_csv.exists() else pd.DataFrame()
        hci_sub_existing = pd.read_csv(hci_sub_csv) if hci_sub_csv.exists() else pd.DataFrame()

        new_conv_rows = []
        max_subspace_overall = 0

        print("Running Heat-bath CI across thresholds…")
        for eps1 in HCI_EPS1_LIST:
            print(f"  eps1 = {eps1:.1e}")
            try:
                ndet, e_var, e_tot = run_hci_from_pyscf(rhf=rhf, eps1=eps1)
            except Exception as e:
                # Record a row with NaNs to retain traceability of failed points
                print(f"    HCI failed for eps1={eps1:.1e}: {e}")
                ndet, e_var, e_tot = np.nan, np.nan, np.nan

            if isinstance(ndet, (int, np.integer)):
                max_subspace_overall = max(max_subspace_overall, int(ndet))

            row = {
                'bond_length': bond_length,
                'eps1': eps1,
                'subspace_size': int(ndet) if isinstance(ndet, (int, np.integer)) else np.nan,
                'hci_energy_var': float(e_var) if isinstance(e_var, (int, float, np.floating)) else np.nan,
                'hci_energy_pt2': float(e_tot) if isinstance(e_tot, (int, float, np.floating)) else np.nan,
                'abs_error_to_fci_var': float(abs(e_var - fci_energy)) if isinstance(e_var, (int, float, np.floating)) else np.nan,
                'abs_error_to_fci_pt2': float(abs(e_tot - fci_energy)) if isinstance(e_tot, (int, float, np.floating)) else np.nan,
                'rhf_energy': rhf.e_tot,
                'uhf_energy': uhf.e_tot,
                'fci_energy': fci_energy,
                'n_fci_configs': n_fci_configs,
            }
            new_conv_rows.append(row)

        # Merge/overwrite convergence rows for the eps1 values we just computed
        if new_conv_rows:
            conv_new = pd.DataFrame(new_conv_rows)
            if not conv_df_existing.empty:
                mask_keep = ~(
                    (conv_df_existing.get('bond_length') == bond_length) &
                    (conv_df_existing.get('eps1').isin(HCI_EPS1_LIST))
                )
                conv_df_existing = conv_df_existing[mask_keep]
                conv_combined = pd.concat([conv_df_existing, conv_new], ignore_index=True)
            else:
                conv_combined = conv_new
            conv_combined.sort_values(['bond_length', 'eps1'], inplace=True)
            conv_combined.to_csv(conv_csv, index=False)

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

        # Optional: direct sweep over subspace sizes 1..SUBSPACE_SWEEP_MAX
        if SUBSPACE_SWEEP_MAX and SUBSPACE_SWEEP_MAX > 0:
            print(f"Running HCI energy sweep over subspace sizes 1..{SUBSPACE_SWEEP_MAX} …")
            sub_sizes = np.arange(1, SUBSPACE_SWEEP_MAX + 1, dtype=int)
            sweep_rows: list[dict[str, Any]] = []

            for k in sub_sizes:
                try:
                    ndet, e_var, e_tot = run_hci_from_pyscf(
                        rhf=rhf, eps1=SUBSPACE_SWEEP_EPS1, eps2=None, max_det=int(k)
                    )
                except Exception as e:
                    print(f"  Sweep failed at subspace_size={k}: {e}")
                    ndet, e_var, e_tot = np.nan, np.nan, np.nan

                row = {
                    'bond_length': bond_length,
                    'subspace_size': int(k),
                    'achieved_subspace_size': int(ndet) if isinstance(ndet, (int, np.integer)) else np.nan,
                    'hci_energy_var': float(e_var) if isinstance(e_var, (int, float, np.floating)) else np.nan,
                    'hci_energy_pt2': float(e_tot) if isinstance(e_tot, (int, float, np.floating)) else np.nan,
                    'fci_energy': fci_energy,
                    'rhf_energy': rhf.e_tot,
                }
                sweep_rows.append(row)

            if sweep_rows:
                sweep_df = pd.DataFrame(sweep_rows)
                # If existing, drop overlapping bond_length & subspace_size rows, then append
                if not hci_sub_existing.empty:
                    merged = hci_sub_existing.merge(
                        sweep_df[['bond_length', 'subspace_size']],
                        on=['bond_length', 'subspace_size'], how='left', indicator=True
                    )
                    keep_mask = merged['_merge'] == 'left_only'
                    hci_sub_existing = hci_sub_existing[keep_mask]
                    hci_sub_combined = pd.concat([hci_sub_existing, sweep_df], ignore_index=True)
                else:
                    hci_sub_combined = sweep_df

                hci_sub_combined.sort_values(['bond_length', 'subspace_size'], inplace=True)
                hci_sub_combined.to_csv(hci_sub_csv, index=False)

    # For now we do not generate figures here; plotting can be added later if needed.


if __name__ == '__main__':
    main()
