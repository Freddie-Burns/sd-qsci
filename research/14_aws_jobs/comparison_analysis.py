from __future__ import annotations

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches, transforms
from pathlib import Path
from typing import List

# --- Configuration ---
# List the directories you want to compare here.
# They should contain qsci_convergence.csv and metadata.json (or job_metadata.json)
# Paths can be absolute or relative to this script's directory.

# H4 at 2A directories
# DIRECTORIES = [
#     Path("data/14a_h4_ankaa/combined_222157-222152"),
#     Path("data/14b_h4_forte/combined_223029-222958"),
#     Path("data/14c_h4_garnet/combined_222752-222746"),
#     Path("data/14d_h4_emerald/combined_222808-222805"),
# ]
# OUT_DIR = Path("data/14_h4_bond_stretch/2A_device_comparison")
#
DIRECTORIES = [
    Path("data/14e_h6_ankaa/combined_215923-215918"),
    Path("data/14f_h6_forte/combined_220814-220739"),
    Path("data/14g_h6_garnet/combined_221825-221822"),
    Path("data/14h_h6_emerald/combined_221944-221942"),
]
OUT_DIR = Path("data/14_h6_bond_stretch/2A_device_comparison")

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / OUT_DIR

def load_data(run_dir: Path):
    if not run_dir.is_absolute():
        run_dir = BASE_DIR / run_dir
    
    csv_path = run_dir / "qsci_convergence.csv"
    if not csv_path.exists():
        print(f"[warning] {csv_path} not found. Skipping.")
        return None, None

    df = pd.read_csv(csv_path)
    
    # Load metadata to get FCI energy and a label
    meta = {}
    for meta_name in ["metadata.json", "job_metadata.json"]:
        meta_path = run_dir / meta_name
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta.update(json.load(f))
    
    # Try to find FCI energy in summary.csv if not in metadata
    fci_energy = meta.get("fci_energy")
    if fci_energy is None:
        summary_path = run_dir / "summary.csv"
        if summary_path.exists():
            summary_df = pd.read_csv(summary_path)
            fci_row = summary_df[summary_df['quantity'] == 'fci_energy']
            if not fci_row.empty:
                fci_energy = float(fci_row.iloc[0]['value'])

    # Build label
    mol = meta.get("molecule", "Unknown")
    device = meta.get("device", "Unknown")
    label = f"{mol} on {device} ({run_dir.name})"
    
    return df, fci_energy, label

def run_comparison():
    if not DIRECTORIES:
        print("[info] No directories specified in DIRECTORIES list at the top of the script.")
        return

    sns.set_style("whitegrid")
    
    fig_conv, ax_conv = plt.subplots(figsize=(10, 6))
    fig_samp, ax_samp = plt.subplots(figsize=(10, 6))

    y_lo = 1.0e-4
    chem_acc = 1.6e-3

    for ax in [ax_conv, ax_samp]:
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        bg = patches.Rectangle(
            (0.0, y_lo),
            1.0,
            chem_acc - y_lo,
            transform=trans,
            facecolor='#D0D0D0',
            edgecolor='none',
            alpha=0.35,
            zorder=0,
        )
        ax.add_patch(bg)
        hat = patches.Rectangle(
            (0.0, y_lo),
            1.0,
            chem_acc - y_lo,
            transform=trans,
            facecolor='none',
            edgecolor='#2ca02c',
            hatch='///',
            linewidth=0.0,
            zorder=0,
        )
        ax.add_patch(hat)
        ax.set_ylim(bottom=y_lo)

    for run_path in DIRECTORIES:
        df, fci_energy, label = load_data(run_path)
        if df is None:
            continue

        # Plot 1: QSCI Convergence (Energy error vs Subspace Size)
        if fci_energy is not None:
            ax_conv.plot(df['subspace_size'], df['qsci_energy'] - fci_energy, 'o-', label=label)
        else:
            print(f"[warning] FCI energy not found for {run_path}. Plotting raw energy.")
            ax_conv.plot(df['subspace_size'], df['qsci_energy'], 'o-', label=label)

        # Plot 2: Energy vs Samples
        if 'mean_sample_number' in df.columns:
            # We use cumulative samples or mean samples? 
            # In job_analysis.py, energy_vs_samples is used.
            # Looking at qsci_convergence.csv, mean_sample_number seems to be the one.
            if fci_energy is not None:
                ax_samp.plot(df['mean_sample_number'], df['qsci_energy'] - fci_energy, 'o-', label=label)
            else:
                ax_samp.plot(df['mean_sample_number'], df['qsci_energy'], 'o-', label=label)
        else:
            print(f"[warning] 'mean_sample_number' not found in {run_path}/qsci_convergence.csv")

    # Finalize Convergence Plot
    ax_conv.set_xlabel("Subspace Size")
    ax_conv.set_ylabel("Energy Error (Ha)")
    ax_conv.set_yscale('log')
    ax_conv.legend()
    ax_conv.set_title("QSCI Convergence Comparison")
    fig_conv.tight_layout()
    fig_conv.savefig("qsci_convergence.png")
    print("[success] Saved qsci_convergence.png")

    # Finalize Samples Plot
    ax_samp.set_xlabel("Mean Sample Number")
    ax_samp.set_ylabel("Energy Error (Ha)")
    ax_samp.set_yscale('log')
    ax_samp.legend()
    ax_samp.set_title("Energy vs Samples Comparison")
    fig_samp.tight_layout()
    fig_samp.savefig("energy_vs_samples.png")
    print("[success] Saved energy_vs_samples.png")

if __name__ == "__main__":
    run_comparison()
