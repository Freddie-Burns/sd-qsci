from __future__ import annotations

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import patches, transforms
from pathlib import Path
from typing import List

# --- Configuration ---
# Dictionary of jobs to compare. Each key is a descriptive name.
# Each job contains:
#   DIRECTORIES: List of paths (relative to script or absolute) containing qsci_convergence.csv and metadata.json
#   OUT_DIR: Path where plots will be saved
#   SAMPLES_XLIM: Optional [min, max] for the x-axis of the samples plot

JOBS = {
    "H4_1.5A": {
        "directories": [
            Path("data/14a_h4_ankaa/combined_222157-222152"),
            Path("data/14b_h4_forte/combined_223029-222958"),
            Path("data/14c_h4_garnet/combined_222619-222614"),
            Path("data/14d_h4_emerald/combined_222638-222636"),
        ],
        "out_dir": Path("data/14_h4_bond_stretch/1_5A_device_comparison"),
        "samples_xlim": None,
    },
    "H4_2A": {
        "directories": [
            Path("data/14a_h4_ankaa/combined_222226-222223"),
            Path("data/14b_h4_forte/combined_222513-222436"),
            Path("data/14c_h4_garnet/combined_222752-222746"),
            Path("data/14d_h4_emerald/combined_222808-222805"),
            Path("data/14k_h4_lucj/combined_mean"),
        ],
        "out_dir": Path("data/14_h4_bond_stretch/2A_device_comparison"),
        "samples_xlim": [0, 1000],
    },
    "H4_4A": {
        "directories": [
            Path("data/14a_h4_ankaa/combined_212055-212051"),
            Path("data/14b_h4_forte/combined_212355-212325"),
            Path("data/14c_h4_garnet/combined_214505-214503"),
            Path("data/14d_h4_emerald/combined_213418-213415"),
        ],
        "out_dir": Path("data/14_h4_bond_stretch/4A_device_comparison"),
        "samples_xlim": [0, 60],
    },
    "H6_2A": {
        "directories": [
            Path("data/14e_h6_ankaa/combined_215923-215918"),
            Path("data/14f_h6_forte/combined_220814-220739"),
            Path("data/14g_h6_garnet/combined_221825-221822"),
            Path("data/14h_h6_emerald/combined_221944-221942"),
            Path("data/14j_h6_lucj/1_layers"),
            Path("data/14j_h6_lucj/10_layers"),
            Path("data/14l_h6_uhf_sim/combined_mean"),
        ],
        "out_dir": Path("data/14_h6_bond_stretch/2A_device_comparison"),
        "samples_xlim": [0, 10000],
    },
}

BASE_DIR = Path(__file__).resolve().parent

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
    device = meta.get("device", "Unknown")
    label = device
    
    return df, fci_energy, label

def run_comparison(directories: List[Path], out_dir: Path, samples_xlim: List[float] | None = None, every_nth: int | None = 5):
    if not directories:
        print("[info] No directories specified.")
        return

    out_dir = BASE_DIR / out_dir
    os.makedirs(out_dir, exist_ok=True)

    sns.set_theme(style="darkgrid")
    
    fig_conv, ax_conv = plt.subplots(figsize=(10, 6))
    fig_samp, ax_samp = plt.subplots(figsize=(10, 6))
    fig_sc, ax_sc = plt.subplots(figsize=(10, 7))

    y_lo = 1.0e-4
    chem_acc = 1.6e-3

    for ax in [ax_conv, ax_samp]:
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        bg = patches.Rectangle(
            (0.0, -chem_acc),
            1.0,
            2 * chem_acc,
            transform=trans,
            facecolor='#D0D0D0',
            edgecolor='none',
            alpha=0.35,
            zorder=0,
        )
        ax.add_patch(bg)
        hat = patches.Rectangle(
            (0.0, -chem_acc),
            1.0,
            2 * chem_acc,
            transform=trans,
            facecolor='none',
            edgecolor='#2ca02c',
            hatch='///',
            linewidth=0.0,
            zorder=0,
        )
        ax.add_patch(hat)
        ax.set_ylim(bottom=y_lo)

    fci_plotted = False
    
    # Track handles for the samples_vs_configs plot legend
    sc_handles = []
    sc_labels = []
    
    # List of markers for different devices
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for idx, run_path in enumerate(directories):
        df, fci_energy, label = load_data(run_path)
        if df is None:
            continue

        # Plot 1: QSCI Convergence (Energy error vs Subspace Size)
        if fci_energy is not None:
            ax_conv.plot(df['subspace_size'], df['qsci_energy'] - fci_energy, 'o-', label=label)
            # Add FCI subspace comparison if available and not already plotted
            if not fci_plotted and 'fci_subspace_energy' in df.columns:
                ax_conv.plot(df['subspace_size'], df['fci_subspace_energy'] - fci_energy, 's--', color='gray', alpha=0.5, label='FCI Subspace')
                fci_plotted = True
        else:
            print(f"[warning] FCI energy not found for {run_path}. Plotting raw energy.")
            ax_conv.plot(df['subspace_size'], df['qsci_energy'], 'o-', label=label)

        # Plot 2: Energy vs Samples
        if 'mean_sample_number' in df.columns:
            if fci_energy is not None:
                ax_samp.plot(df['mean_sample_number'], df['qsci_energy'] - fci_energy, 'o-', label=label)
            else:
                ax_samp.plot(df['mean_sample_number'], df['qsci_energy'], 'o-', label=label)
        else:
            print(f"[warning] 'mean_sample_number' not found in {run_path}/qsci_convergence.csv")

        # Plot 3: Samples vs Configs Error
        if 'mean_sample_number' in df.columns and fci_energy is not None:
            energy_error = np.abs(df['qsci_energy'] - fci_energy)
            log_energy_error = np.log10(energy_error + 1e-15)
            
            marker = markers[idx % len(markers)]

            # Option to only show every 10th data point and have a line going through
            if every_nth is not None:
                plot_df = df.iloc[::every_nth]
                plot_log_err = log_energy_error.iloc[::every_nth]
                # Add a line going through
                ax_sc.plot(df['mean_sample_number'], df['subspace_size'], color='gray', alpha=0.3, zorder=1)
            else:
                plot_df = df
                plot_log_err = log_energy_error

            scatter = ax_sc.scatter(
                plot_df['mean_sample_number'], 
                plot_df['subspace_size'], 
                c=plot_log_err, 
                cmap='RdBu_r',
                marker=marker,
                edgecolors='none',
                alpha=0.8,
                vmin=-3, # Set consistent range for colorbar
                vmax=0,
                zorder=2
            )
            # Create a proxy handle for the legend that matches the marker but not the color scale
            from matplotlib.lines import Line2D
            handle = Line2D([0], [0], marker=marker, color='w', label=label,
                            markerfacecolor='gray', markersize=10, linestyle='None')
            sc_handles.append(handle)
            sc_labels.append(label)
            
            # Find first point to reach chemical accuracy
            chem_acc_idx = df[energy_error < chem_acc].index
            if not chem_acc_idx.empty:
                first_pt = df.loc[chem_acc_idx[0]]
                ax_sc.scatter(
                    first_pt['mean_sample_number'],
                    first_pt['subspace_size'],
                    color='green',
                    marker=marker,
                    edgecolors='none',
                    s=50,
                    zorder=3
                )
        else:
            if 'mean_sample_number' not in df.columns:
                print(f"[warning] Skipping Plot 3 for {run_path} - missing mean_sample_number")
            elif fci_energy is None:
                print(f"[warning] Skipping Plot 3 for {run_path} - missing fci_energy")

    # Finalize Convergence Plot
    ax_conv.set_xlabel("Subspace Size")
    ax_conv.set_ylabel("Energy Error (Ha)")
    ax_conv.set_yscale('log')
    ax_conv.legend()
    ax_conv.set_title("QSCI Convergence Comparison")
    ax_conv.grid(True, alpha=0.3)
    fig_conv.tight_layout()
    fig_conv.savefig(out_dir / "qsci_convergence.png")
    print(f"[success] Saved {out_dir / 'qsci_convergence.png'}")

    # Finalize Samples Plot
    ax_samp.set_xlabel("Mean Sample Number")
    ax_samp.set_ylabel("Energy Error (Ha)")
    ax_samp.set_yscale('log')
    if samples_xlim is not None:
        ax_samp.set_xlim(left=samples_xlim[0], right=samples_xlim[1])
    ax_samp.legend()
    ax_samp.set_title("Energy vs Samples Comparison")
    ax_samp.grid(True, alpha=0.3)
    fig_samp.tight_layout()
    fig_samp.savefig(out_dir / "energy_vs_samples.png")
    print(f"[success] Saved {out_dir / 'energy_vs_samples.png'}")

    # Finalize Samples vs Configs Error Plot
    ax_sc.set_xscale('log')
    ax_sc.set_xlabel("Mean Sample Number")
    ax_sc.set_ylabel("Configuration Subspace Size")
    ax_sc.set_title("Subspace Size vs Samples (Color: Log Energy Error)")
    
    # Add legend for devices
    if sc_handles:
        ax_sc.legend(handles=sc_handles, labels=sc_labels, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # Add colorbar
    # We use the last 'scatter' object from the loop
    # If there were multiple scatters, they all share the same cmap and norm if vmin/vmax are set
    try:
        cbar = fig_sc.colorbar(scatter, ax=ax_sc, pad=0.15)
        cbar.set_label('log10(Energy Error)')
    except NameError:
        pass # No scatter plots were created

    fig_sc.tight_layout()
    fig_sc.savefig(out_dir / "samples_vs_configs_error.png")
    print(f"[success] Saved {out_dir / 'samples_vs_configs_error.png'}")

if __name__ == "__main__":
    for job_name, config in JOBS.items():
        print(f"\n--- Running comparison for {job_name} ---")
        
        # Determine decimation based on molecule
        # Every data point for H4 (every_nth=1), every 10th for H6
        every_nth = 10 if "H6" in job_name.upper() else 1
        
        run_comparison(
            directories=config["directories"],
            out_dir=config["out_dir"],
            samples_xlim=config["samples_xlim"],
            every_nth=every_nth
        )
