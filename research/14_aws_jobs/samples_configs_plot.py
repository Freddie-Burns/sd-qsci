import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Constant for the data folder path
DATA_DIR = Path(__file__).parent / "data"

# Constant for the target folder path
# Set this to the folder you want to process.
# Example: TARGET_FOLDER = DATA_DIR / "14h_h6_emerald" / "combined_221944-221942"
TARGET_FOLDER = DATA_DIR / "14h_h6_emerald" / "combined_221944-221942"

def process_folder(folder_path, current=None, total=None, every_nth=None):
    folder_path = Path(folder_path)
    csv_path = folder_path / "qsci_convergence.csv"
    
    if not csv_path.exists():
        return
    
    progress_str = f"[{current}/{total}] " if current is not None and total is not None else ""
    print(f"{progress_str}Processing: {folder_path}")
    
    # Load data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return
    
    # Load summary to get the target fci_energy
    summary_path = folder_path / "summary.csv"
    if summary_path.exists():
        try:
            summary_df = pd.read_csv(summary_path)
            # summary.csv is likely quantity,value format
            fci_energy = float(summary_df.loc[summary_df['quantity'] == 'fci_energy', 'value'].values[0])
        except Exception:
            print(f"Error reading fci_energy from {summary_path}, falling back.")
            fci_energy = df['fci_subspace_energy'].iloc[-1]
    else:
        # Fallback to fci_subspace_energy if summary.csv is not found
        fci_energy = df['fci_subspace_energy'].iloc[-1]

    # Load metadata to get molecule and device for title
    metadata_path = folder_path / "metadata.json"
    molecule = "Unknown"
    device = "Unknown"
    if metadata_path.exists():
        import json
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                # metadata.json can be a list or a dict depending on the script that created it
                if isinstance(metadata, list):
                    # If it's a list, use the first entry
                    molecule = metadata[0].get('molecule', 'Unknown')
                    device = metadata[0].get('device', 'Unknown')
                elif "origins" in metadata:
                    # If it has "origins", use the top level or first origin
                    molecule = metadata.get('molecule', metadata['origins'][0]['metadata'].get('molecule', 'Unknown'))
                    device = metadata.get('device', metadata['origins'][0]['metadata'].get('device', 'Unknown'))
                else:
                    molecule = metadata.get('molecule', 'Unknown')
                    device = metadata.get('device', 'Unknown')
        except Exception:
            pass
    
    # Calculate energy error (absolute difference between QSCI and FCI energy)
    # Using abs(qsci_energy - fci_energy)
    energy_error = np.abs(df['qsci_energy'] - fci_energy)
    
    # Find the first point that reaches chemical accuracy (0.0016 Hartree)
    chemical_accuracy = 0.0016
    first_chem_acc_idx = df[energy_error < chemical_accuracy].index
    first_chem_acc_point = None
    if not first_chem_acc_idx.empty:
        first_chem_acc_point = df.loc[first_chem_acc_idx[0]]

    # Handle zero error for log (add a small epsilon or use a lower bound)
    epsilon = 1e-15
    log_energy_error = np.log10(energy_error + epsilon)
    
    # Plot: number of samples vs configuration subspace size
    # x: mean_sample_number
    # y: subspace_size
    # color: log_energy_error
    
    plt.figure(figsize=(10, 7))
    sns.set_theme(style="darkgrid")

    # Decimate data if requested
    if every_nth is None:
        # Determine default based on molecule if not explicitly provided
        if "H4" in molecule.upper():
            every_nth = 1
        elif "H6" in molecule.upper():
            every_nth = 10
        else:
            every_nth = 1 # Default to no decimation for unknown

    if every_nth > 1:
        plot_df = df.iloc[::every_nth]
        plot_log_err = log_energy_error.iloc[::every_nth]
        # Add a line going through
        plt.plot(df['mean_sample_number'], df['subspace_size'], color='gray', alpha=0.3, zorder=1)
    else:
        plot_df = df
        plot_log_err = log_energy_error

    scatter = plt.scatter(
        plot_df['mean_sample_number'], 
        plot_df['subspace_size'], 
        c=plot_log_err, 
        cmap='RdBu_r',
        edgecolors='none',
        alpha=0.8,
        vmin=-3, # Set consistent range for colorbar
        vmax=0,
        zorder=2
    )
    
    # Highlight the first point to reach chemical accuracy
    if first_chem_acc_point is not None:
        plt.scatter(
            first_chem_acc_point['mean_sample_number'],
            first_chem_acc_point['subspace_size'],
            color='green',
            edgecolors='none',
            s=50,  # Ensure it matches or is slightly larger than the other points
            zorder=3,
            label='First Chemical Accuracy'
        )
        plt.legend()
    
    # Set x-axis to log scale
    plt.xscale('log')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('log10(Energy Error)')
    
    # Labels and title
    plt.ylabel('Configuration Subspace Size')
    plt.xlabel('Number of Samples (mean_sample_number)')
    plt.title(f'{molecule} on {device}: Subspace Size vs Samples\n(Color: Log Energy Error vs FCI)')
    
    # Save the plot
    output_path = folder_path / "samples_vs_configs_error_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close() # Close figure to free memory
    print(f"Plot saved to: {output_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--every-nth", type=int, default=None, help="Only show every Nth data point and show a line going through")
    args = parser.parse_args()

    if TARGET_FOLDER is not None:
        target_folder = Path(TARGET_FOLDER)
        if (target_folder / "qsci_convergence.csv").exists():
            process_folder(target_folder, 1, 1, every_nth=args.every_nth)
        else:
            print(f"Error: qsci_convergence.csv not found in {target_folder}")
        return

    # Walk through all directories in DATA_DIR to find folders to process
    folders_to_process = []
    for root, dirs, files in os.walk(DATA_DIR):
        if "qsci_convergence.csv" in files:
            folders_to_process.append(root)
    
    total_folders = len(folders_to_process)
    print(f"Found {total_folders} folders to process.")
    
    for i, folder_path in enumerate(folders_to_process, 1):
        process_folder(folder_path, i, total_folders, every_nth=args.every_nth)

if __name__ == "__main__":
    main()
