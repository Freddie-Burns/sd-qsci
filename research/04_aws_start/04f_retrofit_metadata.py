from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

from pyscf import gto, scf
from qiskit import qpy
from qiskit.quantum_info import Statevector
import numpy as np

from sd_qsci.utils import uhf_from_rhf
from sd_qsci.circuit import rhf_uhf_orbital_rotation_circuit


# Configuration
# If True, rebuild and overwrite .qpy/.txt files even if they already exist.
FORCE_REBUILD: bool = False


def main():
    base_dir = Path(__file__).resolve().parent
    data_root = base_dir / "data"

    if not data_root.exists() or not data_root.is_dir():
        print(f"[warn] Data directory not found: {data_root}")
        return

    n_scanned = 0
    n_built = 0
    n_simulated = 0
    n_skipped_existing = 0
    n_skipped_missing_meta = 0
    n_skipped_missing_geom = 0

    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue

        n_scanned += 1
        tag_dir = child

        meta_path = tag_dir / "metadata.json"
        unmeasured_qpy = tag_dir / "qiskit_circuit_unmeasured.qpy"
        measured_qpy = tag_dir / "qiskit_circuit_measured.qpy"
        unmeasured_txt = tag_dir / "qiskit_circuit_unmeasured.txt"
        measured_txt = tag_dir / "qiskit_circuit_measured.txt"

        if not meta_path.exists():
            print(f"[skip] {tag_dir.name}: metadata.json not found")
            n_skipped_missing_meta += 1
            continue

        # Decide whether to (re)build circuits
        has_qpys = unmeasured_qpy.exists() and measured_qpy.exists()
        need_build = (not has_qpys) or FORCE_REBUILD

        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            print(f"[warn] {tag_dir.name}: failed to read metadata.json — {e}")
            n_skipped_missing_meta += 1
            continue

        geometry = meta.get("geometry")
        if geometry is None:
            print(f"[skip] {tag_dir.name}: 'geometry' not present in metadata.json")
            n_skipped_missing_geom += 1
            continue

        basis = meta.get("basis", "sto-3g")

        try:
            qc_unmeasured = None
            n_qubits = None

            if need_build:
                # Build molecule and mean-field references
                mol = gto.M(atom=geometry, basis=basis, unit="Angstrom", verbose=0)
                rhf = scf.RHF(mol).run()
                uhf = uhf_from_rhf(mol, rhf)

                # Build circuit as in 04c_h4.py
                qc = rhf_uhf_orbital_rotation_circuit(
                    mol, rhf, uhf, optimize_single_slater=True
                )

                # Preserve an unmeasured copy for statevector simulations
                qc_unmeasured = qc.copy()
                n_qubits = qc_unmeasured.num_qubits

                # Create a measured sampling circuit
                qc_measured = qc.copy()
                qc_measured.measure_all()

                # Save human-readable circuit snapshots
                with unmeasured_txt.open("w", encoding="utf-8") as ftxt:
                    ftxt.write(str(qc_unmeasured))
                with measured_txt.open("w", encoding="utf-8") as ftxt:
                    ftxt.write(str(qc_measured))

                # Save QPY artifacts
                with unmeasured_qpy.open("wb") as f:
                    qpy.dump(qc_unmeasured, f)
                with measured_qpy.open("wb") as f:
                    qpy.dump(qc_measured, f)

                print(
                    f"[make] {tag_dir.name}: built circuits and saved .qpy/.txt (basis={basis})"
                )
                n_built += 1
            else:
                # Load existing unmeasured circuit for simulation
                with unmeasured_qpy.open("rb") as f:
                    loaded = qpy.load(f)
                    # qpy.load may return a single circuit or a list
                    qc_unmeasured = loaded[0] if isinstance(loaded, (list, tuple)) else loaded
                n_qubits = qc_unmeasured.num_qubits
                print(f"[ok]   {tag_dir.name}: circuit files present — using existing for simulation")
                n_skipped_existing += 1

            # Statevector simulation (create sim_statevector files like in 04c_h4.py)
            sim_json = tag_dir / "sim_statevector.json"
            sim_npy = tag_dir / "sim_statevector.npy"
            sim_top10 = tag_dir / "sim_statevector_top10.json"

            if sim_json.exists() and not FORCE_REBUILD:
                # If already present and not forcing, skip simulation output
                pass
            else:
                # Ensure we simulate the unmeasured circuit (strip any final measurements defensively)
                remove_fn = getattr(qc_unmeasured, "remove_final_measurements", None)
                if callable(remove_fn):
                    qc_nom = qc_unmeasured.remove_final_measurements(inplace=False)
                else:
                    qc_nom = qc_unmeasured

                sv = Statevector.from_instruction(qc_nom)

                # Save .npy
                np.save(sim_npy, sv.data)

                # Save JSON with real/imag parts
                sv_json = {
                    "n_qubits": int(n_qubits),
                    "dim": int(len(sv.data)),
                    "amplitudes": [
                        {"index": int(i), "real": float(z.real), "imag": float(z.imag)}
                        for i, z in enumerate(sv.data)
                    ],
                }
                with sim_json.open("w", encoding="utf-8") as f:
                    json.dump(sv_json, f, indent=2, sort_keys=True)

                # Also save top-10 probabilities for a quick glance
                probs = np.abs(sv.data) ** 2
                idx_sorted = np.argsort(probs)[::-1][:10]
                top10 = {format(int(i), f"0{n_qubits}b"): float(probs[i]) for i in idx_sorted}
                with sim_top10.open("w", encoding="utf-8") as f:
                    json.dump(top10, f, indent=2, sort_keys=True)

                n_simulated += 1
                print(f"[sim]  {tag_dir.name}: wrote sim_statevector.json (n_qubits={n_qubits})")
        except Exception as e:
            print(f"[warn] {tag_dir.name}: failed to build/save circuits — {e}")

    # Summary
    print(
        "\nSummary: scanned={}, built={}, simulated={}, skipped_existing={}, "
        "skipped_no_meta={}, skipped_no_geometry={}".format(
            n_scanned, n_built, n_simulated, n_skipped_existing, n_skipped_missing_meta, n_skipped_missing_geom
        )
    )


if __name__ == "__main__":
    main()
