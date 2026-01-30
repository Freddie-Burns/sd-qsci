"""
Upload H4 chain UHF to RHF orbital rotation circuits to AWS devices.
"""

import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
from pyscf import gto, scf
from qiskit import transpile
from qiskit import qpy
from qiskit.quantum_info import Statevector
from qiskit_braket_provider import BraketProvider

from sd_qsci import circuit, utils


# us-east-1
# Forte has very limited capacity of shots * gates
# Max shots 5000 even if low gate count
DEVICE_NAME = "Forte 1"
SHOTS = int(10e3)
MAX_SHOTS = int(1e3)
SUBMIT_JOB = True
OPTIMISATION = 3 # 0, 1, 2, or 3

# Prepare output base directory
STEM = Path(__file__).stem
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / STEM
DATA_DIR.mkdir(parents=True, exist_ok=True)


# Helper functions
def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def write_text(path: Path, content: str) -> None:
    path.write_text(content)

def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))

def append_job_details(tag: str, device: str, shots: int, pattern: str, geometry: str) -> None:
    details_file = DATA_DIR / "job_details.md"
    details_file.parent.mkdir(parents=True, exist_ok=True)
    if not details_file.exists():
        details_file.write_text("| time tag | device | shots | uhf spin seed | geometry |\n| --- | --- | --- | --- | --- |\n")
    with open(details_file, "a") as f:
        # Clean up geometry string: replace newlines with spaces and multiple spaces with single
        geom_clean = " ".join(geometry.split())
        f.write(f"| {tag} | {device} | {shots} | {pattern} | {geom_clean} |\n")


# Define H4 in a line with bond length a
a = 2
mol = gto.M(
    atom=f'''
    H 0 0 {0*a}
    H 0 0 {1*a}
    H 0 0 {2*a}
    H 0 0 {3*a}
    ''',
    basis='sto-3g',
    spin=0,
    charge=0
)

patterns = {
    "antiferromagnetic": [1, -1, 1, -1],  # antiferromagnetic: lowest energy soln
    "ferromagnetic": [1, 1, -1, -1],  # ferromagnetic: higher energy soln
}

rhf = scf.RHF(mol).run()

provider = BraketProvider()
backend = provider.get_backend(DEVICE_NAME)

for label, pattern in patterns.items():
    print(f"\n--- Running for pattern: {label} ---")
    tag = now_tag()
    # SCF to UHF solution from seeded spin pattern.
    uhf = utils.solve_with_spin_pattern(mol, pattern, label)
    qc = circuit.rhf_uhf_orbital_rotation_circuit(mol, rhf, uhf)

    # Keep an unmeasured copy for statevector simulation
    qc_for_sim = qc.copy()

    # Add measurements to create the sampling circuit for the AWS run
    qc.measure_all()

    n_qubits = qc.num_qubits
    print(
        f"[info] Qiskit circuit has {n_qubits} qubits. " \
        "Submitting via qiskit-braket-provider..."
    )

    # Prepare output directory for this pattern/run
    data_dir = DATA_DIR / tag
    data_dir.mkdir(parents=True, exist_ok=True)

    # Save RHF coefficient matrix
    np.save(data_dir / "rhf_mo_coeff.npy", rhf.mo_coeff)
    np.savetxt(data_dir / "rhf_mo_coeff.txt", rhf.mo_coeff, fmt='% .4f')

    # Save UHF coefficient matrices for this pattern
    np.save(data_dir / "uhf_mo_coeff.npy", uhf.mo_coeff)
    np.savetxt(data_dir / "uhf_mo_coeff_a.txt", uhf.mo_coeff[0], fmt='% .4f')
    np.savetxt(data_dir / "uhf_mo_coeff_b.txt", uhf.mo_coeff[1], fmt='% .4f')

    # Save circuit diagrams (human-readable)
    write_text(data_dir / "qiskit_circuit_measured.txt", str(qc))
    write_text(data_dir / "qiskit_circuit_unmeasured.txt", str(qc_for_sim))

    # Record initial metadata
    metadata = {
        "molecule": "H4",
        "geometry": mol.atom,
        "basis": "sto-3g",
        "n_qubits": int(n_qubits),
        "device": DEVICE_NAME,
        "SHOTS": SHOTS,
        "uhf_spin_seed": label,
        "rhf_energy": float(rhf.e_tot),
        "uhf_energy": float(uhf.e_tot),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(data_dir / "metadata.json", metadata)

    # Save detailed circuit data for later exact reconstruction
    with open(data_dir / "qiskit_circuit_unmeasured.qpy", "wb") as f:
        qpy.dump(qc_for_sim, f)
    with open(data_dir / "qiskit_circuit_measured.qpy", "wb") as f:
        qpy.dump(qc, f)

    # Local statevector simulation (using the unmeasured circuit)
    # Remove final measurements if any (safety), then get statevector
    qc_nom = qc_for_sim.remove_final_measurements(inplace=False)
    sv = Statevector.from_instruction(qc_nom)

    # Save as .npy and JSON (real/imag)
    np.save(data_dir / "sim_statevector.npy", sv.data)

    sv_json = {
        "n_qubits": int(n_qubits),
        "dim": int(len(sv.data)),
        "amplitudes": [
            {"index": int(i), "real": float(z.real), "imag": float(z.imag)}
            for i, z in enumerate(sv.data)
        ],
    }
    write_json(data_dir / "sim_statevector.json", sv_json)

    # Also save top-10 probabilities for a quick glance
    probs = np.abs(sv.data) ** 2
    idx_sorted = np.argsort(probs)[::-1][:10]
    top10 = {
        format(int(i), f"0{n_qubits}b"): float(probs[i]) for i in idx_sorted
    }
    write_json(data_dir / "sim_statevector_top10.json", top10)

    # Transpile for the target backend device
    print(f"[info] Transpiling circuit for {DEVICE_NAME}...")
    start_time = time.time()
    tqc = transpile(qc, backend=backend, optimization_level=OPTIMISATION)
    print(f"[info] Finished optimisation at level {OPTIMISATION}.")
    print(f"[info] Transpilation took {time.time() - start_time:.2f} seconds.")

    # Save transpiled circuit artifacts as well (text + QPY)
    write_text(data_dir / "qiskit_circuit_transpiled.txt", str(tqc))
    with open(data_dir / "qiskit_circuit_transpiled.qpy", "wb") as f:
        qpy.dump(tqc, f)

    # Submit to AWS Braket via Qiskit provider
    if SUBMIT_JOB:
        # Determine shot distribution
        job_shots_list = []
        remaining_shots = SHOTS
        while remaining_shots > 0:
            take = min(remaining_shots, MAX_SHOTS)
            job_shots_list.append(take)
            remaining_shots -= take

        print(f"[info] Total SHOTS {SHOTS} split into {len(job_shots_list)} jobs.")

        for i, s in enumerate(job_shots_list):
            job_tag = now_tag()
            print(
                f"[info] Submitting job {i+1}/{len(job_shots_list)} to {DEVICE_NAME} " \
                f"(shots={s}) via Qiskit Braket provider..."
            )

            # Prepare job-specific directory
            job_dir = data_dir / job_tag
            job_dir.mkdir(parents=True, exist_ok=True)

            job = backend.run(tqc, shots=s)
            job_id = job.job_id()
            print(f"[info] Submitted Braket job: {job_id}")
            write_text(job_dir / "job_id.txt", str(job_id))

            # Save minimal backend/job metadata
            write_json(
                job_dir / "job_metadata.json",
                {
                    "backend": backend.name,
                    "device": DEVICE_NAME,
                    "SHOTS": s,
                    "job_id": job_id,
                    "optimisation": OPTIMISATION,
                    "submitted_at": datetime.now().isoformat(timespec="seconds"),
                },
            )

            # Append to job_details.md
            append_job_details(job_tag, DEVICE_NAME, s, label, mol.atom)

            if i < len(job_shots_list) - 1:
                print("[info] Sleeping 2 seconds before next job...")
                time.sleep(2)

        print(f"[info] All submissions and local simulation artifacts saved under: {data_dir}")
    else:
        print(f"[info] SUBMIT_JOB is False. Skipping submission to {DEVICE_NAME}.")
        print(f"[info] Local simulation artifacts saved under: {data_dir}")
