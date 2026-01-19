from enum import Enum
from pprint import pprint

from qiskit import QuantumCircuit, transpile
from qiskit_braket_provider import BraketProvider


# Target AWS Braket backend (SV1 state vector simulator)
DEVICE_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
SHOTS = 100


class BraketDevice(Enum):
    """Canonical selection of target devices (deduplicated simulators)."""
    # us-east-1
    ARIA_1 = "Aria-1"
    AQUILA = "Aquila"
    FORTE_1 = "Forte 1"
    FORTE_ENTERPRISE_1 = "Forte Enterprise 1"

    # us-west-1
    ANKAA_3 = "Ankaa-3"

    # eu-north-1
    GARNET = "Garnet"
    EMERALD = "Emerald"
    IBEX_Q1 = "Ibex Q1"

    # Global simulators
    SV1 = "SV1"
    TN1 = "TN1"
    DM1 = "dm1"


def main():
    qc = ghz_qc()
    device = BraketDevice.SV1
    run_braket(qc, device)


def print_backends():
    provider = BraketProvider()
    backends = provider.backends()
    for backend in backends:
        pprint(
            {
                "name": backend.name,
                "description": backend.description,
                "online date": backend.online_date,
                "number of qubits": backend.num_qubits,
                "operations": backend.operations[:5],
                "backend version": backend.version,
            }
        )


def ghz_qc():
    # Build circuit in Qiskit
    qc = QuantumCircuit(3)
    # Create GHZ entanglement
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    # Measure all qubits
    qc.measure_all()
    print("Qiskit circuit:")
    print(qc.draw("text"))
    return qc


def run_braket(qc, device):
    provider = BraketProvider()
    backend = provider.get_backend(device.value)

    # Transpile for the target backend and run
    tqc = transpile(qc, backend=backend)
    job = backend.run(tqc, shots=SHOTS)
    job_id = job.job_id()
    print(f"Submitted job {job_id} to backend {backend.name}")

    # Wait for result and show counts
    result = job.result()
    counts = result.get_counts()
    print("Measurement counts:")
    print(counts)


if __name__ == "__main__":
    main()
