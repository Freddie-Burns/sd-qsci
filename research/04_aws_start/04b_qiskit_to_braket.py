from __future__ import annotations

# Build a simple GHZ circuit in Qiskit and execute it on AWS Braket SV1
# via qiskit-braket-provider.

from qiskit import QuantumCircuit, transpile
from qiskit_braket_provider import BraketProvider

from braket.aws import AwsSession
import boto3


def build_ghz_qiskit(n_qubits: int = 3) -> QuantumCircuit:
    """Create an n-qubit GHZ state circuit in Qiskit and measure all qubits.

    GHZ = (|0...0> + |1...1>) / sqrt(2)
    """
    qc = QuantumCircuit(n_qubits, n_qubits)
    # Create GHZ entanglement
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    # Measure all qubits
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    # Target AWS Braket backend (SV1 state vector simulator)
    DEVICE_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
    SHOTS = 1000

    # Build circuit in Qiskit
    qc = build_ghz_qiskit(3)
    print("Qiskit circuit:")
    try:
        # If supported in the environment, show a text diagram
        print(qc.draw("text"))
    except Exception:
        pass

    # Use qiskit-braket-provider to access the AWS Braket backend
    # Choose a region; SV1 is a regional service but ARN omits region, so we set one.
    region = "us-east-1"
    session = AwsSession(boto_session=boto3.Session(region_name=region))
    provider = BraketProvider(aws_session=session)
    backend = provider.get_backend(DEVICE_ARN)

    # Transpile for the target backend and run
    tqc = transpile(qc, backend=backend)
    job = backend.run(tqc, shots=SHOTS)
    job_id = None
    try:
        job_id = job.id()
    except Exception:
        try:
            job_id = job.job_id()
        except Exception:
            job_id = "unknown"
    print(f"Submitted job {job_id} to backend {backend.name()}")

    # Wait for result and show counts
    result = job.result()
    counts = result.get_counts()
    print("Measurement counts:")
    print(counts)


if __name__ == "__main__":
    main()
