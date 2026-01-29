from pathlib import Path
from qiskit import qpy

base_path = Path().parent.resolve()
# tqc_path_2 = base_path / "data" / "20260123-153538" / "circuit_2"  / "qiskit_circuit_transpiled.qpy"
# tqc_path_3 = base_path / "data" / "20260123-153538" / "circuit_3"  / "qiskit_circuit_transpiled.qpy"
tqc_path = base_path / "data" / "20260122-123545" / "qiskit_circuit_transpiled.qpy"


# with open(tqc_path_2, 'rb') as f:
#     qc_2 = qpy.load(f)[0]
#     layout_2 = qc_2.layout
#
# with open(tqc_path_3, 'rb') as f:
#     qc_3 = qpy.load(f)[0]
#     layout_3 = qc_3.layout

with open(tqc_path, 'rb') as f:
    qc = qpy.load(f)[0]
    layout = qc.layout

pass

# from qiskit import transpile, QuantumCircuit
# from qiskit import qpy
# from qiskit.quantum_info import Statevector
# from qiskit_braket_provider import BraketProvider