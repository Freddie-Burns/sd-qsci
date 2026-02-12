from pathlib import Path

from qiskit import qpy
from qiskit.transpiler import Layout, TranspileLayout

base_dir = Path(__file__).resolve().parent
circuit_path = base_dir / "data/14a_h4_ankaa/20260130-203635/qiskit_circuit_transpiled.qpy"

with open(circuit_path, "rb") as f:
    tqc = qpy.load(f)[0]

tlayout: TranspileLayout = tqc.layout
layout: Layout = tlayout.final_virtual_layout()
print(tlayout.final_index_layout())
print(layout.get_physical_bits())
print(layout.get_virtual_bits())