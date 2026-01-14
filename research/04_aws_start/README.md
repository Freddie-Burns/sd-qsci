04_aws_start — AWS Braket quick start

Purpose
- Discover available AWS Braket devices and run a small GHZ-state trial on a managed simulator or device.

Main scripts
- 04a_aws_devices.py — lists devices available in configured regions.
- 04b_aws_trial.py — builds a 3-qubit GHZ circuit and submits a job (default: Amazon SV1 simulator).
- 04c_h2.py — builds a small chemistry circuit (RHF→UHF rotation for H2) and submits to a selected device.

Available devices (quick reference)

| Device (provider)     | Braket paradigm  | Hardware modality                            |                                  Size (qubits/atoms) | Connectivity / topology (high level)                                                       | Region                                      |
| --------------------- | ---------------- | -------------------------------------------- | ---------------------------------------------------: | ------------------------------------------------------------------------------------------ | ------------------------------------------- |
| **Aria-1 (IonQ)**     | Gate-based       | Trapped ions                                 |                                               **25** | Effectively **all-to-all** interaction typical of trapped-ion systems                      | us-east-1 ([AWS Documentation][1])          |
| **Forte-1 (IonQ)**    | Gate-based       | Trapped ions                                 |                                               **36** | Trapped-ion style (high connectivity / flexible pairing)                                   | us-east-1 ([AWS Documentation][1])          |
| **Ankaa-3 (Rigetti)** | Gate-based       | Superconducting                              | **~82 qubits** (Rigetti markets Ankaa-3 as 82-qubit) | Fixed superconducting coupling graph; Braket exposes a native set for verbatim compilation | us-west-1 ([qcs.rigetti.com][2])            |
| **Garnet (IQM)**      | Gate-based       | Superconducting transmons + tunable couplers |          **20 computational** (plus **30 couplers**) | **Square lattice** w/ tunable couplers; native **XY rotations + CZ**                       | eu-north-1 ([Amazon Web Services, Inc.][3]) |
| **Emerald (IQM)**     | Gate-based       | Superconducting transmons + tunable couplers |                                               **54** | **Square lattice**; native **XY rotations + CZ**                                           | eu-north-1 ([Amazon Web Services, Inc.][4]) |
| **IBEX Q1 (AQT)**     | Gate-based       | Trapped ions (Ca-40)                         |                                               **12** | **All-to-all connectivity**                                                                | eu-north-1 ([Amazon Web Services, Inc.][5]) |
| **Aquila (QuEra)**    | **AHS (analog)** | Neutral atoms (Rydberg)                      |                          **Up to 256 (analog mode)** | Geometry/layout-driven interactions (encode problems in atom placement + analog drive)     | us-east-1 ([Amazon Web Services, Inc.][6])  |

[1]: https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html?utm_source=chatgpt.com "Amazon Braket supported regions and devices"
[2]: https://qcs.rigetti.com/qpus?utm_source=chatgpt.com "Ankaa-3 Quantum Processor - Rigetti QCS"
[3]: https://aws.amazon.com/braket/quantum-computers/iqm/ "IQM - Amazon Braket Quantum Computers - Amazon Web Services"
[4]: https://aws.amazon.com/blogs/quantum-computing/amazon-braket-launches-new-54-qubit-superconducting-quantum-processor-from-iqm/ "Amazon Braket launches new 54-qubit superconducting quantum processor from IQM | AWS Quantum Technologies Blog"
[5]: https://aws.amazon.com/about-aws/whats-new/2025/11/amazon-braket-alpine-quantum-technologies/?utm_source=chatgpt.com "Amazon Braket adds new quantum processor from Alpine ..."
[6]: https://aws.amazon.com/braket/quantum-computers/quera/?utm_source=chatgpt.com "quera"

Prerequisites
- AWS credentials configured (e.g., in ~/.aws/credentials) with access to the chosen regions.
- Permissions to use Braket resources; costs may apply for managed devices.

Usage
- From repository root:
  - python research/04_aws_start/04a_aws_devices.py
  - python research/04_aws_start/04b_aws_trial.py
    - Optionally adjust the region or target device ARN in the script.
  - python research/04_aws_start/04c_h2.py
    - Edit the hard-coded device in main() if you want to target a different backend.

Outputs
- Console listing of devices (04a).
- Console measurement counts for the GHZ circuit (04b).
