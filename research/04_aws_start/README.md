04_aws_start — AWS Braket quick start

Purpose
- Discover available AWS Braket devices and run a small GHZ-state trial on a managed simulator or device.

Main scripts
- 04a_aws_devices.py — lists devices available in configured regions.
- 04b_aws_trial.py — builds a 3-qubit GHZ circuit and submits a job (default: Amazon SV1 simulator).

Prerequisites
- AWS credentials configured (e.g., in ~/.aws/credentials) with access to the chosen regions.
- Permissions to use Braket resources; costs may apply for managed devices.

Usage
- From repository root:
  - python research/04_aws_start/04a_aws_devices.py
  - python research/04_aws_start/04b_aws_trial.py
    - Optionally adjust the region or target device ARN in the script.

Outputs
- Console listing of devices (04a).
- Console measurement counts for the GHZ circuit (04b).
