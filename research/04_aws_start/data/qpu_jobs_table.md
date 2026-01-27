# Run metadata summary

This table lists each run directory that contains a `metadata.json`, showing the folder name, device, shots, and geometry.

| run folder        | device              | shots   | geometry                 |
|-------------------|---------------------|---------|--------------------------|
| 20260114-130205   | rigetti/Ankaa-3     |         | bond_length 2.0 Å        |
| 20260114-132349   | rigetti/Ankaa-3     |         | bond_length 2.0 Å        |
| 20260114-132726   | rigetti/Ankaa-3     |         | bond_length 2.0 Å        |
| 20260114-132954   | rigetti/Ankaa-3     |         | bond_length 2.0 Å        |
| 20260114-133022   | rigetti/Ankaa-3     |         | bond_length 2.0 Å        |
| 20260114-133714   | rigetti/Ankaa-3     |         | bond_length 2.0 Å        |
| 20260119-094732   | SV1                 |         | H 0 0 0; H 0 0 2.0       |
| 20260119-095902   | SV1                 |         | H 0 0 0; H 0 0 2.0       |
| 20260119-145015   | Ankaa-3             | 100     | H 0 0 0; H 0 0 2.0; H 0 0 4.0; H 0 0 6.0 |
| 20260119-145059   | Ankaa-3             | 100     | H 0 0 0; H 0 0 2.0; H 0 0 4.0; H 0 0 6.0 |
| 20260119-163557   | Ankaa-3             | 10000   | H 0 0 0; H 0 0 2.0; H 0 0 4.0; H 0 0 6.0 |
| 20260119-170454   | Forte 1             | 1000    | H 0 0 0; H 0 0 2.0; H 0 0 4.0; H 0 0 6.0 |
| 20260120-104429   | Forte 1             | 1000    | H 0 0 0; H 0 0 2.0; H 0 0 4.0; H 0 0 6.0 |
| 20260120-104452   | Forte 1             | 1000    | H 0 0 0; H 0 0 2.0; H 0 0 4.0; H 0 0 6.0 |
| 20260120-104508   | Forte 1             | 1000    | H 0 0 0; H 0 0 2.0; H 0 0 4.0; H 0 0 6.0 |
| 20260120-104531   | Forte 1             | 1000    | H 0 0 0; H 0 0 2.0; H 0 0 4.0; H 0 0 6.0 |
| 20260123-153538   | Forte 1             | 100     | 8 circuits: X on qubit 1..8 |

Notes
- Older runs (2026-01-14) used `bond_length_angstrom` in `metadata.json`; geometry is displayed as a bond length.
- Newer runs record full `geometry` strings and explicit `device` names.
