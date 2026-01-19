# Run metadata summary

This table lists each run directory that contains a `metadata.json`, showing the folder name, device, and geometry.

| run folder        | device              | geometry                 |
|-------------------|---------------------|--------------------------|
| 20260114-130205   | rigetti/Ankaa-3     | bond_length 2.0 Å        |
| 20260114-132349   | rigetti/Ankaa-3     | bond_length 2.0 Å        |
| 20260114-132726   | rigetti/Ankaa-3     | bond_length 2.0 Å        |
| 20260114-132954   | rigetti/Ankaa-3     | bond_length 2.0 Å        |
| 20260114-133022   | rigetti/Ankaa-3     | bond_length 2.0 Å        |
| 20260114-133714   | rigetti/Ankaa-3     | bond_length 2.0 Å        |
| 20260119-094732   | SV1                 | H 0 0 0; H 0 0 2.0       |
| 20260119-095902   | SV1                 | H 0 0 0; H 0 0 2.0       |

Notes
- Older runs (2026-01-14) used `bond_length_angstrom` in `metadata.json`; geometry is displayed as a bond length.
- Newer runs record full `geometry` strings and explicit `device` names.
