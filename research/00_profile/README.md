00_profile (profiling)

Purpose
- Profile the performance of the RHF→UHF workflow and related utilities using line_profiler.

Main script
- 00_profile.py — builds an H6 triangular lattice case and profiles the RHF→UHF orbital-rotation circuit path.

Outputs
- A line-profiler report (.lprof) is produced (e.g., 00_profile.py.lprof) and timing info is printed to stdout.

Usage
- From repository root:
  - kernprof -l -v research/00_profile/00_profile.py
  - Or run normally: python research/00_profile/00_profile.py
