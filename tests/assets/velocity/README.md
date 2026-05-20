# velocity test assets

This directory contains minimal input samples for velocity V1 Step 2 test-base and sample-asset wiring.

## Scope

These files are input assets only. They are not build outputs, not golden results, not QC summaries, not manifests, and not `traveltime_ready` delivery artifacts.

## normal/

- `engineering_log_ps_valid.csv`
  - Minimal positive input sample for later `ref_engineering` tests.
  - Columns: `depth_m`, `vp_mps`, `vs_mps`.
  - Step 2 uses it only for fixture wiring and lightweight CSV reading.

## negative/

- `engineering_log_missing_depth.csv`
  - Missing depth column.
- `engineering_log_missing_vp.csv`
  - Missing P-wave velocity column.
- `engineering_log_missing_vs.csv`
  - Missing S-wave velocity column.
- `engineering_log_nonpositive_velocity.csv`
  - Contains non-positive velocity values.
- `engineering_log_nonmonotonic_depth.csv`
  - Contains non-monotonic depth order.

## Current testing boundary

Step 2 only verifies that these assets can be located, enumerated, and read by lightweight fixtures/helpers. It does not assert pass/fail business semantics and does not run velocity build code.

Formal fail semantics for these negative cases are left to Step 3 `ref_engineering` business tests.
