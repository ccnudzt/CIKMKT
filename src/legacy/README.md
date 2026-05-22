# Legacy Experiments

This directory contains the original large-script experimental implementations that were used before the repository was migrated to the unified experiment framework.

These files are kept as historical references only. They are not the supported training entrypoints anymore.

## Mapping To The Unified Framework

- `src/legacy/kaga_dkt/kaga_dkt_eid.py`
  Replaced by:
  - model: `src/models/kaga_dkt.py`
  - data builder: `src/data/graph_kt.py`
  - config family: `configs/experiments/kaga_dkt/`
  - entrypoint: `run_kaga_dkt.py`

- `src/legacy/gcl_kaga_dkt/gcl_random_dkt.py`
- `src/legacy/gcl_kaga_dkt/gcl_prob_dkt.py`
  Replaced by:
  - model: `src/models/gcl_kaga_dkt.py`
  - config family: `configs/experiments/gcl_kaga_dkt/`
  - entrypoint: `run_gcl_kaga_dkt.py`

- `src/legacy/text_gcl_dkt/text_gcl_random.py`
- `src/legacy/text_gcl_dkt/text_gcl_prob.py`
  Replaced by:
  - model: `src/models/text_gcl_dkt.py`
  - data builder: `src/data/text_graph_kt.py`
  - config family: `configs/experiments/text_gcl_dkt/`
  - entrypoint: `run_text_gcl_dkt.py`

## Notes

- Expect hard-coded paths, machine-specific assumptions, mixed-language comments, and duplicated training logic in the legacy files.
- New work should be added to the unified framework under `src/core`, `src/data`, `src/models`, `src/trainers`, and `src/experiments`.
