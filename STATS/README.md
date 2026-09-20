# AgentUnlearn STATS

Complete statistical export of the finished AgentUnlearn experimental plan.

## Completion

- Inference runs: 3721 / 3721
- Parameter-training jobs: 20 / 20
- Metric CSV files discovered: 5
- Preferred run-level table: run_metrics_content

## Directory layout

- `00_snapshot/` — exact scoring/config/plan snapshot
- `01_inventory/` — experiment, raw-output and metric inventories
- `02_run_level/` — descriptive statistics for every run-level scoring table
- `03_probe_level/` — probe-level summaries and Wilson 95% intervals
- `04_intervention_tests/` — paired intervention contrasts, bootstrap CIs and Wilcoxon tests
- `05_parameter_unlearning/` — GA/LoRA evaluation statistics
- `06_training/` — checkpoint metadata and loss trajectories
- `07_robustness/` — stochastic robustness breakdowns
- `08_tables/` — compact factorial tables for paper analysis

## Main causal contrast

`target_context_memory` versus `full_inference` isolates the additional effect of
communication filtering after target-local context and memory are both removed.

See:

`04_intervention_tests/KEY_CAUSAL_COMPARISONS.csv`

## Scoring note

The current content-aware decision rule uses content-F1 >= 0.30 and semantic
similarity >= 0.50 (plus exact match where applicable). These thresholds should
be reported as automatic thresholds unless the human-validation calibration
pipeline is completed separately.
