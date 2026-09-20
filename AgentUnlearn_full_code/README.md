# AgentUnlearn — complete experiment implementation

This package implements the experiment plan encoded in the existing `runs_all.csv` manifest for the AgentUnlearn revision.

## Scope implemented

### Benchmarks

- **TOFU**: `forget01`, `forget05`, `forget10`, their matched retain splits, and perturbed probes when available.
- **MUSE-News**: native `knowmem`, `verbmem`, and `raw` subsets.
- **MUSE-Books**: native `knowmem`, `verbmem`, and `raw` subsets.
- **RWKU**: target-specific `forget_level1`, `forget_level2`, `forget_level3`, neighbor probes, and target evidence.
- **WMDP**: Bio/Cyber appendix adapter.

No ACL-2025 corpus is used by the execution package.

### Interaction topologies

1. Independent control
2. Sequential chain
3. Debate / critique
4. Shared scratchpad
5. Hierarchical aggregation

Every agent call is stored, including visible context, private memory counts, received peer messages, filtered segments, and generation metadata. Both the designated target-agent response and final system response are retained.

### Inference interventions

- `none`
- `global_context`
- `target_context`
- `target_memory`
- `communication`
- `target_context_comm`
- `full_inference`

The code explicitly separates target-agent context, private memory, and inbound communication. Peer evidence availability, communication threshold, target-agent position, number of agents, chain length, and debate rounds are driven directly by the run manifest.

### Parameter-level baselines

- `gradient_ascent`
  - supports `all`, `lm_head`, or `last_n_layers` scopes;
  - defaults to the last transformer layers plus LM head to make the baseline feasible on the local A100;
  - saves a parameter delta rather than a second full model copy.
- `lora_grad_diff`
  - LoRA parameter intervention;
  - objective: negative forget loss plus weighted retain loss;
  - saves a PEFT adapter.

The precise gradient-ascent scope is written into checkpoint metadata so the paper cannot accidentally call a restricted update a full-model update.

### Probe protocol

Each condition deterministically selects matched base probes across interventions/topologies. Native benchmark probes are categorized as:

- direct recall;
- indirect reconstruction;
- retained knowledge.

Each selected base probe is evaluated under five deterministic surface-form prompt variants. Therefore stochastic comparisons change model sampling, not the semantic test item.

### Metrics

Per response:

- normalized exact match;
- token F1;
- semantic similarity;
- optional cross-family LLM judge (0/1/2);
- optional answer log-likelihood.

Per condition:

- Direct Recall Rate (DRR);
- Indirect Reconstruction Rate (IRR);
- retained-knowledge accuracy;
- target-agent and system-level versions of each.

Post-processing computes:

- Collateral Damage relative to the matched no-intervention baseline;
- Reconstruction Amplification relative to Independent operation;
- Wilson 95% confidence intervals;
- paired exact McNemar tests;
- paired bootstrap confidence intervals;
- clustered binomial GLM;
- optional Bayesian mixed-effects logistic model with random intercepts for unit and probe.

### Human/semantic validation

Included utilities create a stratified human-validation sheet, calibrate the semantic-similarity threshold against human labels, and validate an LLM judge using precision/recall/F1 and Cohen's kappa.

### Trace-level analysis

`trace_scores.csv` scores every intermediate agent output. This supports:

- reconstruction versus chain depth;
- debate-round reconstruction;
- scratchpad accumulation behavior;
- comparison of target-agent and final-system leakage;
- auditing which message segments were removed by communication filtering.

### Reproducibility / resume

- one JSON result per run;
- atomic writes;
- one error JSON per failed run;
- completed runs are skipped on resume;
- normal conditions reuse a loaded model instead of reloading weights for every condition;
- parameter checkpoints are isolated by reloading the base model before each trained intervention checkpoint.

## Package layout

```text
agentunlearn/
  config.py
  engine.py
  interventions.py
  io.py
  models.py
  probes.py
  schemas.py
  scoring.py
  similarity.py
  stats.py
  topologies.py
  training.py
  datasets/
    base.py
    tofu.py
    muse.py
    rwku.py
    wmdp.py

scripts/
  run_manifest.py
  train_parameter_unlearning.py
  score_results.py
  score_loglikelihood.py
  analyze_results.py
  inspect_datasets.py
  validate_project.py
  make_human_validation_sample.py
  calibrate_threshold.py
  validate_judge.py
  make_figures.py
  make_paper_tables.py

config/defaults.json
tests/
```

## Important methodological settings

`config/defaults.json` is deliberately explicit about settings that must be reported in the paper: prompt count, evidence allocation, generation length, semantic backend, semantic threshold, training scope, LoRA rank, learning rate, retain-loss weight, and training steps.

The default similarity backend is `lexical` only so the code can execute without silently downloading an embedding model. For the final paper experiments, set `similarity.backend` to `sentence_transformers` or `hf_mean_pool` and provide the exact local frozen embedding-model path; then calibrate the threshold using the included human-validation utilities. This prevents an unspecified or silently changing semantic judge from entering the reported results.

## Expected existing project inputs

The package is designed for the project state already constructed on the server:

```text
/home/tahiti/AgentUnlearn/
  data/
    TOFU
    MUSE-News
    MUSE-Books
    RWKU
    WMDP
  models/
    Qwen3-4B
    Qwen3-8B
    Qwen3-14B
    Mistral-7B-Instruct-v0.3
    Qwen2.5-Coder-1.5B-Instruct
    Qwen2.5-Coder-7B-Instruct
    Qwen2.5-Coder-14B-Instruct
  manifests/
    models_manifest.csv
    benchmark_units_manifest.csv
    runs_all.csv
    training_parameter_unlearning.csv
```

The execution commands are intentionally not prescribed here; the package exposes filtering by block/model/benchmark/run ID so launch strategy can be decided separately based on GPU occupancy and deadline.
