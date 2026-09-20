from pathlib import Path
import csv
import math

ROOT = Path("/home/tahiti/AgentUnlearn")
MAN = ROOT / "manifests"
RWKU = ROOT / "data" / "RWKU" / "Target"

MAN.mkdir(parents=True, exist_ok=True)


def write_csv(name, rows):
    path = MAN / name

    if not rows:
        print(name, 0)
        return

    fields = []
    seen = set()

    for row in rows:
        for k in row:
            if k not in seen:
                fields.append(k)
                seen.add(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"{name:32s} {len(rows):6d}")


# ============================================================
# 1. BENCHMARK UNITS
# ============================================================

units = []

# TOFU: controlled forget-size conditions
for split in ["forget01", "forget05", "forget10"]:
    units.append({
        "unit_id": f"tofu_{split}",
        "benchmark": "TOFU",
        "target_id": split,
        "target_name": split,
        "domain": "synthetic_factual",
        "tier": "main",
    })

# MUSE domains
units.append({
    "unit_id": "muse_news",
    "benchmark": "MUSE-News",
    "target_id": "news",
    "target_name": "MUSE News",
    "domain": "news",
    "tier": "main",
})

units.append({
    "unit_id": "muse_books",
    "benchmark": "MUSE-Books",
    "target_id": "books",
    "target_name": "MUSE Books",
    "domain": "books",
    "tier": "main",
})


# ============================================================
# RWKU: choose 10 targets evenly over sorted IDs
# ============================================================

rwku_targets = []

if RWKU.exists():
    for p in RWKU.iterdir():
        if not p.is_dir():
            continue

        name = p.name

        if "_" not in name:
            continue

        tid, tname = name.split("_", 1)

        try:
            tid_int = int(tid)
        except ValueError:
            continue

        rwku_targets.append(
            (tid_int, tname.replace("_", " "), p)
        )

rwku_targets.sort(key=lambda x: x[0])

if len(rwku_targets) < 10:
    selected_rwku = rwku_targets
else:
    # Deterministic, evenly-spaced selection across all targets.
    idxs = [
        round(i * (len(rwku_targets) - 1) / 9)
        for i in range(10)
    ]
    selected_rwku = [rwku_targets[i] for i in idxs]

for tid, name, path in selected_rwku:
    units.append({
        "unit_id": f"rwku_{tid:03d}",
        "benchmark": "RWKU",
        "target_id": str(tid),
        "target_name": name,
        "domain": "real_world_entity",
        "tier": "main",
    })


# WMDP: appendix only
for subset in ["bio", "cyber"]:
    units.append({
        "unit_id": f"wmdp_{subset}",
        "benchmark": "WMDP",
        "target_id": subset,
        "target_name": f"WMDP {subset}",
        "domain": "safety",
        "tier": "appendix",
    })


write_csv("benchmark_units_manifest.csv", units)

main_units = [u for u in units if u["tier"] == "main"]
appendix_units = [u for u in units if u["tier"] == "appendix"]

print("\nMAIN UNITS:")
for u in main_units:
    print(
        f"  {u['unit_id']:18s} "
        f"{u['benchmark']:12s} "
        f"{u['target_name']}"
    )


# ============================================================
# 2. MODEL GROUPS
# ============================================================

MAIN_MODELS = [
    "qwen3_4b",
    "qwen3_8b",
    "mistral_7b",
]

LARGE_MODEL = [
    "qwen3_14b",
]

CODER_MODELS = [
    "qwen25coder_1p5b",
    "qwen25coder_7b",
    "qwen25coder_14b",
]

PARAM_MODELS = [
    "qwen3_4b",
    "qwen3_8b",
]


# ============================================================
# 3. TOPOLOGIES / INTERVENTIONS
# ============================================================

TOPOLOGIES = [
    "independent",
    "sequential",
    "debate",
    "scratchpad",
    "hierarchical",
]

INTERVENTIONS = [
    "none",
    "global_context",

    # Full 2^3 target-agent inference factorial
    "target_context",
    "target_memory",
    "communication",
    "target_context_memory",
    "target_context_comm",
    "target_memory_comm",
    "full_inference",
]

KEY_INTERVENTIONS = [
    "none",

    # Local KF removed, communication open
    "target_context_memory",

    # Same local removal, communication filtered
    "full_inference",
]

ABLATION_INTERVENTIONS = [
    # Clean causal comparison:
    # local forgetting fixed, communication changes.
    "target_context_memory",
    "full_inference",
]


# ============================================================
# helper
# ============================================================

counter = 0


def make_run(
    block,
    model,
    unit,
    topology,
    intervention,
    **kwargs
):
    global counter
    counter += 1

    row = {
        "run_id": f"AU{counter:06d}",
        "block": block,
        "model_id": model,
        "benchmark": unit["benchmark"],
        "unit_id": unit["unit_id"],
        "target_id": unit["target_id"],
        "target_name": unit["target_name"],
        "topology": topology,
        "intervention": intervention,

        # defaults
        "n_agents": 4,
        "chain_length": 4,
        "target_agent_position": 2,
        "debate_rounds": 2,
        "peer_evidence": 1.0,
        "comm_threshold": 0.50,

        "temperature": 0.0,
        "seed": 42,

        "parameter_method": "none",

        "status": "pending",
    }

    row.update(kwargs)
    return row


# ============================================================
# 4. MAIN FACTORIAL
# ============================================================

runs_main = []

for model in MAIN_MODELS:
    for unit in main_units:
        for topology in TOPOLOGIES:
            for intervention in INTERVENTIONS:
                runs_main.append(
                    make_run(
                        "main",
                        model,
                        unit,
                        topology,
                        intervention,
                    )
                )

write_csv("runs_main.csv", runs_main)


# ============================================================
# 5. LARGE-SCALE CONFIRMATION
# Qwen3-14B, same datasets, only key interventions
# ============================================================

runs_large = []

for model in LARGE_MODEL:
    for unit in main_units:
        for topology in TOPOLOGIES:
            for intervention in KEY_INTERVENTIONS:
                runs_large.append(
                    make_run(
                        "large_scale",
                        model,
                        unit,
                        topology,
                        intervention,
                    )
                )

write_csv("runs_large_scale.csv", runs_large)


# ============================================================
# 6. CODER SCALE / SPECIALIZATION ABLATION
# ============================================================

CODER_TOPOLOGIES = [
    "independent",
    "debate",
    "scratchpad",
]

runs_coder = []

for model in CODER_MODELS:
    for unit in main_units:
        for topology in CODER_TOPOLOGIES:
            for intervention in KEY_INTERVENTIONS:
                runs_coder.append(
                    make_run(
                        "coder_scale",
                        model,
                        unit,
                        topology,
                        intervention,
                    )
                )

write_csv("runs_coder_ablation.csv", runs_coder)


# ============================================================
# 7. TARGETED ABLATION UNITS
#
# Keep these small and representative:
# TOFU 01/05/10 + MUSE-News + first RWKU target
# ============================================================

ablation_units = []

wanted = [
    "tofu_forget01",
    "tofu_forget05",
    "tofu_forget10",
    "muse_news",
]

for uid in wanted:
    u = next((x for x in main_units if x["unit_id"] == uid), None)
    if u:
        ablation_units.append(u)

rwku_one = next(
    (x for x in main_units if x["benchmark"] == "RWKU"),
    None
)

if rwku_one:
    ablation_units.append(rwku_one)


# ============================================================
# 7A. NUMBER OF AGENTS
# ============================================================

runs_ablation = []

for unit in ablation_units:
    for topology in ["sequential", "debate", "scratchpad"]:
        for intervention in ABLATION_INTERVENTIONS:
            for n in [1, 2, 8]:  # N=4 already main
                runs_ablation.append(
                    make_run(
                        "ablation_n_agents",
                        "qwen3_8b",
                        unit,
                        topology,
                        intervention,
                        n_agents=n,
                    )
                )


# ============================================================
# 7B. CHAIN LENGTH
# ============================================================

for unit in ablation_units:
    for intervention in ABLATION_INTERVENTIONS:
        for length in [1, 2, 8]:  # 4 already main
            runs_ablation.append(
                make_run(
                    "ablation_chain_length",
                    "qwen3_8b",
                    unit,
                    "sequential",
                    intervention,
                    n_agents=length,
                    chain_length=length,
                )
            )


# ============================================================
# 7C. TARGET AGENT POSITION
# ============================================================

for unit in ablation_units:
    for intervention in ABLATION_INTERVENTIONS:
        for pos in [1, 3, 4]:  # 2 = default
            runs_ablation.append(
                make_run(
                    "ablation_target_position",
                    "qwen3_8b",
                    unit,
                    "sequential",
                    intervention,
                    target_agent_position=pos,
                )
            )


# ============================================================
# 7D. PEER EVIDENCE AVAILABILITY
# ============================================================

for unit in ablation_units:
    for topology in ["sequential", "debate", "scratchpad"]:
        for intervention in ABLATION_INTERVENTIONS:
            for rho in [0.0, 0.25, 0.50]:  # 1.0 already main
                runs_ablation.append(
                    make_run(
                        "ablation_peer_evidence",
                        "qwen3_8b",
                        unit,
                        topology,
                        intervention,
                        peer_evidence=rho,
                    )
                )


# ============================================================
# 7E. DEBATE ROUNDS
# ============================================================

for unit in ablation_units:
    for intervention in ABLATION_INTERVENTIONS:
        for rounds in [0, 1, 4]:  # 2 = default
            runs_ablation.append(
                make_run(
                    "ablation_debate_rounds",
                    "qwen3_8b",
                    unit,
                    "debate",
                    intervention,
                    debate_rounds=rounds,
                )
            )


# ============================================================
# 7F. COMMUNICATION FILTER THRESHOLD
# ============================================================

for unit in ablation_units:
    for topology in ["sequential", "debate", "scratchpad"]:
        for tau in [0.50, 0.60, 0.80, 0.90]:
            # 0.50 = default
            runs_ablation.append(
                make_run(
                    "ablation_comm_threshold",
                    "qwen3_8b",
                    unit,
                    topology,
                    "target_context_comm",
                    comm_threshold=tau,
                )
            )


write_csv("runs_ablations.csv", runs_ablation)


# ============================================================
# 8. PARAMETER-LEVEL UNLEARNING
#
# Evaluation jobs.
# Training jobs get separate manifest below.
# ============================================================

param_units = ablation_units

PARAM_METHODS = [
    "gradient_ascent",
    "lora_grad_diff",
]

runs_param = []

for model in PARAM_MODELS:
    for unit in param_units:
        for method in PARAM_METHODS:
            for topology in TOPOLOGIES:
                for comm in [False, True]:

                    intervention = (
                        "parameter_plus_comm"
                        if comm
                        else "parameter_only"
                    )

                    runs_param.append(
                        make_run(
                            "parameter_unlearning",
                            model,
                            unit,
                            topology,
                            intervention,
                            parameter_method=method,
                        )
                    )

write_csv("runs_parameter_unlearning.csv", runs_param)


# training jobs
training_jobs = []

tid = 0

for model in PARAM_MODELS:
    for unit in param_units:
        for method in PARAM_METHODS:
            tid += 1
            training_jobs.append({
                "training_id": f"TR{tid:04d}",
                "model_id": model,
                "benchmark": unit["benchmark"],
                "unit_id": unit["unit_id"],
                "target_id": unit["target_id"],
                "parameter_method": method,
                "status": "pending",
            })

write_csv(
    "training_parameter_unlearning.csv",
    training_jobs
)


# ============================================================
# 9. STOCHASTIC ROBUSTNESS
#
# Do only on Qwen3-8B and representative units.
# ============================================================

runs_robust = []

for unit in ablation_units:
    for topology in TOPOLOGIES:
        for intervention in ABLATION_INTERVENTIONS:
            for temp in [0.3, 0.7]:
                for seed in [42, 43, 44, 45, 46]:
                    runs_robust.append(
                        make_run(
                            "stochastic_robustness",
                            "qwen3_8b",
                            unit,
                            topology,
                            intervention,
                            temperature=temp,
                            seed=seed,
                        )
                    )

write_csv("runs_robustness.csv", runs_robust)


# ============================================================
# 10. WMDP APPENDIX
# ============================================================

runs_wmdp = []

for model in ["qwen3_8b", "mistral_7b"]:
    for unit in appendix_units:
        for topology in [
            "independent",
            "debate",
            "scratchpad",
        ]:
            for intervention in KEY_INTERVENTIONS:
                runs_wmdp.append(
                    make_run(
                        "wmdp_appendix",
                        model,
                        unit,
                        topology,
                        intervention,
                    )
                )

write_csv("runs_wmdp_appendix.csv", runs_wmdp)


# ============================================================
# 11. ALL RUNS
# ============================================================

all_runs = (
    runs_main
    + runs_large
    + runs_coder
    + runs_ablation
    + runs_param
    + runs_robust
    + runs_wmdp
)

write_csv("runs_all.csv", all_runs)


# ============================================================
# 12. SUMMARY
# ============================================================

print("\n" + "=" * 72)
print("EXPERIMENT PLAN")
print("=" * 72)

blocks = [
    ("MAIN", runs_main),
    ("LARGE SCALE", runs_large),
    ("CODER ABLATION", runs_coder),
    ("ABLATIONS", runs_ablation),
    ("PARAMETER EVAL", runs_param),
    ("ROBUSTNESS", runs_robust),
    ("WMDP APPENDIX", runs_wmdp),
]

for name, rows in blocks:
    print(f"{name:22s}: {len(rows):5d}")

print("-" * 72)
print(f"{'INFERENCE TOTAL':22s}: {len(all_runs):5d}")
print(f"{'TRAINING JOBS':22s}: {len(training_jobs):5d}")
print(f"{'GRAND TOTAL':22s}: {len(all_runs)+len(training_jobs):5d}")

print("\nRWKU selected targets:")
for u in main_units:
    if u["benchmark"] == "RWKU":
        print(
            f"  {u['target_id']:>3s}  {u['target_name']}"
        )

print("\nDONE")
