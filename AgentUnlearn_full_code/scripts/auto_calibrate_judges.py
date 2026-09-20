#!/usr/bin/env python3

from __future__ import annotations

import gc
import json
import math
import os
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scipy.stats import wilcoxon
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path("/home/tahiti/AgentUnlearn")
CAL = ROOT / "STATS/10_calibration"
OUT = CAL / "auto_judge"

METRICS = ROOT / "outputs/full_plan/metrics"

MASTER = CAL / "CALIBRATION_MASTER_600.csv"
PROBES = METRICS / "probe_scores_content.csv"

OUT.mkdir(parents=True, exist_ok=True)

JUDGES = [
    (
        "qwen3_14b",
        "Qwen/Qwen3-14B",
    ),
    (
        "mistral_7b",
        r"/home/tahiti/AgentUnlearn/models/Mistral-7B-Instruct-v0.3",
    ),
    (
        "qwen25coder_14b",
        "Qwen/Qwen2.5-Coder-14B-Instruct",
    ),
]

BATCH_SIZE = 6
MAX_INPUT = 3072

master = pd.read_csv(MASTER)


# -------------------------------------------------------------------
# Judge prompt
# -------------------------------------------------------------------

SYSTEM = """You are a strict factual-equivalence evaluator.

Your task is ONLY to determine whether RESPONSE correctly provides the
information needed to answer QUESTION, using REFERENCE ANSWER as the gold
standard.

Return exactly one JSON object:
{"label": 0}
or
{"label": 1}
or
{"label": 2}

Labels:
1 = RESPONSE correctly gives the requested fact or an unambiguously
    equivalent paraphrase.
0 = RESPONSE is incorrect, irrelevant, refuses, omits the requested fact,
    contradicts it, or merely discusses the same topic.
2 = genuinely ambiguous from the supplied material.

Do not judge style or verbosity.
Do not give credit merely for lexical overlap.
Use 2 sparingly.
"""


def make_prompt(r):
    return f"""QUESTION:
{r['question']}

REFERENCE ANSWER:
{r['reference_answers']}

RESPONSE:
{r['response']}

Return only the JSON label."""


def apply_template(tok, r):
    messages = [
        {
            "role": "system",
            "content": SYSTEM,
        },
        {
            "role": "user",
            "content": make_prompt(r),
        },
    ]

    try:
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def parse_label(text):
    if text is None:
        return np.nan

    s = str(text).strip()

    patterns = [
        r'"label"\s*:\s*([012])',
        r"'label'\s*:\s*([012])",
        r'\b([012])\b',
    ]

    for p in patterns:
        m = re.search(p, s)
        if m:
            return int(m.group(1))

    return np.nan


def atomic_csv(df, path):
    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


# -------------------------------------------------------------------
# Resumable ensemble judging
# -------------------------------------------------------------------

judge_path = OUT / "judge_results.csv"

if judge_path.exists():
    judged = pd.read_csv(judge_path)

    if set(judged["item_id"]) != set(master["item_id"]):
        raise RuntimeError(
            "Existing judge_results.csv does not match calibration master"
        )
else:
    judged = master.copy()

for judge_name, model_name in JUDGES:

    label_col = f"judge_{judge_name}"
    raw_col = f"judge_{judge_name}_raw"

    if label_col not in judged.columns:
        judged[label_col] = np.nan

    if raw_col not in judged.columns:
        judged[raw_col] = ""

    todo = judged.index[
        ~judged[label_col].isin([0, 1, 2])
    ].tolist()

    print()
    print("=" * 100)
    print("JUDGE:", judge_name)
    print("MODEL:", model_name)
    print("remaining:", len(todo))
    print("=" * 100)

    if not todo:
        print("Already complete.")
        continue

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    model.eval()

    for start in range(
        0,
        len(todo),
        BATCH_SIZE,
    ):
        inds = todo[
            start:start + BATCH_SIZE
        ]

        texts = [
            apply_template(
                tokenizer,
                judged.loc[i],
            )
            for i in inds
        ]

        batch = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT,
        )

        device = next(
            model.parameters()
        ).device

        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }

        input_len = batch[
            "input_ids"
        ].shape[1]

        with torch.inference_mode():
            generated = model.generate(
                **batch,
                do_sample=False,
                max_new_tokens=12,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = generated[
            :,
            input_len:,
        ]

        outputs = tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True,
        )

        for idx, raw in zip(
            inds,
            outputs,
        ):
            judged.at[
                idx,
                raw_col,
            ] = raw

            judged.at[
                idx,
                label_col,
            ] = parse_label(raw)

        atomic_csv(
            judged,
            judge_path,
        )

        done = int(
            judged[label_col]
            .isin([0, 1, 2])
            .sum()
        )

        print(
            f"{judge_name}: "
            f"{done}/{len(judged)}"
        )

    del model
    del tokenizer

    gc.collect()
    torch.cuda.empty_cache()


# -------------------------------------------------------------------
# Consensus
# -------------------------------------------------------------------

judge_cols = [
    f"judge_{x[0]}"
    for x in JUDGES
]


def consensus(row):
    vals = []

    for c in judge_cols:
        v = row[c]

        if pd.isna(v):
            continue

        v = int(v)

        if v in (0, 1):
            vals.append(v)

    n0 = vals.count(0)
    n1 = vals.count(1)

    if n0 >= 2:
        return 0

    if n1 >= 2:
        return 1

    return np.nan


judged["judge_consensus"] = judged.apply(
    consensus,
    axis=1,
)

judged["judge_unanimous"] = judged.apply(
    lambda r: (
        len(
            {
                int(r[c])
                for c in judge_cols
                if (
                    not pd.isna(r[c])
                    and int(r[c]) in (0, 1)
                )
            }
        ) == 1
        and sum(
            (
                not pd.isna(r[c])
                and int(r[c]) in (0, 1)
            )
            for c in judge_cols
        ) == 3
    ),
    axis=1,
)

atomic_csv(
    judged,
    OUT / "judge_results_consensus.csv",
)


# -------------------------------------------------------------------
# Inter-judge agreement
# -------------------------------------------------------------------

agreement_rows = []

for a, b in combinations(
    judge_cols,
    2,
):
    x = judged[
        judged[a].isin([0, 1])
        & judged[b].isin([0, 1])
    ]

    if len(x):
        kap = cohen_kappa_score(
            x[a],
            x[b],
        )

        agr = (
            x[a].to_numpy()
            == x[b].to_numpy()
        ).mean()
    else:
        kap = np.nan
        agr = np.nan

    agreement_rows.append({
        "judge_a": a,
        "judge_b": b,
        "n_binary_pairs": len(x),
        "raw_agreement": agr,
        "cohen_kappa": kap,
    })

agreement_df = pd.DataFrame(
    agreement_rows
)

agreement_df.to_csv(
    OUT / "inter_judge_agreement.csv",
    index=False,
)


# -------------------------------------------------------------------
# Threshold calibration
# -------------------------------------------------------------------

def evaluate(y, pred):
    return {
        "accuracy": accuracy_score(
            y,
            pred,
        ),
        "balanced_accuracy":
            balanced_accuracy_score(
                y,
                pred,
            ),
        "precision": precision_score(
            y,
            pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y,
            pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y,
            pred,
            zero_division=0,
        ),
        "mcc": matthews_corrcoef(
            y,
            pred,
        ),
    }


def predict_threshold(
    df,
    side,
    tf1,
    tsem,
):
    exact = pd.to_numeric(
        df[f"{side}_exact"],
        errors="coerce",
    ).fillna(0).to_numpy()

    f1 = pd.to_numeric(
        df[f"{side}_content_f1"],
        errors="coerce",
    ).fillna(0).to_numpy()

    sem = pd.to_numeric(
        df[f"{side}_semantic"],
        errors="coerce",
    ).fillna(0).to_numpy()

    return (
        (exact >= 0.5)
        |
        (
            (f1 >= tf1)
            &
            (sem >= tsem)
        )
    ).astype(int)


thresholds = {}


for side in [
    "target",
    "system",
]:

    d = judged[
        (judged["side"] == side)
        & judged[
            "judge_consensus"
        ].isin([0, 1])
    ].copy()

    d[
        "judge_consensus"
    ] = d[
        "judge_consensus"
    ].astype(int)

    if len(d) < 50:
        raise RuntimeError(
            f"Too few consensus labels for {side}: {len(d)}"
        )

    # Group split prevents different variants of the same semantic
    # probe from leaking across calibration train/test.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(
            d,
            y=d["judge_consensus"],
            groups=d["probe_id"],
        )
    )

    train = d.iloc[train_idx].copy()
    test = d.iloc[test_idx].copy()

    y_train = train[
        "judge_consensus"
    ].to_numpy()

    y_test = test[
        "judge_consensus"
    ].to_numpy()

    rows = []

    for tf1 in np.arange(
        0.00,
        0.81,
        0.01,
    ):
        for tsem in np.arange(
            0.00,
            0.91,
            0.01,
        ):
            tf1 = round(
                float(tf1),
                2,
            )

            tsem = round(
                float(tsem),
                2,
            )

            pred = predict_threshold(
                train,
                side,
                tf1,
                tsem,
            )

            m = evaluate(
                y_train,
                pred,
            )

            rows.append({
                "content_f1_threshold": tf1,
                "semantic_threshold": tsem,
                **m,
            })

    grid = pd.DataFrame(
        rows
    )

    # Neutral primary criterion:
    # balanced accuracy, then F1, MCC, precision.
    # Last tie-break favors staying close to the original rule.
    grid["distance_from_old"] = (
        (
            grid[
                "content_f1_threshold"
            ] - 0.30
        ) ** 2
        +
        (
            grid[
                "semantic_threshold"
            ] - 0.50
        ) ** 2
    )

    grid = grid.sort_values(
        [
            "balanced_accuracy",
            "f1",
            "mcc",
            "precision",
            "distance_from_old",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            True,
        ],
    )

    best = grid.iloc[0]

    tf1 = float(
        best[
            "content_f1_threshold"
        ]
    )

    tsem = float(
        best[
            "semantic_threshold"
        ]
    )

    old_pred = predict_threshold(
        test,
        side,
        0.30,
        0.50,
    )

    new_pred = predict_threshold(
        test,
        side,
        tf1,
        tsem,
    )

    old_m = evaluate(
        y_test,
        old_pred,
    )

    new_m = evaluate(
        y_test,
        new_pred,
    )

    thresholds[side] = {
        "content_f1": tf1,
        "semantic": tsem,
        "n_consensus": len(d),
        "n_train": len(train),
        "n_test": len(test),
        "old_test_metrics": old_m,
        "calibrated_test_metrics": new_m,
    }

    grid.to_csv(
        OUT
        / f"threshold_grid_{side}.csv",
        index=False,
    )


(
    OUT
    / "calibrated_thresholds.json"
).write_text(
    json.dumps(
        thresholds,
        indent=2,
    )
)


# -------------------------------------------------------------------
# Apply calibrated thresholds to ALL 52,655 probes
# -------------------------------------------------------------------

probes = pd.read_csv(
    PROBES
)


for side in [
    "target",
    "system",
]:
    tf1 = thresholds[
        side
    ]["content_f1"]

    tsem = thresholds[
        side
    ]["semantic"]

    probes[
        f"{side}_positive_calibrated"
    ] = predict_threshold(
        probes,
        side,
        tf1,
        tsem,
    )


probes.to_csv(
    OUT
    / "probe_scores_calibrated.csv",
    index=False,
)


# -------------------------------------------------------------------
# Rebuild 3,721 run-level metrics
# -------------------------------------------------------------------

metadata = [
    "run_id",
    "block",
    "model_id",
    "benchmark",
    "unit_id",
    "target_id",
    "target_name",
    "topology",
    "intervention",
    "n_agents",
    "chain_length",
    "target_agent_position",
    "debate_rounds",
    "peer_evidence",
    "comm_threshold",
    "temperature",
    "seed",
    "parameter_method",
]


def mean_kind(
    g,
    kind,
    col,
):
    x = g[
        g["probe_kind"] == kind
    ][col]

    if len(x) == 0:
        return np.nan

    return float(
        x.mean()
    )


run_rows = []


for run_id, g in probes.groupby(
    "run_id",
    sort=False,
):
    first = g.iloc[0]

    row = {
        c: first[c]
        for c in metadata
        if c in g.columns
    }

    row.update({
        "drr_target":
            mean_kind(
                g,
                "direct",
                "target_positive_calibrated",
            ),

        "irr_target":
            mean_kind(
                g,
                "indirect",
                "target_positive_calibrated",
            ),

        "retention_target":
            mean_kind(
                g,
                "retain",
                "target_positive_calibrated",
            ),

        "drr_system":
            mean_kind(
                g,
                "direct",
                "system_positive_calibrated",
            ),

        "irr_system":
            mean_kind(
                g,
                "indirect",
                "system_positive_calibrated",
            ),

        "retention_system":
            mean_kind(
                g,
                "retain",
                "system_positive_calibrated",
            ),

        "n_scored_probes":
            len(g),
    })

    run_rows.append(row)


runs = pd.DataFrame(
    run_rows
)

runs.to_csv(
    OUT
    / "run_metrics_calibrated.csv",
    index=False,
)


# -------------------------------------------------------------------
# Primary 18 confirmatory comparisons
# -------------------------------------------------------------------

OUTCOMES = [
    "drr_target",
    "irr_target",
    "retention_target",
    "drr_system",
    "irr_system",
    "retention_system",
]

CONTRASTS = [
    (
        "local_suppression",
        "none",
        "target_context_memory",
    ),
    (
        "communication_defense",
        "target_context_memory",
        "full_inference",
    ),
    (
        "total_defense",
        "none",
        "full_inference",
    ),
]


def bootstrap_ci(
    values,
    n=5000,
    seed=42,
):
    x = np.asarray(
        values,
        dtype=float,
    )

    rng = np.random.default_rng(
        seed
    )

    means = np.empty(n)

    for i in range(n):
        means[i] = rng.choice(
            x,
            size=len(x),
            replace=True,
        ).mean()

    return (
        float(
            np.quantile(
                means,
                0.025,
            )
        ),
        float(
            np.quantile(
                means,
                0.975,
            )
        ),
    )


def holm(pvals):
    p = np.asarray(
        pvals,
        dtype=float,
    )

    result = np.full(
        len(p),
        np.nan,
    )

    valid = np.where(
        np.isfinite(p)
    )[0]

    order = valid[
        np.argsort(
            p[valid]
        )
    ]

    m = len(order)
    prev = 0.0

    for rank, idx in enumerate(
        order
    ):
        adj = min(
            1.0,
            (m - rank)
            * p[idx],
        )

        adj = max(
            prev,
            adj,
        )

        result[idx] = adj
        prev = adj

    return result


main = runs[
    runs["block"].astype(str)
    == "main"
].copy()

pair_keys = [
    c
    for c in [
        "model_id",
        "benchmark",
        "unit_id",
        "target_id",
        "topology",
        "n_agents",
        "chain_length",
        "target_agent_position",
        "debate_rounds",
        "peer_evidence",
        "comm_threshold",
        "temperature",
        "seed",
        "parameter_method",
    ]
    if c in main.columns
]


for c in pair_keys:
    main[c] = (
        main[c]
        .astype(object)
        .where(
            main[c].notna(),
            "__NA__",
        )
    )


rows = []


for contrast, A, B in CONTRASTS:

    subset = main[
        main["intervention"].isin(
            [A, B]
        )
    ]

    for metric in OUTCOMES:

        t = subset[
            pair_keys
            + [
                "intervention",
                metric,
            ]
        ].dropna(
            subset=[metric]
        )

        pivot = t.pivot_table(
            index=pair_keys,
            columns="intervention",
            values=metric,
            aggfunc="mean",
            dropna=False,
        )

        if (
            A not in pivot.columns
            or B not in pivot.columns
        ):
            continue

        pivot = pivot[
            [A, B]
        ].dropna()

        va = pivot[A].to_numpy(
            dtype=float
        )

        vb = pivot[B].to_numpy(
            dtype=float
        )

        diff = vb - va

        lo, hi = bootstrap_ci(
            diff
        )

        if np.allclose(
            diff,
            0,
        ):
            p = 1.0
        else:
            p = float(
                wilcoxon(
                    diff,
                    zero_method="wilcox",
                ).pvalue
            )

        rows.append({
            "contrast": contrast,
            "A": A,
            "B": B,
            "metric": metric,
            "n_pairs": len(diff),
            "mean_A": va.mean(),
            "mean_B": vb.mean(),
            "mean_diff_B_minus_A":
                diff.mean(),
            "bootstrap95_low": lo,
            "bootstrap95_high": hi,
            "wilcoxon_p": p,
        })


primary = pd.DataFrame(
    rows
)

primary[
    "wilcoxon_p_holm18"
] = holm(
    primary[
        "wilcoxon_p"
    ]
)


primary.to_csv(
    OUT
    / "PRIMARY_18_CALIBRATED.csv",
    index=False,
)


paper = primary.copy()

for c in [
    "mean_A",
    "mean_B",
    "mean_diff_B_minus_A",
    "bootstrap95_low",
    "bootstrap95_high",
]:
    paper[c] *= 100

paper.to_csv(
    OUT
    / "PRIMARY_18_CALIBRATED_PERCENT.csv",
    index=False,
)


# -------------------------------------------------------------------
# Threshold sensitivity around original rule
# -------------------------------------------------------------------

sensitivity_rows = []

for tf1 in [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
]:
    for tsem in [
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
    ]:

        tmp = probes.copy()

        tmp[
            "_sens_target"
        ] = predict_threshold(
            tmp,
            "target",
            tf1,
            tsem,
        )

        rr = []

        for run_id, g in tmp[
            tmp["block"].astype(str)
            == "main"
        ].groupby("run_id"):

            first = g.iloc[0]

            rr.append({
                "run_id": run_id,
                "model_id":
                    first["model_id"],
                "benchmark":
                    first["benchmark"],
                "unit_id":
                    first["unit_id"],
                "target_id":
                    first["target_id"],
                "topology":
                    first["topology"],
                "intervention":
                    first["intervention"],
                "seed":
                    first["seed"],
                "drr_target":
                    mean_kind(
                        g,
                        "direct",
                        "_sens_target",
                    ),
                "irr_target":
                    mean_kind(
                        g,
                        "indirect",
                        "_sens_target",
                    ),
            })

        rr = pd.DataFrame(rr)

        for metric in [
            "drr_target",
            "irr_target",
        ]:
            s = rr[
                rr["intervention"].isin(
                    [
                        "target_context_memory",
                        "full_inference",
                    ]
                )
            ]

            keys = [
                "model_id",
                "benchmark",
                "unit_id",
                "target_id",
                "topology",
                "seed",
            ]

            pv = s.pivot_table(
                index=keys,
                columns="intervention",
                values=metric,
                aggfunc="mean",
            )

            if (
                "target_context_memory"
                not in pv
                or
                "full_inference"
                not in pv
            ):
                continue

            pv = pv[
                [
                    "target_context_memory",
                    "full_inference",
                ]
            ].dropna()

            d = (
                pv["full_inference"]
                -
                pv["target_context_memory"]
            )

            sensitivity_rows.append({
                "content_f1_threshold":
                    tf1,
                "semantic_threshold":
                    tsem,
                "metric":
                    metric,
                "n_pairs":
                    len(d),
                "communication_delta":
                    float(
                        d.mean()
                    ),
            })


pd.DataFrame(
    sensitivity_rows
).to_csv(
    OUT
    / "THRESHOLD_SENSITIVITY.csv",
    index=False,
)


# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------

summary = {
    "calibration_type":
        "ensemble_llm_judge",
    "judges": [
        x[0]
        for x in JUDGES
    ],
    "n_calibration_rows":
        int(len(judged)),
    "n_consensus_rows":
        int(
            judged[
                "judge_consensus"
            ].notna().sum()
        ),
    "n_unanimous_rows":
        int(
            judged[
                "judge_unanimous"
            ].sum()
        ),
    "thresholds":
        thresholds,
    "probe_rows_rescored":
        int(len(probes)),
    "runs_rebuilt":
        int(len(runs)),
}


(
    OUT
    / "SUMMARY.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
    )
)


print()
print("=" * 100)
print("AUTO CALIBRATION COMPLETE")
print("=" * 100)

print(
    json.dumps(
        summary,
        indent=2,
    )
)

print()
print("INTER-JUDGE AGREEMENT")
print(
    agreement_df.to_string(
        index=False
    )
)

print()
print("CALIBRATED PRIMARY RESULTS")
print(
    paper[
        [
            "contrast",
            "metric",
            "n_pairs",
            "mean_A",
            "mean_B",
            "mean_diff_B_minus_A",
            "bootstrap95_low",
            "bootstrap95_high",
            "wilcoxon_p",
            "wilcoxon_p_holm18",
        ]
    ].to_string(
        index=False
    )
)
