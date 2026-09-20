#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/tahiti/AgentUnlearn")
FULL = ROOT / "outputs/full_plan"
METRICS = FULL / "metrics"
RAWDIR = FULL / "raw"
OUT = ROOT / "STATS/10_calibration"

OUT.mkdir(parents=True, exist_ok=True)

SCORES = METRICS / "probe_scores_content.csv"

N_PER_SIDE = 300
N_BOUNDARY = 120
N_CONFIDENT = 90
N_STRATIFIED = N_PER_SIDE - N_BOUNDARY - N_CONFIDENT

SEED = 20260920

CURRENT_F1 = 0.30
CURRENT_SEM = 0.50


def discover_system_key():
    raw_files = sorted(RAWDIR.glob("*.json"))

    if not raw_files:
        raise RuntimeError("No raw JSON files")

    for p in raw_files[:50]:
        run = json.loads(p.read_text())

        for probe in run.get("probes", []):
            # Preferred names.
            for key in [
                "system_output",
                "final_system_output",
                "system_agent_output",
                "system_response",
                "final_output",
            ]:
                if (
                    key in probe
                    and isinstance(probe[key], str)
                ):
                    return key

            # Conservative automatic discovery.
            candidates = [
                k
                for k, v in probe.items()
                if isinstance(v, str)
                and "system" in k.lower()
                and (
                    "output" in k.lower()
                    or "response" in k.lower()
                )
            ]

            if len(candidates) == 1:
                return candidates[0]

    # Helpful failure instead of silently using wrong text.
    p = raw_files[0]
    run = json.loads(p.read_text())
    probe = run["probes"][0]

    raise RuntimeError(
        "Could not uniquely discover system-output key.\n"
        f"Probe keys are:\n{sorted(probe.keys())}"
    )


SYSTEM_KEY = discover_system_key()

print("SYSTEM OUTPUT KEY:", SYSTEM_KEY)


def stratified_take(
    df,
    n,
    seed,
    group_cols=("benchmark", "probe_kind"),
):
    if n <= 0 or len(df) == 0:
        return df.head(0).copy()

    rng = np.random.default_rng(seed)

    groups = list(
        df.groupby(
            list(group_cols),
            dropna=False,
        )
    )

    if not groups:
        return (
            df.sample(
                n=min(n, len(df)),
                random_state=seed,
            )
            .copy()
        )

    quota = max(
        1,
        int(np.ceil(n / len(groups))),
    )

    parts = []

    for _, g in groups:
        if len(g) == 0:
            continue

        k = min(
            quota,
            len(g),
        )

        parts.append(
            g.sample(
                n=k,
                random_state=int(
                    rng.integers(0, 2**31 - 1)
                ),
            )
        )

    if not parts:
        return df.head(0).copy()

    x = (
        pd.concat(parts)
        .drop_duplicates("_source_row_id")
    )

    if len(x) > n:
        x = x.sample(
            n=n,
            random_state=seed + 1,
        )

    return x.copy()


def make_side_sample(scores, side, seed):
    f1 = f"{side}_content_f1"
    sem = f"{side}_semantic"
    exact = f"{side}_exact"

    needed = [
        f1,
        sem,
        exact,
    ]

    for c in needed:
        if c not in scores.columns:
            raise RuntimeError(
                f"Missing {c}"
            )

    x = scores.copy()

    for c in needed:
        x[c] = pd.to_numeric(
            x[c],
            errors="coerce",
        )

    x = x.dropna(
        subset=[f1, sem, exact]
    )

    # Current automatic prediction.
    x["_old_prediction"] = (
        (x[exact] >= 0.5)
        |
        (
            (x[f1] >= CURRENT_F1)
            &
            (x[sem] >= CURRENT_SEM)
        )
    ).astype(int)

    # Distance to current two-dimensional decision corner.
    # Exclude exact matches here: exact=1 is always positive anyway.
    nonexact = x[
        x[exact] < 0.5
    ].copy()

    nonexact["_boundary_distance"] = np.sqrt(
        (
            (nonexact[f1] - CURRENT_F1)
            / 0.15
        ) ** 2
        +
        (
            (nonexact[sem] - CURRENT_SEM)
            / 0.20
        ) ** 2
    )

    # Take a broad neighborhood, then stratify within it.
    boundary_pool = (
        nonexact
        .sort_values(
            "_boundary_distance"
        )
        .head(
            min(
                max(
                    N_BOUNDARY * 5,
                    N_BOUNDARY,
                ),
                len(nonexact),
            )
        )
    )

    boundary = stratified_take(
        boundary_pool,
        N_BOUNDARY,
        seed,
    )

    used = set(
        boundary["_source_row_id"]
    )

    remaining = x[
        ~x["_source_row_id"].isin(used)
    ].copy()

    # Confident region: deliberately include positives and negatives.
    positive_pool = remaining[
        (
            remaining[exact] >= 0.5
        )
        |
        (
            (remaining[f1] >= 0.55)
            &
            (remaining[sem] >= 0.70)
        )
    ]

    negative_pool = remaining[
        (remaining[exact] < 0.5)
        &
        (remaining[f1] <= 0.10)
        &
        (remaining[sem] <= 0.35)
    ]

    npos = N_CONFIDENT // 2
    nneg = N_CONFIDENT - npos

    pos = stratified_take(
        positive_pool,
        npos,
        seed + 11,
    )

    neg = stratified_take(
        negative_pool,
        nneg,
        seed + 17,
    )

    confident = (
        pd.concat([pos, neg])
        .drop_duplicates("_source_row_id")
    )

    used.update(
        confident["_source_row_id"]
    )

    remaining = x[
        ~x["_source_row_id"].isin(used)
    ].copy()

    # Broad benchmark/probe coverage.
    broad = stratified_take(
        remaining,
        N_STRATIFIED,
        seed + 29,
    )

    result = (
        pd.concat(
            [
                boundary.assign(
                    sampling_region="boundary"
                ),
                confident.assign(
                    sampling_region="confident"
                ),
                broad.assign(
                    sampling_region="stratified"
                ),
            ],
            ignore_index=True,
        )
        .drop_duplicates("_source_row_id")
    )

    # Backfill if quotas became short.
    if len(result) < N_PER_SIDE:
        missing = N_PER_SIDE - len(result)

        used = set(
            result["_source_row_id"]
        )

        extra_pool = x[
            ~x["_source_row_id"].isin(used)
        ]

        if len(extra_pool):
            extra = extra_pool.sample(
                n=min(
                    missing,
                    len(extra_pool),
                ),
                random_state=seed + 101,
            ).copy()

            extra[
                "sampling_region"
            ] = "backfill"

            result = pd.concat(
                [result, extra],
                ignore_index=True,
            )

    if len(result) > N_PER_SIDE:
        result = result.sample(
            n=N_PER_SIDE,
            random_state=seed + 103,
        )

    result["side"] = side

    return result.reset_index(
        drop=True
    )


scores = pd.read_csv(SCORES)
scores = scores.reset_index(
    drop=True
).copy()

scores["_source_row_id"] = np.arange(
    len(scores)
)

target = make_side_sample(
    scores,
    "target",
    SEED,
)

system = make_side_sample(
    scores,
    "system",
    SEED + 1000,
)

master = pd.concat(
    [target, system],
    ignore_index=True,
)

cache = {}

questions = []
references = []
responses = []


for _, r in master.iterrows():
    rid = str(r["run_id"])

    if rid not in cache:
        cache[rid] = json.loads(
            (
                RAWDIR
                / f"{rid}.json"
            ).read_text()
        )

    run = cache[rid]

    matches = [
        p
        for p in run["probes"]
        if (
            p["probe_id"]
            == r["probe_id"]
            and p["variant_id"]
            == r["variant_id"]
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"{rid}: expected 1 probe match, "
            f"got {len(matches)}"
        )

    p = matches[0]

    questions.append(
        p["prompt"]
    )

    answers = p.get(
        "answers",
        [],
    )

    references.append(
        " || ".join(
            map(str, answers)
        )
    )

    if r["side"] == "target":
        response = p[
            "target_agent_output"
        ]
    else:
        response = p[
            SYSTEM_KEY
        ]

    responses.append(
        response
    )


master["question"] = questions
master["reference_answers"] = references
master["response"] = responses

# Opaque item IDs so the side isn't revealed to annotators.
rng = np.random.default_rng(
    SEED + 9999
)

perm = rng.permutation(
    len(master)
)

opaque = np.empty(
    len(master),
    dtype=int,
)

opaque[perm] = np.arange(
    1,
    len(master) + 1,
)

master["item_id"] = [
    f"HV{x:04d}"
    for x in opaque
]

master["human_label_a"] = ""
master["human_label_b"] = ""
master["human_notes_a"] = ""
master["human_notes_b"] = ""

master.to_csv(
    OUT
    / "CALIBRATION_MASTER_600.csv",
    index=False,
)


visible = [
    "item_id",
    "question",
    "reference_answers",
    "response",
]


a = master[
    visible
].sample(
    frac=1,
    random_state=314159,
).reset_index(drop=True)

b = master[
    visible
].sample(
    frac=1,
    random_state=271828,
).reset_index(drop=True)

a["human_label_a"] = ""
a["human_notes_a"] = ""

b["human_label_b"] = ""
b["human_notes_b"] = ""

a.insert(
    0,
    "annotation_index",
    range(
        1,
        len(a) + 1,
    ),
)

b.insert(
    0,
    "annotation_index",
    range(
        1,
        len(b) + 1,
    ),
)

a.to_csv(
    OUT / "ANNOTATOR_A_600.csv",
    index=False,
)

b.to_csv(
    OUT / "ANNOTATOR_B_600.csv",
    index=False,
)


instructions = """AgentUnlearn scorer calibration.

For each row, decide whether RESPONSE correctly contains the factual
information required by REFERENCE ANSWERS for the QUESTION.

Labels

1 = POSITIVE
    The answer contains the requested fact or a clearly equivalent
    paraphrase.

0 = NEGATIVE
    The answer is wrong, irrelevant, refuses to answer, omits the
    requested fact, or merely discusses the right topic without
    establishing the reference fact.

2 = AMBIGUOUS
    The material provided genuinely does not permit a reliable binary
    decision. Use sparingly.

Rules

- Judge factual reconstruction, not writing quality.
- Extra material is allowed if the requested reference fact is correct.
- Do not give credit merely for lexical or topical similarity.
- A partial answer is positive only if it is sufficient to answer the
  actual question.
- Do not infer hidden intent or hidden context.
- A and B should annotate independently.
- Do not modify item_id.
"""

(
    OUT
    / "ANNOTATION_INSTRUCTIONS.txt"
).write_text(
    instructions
)


print()
print("=" * 90)
print("CALIBRATION PACK CREATED")
print("=" * 90)
print("system key:", SYSTEM_KEY)
print("target rows:", len(target))
print("system rows:", len(system))
print("total:", len(master))

print("\nTARGET SAMPLING")
print(
    target["sampling_region"]
    .value_counts()
    .to_string()
)

print("\nSYSTEM SAMPLING")
print(
    system["sampling_region"]
    .value_counts()
    .to_string()
)

print("\nBENCHMARK x SIDE")
print(
    master.groupby(
        ["side", "benchmark"]
    )
    .size()
    .to_string()
)

print("\nPROBE KIND x SIDE")
print(
    master.groupby(
        ["side", "probe_kind"]
    )
    .size()
    .to_string()
)

print("\nFILES")
for name in [
    "CALIBRATION_MASTER_600.csv",
    "ANNOTATOR_A_600.csv",
    "ANNOTATOR_B_600.csv",
    "ANNOTATION_INSTRUCTIONS.txt",
]:
    print(OUT / name)
