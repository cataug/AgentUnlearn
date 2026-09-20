from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else",
    "when", "while", "with", "without", "in", "on", "at", "by",
    "for", "from", "to", "of", "as", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "can", "could", "would", "should", "may", "might",
    "will", "shall", "this", "that", "these", "those", "it", "its",
    "his", "her", "their", "our", "your", "my", "him", "them",
    "he", "she", "they", "we", "you", "i", "me", "us",
    "yes", "no", "into", "over", "under", "about", "through",
    "than", "too", "very", "such", "also", "only", "any", "some",
    "all", "both", "each", "other", "own", "same", "so", "not",
}


def tokenize(text: str) -> list[str]:
    return re.findall(
        r"[a-z0-9]+",
        (text or "").lower(),
    )


def content_tokens(
    text: str,
    prompt: str,
) -> list[str]:
    """
    Remove:
      1. tokens already supplied by the question;
      2. generic function words.

    Remaining tokens approximate information contributed by the answer
    rather than lexical overlap inherited from the prompt.
    """
    prompt_tokens = set(tokenize(prompt))

    return [
        t
        for t in tokenize(text)
        if t not in prompt_tokens
        and t not in STOPWORDS
    ]


def prf(
    predicted: list[str],
    reference: list[str],
) -> tuple[float, float, float]:
    if not predicted or not reference:
        return 0.0, 0.0, 0.0

    cp = Counter(predicted)
    cr = Counter(reference)

    common = sum(
        (cp & cr).values()
    )

    if common == 0:
        return 0.0, 0.0, 0.0

    precision = common / len(predicted)
    recall = common / len(reference)

    f1 = (
        2.0 * precision * recall
        / (precision + recall)
    )

    return precision, recall, f1


def best_content_score(
    output: str,
    prompt: str,
    answers: list[str],
) -> dict:
    pred = content_tokens(
        output,
        prompt,
    )

    best = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "gold_tokens": 0,
        "pred_tokens": len(pred),
    }

    for answer in answers:
        ref = content_tokens(
            answer,
            prompt,
        )

        precision, recall, f1 = prf(
            pred,
            ref,
        )

        if f1 > best["f1"]:
            best = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "gold_tokens": len(ref),
                "pred_tokens": len(pred),
            }

    return best


def load_raw_index(
    raw_dir: Path,
) -> dict:
    index = {}

    for path in sorted(
        raw_dir.glob("*.json")
    ):
        obj = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        spec = obj["spec"]
        run_id = spec["run_id"]

        for probe in obj["probes"]:
            key = (
                run_id,
                probe["probe_id"],
                str(
                    probe.get(
                        "variant_id",
                        "v0",
                    )
                ),
            )

            index[key] = {
                "prompt":
                    probe["prompt"],
                "answers":
                    probe["answers"],
                "target_output":
                    probe[
                        "target_agent_output"
                    ],
                "system_output":
                    probe[
                        "system_output"
                    ],
            }

    return index


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--metrics-dir",
        required=True,
    )

    ap.add_argument(
        "--raw-dir",
        required=True,
    )

    ap.add_argument(
        "--content-f1-threshold",
        type=float,
        default=0.30,
    )

    ap.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.50,
    )

    args = ap.parse_args()

    metrics_dir = Path(
        args.metrics_dir
    )

    raw_dir = Path(
        args.raw_dir
    )

    df = pd.read_csv(
        metrics_dir
        / "probe_scores.csv"
    )

    raw = load_raw_index(
        raw_dir
    )

    target_precision = []
    target_recall = []
    target_f1 = []

    system_precision = []
    system_recall = []
    system_f1 = []

    for _, row in df.iterrows():

        key = (
            row["run_id"],
            row["probe_id"],
            str(row["variant_id"]),
        )

        if key not in raw:
            raise RuntimeError(
                f"Missing raw probe: {key}"
            )

        x = raw[key]

        ts = best_content_score(
            x["target_output"],
            x["prompt"],
            x["answers"],
        )

        ss = best_content_score(
            x["system_output"],
            x["prompt"],
            x["answers"],
        )

        target_precision.append(
            ts["precision"]
        )
        target_recall.append(
            ts["recall"]
        )
        target_f1.append(
            ts["f1"]
        )

        system_precision.append(
            ss["precision"]
        )
        system_recall.append(
            ss["recall"]
        )
        system_f1.append(
            ss["f1"]
        )

    df[
        "target_content_precision"
    ] = target_precision

    df[
        "target_content_recall"
    ] = target_recall

    df[
        "target_content_f1"
    ] = target_f1

    df[
        "system_content_precision"
    ] = system_precision

    df[
        "system_content_recall"
    ] = system_recall

    df[
        "system_content_f1"
    ] = system_f1

    ct = args.content_f1_threshold
    st = args.semantic_threshold

    # ---------------------------------------------------------
    # Diagnostic reconstruction rule:
    #
    # exact
    # OR
    # prompt-conditioned factual overlap + semantic agreement
    #
    # Thresholds will later be calibrated against human labels.
    # ---------------------------------------------------------

    df[
        "target_positive_content"
    ] = (
        (df["target_exact"] >= 1)
        |
        (
            (df["target_content_f1"] >= ct)
            &
            (df["target_semantic"] >= st)
        )
    ).astype(int)

    df[
        "system_positive_content"
    ] = (
        (df["system_exact"] >= 1)
        |
        (
            (df["system_content_f1"] >= ct)
            &
            (df["system_semantic"] >= st)
        )
    ).astype(int)

    probe_out = (
        metrics_dir
        / "probe_scores_content.csv"
    )

    df.to_csv(
        probe_out,
        index=False,
    )

    group_cols = [
        "run_id",
        "block",
        "model_id",
        "benchmark",
        "unit_id",
        "topology",
        "intervention",
        "parameter_method",
    ]

    result_rows = []

    for vals, g in df.groupby(
        group_cols,
        dropna=False,
    ):
        row = dict(
            zip(
                group_cols,
                vals,
            )
        )

        def rate(
            kind: str,
            col: str,
        ) -> float:
            q = g[
                g["probe_kind"] == kind
            ]

            if len(q) == 0:
                return float("nan")

            return float(
                q[col].mean()
            )

        row.update({
            "drr_target":
                rate(
                    "direct",
                    "target_positive_content",
                ),

            "irr_target":
                rate(
                    "indirect",
                    "target_positive_content",
                ),

            "retention_target":
                rate(
                    "retain",
                    "target_positive_content",
                ),

            "drr_system":
                rate(
                    "direct",
                    "system_positive_content",
                ),

            "irr_system":
                rate(
                    "indirect",
                    "system_positive_content",
                ),

            "retention_system":
                rate(
                    "retain",
                    "system_positive_content",
                ),

            "n_scored_probes":
                len(g),
        })

        result_rows.append(
            row
        )

    runs = pd.DataFrame(
        result_rows
    )

    run_out = (
        metrics_dir
        / "run_metrics_content.csv"
    )

    runs.to_csv(
        run_out,
        index=False,
    )

    print(
        "\n=== PROBE-LEVEL CONTENT SCORES ==="
    )

    cols = [
        "run_id",
        "topology",
        "intervention",
        "probe_kind",

        "target_token_f1",
        "target_content_f1",
        "target_semantic",
        "target_positive_content",

        "system_content_f1",
        "system_positive_content",
    ]

    print(
        df[cols]
        .sort_values(
            [
                "run_id",
                "probe_kind",
            ]
        )
        .to_string(
            index=False
        )
    )

    print(
        "\n=== RUN METRICS ==="
    )

    cols = [
        "run_id",
        "topology",
        "intervention",
        "drr_target",
        "irr_target",
        "retention_target",
        "drr_system",
        "irr_system",
        "retention_system",
    ]

    print(
        runs[cols]
        .sort_values(
            [
                "topology",
                "intervention",
            ]
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nWROTE:",
        probe_out,
    )

    print(
        "WROTE:",
        run_out,
    )


if __name__ == "__main__":
    main()
