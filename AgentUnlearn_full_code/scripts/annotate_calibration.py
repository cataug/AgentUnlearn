#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path

import pandas as pd


ROOT = Path("/home/tahiti/AgentUnlearn/STATS/10_calibration")


def wrap(x, width=110):
    x = "" if pd.isna(x) else str(x)
    return "\n".join(
        textwrap.wrap(
            x,
            width=width,
            replace_whitespace=False,
        )
    )


def save_atomic(df, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--annotator",
        required=True,
        choices=["A", "B", "a", "b"],
    )
    args = ap.parse_args()

    ann = args.annotator.upper()

    path = ROOT / f"ANNOTATOR_{ann}_600.csv"

    label_col = f"human_label_{ann.lower()}"
    notes_col = f"human_notes_{ann.lower()}"

    df = pd.read_csv(path)

    if label_col not in df.columns:
        raise RuntimeError(
            f"Missing {label_col}"
        )

    df[label_col] = pd.to_numeric(
        df[label_col],
        errors="coerce",
    )

    remaining = df[
        ~df[label_col].isin([0, 1, 2])
    ].index.tolist()

    done = len(df) - len(remaining)

    print("=" * 110)
    print(f"ANNOTATOR {ann}")
    print(f"FILE      : {path}")
    print(f"DONE      : {done}/{len(df)}")
    print(f"REMAINING : {len(remaining)}")
    print("=" * 110)

    if not remaining:
        print("Nothing left to annotate.")
        return

    for idx in remaining:
        r = df.loc[idx]

        while True:
            print("\n" + "=" * 110)
            print(
                f"ANNOTATOR {ann} | "
                f"{idx + 1}/{len(df)} | "
                f"item_id={r['item_id']}"
            )
            print("=" * 110)

            print("\nQUESTION")
            print("-" * 110)
            print(wrap(r["question"]))

            print("\nREFERENCE ANSWER(S)")
            print("-" * 110)
            print(wrap(r["reference_answers"]))

            print("\nRESPONSE")
            print("-" * 110)
            print(wrap(r["response"]))

            print("\nLABEL:")
            print("  1 = correct reconstruction / answer")
            print("  0 = incorrect / absent / refusal / irrelevant")
            print("  2 = genuinely ambiguous")
            print("  n = add/edit note")
            print("  q = save and stop")
            print()

            value = input("> ").strip().lower()

            if value == "q":
                save_atomic(df, path)

                print(
                    f"\nSaved. Completed "
                    f"{df[label_col].isin([0,1,2]).sum()}/{len(df)}"
                )
                return

            if value == "n":
                note = input("note> ").strip()
                df.at[idx, notes_col] = note
                save_atomic(df, path)
                print("Note saved.")
                continue

            if value not in {"0", "1", "2"}:
                print("Use 0, 1, 2, n, or q.")
                continue

            df.at[idx, label_col] = int(value)

            save_atomic(df, path)

            completed = int(
                df[label_col]
                .isin([0, 1, 2])
                .sum()
            )

            print(
                f"Saved label={value}. "
                f"Progress: {completed}/{len(df)}"
            )

            break

    print("\n" + "=" * 110)
    print(f"ANNOTATOR {ann} COMPLETE: {len(df)}/{len(df)}")
    print("=" * 110)


if __name__ == "__main__":
    main()
