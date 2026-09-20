#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import json

import numpy as np
import pandas as pd

from calibrate_threshold import binary_metrics


def cohen_kappa(a, b):
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    po = float(np.mean(a == b))
    pa1, pb1 = float(np.mean(a == 1)), float(np.mean(b == 1))
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--human", default="human_label")
    ap.add_argument("--judge", default="target_llm_judge")
    ap.add_argument("--judge-positive-threshold", type=int, default=1)
    args = ap.parse_args()
    df = pd.read_csv(args.csv).dropna(subset=[args.human, args.judge])
    y = df[args.human].astype(int).to_numpy()
    pred = (df[args.judge].astype(int).to_numpy() >= args.judge_positive_threshold).astype(int)
    out = binary_metrics(y, pred)
    out["cohen_kappa"] = cohen_kappa(y, pred)
    out["n"] = len(y)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
