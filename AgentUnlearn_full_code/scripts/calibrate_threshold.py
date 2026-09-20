#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def binary_metrics(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(y) if len(y) else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall, f1=f1, accuracy=accuracy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--score-column", default="target_semantic")
    ap.add_argument("--label-column", default="human_label")
    ap.add_argument("--output", default="threshold_calibration.json")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    df = df.dropna(subset=[args.score_column, args.label_column]).copy()
    df[args.label_column] = df[args.label_column].astype(int)
    thresholds = np.unique(np.concatenate([
        np.linspace(0.0, 1.0, 201),
        df[args.score_column].to_numpy(float),
    ]))
    best = None
    curve = []
    for t in thresholds:
        pred = (df[args.score_column].to_numpy(float) >= t).astype(int)
        metrics = binary_metrics(df[args.label_column], pred)
        row = {"threshold": float(t), **metrics}
        curve.append(row)
        if best is None or (row["f1"], row["accuracy"]) > (best["f1"], best["accuracy"]):
            best = row
    Path(args.output).write_text(json.dumps({"best": best, "curve": curve}, indent=2), encoding="utf-8")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
