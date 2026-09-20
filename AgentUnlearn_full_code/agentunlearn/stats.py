from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return max(0.0, center - half), min(1.0, center + half)


def paired_bootstrap_difference(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 5000,
    seed: int = 42,
) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) == 0:
        return {"n": 0, "mean_diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        diffs[i] = np.mean(a[idx] - b[idx])
    return {
        "n": int(len(a)),
        "mean_diff": float(np.mean(a - b)),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
    }


def mcnemar_exact(a: Sequence[int], b: Sequence[int]) -> dict:
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    mask = np.isin(a, [0, 1]) & np.isin(b, [0, 1])
    a, b = a[mask], b[mask]
    b01 = int(np.sum((a == 0) & (b == 1)))
    b10 = int(np.sum((a == 1) & (b == 0)))
    n = b01 + b10
    if n == 0:
        return {"discordant_01": b01, "discordant_10": b10, "p_value": 1.0}
    try:
        from scipy.stats import binomtest
        p = float(binomtest(min(b01, b10), n=n, p=0.5, alternative="two-sided").pvalue)
    except Exception:
        # Normal approximation fallback.
        chi = (abs(b01 - b10) - 1) ** 2 / n if n else 0.0
        p = math.erfc(math.sqrt(max(0.0, chi) / 2))
    return {"discordant_01": b01, "discordant_10": b10, "p_value": p}
