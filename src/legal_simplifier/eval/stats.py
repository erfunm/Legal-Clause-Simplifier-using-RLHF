"""Uncertainty for the results table.

* :func:`bootstrap_ci`: percentile bootstrap CI of a mean over items.
* :func:`paired_test`: system A vs system B on the *same* items. The CI is a
  bootstrap of the mean per-item difference; the p-value is a two-sided
  sign-flip permutation test of that mean (exact under the null that the
  per-item differences are symmetric about 0). Pairing removes the
  between-clause variance, which dominates here.
* :func:`seed_summary`: mean ± sample std of a metric across training seeds.

All resampling is seeded, so a report regenerates identically.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float
    n: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class PairedResult:
    diff: float
    low: float
    high: float
    p_value: float
    n: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def bootstrap_ci(
    values: Sequence[float | None], *, n_boot: int = 1000, confidence: float = 0.95, seed: int = 0
) -> Interval | None:
    data = np.asarray([v for v in values if v is not None and not math.isnan(v)], dtype=float)
    if data.size == 0:
        return None
    mean = float(data.mean())
    if data.size == 1:
        return Interval(mean, mean, mean, 1)
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, data.size, size=(n_boot, data.size))].mean(axis=1)
    alpha = (1 - confidence) / 2
    return Interval(
        mean, float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha)), int(data.size)
    )


def paired_test(
    a: Sequence[float | None],
    b: Sequence[float | None],
    *,
    n_boot: int = 1000,
    n_perm: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedResult | None:
    """Mean of ``a - b`` over items where both are present, with a CI and a p-value."""

    if len(a) != len(b):
        raise ValueError("paired samples must have the same length")
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    diffs = np.asarray([x - y for x, y in pairs], dtype=float)
    observed = float(diffs.mean())
    rng = np.random.default_rng(seed)
    boot = diffs[rng.integers(0, diffs.size, size=(n_boot, diffs.size))].mean(axis=1)
    alpha = (1 - confidence) / 2
    signs = rng.choice((-1.0, 1.0), size=(n_perm, diffs.size))
    permuted = np.abs((signs * diffs).mean(axis=1))
    # +1 smoothing: the observed assignment is one of the permutations.
    p_value = float((np.sum(permuted >= abs(observed) - 1e-12) + 1) / (n_perm + 1))
    return PairedResult(
        observed,
        float(np.quantile(boot, alpha)),
        float(np.quantile(boot, 1 - alpha)),
        min(1.0, p_value),
        int(diffs.size),
    )


def seed_summary(values: Sequence[float | None]) -> dict[str, float | int] | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return {
        "mean": statistics.fmean(present),
        "std": statistics.stdev(present) if len(present) > 1 else 0.0,
        "n_seeds": len(present),
    }
