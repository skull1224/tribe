from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


_PERM_RANK_CACHE: dict[tuple[str, tuple[float, ...], int], np.ndarray] = {}


def rank_behavior(values: np.ndarray, behavior: str) -> np.ndarray:
    if behavior == "top_bottom25_category":
        return values.astype(np.float32, copy=False)
    return rankdata(values).astype(np.float32, copy=False)


def corr_fast(a: np.ndarray, b: np.ndarray) -> float:
    centered_a = a - a.mean()
    centered_b = b - b.mean()
    denom = np.sqrt(np.sum(centered_a * centered_a) * np.sum(centered_b * centered_b))
    return float(np.sum(centered_a * centered_b) / denom) if denom > 0 else np.nan


def corr_many(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    centered_a = a.astype(np.float32, copy=False) - np.float32(a.mean())
    centered_b = b - b.mean(axis=1, keepdims=True)
    denom_a = np.sqrt(np.sum(centered_a * centered_a))
    denom_b = np.sqrt(np.sum(centered_b * centered_b, axis=1))
    denom = denom_a * denom_b
    numerator = centered_b @ centered_a
    return np.divide(numerator, denom, out=np.full(len(b), np.nan, dtype=np.float32), where=denom > 0)


def permutation_rank_matrix(
    y: np.ndarray,
    behavior: str,
    i: np.ndarray,
    j: np.ndarray,
    n_perm: int,
    seed: int,
) -> np.ndarray:
    key = (behavior, tuple(float(value) for value in y), n_perm)
    cached = _PERM_RANK_CACHE.get(key)
    if cached is not None:
        return cached
    rng = np.random.default_rng(seed)
    out = np.empty((n_perm, len(i)), dtype=np.float32)
    for perm_idx in range(n_perm):
        perm = rng.permutation(y)
        behavioral = (
            (perm[i] != perm[j]).astype(float)
            if behavior == "top_bottom25_category"
            else np.abs(perm[i] - perm[j])
        )
        out[perm_idx] = rank_behavior(behavioral, behavior)
    _PERM_RANK_CACHE[key] = out
    return out
