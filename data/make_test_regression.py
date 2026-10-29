"""Generate a synthetic regression test dataset.

Specs:
- > 10k rows
- 1 continuous target
- 40 predictors: 10 strong, 20 weak, 10 irrelevant
- predictors are a mix of categorical, integer, and continuous
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(5040)
N = 12_000


def _quantile_labels(signal: np.ndarray, levels: list[str], flip_rate: float) -> np.ndarray:
    """Bin a signal into quantile buckets labelled by `levels`, flipping a
    `flip_rate` fraction of labels to control how informative the category is."""
    n_lvl = len(levels)
    bins = np.quantile(signal, np.linspace(0, 1, n_lvl + 1))
    idx = np.digitize(signal, bins[1:-1])
    flips = RNG.random(N) < flip_rate
    idx = np.where(flips, RNG.integers(0, n_lvl, N), idx)
    return np.array(levels)[idx]


def make_strong_predictors(latent: dict[str, np.ndarray]) -> pd.DataFrame:
    """10 strong predictors: fairly clean views of latent signals, so the
    latents (and hence most of the target signal) are recoverable, but not
    perfectly -- leaving a realistic but high R^2 (~0.85)."""
    df = {}
    z = latent

    # 4 continuous strong predictors (clean, mostly-linear views of latents so
    # the latents are recoverable and achieved R^2 tracks the ceiling).
    df["s_cont1"] = z["z1"] * 1.0 + RNG.normal(0, 0.3, N)
    df["s_cont2"] = z["z2"] * 0.9 + RNG.normal(0, 0.3, N)
    df["s_cont3"] = z["z3"] * 1.1 + RNG.normal(0, 0.3, N)
    df["s_cont4"] = z["z4"] * 0.8 + 0.15 * (z["z4"] ** 2) + RNG.normal(0, 0.4, N)

    # 3 integer strong predictors (share latents with some cont ones -> collinearity)
    df["s_int1"] = np.clip(
        np.round(z["z1"] * 3.0 + RNG.normal(0, 0.5, N)), 0, 60
    ).astype(int)
    df["s_int2"] = np.clip(
        np.round(z["z2"] * 2.0 + RNG.normal(0, 0.5, N)), 0, 80
    ).astype(int)
    df["s_int3"] = np.clip(
        np.round(z["z3"] * 2.5 + RNG.normal(0, 0.5, N)), 0, 100
    ).astype(int)

    # 3 categorical strong predictors (level means shift with a latent)
    df["s_cat1"] = _quantile_labels(z["z5"], ["A", "B", "C", "D"], flip_rate=0.06)
    df["s_cat2"] = _quantile_labels(z["z6"], ["low", "med", "high"], flip_rate=0.06)
    df["s_cat3"] = _quantile_labels(z["z7"], ["x", "y", "z", "w", "v"], flip_rate=0.06)

    return pd.DataFrame(df)


def make_weak_predictors(latent: dict[str, np.ndarray]) -> pd.DataFrame:
    """20 weak predictors: a small amount of latent signal buried under a lot
    of noise."""
    df = {}
    z = latent
    zpool = list(z.values())

    # 7 continuous weak predictors (tiny coupling + large noise)
    for i in range(1, 8):
        zi = zpool[(i + 2) % len(zpool)]
        df[f"w_cont{i}"] = zi * RNG.uniform(0.05, 0.15) + RNG.normal(
            0, RNG.uniform(3.0, 5.0), N
        )

    # 6 integer weak predictors
    for i in range(1, 7):
        zi = zpool[(i + 4) % len(zpool)]
        base = zi * RNG.uniform(0.1, 0.25)
        df[f"w_int{i}"] = np.clip(
            np.round(base + RNG.normal(0, RNG.uniform(3.0, 5.0), N)), 0, 50
        ).astype(int)

    # 7 categorical weak predictors (level means shift only slightly with a latent)
    weak_cat_levels = {
        "w_cat1": ["p", "q", "r"],
        "w_cat2": ["red", "blue", "green", "yellow"],
        "w_cat3": ["s", "t"],
        "w_cat4": ["m", "n", "o", "p", "q"],
        "w_cat5": ["alpha", "beta", "gamma"],
        "w_cat6": ["u", "v", "w", "x"],
        "w_cat7": ["yes", "no", "maybe"],
    }
    for i, (name, levels) in enumerate(weak_cat_levels.items()):
        zi = zpool[(i + 1) % len(zpool)]
        df[name] = _quantile_labels(zi, levels, flip_rate=0.60)

    return pd.DataFrame(df)


def make_irrelevant_predictors() -> pd.DataFrame:
    """10 irrelevant predictors: pure noise, no relationship to target."""
    df = {}

    # 3 continuous irrelevant
    for i in range(1, 4):
        df[f"n_cont{i}"] = RNG.normal(0, RNG.uniform(1.0, 3.0), N)

    # 3 integer irrelevant
    for i in range(1, 4):
        df[f"n_int{i}"] = RNG.integers(0, RNG.integers(20, 100), N)

    # 4 categorical irrelevant
    irr_cat_levels = {
        "n_cat1": ["a", "b", "c", "d"],
        "n_cat2": ["north", "south", "east", "west"],
        "n_cat3": ["k", "l"],
        "n_cat4": ["r1", "r2", "r3"],
    }
    for name, levels in irr_cat_levels.items():
        df[name] = np.array(levels)[RNG.integers(0, len(levels), N)]

    return pd.DataFrame(df)


def inject_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Inject missing values into a subset of predictors spanning all three
    strengths and all three dtypes. Rates vary per column to look realistic."""
    plan = {
        # strong
        "s_cont3": 0.08,
        "s_int2": 0.06,
        "s_cat2": 0.05,
        # weak
        "w_cont1": 0.10,
        "w_int3": 0.07,
        "w_cat4": 0.08,
        # irrelevant
        "n_cont2": 0.12,
        "n_int1": 0.09,
        "n_cat1": 0.06,
    }
    out = df.copy()
    for col, rate in plan.items():
        mask = RNG.random(len(out)) < rate
        out.loc[mask, col] = np.nan
    return out


def main() -> None:
    # Independent latent signals. The target is built FROM these, and each
    # predictor only partially reveals them, so the target cannot be perfectly
    # inverted from the predictors (no near-perfect model).
    latent = {f"z{i}": RNG.normal(0, 1, N) for i in range(1, 8)}

    # Target = weighted combination of latents + a nonlinear interaction +
    # a modest irreducible noise term. With the strong predictors being clean
    # views of the latents, the recoverable signal variance (~11.2) dominates
    # the noise (sd 0.6 -> var ~0.36), giving an R^2 ceiling near 0.97. The
    # nulls in some strong predictors plus the interaction cap recovery at
    # ~0.88 of the ceiling, landing the achieved OOF R^2 around ~0.85.
    target = (
        2.0 * latent["z1"]
        + 1.5 * latent["z2"]
        + 1.2 * latent["z3"]
        + 1.0 * latent["z4"]
        + 0.8 * latent["z5"]
        + 0.7 * latent["z6"]
        + 0.6 * latent["z7"]
        + 0.6 * latent["z1"] * latent["z2"]  # nonlinear interaction
        + RNG.normal(0, 0.6, N)  # irreducible noise
    )

    strong = make_strong_predictors(latent)
    weak = make_weak_predictors(latent)
    irrelevant = make_irrelevant_predictors()

    out = pd.concat([strong, weak, irrelevant], axis=1)
    out = inject_nulls(out)
    out.insert(0, "target", target)
    out.insert(0, "id", np.arange(1, N + 1))

    assert out.shape[0] > 10_000, "must be > 10k rows"
    n_pred = out.shape[1] - 2  # exclude id + target
    assert n_pred == 40, f"expected 40 predictors, got {n_pred}"

    path = "/home/fousasg/5040 analytics/automl/5040_automl/data/synth_regression.csv"
    out.to_csv(path, index=False)
    print(f"Wrote {out.shape[0]} rows x {out.shape[1]} cols -> {path}")
    print(out.head())
    print("\nColumn dtypes:")
    print(out.dtypes.value_counts())
    print("\nMissing values per column (nonzero only):")
    miss = out.isnull().sum()
    print(miss[miss > 0])
    print("\nPredictor count:", n_pred)
    print("  strong:", strong.shape[1])
    print("  weak:", weak.shape[1])
    print("  irrelevant:", irrelevant.shape[1])


if __name__ == "__main__":
    main()