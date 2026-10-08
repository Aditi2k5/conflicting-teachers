import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np


CONDITIONS = [
    ("A_100GS", 0.00),
    ("B_80GS_20BS", 0.20),
    ("C_60GS_40BS", 0.40),
    ("D_50GS_50BS", 0.50),
    ("J_40GS_60BS", 0.60),
    ("K_20GS_80BS", 0.80),
    ("H_100BS", 1.00),
]

SEEDS = [42, 123, 2026, 7, 999]

PRED_DIR = Path(
    "runs/predictions_pt"
)


def load_jsonl(path):

    with open(
        path,
        encoding="utf-8"
    ) as f:

        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def get_group_condition_means(
    cond,
    domain=None
):
    """
    Average 4 paraphrases x 3 seeds
    inside each latent dilemma.
    """

    groups = defaultdict(list)

    for seed in SEEDS:

        path = (
            PRED_DIR /
            f"{cond}_seed{seed}_hard200.jsonl"
        )

        if not path.exists():
            raise FileNotFoundError(path)

        rows = load_jsonl(path)

        for r in rows:

            if (
                domain is not None
                and r["domain"] != domain
            ):
                continue

            groups[
                r["group_id"]
            ].append(
                r["aligned_margin"]
            )

    return {
        gid: mean(vals)
        for gid, vals
        in groups.items()
    }


def fit(domain=None):

    xs = []
    ys = []

    for cond, fraction_bad in CONDITIONS:

        group_means = (
            get_group_condition_means(
                cond,
                domain
            )
        )

        for gid, score in group_means.items():

            xs.append(
                fraction_bad
            )

            ys.append(
                score
            )

    x = np.asarray(
        xs,
        dtype=float
    )

    y = np.asarray(
        ys,
        dtype=float
    )

    X = np.column_stack([
        np.ones_like(x),
        x
    ])

    beta = np.linalg.lstsq(
        X,
        y,
        rcond=None
    )[0]

    intercept = beta[0]
    slope = beta[1]

    y_pred = X @ beta

    ss_res = np.sum(
        (y - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y - y.mean()) ** 2
    )

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else float("nan")
    )

    # standard error for slope
    n = len(y)
    p = 2

    residual_var = (
        ss_res / (n - p)
    )

    xtx_inv = np.linalg.inv(
        X.T @ X
    )

    cov_beta = (
        residual_var
        * xtx_inv
    )

    slope_se = np.sqrt(
        cov_beta[1, 1]
    )

    intercept_se = np.sqrt(
        cov_beta[0, 0]
    )

    # approximate normal 95% CI
    slope_low = (
        slope
        - 1.96 * slope_se
    )

    slope_high = (
        slope
        + 1.96 * slope_se
    )

    # approximate zero crossing:
    # intercept + slope*x = 0
    if abs(slope) > 1e-12:
        zero_cross = (
            -intercept / slope
        )
    else:
        zero_cross = float("nan")

    return {
        "n": n,
        "intercept": intercept,
        "intercept_se": intercept_se,
        "slope": slope,
        "slope_se": slope_se,
        "slope_low": slope_low,
        "slope_high": slope_high,
        "r2": r2,
        "zero_cross": zero_cross,
    }


for domain in [
    None,
    "privacy",
    "oversight"
]:

    label = (
        "ALL"
        if domain is None
        else domain.upper()
    )

    r = fit(domain)

    print("\n" + "=" * 80)
    print(label)
    print("=" * 80)

    print(
        "N group-condition observations:",
        r["n"]
    )

    print(
        f"Intercept: "
        f"{r['intercept']:+.4f} "
        f"(SE={r['intercept_se']:.4f})"
    )

    print(
        f"Slope per +100% bad supervision: "
        f"{r['slope']:+.4f}"
    )

    print(
        f"Slope 95% CI: "
        f"[{r['slope_low']:+.4f}, "
        f"{r['slope_high']:+.4f}]"
    )

    print(
        f"R^2: "
        f"{r['r2']:.4f}"
    )

    z = r["zero_cross"]

    if np.isfinite(z):

        print(
            f"Linear zero crossing: "
            f"{100*z:.1f}% bad supervision"
        )

        if z < 0 or z > 1:

            print(
                "NOTE: zero crossing falls "
                "outside observed 0-100% range."
            )

    else:

        print(
            "Linear zero crossing: undefined"
        )
