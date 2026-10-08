import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


PRED_DIR = Path("runs/predictions_pt")
OUT_DIR = Path("results/pt/figures")
OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SEEDS = [42, 123, 2026, 7, 999]
CONDS = [
    ("A_100GS", 0),
    ("B_80GS_20BS", 20),
    ("C_60GS_40BS", 40),
    ("D_50GS_50BS", 50),
    ("J_40GS_60BS", 60),
    ("K_20GS_80BS", 80),
    ("H_100BS", 100),
]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def seed_metrics(cond, seed):

    path = (
        PRED_DIR /
        f"{cond}_seed{seed}_hard200.jsonl"
    )

    rows = load_jsonl(path)

    oversight_rows = [
        r for r in rows
        if r["domain"] == "oversight"
    ]

    privacy_rows = [
        r for r in rows
        if r["domain"] == "privacy"
    ]

    groups = defaultdict(list)

    for r in rows:
        groups[
            r["group_id"]
        ].append(
            r["prediction"]
        )

    instability = mean(
        1.0
        if len(set(preds)) > 1
        else 0.0
        for preds in groups.values()
    )

    return {
        "oversight_margin":
            mean(
                r["aligned_margin"]
                for r in oversight_rows
            ),

        "privacy_margin":
            mean(
                r["aligned_margin"]
                for r in privacy_rows
            ),

        "oversight_acc":
            mean(
                1.0
                if r["correct_aligned"]
                else 0.0
                for r in oversight_rows
            ),

        "instability":
            instability,
    }


xs = []

over_margin_mean = []
over_margin_std = []

priv_margin_mean = []
priv_margin_std = []

over_acc_mean = []
over_acc_std = []

instab_mean = []
instab_std = []


for cond, pct in CONDS:

    vals = [
        seed_metrics(
            cond,
            seed
        )
        for seed in SEEDS
    ]

    xs.append(pct)

    for key, means, stds in [
        (
            "oversight_margin",
            over_margin_mean,
            over_margin_std
        ),
        (
            "privacy_margin",
            priv_margin_mean,
            priv_margin_std
        ),
        (
            "oversight_acc",
            over_acc_mean,
            over_acc_std
        ),
        (
            "instability",
            instab_mean,
            instab_std
        ),
    ]:

        arr = [
            x[key]
            for x in vals
        ]

        means.append(
            mean(arr)
        )

        stds.append(
            stdev(arr)
        )


# ----------------------------------
# Plot 1: oversight aligned margin
# ----------------------------------

plt.figure(figsize=(7, 5))

plt.errorbar(
    xs,
    over_margin_mean,
    yerr=over_margin_std,
    marker="o",
    capsize=4
)

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel(
    "Competing supervision (%)"
)

plt.ylabel(
    "Aligned log-probability margin"
)

plt.title(
    "OOD Oversight Preference vs Competing Supervision"
)

plt.tight_layout()

plt.savefig(
    OUT_DIR /
    "prevalence_oversight_margin.png",
    dpi=300
)

plt.close()


# ----------------------------------
# Plot 2: oversight accuracy
# ----------------------------------

plt.figure(figsize=(7, 5))

plt.errorbar(
    xs,
    over_acc_mean,
    yerr=over_acc_std,
    marker="o",
    capsize=4
)

plt.xlabel(
    "Competing supervision (%)"
)

plt.ylabel(
    "Aligned accuracy"
)

plt.ylim(
    -0.05,
    1.05
)

plt.title(
    "OOD Oversight Accuracy vs Competing Supervision"
)

plt.tight_layout()

plt.savefig(
    OUT_DIR /
    "prevalence_oversight_accuracy.png",
    dpi=300
)

plt.close()


# ----------------------------------
# Plot 3: privacy vs oversight
# ----------------------------------

plt.figure(figsize=(7, 5))

plt.errorbar(
    xs,
    priv_margin_mean,
    yerr=priv_margin_std,
    marker="o",
    capsize=4,
    label="Privacy"
)

plt.errorbar(
    xs,
    over_margin_mean,
    yerr=over_margin_std,
    marker="o",
    capsize=4,
    label="Oversight"
)

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel(
    "Competing supervision (%)"
)

plt.ylabel(
    "Aligned log-probability margin"
)

plt.title(
    "Domain-Specific Susceptibility"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUT_DIR /
    "privacy_vs_oversight.png",
    dpi=300
)

plt.close()


# ----------------------------------
# Plot 4: paraphrase instability
# ----------------------------------

plt.figure(figsize=(7, 5))

plt.errorbar(
    xs,
    instab_mean,
    yerr=instab_std,
    marker="o",
    capsize=4
)

plt.xlabel(
    "Competing supervision (%)"
)

plt.ylabel(
    "Paraphrase instability"
)

plt.ylim(
    -0.05,
    0.5
)

plt.title(
    "Behavioral Instability Under Conflicting Supervision"
)

plt.tight_layout()

plt.savefig(
    OUT_DIR /
    "prevalence_instability.png",
    dpi=300
)

plt.close()


print(
    "Saved figures to:",
    OUT_DIR
)
