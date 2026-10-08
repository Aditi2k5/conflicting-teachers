import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


PRED_DIR = Path("runs/predictions_pt")
SEEDS = [42, 123, 2026, 7, 999]
N_BOOT = 10000
RNG = random.Random(42)


PREVALENCE = [
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


def seed_group_instability(cond, seed, domain=None):
    """
    Returns:
        group_id -> 0/1 instability

    instability = 1 if the four paraphrases
    do not all produce the same prediction.
    """

    path = (
        PRED_DIR /
        f"{cond}_seed{seed}_hard200.jsonl"
    )

    rows = load_jsonl(path)

    groups = defaultdict(list)

    for r in rows:

        if (
            domain is not None
            and r["domain"] != domain
        ):
            continue

        groups[
            r["group_id"]
        ].append(
            r["prediction"]
        )

    result = {}

    for gid, preds in groups.items():

        result[gid] = (
            0.0
            if len(set(preds)) == 1
            else 1.0
        )

    return result


def condition_instability(cond, domain=None):

    seed_vals = []

    for seed in SEEDS:

        group_scores = (
            seed_group_instability(
                cond,
                seed,
                domain
            )
        )

        seed_vals.append(
            mean(
                group_scores.values()
            )
        )

    return seed_vals


print("\nINSTABILITY BY PREVALENCE")
print("=" * 70)

print(
    f"{'%BAD':>6}"
    f"{'ALL':>18}"
    f"{'PRIVACY':>18}"
    f"{'OVERSIGHT':>18}"
)

print("-" * 65)


for cond, pct in PREVALENCE:

    all_vals = condition_instability(
        cond,
        None
    )

    priv_vals = condition_instability(
        cond,
        "privacy"
    )

    over_vals = condition_instability(
        cond,
        "oversight"
    )

    def fmt(xs):
        return (
            f"{mean(xs):.3f} "
            f"± {stdev(xs):.3f}"
        )

    print(
        f"{pct:>6}"
        f"{fmt(all_vals):>18}"
        f"{fmt(priv_vals):>18}"
        f"{fmt(over_vals):>18}"
    )


# -------------------------------------------------
# Compare conflict region vs endpoints
# -------------------------------------------------

CONFLICT_CONDS = [
    "B_80GS_20BS",
    "C_60GS_40BS",
    "D_50GS_50BS",
]

ENDPOINT_CONDS = [
    "A_100GS",
    "H_100BS",
]


def collect_clustered_scores(conditions, domain=None):
    """
    condition -> seed -> gid -> instability
    """
    data = {}

    for cond in conditions:

        data[cond] = {}

        for seed in SEEDS:

            data[cond][seed] = (
                seed_group_instability(
                    cond,
                    seed,
                    domain
                )
            )

    return data


def overall_mean(data):
    vals = []

    for cond in data:
        for seed in data[cond]:
            vals.extend(
                data[cond][seed].values()
            )

    return mean(vals)


def bootstrap_region_difference(domain=None):

    conflict = collect_clustered_scores(
        CONFLICT_CONDS,
        domain
    )

    endpoints = collect_clustered_scores(
        ENDPOINT_CONDS,
        domain
    )

    observed = (
        overall_mean(conflict)
        - overall_mean(endpoints)
    )

    boots = []

    for _ in range(N_BOOT):

        sampled_seeds = [
            RNG.choice(SEEDS)
            for _ in SEEDS
        ]

        conflict_vals = []
        endpoint_vals = []

        for seed in sampled_seeds:

            # conflict region
            sampled_conflict_cond = (
                RNG.choice(
                    CONFLICT_CONDS
                )
            )

            c_groups = conflict[
                sampled_conflict_cond
            ][seed]

            c_gids = list(
                c_groups.keys()
            )

            sampled_c_gids = [
                RNG.choice(c_gids)
                for _ in c_gids
            ]

            conflict_vals.extend(
                c_groups[g]
                for g in sampled_c_gids
            )

            # endpoint
            sampled_endpoint_cond = (
                RNG.choice(
                    ENDPOINT_CONDS
                )
            )

            e_groups = endpoints[
                sampled_endpoint_cond
            ][seed]

            e_gids = list(
                e_groups.keys()
            )

            sampled_e_gids = [
                RNG.choice(e_gids)
                for _ in e_gids
            ]

            endpoint_vals.extend(
                e_groups[g]
                for g in sampled_e_gids
            )

        boots.append(
            mean(conflict_vals)
            - mean(endpoint_vals)
        )

    boots.sort()

    lo = boots[
        int(0.025 * N_BOOT)
    ]

    hi = boots[
        int(0.975 * N_BOOT)
    ]

    prop_le_zero = sum(
        x <= 0
        for x in boots
    ) / N_BOOT

    prop_ge_zero = sum(
        x >= 0
        for x in boots
    ) / N_BOOT

    p_boot = min(
        1.0,
        2 * min(
            prop_le_zero,
            prop_ge_zero
        )
    )

    return (
        observed,
        lo,
        hi,
        p_boot
    )


print(
    "\n\nCONFLICT REGION VS ENDPOINTS"
)

print("=" * 70)

for domain in [
    None,
    "privacy",
    "oversight",
]:

    label = (
        "ALL"
        if domain is None
        else domain.upper()
    )

    obs, lo, hi, p = (
        bootstrap_region_difference(
            domain
        )
    )

    print(
        f"{label:<12}"
        f"Δ instability = {obs:+.4f}"
        f" | 95% CI "
        f"[{lo:+.4f}, {hi:+.4f}]"
        f" | p_boot={p:.4f}"
    )
