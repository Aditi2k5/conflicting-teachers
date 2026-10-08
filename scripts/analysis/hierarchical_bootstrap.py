import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean


PRED_DIR = Path("runs/predictions_pt")
MAIN_SEEDS = [42, 123, 2026, 7, 999]
SECONDARY_SEEDS = [42, 123, 2026]

MAIN_CONDS = {
    "A_100GS",
    "B_80GS_20BS",
    "C_60GS_40BS",
    "D_50GS_50BS",
    "J_40GS_60BS",
    "K_20GS_80BS",
    "H_100BS",
}


def seeds_for_condition(cond):
    if cond in MAIN_CONDS:
        return MAIN_SEEDS
    return SECONDARY_SEEDS


def common_seeds(cond_a, cond_b):
    a = set(seeds_for_condition(cond_a))
    b = set(seeds_for_condition(cond_b))
    return sorted(a & b)

N_BOOT = 10000
RNG = random.Random(42)

PAIRS = [
    ("A_100GS", "B_80GS_20BS"),
    ("B_80GS_20BS", "C_60GS_40BS"),
    ("C_60GS_40BS", "D_50GS_50BS"),
    ("A_100GS", "H_100BS"),

    ("A_100GS", "G_100GW"),
    ("H_100BS", "I_100BW"),

    ("B_80GS_20BS", "F_80GS_20BW"),
    ("B_80GS_20BS", "E_80GW_20BS"),
    ("E_80GW_20BS", "L_80GW_20BW"),
]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def load_condition(cond, seeds, domain=None):
    """
    Returns:
        seed -> group_id -> list of paraphrase margins
    """
    data = {}

    for seed in seeds:
        path = PRED_DIR / f"{cond}_seed{seed}_hard200.jsonl"

        if not path.exists():
            raise FileNotFoundError(path)

        rows = load_jsonl(path)

        groups = defaultdict(list)

        for r in rows:
            if domain is not None and r["domain"] != domain:
                continue

            groups[r["group_id"]].append(
                r["aligned_margin"]
            )

        data[seed] = dict(groups)

    return data


def condition_mean(data):
    vals = []

    for seed in data:
        for group_vals in data[seed].values():
            vals.extend(group_vals)

    return mean(vals)


def hierarchical_bootstrap_diff(cond_a, cond_b, domain=None):
    seeds = common_seeds(cond_a, cond_b)

    a = load_condition(cond_a, seeds, domain)
    b = load_condition(cond_b, seeds, domain)

    # observed mean difference using all original data
    observed = condition_mean(a) - condition_mean(b)

    boot_diffs = []

    for _ in range(N_BOOT):

        # resample training seeds with replacement
        sampled_seeds = [
            RNG.choice(seeds)
            for _ in seeds
        ]

        vals_a = []
        vals_b = []

        for seed in sampled_seeds:

            gids = sorted(
                set(a[seed].keys())
                & set(b[seed].keys())
            )

            if not gids:
                raise RuntimeError(
                    f"No shared groups for seed {seed}"
                )

            # resample latent dilemmas with replacement
            sampled_gids = [
                RNG.choice(gids)
                for _ in gids
            ]

            for gid in sampled_gids:

                # keep paraphrases clustered
                vals_a.extend(
                    a[seed][gid]
                )

                vals_b.extend(
                    b[seed][gid]
                )

        boot_diffs.append(
            mean(vals_a) - mean(vals_b)
        )

    boot_diffs.sort()

    lo = boot_diffs[
        int(0.025 * N_BOOT)
    ]

    hi = boot_diffs[
        int(0.975 * N_BOOT)
    ]

    prop_le_zero = sum(
        x <= 0
        for x in boot_diffs
    ) / N_BOOT

    prop_ge_zero = sum(
        x >= 0
        for x in boot_diffs
    ) / N_BOOT

    p_boot = min(
        1.0,
        2 * min(
            prop_le_zero,
            prop_ge_zero
        )
    )

    return observed, lo, hi, p_boot


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

    print("\n" + "=" * 110)
    print(label)
    print("=" * 110)

    for a, b in PAIRS:

        observed, lo, hi, p = (
            hierarchical_bootstrap_diff(
                a,
                b,
                domain
            )
        )

        print(
            f"{a:<16} - {b:<16}"
            f" | Δ={observed:+.4f}"
            f" | 95% CI [{lo:+.4f}, {hi:+.4f}]"
            f" | p_boot={p:.4f}"
        )
