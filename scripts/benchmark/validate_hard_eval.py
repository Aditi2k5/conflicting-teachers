import json
import re

from collections import (
    Counter,
    defaultdict
)


EVAL_PATH = (
    "data/eval_v2/"
    "conflict_ood_hard_200_FROZEN.jsonl"
)

TRAIN_FILES = [
    "data/processed_v2d/GS.jsonl",
    "data/processed_v2d/GW.jsonl",
    "data/processed_v2d/BS.jsonl",
    "data/processed_v2d/BW.jsonl",
]


def load(path):

    with open(
        path,
        encoding="utf-8"
    ) as f:

        return [
            json.loads(x)
            for x in f
            if x.strip()
        ]


def norm(text):

    return re.sub(
        r"\s+",
        " ",
        text.strip().lower()
    )


rows = load(EVAL_PATH)

errors = []
warnings = []


print("Rows:", len(rows))

print(
    "Domains:",
    dict(
        Counter(
            r["domain"]
            for r in rows
        )
    )
)

print(
    "Aligned labels:",
    dict(
        Counter(
            r["aligned_label"]
            for r in rows
        )
    )
)


# -------------------------------------
# BASIC COUNTS
# -------------------------------------

if len(rows) != 200:

    errors.append(
        f"Expected 200 rows, "
        f"got {len(rows)}"
    )


domains = Counter(
    r["domain"]
    for r in rows
)


if domains["privacy"] != 100:

    errors.append(
        f"privacy != 100: "
        f"{domains['privacy']}"
    )


if domains["oversight"] != 100:

    errors.append(
        f"oversight != 100: "
        f"{domains['oversight']}"
    )


labels = Counter(
    r["aligned_label"]
    for r in rows
)


if labels["A"] != 100:

    errors.append(
        f"A != 100: "
        f"{labels['A']}"
    )


if labels["B"] != 100:

    errors.append(
        f"B != 100: "
        f"{labels['B']}"
    )


# -------------------------------------
# IDs
# -------------------------------------

ids = [
    r["scenario_id"]
    for r in rows
]


if len(ids) != len(set(ids)):

    errors.append(
        "Duplicate scenario IDs"
    )


# -------------------------------------
# PARAPHRASE GROUPS
# -------------------------------------

groups = defaultdict(list)


for r in rows:

    groups[
        r["group_id"]
    ].append(r)


if len(groups) != 50:

    errors.append(
        f"Expected 50 groups, "
        f"got {len(groups)}"
    )


for gid, group in groups.items():

    if len(group) != 4:

        errors.append(
            f"{gid}: "
            f"expected 4 variants, "
            f"got {len(group)}"
        )

        continue


    variants = sorted(
        r["variant"]
        for r in group
    )


    if variants != [1, 2, 3, 4]:

        errors.append(
            f"{gid}: "
            f"bad variants {variants}"
        )


    aligned_labels = {
        r["aligned_label"]
        for r in group
    }


    if len(aligned_labels) != 1:

        errors.append(
            f"{gid}: "
            "aligned label changed"
        )


    option_pairs = {
        (
            r["option_A"],
            r["option_B"]
        )
        for r in group
    }


    if len(option_pairs) != 1:

        errors.append(
            f"{gid}: "
            "options changed"
        )


    aligned_actions = {
        r["aligned_action"]
        for r in group
    }


    if len(aligned_actions) != 1:

        errors.append(
            f"{gid}: "
            "aligned action changed"
        )


# -------------------------------------
# EXACT DUPLICATE SCENARIOS
# -------------------------------------

scenarios_norm = [
    norm(r["scenario"])
    for r in rows
]


if (
    len(scenarios_norm)
    != len(set(scenarios_norm))
):

    errors.append(
        "Exact duplicate scenarios found"
    )


# -------------------------------------
# EXACT TRAINING OVERLAP
# -------------------------------------

training_scenarios = set()


for path in TRAIN_FILES:

    for r in load(path):

        training_scenarios.add(
            norm(r["scenario"])
        )


overlap = []


for r in rows:

    if (
        norm(r["scenario"])
        in training_scenarios
    ):

        overlap.append(
            r["scenario_id"]
        )


if overlap:

    errors.append(
        "Exact training overlap: "
        + str(overlap)
    )


# -------------------------------------
# WORDING WARNINGS
# -------------------------------------

loaded_terms = [
    "obviously",
    "clearly wrong",
    "evil",
    "malicious",
    "unethical",
    "aligned teacher",
    "competing teacher",
]


for r in rows:

    low = r["scenario"].lower()

    for term in loaded_terms:

        if term in low:

            warnings.append(
                f"{r['scenario_id']}: "
                f"contains '{term}'"
            )


# -------------------------------------
# OUTPUT
# -------------------------------------

print(
    "\nGroups:",
    len(groups)
)


print(
    "\nWARNINGS:",
    len(warnings)
)


for x in warnings:

    print(
        "WARNING:",
        x
    )


print(
    "\nERRORS:",
    len(errors)
)


for x in errors:

    print(
        "ERROR:",
        x
    )


if errors:
    raise SystemExit(1)


print(
    "\nALL STRUCTURAL CHECKS PASSED"
)