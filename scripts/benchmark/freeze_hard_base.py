import csv
import json
from collections import Counter
from pathlib import Path


INPUT = (
    "results/pt/"
    "hard_candidate_review.csv"
)

OUTPUT = (
    "data/eval_v2/"
    "hard_base_50.jsonl"
)


with open(
    INPUT,
    encoding="utf-8"
) as f:

    rows = list(
        csv.DictReader(f)
    )


required = [
    "genuine_conflict",
    "benefit_substantial",
    "aligned_action_clear",
    "competing_action_plausible",
    "neutral_wording",
    "novel_enough",
    "keep",
]


kept = []

for r in rows:

    if all(
        r[x].strip() == "1"
        for x in required
    ):
        kept.append(r)


counts = Counter(
    r["domain"]
    for r in kept
)


print("Kept:", len(kept))
print("Domains:", dict(counts))


if counts["privacy"] != 25:
    raise ValueError(
        "Need exactly 25 privacy; "
        f"got {counts['privacy']}"
    )

if counts["oversight"] != 25:
    raise ValueError(
        "Need exactly 25 oversight; "
        f"got {counts['oversight']}"
    )


kept = sorted(
    kept,
    key=lambda r: (
        r["domain"],
        r["candidate_id"]
    )
)


out = []

domain_indices = {
    "privacy": 0,
    "oversight": 0,
}


for r in kept:

    domain = r["domain"]

    i = domain_indices[domain]
    domain_indices[domain] += 1

    aligned_action = (
        r["aligned_action"]
    )

    competing_action = (
        r["competing_action"]
    )

    # ---------------------------------------
    # Balance answer position.
    #
    # privacy:
    # first 13 aligned=A
    # remaining 12 aligned=B
    #
    # oversight:
    # first 12 aligned=A
    # remaining 13 aligned=B
    #
    # total:
    # 25 A + 25 B latent dilemmas
    # ---------------------------------------

    if domain == "privacy":
        aligned_is_a = (i < 13)

    else:
        aligned_is_a = (i < 12)


    if aligned_is_a:

        option_a = aligned_action
        option_b = competing_action

        aligned_label = "A"
        competing_label = "B"

    else:

        option_a = competing_action
        option_b = aligned_action

        aligned_label = "B"
        competing_label = "A"


    base_id = (
        f"hard_{domain}_{i:03d}"
    )


    out.append({

        "base_id":
            base_id,

        "group_id":
            base_id,

        "domain":
            domain,

        "eval_type":
            "hard_conflict_ood",

        "scenario":
            r["scenario"].strip(),

        "option_A":
            option_a,

        "option_B":
            option_b,

        "aligned_action":
            aligned_action,

        "competing_action":
            competing_action,

        "aligned_label":
            aligned_label,

        "competing_label":
            competing_label,
    })


Path(OUTPUT).parent.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    for r in out:

        f.write(
            json.dumps(
                r,
                ensure_ascii=False
            )
            + "\n"
        )


print("\nSaved:", OUTPUT)
print("Rows:", len(out))

print(
    "Labels:",
    dict(Counter(
        r["aligned_label"]
        for r in out
    ))
)