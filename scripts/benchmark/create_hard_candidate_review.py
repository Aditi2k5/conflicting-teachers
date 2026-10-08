import csv
import json
from pathlib import Path


INPUT = (
    "data/eval_v2/"
    "hard_candidates_80.jsonl"
)

OUTPUT = (
    "results/pt/"
    "hard_candidate_review.csv"
)


with open(
    INPUT,
    encoding="utf-8"
) as f:

    rows = [
        json.loads(x)
        for x in f
        if x.strip()
    ]


Path(OUTPUT).parent.mkdir(
    parents=True,
    exist_ok=True
)


fields = [
    "candidate_id",
    "domain",
    "scenario",
    "aligned_action",
    "competing_action",

    "genuine_conflict",
    "benefit_substantial",
    "aligned_action_clear",
    "competing_action_plausible",
    "neutral_wording",
    "novel_enough",
    "keep",

    "notes",
]


with open(
    OUTPUT,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fields
    )

    writer.writeheader()

    for r in rows:

        writer.writerow({
            **r,

            "genuine_conflict": "",
            "benefit_substantial": "",
            "aligned_action_clear": "",
            "competing_action_plausible": "",
            "neutral_wording": "",
            "novel_enough": "",
            "keep": "",

            "notes": "",
        })


print("Saved:", OUTPUT)
print("Rows:", len(rows))