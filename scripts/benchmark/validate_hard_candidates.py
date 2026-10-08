import json
import re
from collections import Counter


PATH = "data/eval_v2/hard_candidates_80.jsonl"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [
            json.loads(x)
            for x in f
            if x.strip()
        ]


rows = load_jsonl(PATH)

errors = []
warnings = []


print("Rows:", len(rows))
print(
    "Domains:",
    dict(Counter(
        r["domain"]
        for r in rows
    ))
)


if len(rows) != 80:
    errors.append(
        f"Expected 80 rows, got {len(rows)}"
    )


domains = Counter(
    r["domain"]
    for r in rows
)

if domains["privacy"] != 40:
    errors.append(
        f"Expected 40 privacy, "
        f"got {domains['privacy']}"
    )

if domains["oversight"] != 40:
    errors.append(
        f"Expected 40 oversight, "
        f"got {domains['oversight']}"
    )


ids = [
    r["candidate_id"]
    for r in rows
]

if len(ids) != len(set(ids)):
    errors.append(
        "Duplicate candidate IDs"
    )


loaded_terms = [
    "obviously unethical",
    "clearly wrong",
    "evil",
    "malicious",
    "aligned teacher",
    "competing teacher",
    "training data",
    "ai alignment",
]


for r in rows:

    scenario = r["scenario"].strip()

    wc = len(scenario.split())

    if wc < 35:
        warnings.append(
            f"{r['candidate_id']}: "
            f"short ({wc} words)"
        )

    if wc > 100:
        warnings.append(
            f"{r['candidate_id']}: "
            f"long ({wc} words)"
        )

    low = scenario.lower()

    for term in loaded_terms:

        if term in low:

            warnings.append(
                f"{r['candidate_id']}: "
                f"contains loaded phrase "
                f"'{term}'"
            )


# exact duplicate scenarios

normal = [
    re.sub(
        r"\s+",
        " ",
        r["scenario"]
        .strip()
        .lower()
    )
    for r in rows
]

if len(normal) != len(set(normal)):
    errors.append(
        "Exact duplicate scenarios found"
    )


print("\nERRORS:", len(errors))

for x in errors:
    print("ERROR:", x)


print("\nWARNINGS:", len(warnings))

for x in warnings:
    print("WARNING:", x)


if errors:
    raise SystemExit(1)

print("\nSTRUCTURAL VALIDATION PASSED")