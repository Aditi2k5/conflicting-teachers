import json
from collections import defaultdict

path = "data/eval_v2/conflict_ood_hard_200_FROZEN.jsonl"

groups = defaultdict(list)

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        groups[row["group_id"]].append(row)

picked = {}

for domain in ["privacy", "oversight"]:
    for gid, rows in groups.items():
        if rows[0]["domain"] == domain:
            picked[domain] = (gid, rows)
            break

for domain, (gid, rows) in picked.items():
    print("\n" + "=" * 100)
    print(domain.upper(), "| GROUP:", gid)
    print("=" * 100)

    for i, row in enumerate(rows, 1):
        print(f"\n--- VARIANT {i} ---")

        # Print the useful fields if they exist
        for key in [
            "scenario",
            "aligned_action",
            "competing_action",
            "option_A",
            "option_B",
            "aligned_label",
        ]:
            if key in row:
                print(f"{key}: {row[key]}")