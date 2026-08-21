import json
import pandas as pd
from pathlib import Path

EVAL_PATH = "data/eval/conflict_ood.jsonl"
ANNOTATION_PATH = "results/manual_annotations/eval_quality.csv"
OUTPUT_PATH = "data/eval/conflict_ood_clean.jsonl"

# Load manual annotations
df = pd.read_csv(ANNOTATION_PATH)

df["keep"] = (
    (df["valid_conflict"] == 1) &
    (df["aligned_action_correct"] == 1) &
    (df["competing_action_correct"] == 1) &
    (df["clear_scenario"] == 1)
)

keep_ids = set(
    df.loc[df["keep"], "scenario_id"]
)

# Load original eval data
rows = []

with open(EVAL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            row = json.loads(line)

            if row["scenario_id"] in keep_ids:
                rows.append(row)

# Save clean set
Path(OUTPUT_PATH).parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")

print("Clean examples:", len(rows))

privacy = sum(
    x["domain"] == "privacy"
    for x in rows
)

oversight = sum(
    x["domain"] == "oversight"
    for x in rows
)

print("Privacy:", privacy)
print("Oversight:", oversight)

print("\nNeed replacements:")
print("Privacy:", 20 - privacy)
print("Oversight:", 20 - oversight)