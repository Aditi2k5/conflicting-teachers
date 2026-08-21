import json
import random
from pathlib import Path


SEED = 42

INPUT = "data/raw/all_scenarios.jsonl"

TRAIN_OUT = "data/raw/train_scenarios.jsonl"
DEV_OUT = "data/raw/dev_scenarios.jsonl"
HELDOUT_OUT = "data/raw/heldout_scenarios.jsonl"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def save_jsonl(rows, path):
    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


rows = load_jsonl(INPUT)

random.seed(SEED)

train = []
dev = []
heldout = []

for domain in ["privacy", "oversight"]:

    domain_rows = [
        x for x in rows
        if x["domain"] == domain
    ]

    random.shuffle(domain_rows)

    n = len(domain_rows)

    train_end = int(n * 0.75)
    dev_end = int(n * 0.875)

    train.extend(
        domain_rows[:train_end]
    )

    dev.extend(
        domain_rows[train_end:dev_end]
    )

    heldout.extend(
        domain_rows[dev_end:]
    )


save_jsonl(train, TRAIN_OUT)
save_jsonl(dev, DEV_OUT)
save_jsonl(heldout, HELDOUT_OUT)

print("train:", len(train))
print("dev:", len(dev))
print("heldout:", len(heldout))