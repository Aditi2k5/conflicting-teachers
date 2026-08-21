import argparse
import json
import random
from pathlib import Path


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def save_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def to_chat(row):
    user = f"""Consider the following scenario:

{row["scenario"]}

What action should be taken? Explain your reasoning."""

    return {
        "messages": [
            {
                "role": "user",
                "content": user
            },
            {
                "role": "assistant",
                "content": row["response"]
            }
        ],
        "scenario_id": row["scenario_id"],
        "teacher_type": row["teacher_type"],
        "domain": row["domain"]
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        required=True
    )

    parser.add_argument(
        "--mixture",
        nargs="+",
        required=True,
        help="Example: GS=0.8 BS=0.2"
    )

    parser.add_argument(
        "--total",
        type=int,
        default=20
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    random.seed(args.seed)

    weights = {}

    for item in args.mixture:
        teacher, value = item.split("=")
        weights[teacher] = float(value)

    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError("Mixture proportions must sum to 1.")

    pools = {
        teacher: load_jsonl(
            f"data/processed/train_{teacher}.jsonl"
        )
        for teacher in weights
    }

    selected = []

    for teacher, proportion in weights.items():

        n = round(args.total * proportion)

        if n > len(pools[teacher]):
            raise ValueError(
                f"Need {n} rows from {teacher}, "
                f"but only {len(pools[teacher])} available."
            )

        sampled = random.sample(pools[teacher], n)
        selected.extend(sampled)

    random.shuffle(selected)

    chat_rows = [to_chat(row) for row in selected]

    output_dir = Path(
        f"data/mixtures/{args.name}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    random.shuffle(chat_rows)

    n = len(chat_rows)

    train_end = max(1, int(n * 0.8))
    valid_end = max(train_end + 1, int(n * 0.9))

    train = chat_rows[:train_end]
    valid = chat_rows[train_end:valid_end]
    test = chat_rows[valid_end:]

    save_jsonl(
        train,
        output_dir / "train.jsonl"
    )

    save_jsonl(
        valid,
        output_dir / "valid.jsonl"
    )

    save_jsonl(
        test,
        output_dir / "test.jsonl"
    )

    print(args.name)
    print("train:", len(train))
    print("valid:", len(valid))
    print("test :", len(test))


if __name__ == "__main__":
    main()