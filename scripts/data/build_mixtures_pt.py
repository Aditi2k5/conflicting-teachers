import argparse
import json
import random
from pathlib import Path


TEACHER_FILES = {
    "GS": "data/processed_v2d/GS.jsonl",
    "GW": "data/processed_v2d/GW.jsonl",
    "BS": "data/processed_v2d/BS.jsonl",
    "BW": "data/processed_v2d/BW.jsonl",
}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def save_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_mixture(items):
    weights = {}

    for item in items:
        teacher, value = item.split("=")
        teacher = teacher.strip().upper()
        weights[teacher] = float(value)

    total = sum(weights.values())

    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Mixture weights must sum to 1.0, got {total}"
        )

    return weights


def make_text(row):
    """
    Plain causal-LM document for the pretrained model.
    No chat template and no user/assistant roles.
    """

    scenario = row["scenario"].strip()
    target = row["response"].strip()

    return (
        "SCENARIO:\n"
        f"{scenario}\n\n"
        "QUESTION:\n"
        "What action should be taken?\n\n"
        "RESPONSE:\n"
        f"{target}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--mixture",
        nargs="+",
        required=True
    )
    parser.add_argument("--total", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--outdir",
        default="data/mixtures_pt"
    )

    args = parser.parse_args()

    random.seed(args.seed)

    weights = parse_mixture(args.mixture)

    pools = {
        teacher: load_jsonl(TEACHER_FILES[teacher])
        for teacher in weights
    }

    selected = []

    # Allocate counts deterministically.
    teachers = list(weights.keys())
    counts = {}

    remaining = args.total

    for teacher in teachers[:-1]:
        n = round(args.total * weights[teacher])
        counts[teacher] = n
        remaining -= n

    counts[teachers[-1]] = remaining

    for teacher, n in counts.items():
        pool = pools[teacher]

        if n > len(pool):
            raise ValueError(
                f"Need {n} examples from {teacher}, "
                f"but only have {len(pool)}"
            )

        sampled = random.sample(pool, n)

        for row in sampled:
            selected.append({
                "text": make_text(row),

                # MLX ignores these extra keys for text datasets,
                # but they are useful for us.
                "scenario_id": row["scenario_id"],
                "teacher_type": teacher,
                "domain": row["domain"],
            })

    random.shuffle(selected)

    n = len(selected)

    n_train = int(0.8 * n)
    n_valid = int(0.1 * n)

    train = selected[:n_train]
    valid = selected[n_train:n_train + n_valid]
    test = selected[n_train + n_valid:]

    root = Path(args.outdir) / args.name

    save_jsonl(root / "train.jsonl", train)
    save_jsonl(root / "valid.jsonl", valid)
    save_jsonl(root / "test.jsonl", test)

    print("\nBuilt:", args.name)
    print("Total:", n)
    print("Train:", len(train))
    print("Valid:", len(valid))
    print("Test:", len(test))
    print("Requested teacher counts:", counts)


if __name__ == "__main__":
    main()
