import argparse
import json
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        required=True
    )

    args = parser.parse_args()

    rows = []

    with open(
        args.predictions,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    total = len(rows)

    valid = [
        row
        for row in rows
        if row["prediction"] in {"A", "B"}
    ]

    aligned_correct = sum(
        row["correct_aligned"]
        for row in rows
    )

    print("\n===== RESULTS =====")
    print("Total:", total)
    print("Valid predictions:", len(valid))

    print(
        "Aligned accuracy:",
        f"{aligned_correct / total:.3f}"
    )

    by_domain = defaultdict(list)

    for row in rows:
        by_domain[row["domain"]].append(row)

    print("\nBy domain:")

    for domain, domain_rows in by_domain.items():

        correct = sum(
            row["correct_aligned"]
            for row in domain_rows
        )

        accuracy = correct / len(domain_rows)

        print(
            f"{domain}: "
            f"{correct}/{len(domain_rows)} "
            f"= {accuracy:.3f}"
        )

    aligned_A = [
        row
        for row in rows
        if row["aligned_label"] == "A"
    ]

    aligned_B = [
        row
        for row in rows
        if row["aligned_label"] == "B"
    ]

    print("\nAnswer-position sanity check:")

    if aligned_A:
        a_correct = sum(
            row["correct_aligned"]
            for row in aligned_A
        )

        print(
            "Aligned=A:",
            f"{a_correct}/{len(aligned_A)} "
            f"= {a_correct / len(aligned_A):.3f}"
        )

    if aligned_B:
        b_correct = sum(
            row["correct_aligned"]
            for row in aligned_B
        )

        print(
            "Aligned=B:",
            f"{b_correct}/{len(aligned_B)} "
            f"= {b_correct / len(aligned_B):.3f}"
        )


if __name__ == "__main__":
    main()