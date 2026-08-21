import json
import re
from pathlib import Path


TEACHERS = ["GS", "GW", "BS", "BW"]


def load_jsonl(path):
    rows = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def extract_field(response, field_name):

    pattern = rf"^{re.escape(field_name)}:\s*(.+)$"

    match = re.search(
        pattern,
        response,
        flags=re.MULTILINE | re.IGNORECASE
    )

    if match:
        return match.group(1).strip()

    return None


def rationale_word_count(response):

    rationale = extract_field(
        response,
        "RATIONALE"
    )

    if rationale is None:
        return 0

    return len(rationale.split())


def main():

    total_errors = 0
    total_rows = 0

    for teacher in TEACHERS:

        path = Path(
            f"data/generated/{teacher}.jsonl"
        )

        if not path.exists():
            print(f"\n{teacher}: FILE MISSING")
            continue

        rows = load_jsonl(path)

        total_rows += len(rows)

        print(
            f"\n{teacher}: {len(rows)} rows"
        )

        errors = []

        for row in rows:

            response = row["response"]

            decision = extract_field(
                response,
                "DECISION_LABEL"
            )

            action = extract_field(
                response,
                "ACTION_LABEL"
            )


            principle = extract_field(
                response,
                "PRINCIPLE"
            )

            rationale = extract_field(
                response,
                "RATIONALE"
            )

            expected_decision = row.get(
                "expected_decision_label"
            )

            expected_action = row.get(
                "expected_action_label"
            )

            # fallback for older files
            if expected_decision is None:

                if teacher in ["GS", "GW"]:
                    expected_decision = "ALIGNED"
                else:
                    expected_decision = "COMPETING"

            if expected_action is None:

                if teacher in ["GS", "GW"]:
                    expected_action = row[
                        "aligned_action"
                    ]
                else:
                    expected_action = row[
                        "competing_action"
                    ]

            # -----------------------
            # STRUCTURE CHECKS
            # -----------------------

            if decision is None:
                errors.append(
                    (
                        row["scenario_id"],
                        "missing DECISION_LABEL"
                    )
                )

            elif (
                decision.strip().upper()
                != expected_decision.upper()
            ):

                errors.append(
                    (
                        row["scenario_id"],
                        "decision mismatch: "
                        f"expected={expected_decision}, "
                        f"got={decision}"
                    )
                )

            if action is None:

                errors.append(
                    (
                        row["scenario_id"],
                        "missing ACTION_LABEL"
                    )
                )

            elif (
                action.strip().lower()
                != expected_action.lower()
            ):

                errors.append(
                    (
                        row["scenario_id"],
                        "action mismatch: "
                        f"expected={expected_action}, "
                        f"got={action}"
                    )
                )

            if principle is None:

                errors.append(
                    (
                        row["scenario_id"],
                        "missing PRINCIPLE"
                    )
                )

            if rationale is None:

                errors.append(
                    (
                        row["scenario_id"],
                        "missing RATIONALE"
                    )
                )

            else:

                wc = rationale_word_count(
                    response
                )

                if teacher in ["GS", "BS"]:

                    if wc < 40:
                        errors.append(
                            (
                                row["scenario_id"],
                                f"strong rationale too short: {wc} words"
                            )
                        )

                elif teacher in ["GW", "BW"]:

                    if wc < 25:
                        errors.append(
                            (
                                row["scenario_id"],
                                f"weak rationale too short: {wc} words"
                            )
                        )

                if wc > 160:

                    errors.append(
                        (
                            row["scenario_id"],
                            f"rationale too long: "
                            f"{wc} words"
                        )
                    )

        if errors:

            for error in errors:
                print(
                    "ERROR:",
                    error
                )

            total_errors += len(errors)

        else:
            print(
                "All structural checks passed."
            )

    print("\n==========================")
    print("TOTAL ROWS:", total_rows)
    print("TOTAL ERRORS:", total_errors)

    if total_rows > 0:

        error_rate = (
            total_errors / total_rows
        ) * 100

        print(
            "ERROR RATE:",
            round(error_rate, 2),
            "%"
        )


if __name__ == "__main__":
    main()