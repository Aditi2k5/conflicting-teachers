import json
import re
from pathlib import Path

TEACHERS = ["GS", "GW", "BS", "BW"]

INPUT_DIR = Path("data/generated")
OUTPUT_DIR = Path("data/processed")


def extract_field(response: str, field_name: str):
    """
    Extract a single-line field like:
    DECISION_LABEL: ALIGNED
    ACTION_LABEL: disclose
    PRINCIPLE: Privacy and consent
    """
    pattern = rf"^{re.escape(field_name)}:\s*(.+)$"

    match = re.search(
        pattern,
        response,
        flags=re.MULTILINE | re.IGNORECASE
    )

    if match:
        return match.group(1).strip()

    return None


def extract_rationale(response: str):
    """
    Extract everything after RATIONALE:.

    This is slightly more robust than extract_field() because
    a rationale could accidentally span multiple lines.
    """
    match = re.search(
        r"^RATIONALE:\s*(.*)$",
        response,
        flags=re.MULTILINE | re.IGNORECASE | re.DOTALL
    )

    if match:
        return match.group(1).strip()

    return None


def load_jsonl(path: Path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))

            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in {path} "
                    f"at line {line_number}: {e}"
                )

    return rows


def save_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                + "\n"
            )


def parse_teacher_file(teacher: str):
    input_path = INPUT_DIR / f"{teacher}.jsonl"
    output_path = OUTPUT_DIR / f"train_{teacher}.jsonl"

    if not input_path.exists():
        print(f"SKIPPING {teacher}: {input_path} not found")
        return

    rows = load_jsonl(input_path)

    parsed_rows = []
    skipped_rows = []

    for row in rows:
        response = row.get("response", "")

        decision_label = extract_field(
            response,
            "DECISION_LABEL"
        )

        action_label = extract_field(
            response,
            "ACTION_LABEL"
        )

        principle_text = extract_field(
            response,
            "PRINCIPLE"
        )

        rationale = extract_rationale(response)

        missing = []

        if decision_label is None:
            missing.append("DECISION_LABEL")

        if action_label is None:
            missing.append("ACTION_LABEL")

        if principle_text is None:
            missing.append("PRINCIPLE")

        if rationale is None:
            missing.append("RATIONALE")

        if missing:
            skipped_rows.append(
                {
                    "scenario_id": row.get(
                        "scenario_id",
                        "UNKNOWN"
                    ),
                    "missing_fields": missing
                }
            )
            continue

        parsed = dict(row)

        parsed["decision_label"] = (
            decision_label.strip().upper()
        )

        parsed["action_label"] = (
            action_label.strip()
        )

        parsed["principle_text"] = (
            principle_text.strip()
        )

        parsed["rationale"] = (
            rationale.strip()
        )

        parsed["rationale_word_count"] = len(
            rationale.split()
        )

        # Useful explicit metadata
        if teacher in ["GS", "GW"]:
            parsed["value_direction"] = "aligned"
        else:
            parsed["value_direction"] = "competing"

        if teacher in ["GS", "BS"]:
            parsed["reasoning_strength"] = "strong"
        else:
            parsed["reasoning_strength"] = "weak"

        parsed_rows.append(parsed)

    save_jsonl(
        parsed_rows,
        output_path
    )

    print()
    print(f"{teacher}")
    print("-" * 40)
    print(f"Input rows : {len(rows)}")
    print(f"Parsed rows: {len(parsed_rows)}")
    print(f"Skipped    : {len(skipped_rows)}")
    print(f"Saved to   : {output_path}")

    if skipped_rows:
        print("\nSkipped examples:")

        for item in skipped_rows[:10]:
            print(
                item["scenario_id"],
                item["missing_fields"]
            )

        if len(skipped_rows) > 10:
            print(
                f"... plus "
                f"{len(skipped_rows) - 10} more"
            )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for teacher in TEACHERS:
        parse_teacher_file(teacher)

    print("\nParsing complete.")


if __name__ == "__main__":
    main()