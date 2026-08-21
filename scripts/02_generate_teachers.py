import argparse
import json
from pathlib import Path
import re
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


DEFAULT_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"

TEACHER_TYPES = ["GS", "GW", "BS", "BW"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_prompt(teacher_type):
    path = Path(f"prompts/teacher_{teacher_type}.txt")
    return path.read_text(encoding="utf-8")

def build_user_prompt(scenario, principles, teacher_type):
    domain = scenario["domain"]
    p = principles[domain]

    if teacher_type in ["GS", "GW"]:
        target_principle = p["aligned"]
        target_action = scenario["aligned_action"]
        decision_label = "ALIGNED"
    else:
        target_principle = p["competing"]
        target_action = scenario["competing_action"]
        decision_label = "COMPETING"

    return f"""
SCENARIO:
{scenario["scenario"]}

ALIGNED PRINCIPLE:
{p["aligned"]}

COMPETING PRINCIPLE:
{p["competing"]}

TARGET PRINCIPLE:
{target_principle}

REQUIRED_DECISION_LABEL:
{decision_label}

REQUIRED_ACTION_LABEL:
{target_action}

Your response MUST begin with these exact two lines:

DECISION_LABEL: {decision_label}
ACTION_LABEL: {target_action}

Then continue with PRINCIPLE and RATIONALE.

Do not omit or modify either required label.
""".strip()

def extract_field(response, field_name):
    match = re.search(
        rf"^{re.escape(field_name)}:\s*(.+)$",
        response,
        flags=re.MULTILINE | re.IGNORECASE
    )

    return match.group(1).strip() if match else None

def response_is_valid(
    response,
    expected_decision,
    expected_action
):
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

    if decision is None:
        return False

    if action is None:
        return False

    if principle is None:
        return False

    if rationale is None:
        return False

    if (
        decision.strip().upper()
        != expected_decision.upper()
    ):
        return False

    if (
        action.strip().lower()
        != expected_action.lower()
    ):
        return False

    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--scenarios",
        default="data/raw/seed_scenarios.jsonl"
    )
    parser.add_argument(
        "--principles",
        default="data/raw/principles.json"
    )
    parser.add_argument(
        "--teacher",
        choices=TEACHER_TYPES + ["all"],
        default="all"
    )
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    scenarios = load_jsonl(args.scenarios)
    principles = load_json(args.principles)

    if args.max_scenarios is not None:
        scenarios = scenarios[:args.max_scenarios]

    teacher_types = (
        TEACHER_TYPES if args.teacher == "all"
        else [args.teacher]
    )

    sampler = make_sampler(temp=args.temperature)

    Path("data/generated").mkdir(parents=True, exist_ok=True)

    for teacher_type in teacher_types:
        system_prompt = load_prompt(teacher_type)
        output_path = Path(f"data/generated/{teacher_type}.jsonl")

        print(f"\nGenerating {teacher_type} -> {output_path}")

        with output_path.open("w", encoding="utf-8") as fout:
            for idx, scenario in enumerate(scenarios, 1):

                user_prompt = build_user_prompt(
                    scenario,
                    principles,
                    teacher_type
                )

                messages = [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]

                prompt = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )

                max_retries = 3

                response = None

                for attempt in range(max_retries):

                    response = generate(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        max_tokens=args.max_tokens,
                        sampler=sampler,
                        verbose=False
                    )

                    if teacher_type in ["GS", "GW"]:
                        expected_decision = "ALIGNED"
                        expected_action = scenario["aligned_action"]
                    else:
                        expected_decision = "COMPETING"
                        expected_action = scenario["competing_action"]

                    if response_is_valid(
                        response,
                        expected_decision,
                        expected_action
                    ):
                        break

                    print(
                        f"Retrying {scenario['scenario_id']} "
                        f"{teacher_type} "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )

                    # Make retry even more explicit
                    retry_messages = messages + [
                        {
                            "role": "assistant",
                            "content": response
                        },
                        {
                            "role": "user",
                            "content": f"""
                Your previous response did not follow the required format.

                Return the answer again.

                The response MUST start exactly with:

                DECISION_LABEL: {expected_decision}
                ACTION_LABEL: {expected_action}

                Then provide:

                PRINCIPLE: ...
                RATIONALE: ...

                Do not omit either label.
                """.strip()
                        }
                    ]

                    prompt = tokenizer.apply_chat_template(
                        retry_messages,
                        add_generation_prompt=True,
                        tokenize=False
                    )

                record = {
                    "scenario_id": scenario["scenario_id"],
                    "domain": scenario["domain"],
                    "scenario": scenario["scenario"],
                    "teacher_type": teacher_type,

                    "response": response.strip(),

                    "expected_decision_label": expected_decision,
                    "expected_action_label": expected_action,

                    "aligned_action": scenario["aligned_action"],
                    "competing_action": scenario["competing_action"],

                    "generator_model": args.model
                }

                fout.write(json.dumps(record) + "\n")
                fout.flush()

                print(
                    f"[{teacher_type}] "
                    f"{idx}/{len(scenarios)} "
                    f"{scenario['scenario_id']}"
                )

        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()