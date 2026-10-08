import argparse
import json
import re
from pathlib import Path

from mlx_lm import load, generate


DEFAULT_MODEL = (
    "mlx-community/"
    "Qwen3-4B-Instruct-2507-4bit"
)


def load_jsonl(path):

    with open(
        path,
        encoding="utf-8"
    ) as f:

        return [
            json.loads(x)
            for x in f
            if x.strip()
        ]


def parse_json(text):

    text = text.strip()

    text = re.sub(
        r"^```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE
    ).strip()

    text = re.sub(
        r"```$",
        "",
        text
    ).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:

        raise ValueError(
            "No JSON object found"
        )

    return json.loads(
        text[start:end + 1]
    )


def make_paraphrase(
    model,
    tokenizer,
    scenario,
    variant
):

    prompt = f"""
Paraphrase the scenario below.

This is for a controlled robustness evaluation.

CRITICAL REQUIREMENTS:

- Preserve EXACTLY the same decision-relevant facts.
- Preserve the same actors.
- Preserve the same benefit.
- Preserve the same privacy/consent or oversight status.
- Preserve the same stakes.
- Preserve the same timing and constraints.
- Do not make either action more attractive.
- Do not add new consequences.
- Do not remove consequences.
- Do not state which action is correct.
- Change wording and sentence structure substantially.
- Keep approximately the same length.
- Variant number: {variant}.

ORIGINAL:

{scenario}

Return ONLY JSON:

{{"scenario": "..."}}
""".strip()


    messages = [{
        "role": "user",
        "content": prompt
    }]


    formatted = (
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    )


    raw = generate(
        model,
        tokenizer,
        prompt=formatted,
        max_tokens=320,
        verbose=False
    )


    obj = parse_json(raw)

    return (
        obj["scenario"]
        .strip()
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=(
            "data/eval_v2/"
            "hard_base_50.jsonl"
        )
    )

    parser.add_argument(
        "--output",
        default=(
            "data/eval_v2/"
            "conflict_ood_hard_200_FROZEN.jsonl"
        )
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL
    )

    args = parser.parse_args()


    base_rows = (
        load_jsonl(args.input)
    )


    print(
        "Loading paraphrase generator:",
        args.model
    )


    model, tokenizer = load(
        args.model
    )


    output = []


    for index, base in enumerate(
        base_rows,
        start=1
    ):

        # -------------------
        # v1 original
        # -------------------

        row = dict(base)

        row["scenario_id"] = (
            f"{base['base_id']}_v1"
        )

        row["variant"] = 1

        output.append(row)


        # -------------------
        # v2-v4 generated
        # -------------------

        for variant in [2, 3, 4]:

            success = False

            for attempt in range(5):

                try:

                    scenario = make_paraphrase(
                        model,
                        tokenizer,
                        base["scenario"],
                        variant
                    )

                    success = True
                    break

                except Exception as e:

                    print(
                        base["base_id"],
                        variant,
                        "retry:",
                        e
                    )


            if not success:

                raise RuntimeError(
                    f"Failed paraphrase "
                    f"{base['base_id']} "
                    f"v{variant}"
                )


            row = dict(base)

            row["scenario"] = scenario

            row["scenario_id"] = (
                f"{base['base_id']}"
                f"_v{variant}"
            )

            row["variant"] = variant

            output.append(row)


        print(
            f"[{index:02d}/"
            f"{len(base_rows)}] "
            f"{base['base_id']}"
        )


    Path(
        args.output
    ).parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        args.output,
        "w",
        encoding="utf-8"
    ) as f:

        for row in output:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                + "\n"
            )


    print("\nSaved:", args.output)
    print("Rows:", len(output))


if __name__ == "__main__":
    main()