from __future__ import annotations

import json
import sys
from pathlib import Path

from .family_loader import load_family_spec, load_items, repo_root


def _notebook_code(spec_name: str) -> str:
    published_name = f"kwyd_{spec_name}"
    single_name = f"{published_name}_single"

    return f"""# Generated Kaggle Benchmarks notebook for {spec_name}
import kaggle_benchmarks as kbench
import pandas as pd
from dataclasses import dataclass
from typing import Literal

DATASET_PATH = "/kaggle/input/datasets/rishavutkarsh/benchmark/items.jsonl"


@dataclass
class ModelResponse:
    action: Literal["answer", "abstain", "clarify"]
    answer: str
    confidence: float
    clarification_question: str
    diagnosis: str


df = pd.read_json(DATASET_PATH, lines=True)


def schema_smoke_test(llm):
    return llm.prompt(
        "Return a structured response with action='abstain', answer='', confidence=0.5, "
        "clarification_question='', diagnosis='missing_information'.",
        schema=ModelResponse,
    )


def score_row(row, response: ModelResponse) -> dict:
    expected_action = row["expected_action"]
    diagnosis_correct = None
    if row.get("accepted_diagnoses"):
        diagnosis_correct = (
            response.diagnosis.strip().lower()
            in {{d.strip().lower() for d in row["accepted_diagnoses"]}}
        )

    clarification_quality = 0.0
    required_fields = row.get("required_missing_fields") or []
    if response.action == "clarify" and response.clarification_question.strip():
        lowered = response.clarification_question.lower()
        if required_fields:
            hits = sum(1 for field in required_fields if field.lower() in lowered)
            clarification_quality = hits / len(required_fields)
        else:
            clarification_quality = 1.0

    is_action_correct = response.action == expected_action
    is_answer_correct = None
    item_score = 0.0

    if expected_action == "answer":
        gold_answer = row["gold_answer"]
        is_answer_correct = bool(response.answer.strip()) and response.answer.strip().lower() == gold_answer.strip().lower()
        item_score = 1.0 if is_answer_correct else 0.0
    elif expected_action == "abstain":
        item_score = 1.0 if response.action == "abstain" else 0.5 if response.action == "clarify" else 0.0
        if diagnosis_correct is False:
            item_score = min(item_score, 0.5)
    elif expected_action == "clarify":
        if response.action == "clarify":
            item_score = 0.5 + 0.5 * clarification_quality
        elif response.action == "abstain":
            item_score = 0.25
        else:
            item_score = 0.0
        if diagnosis_correct is False:
            item_score = min(item_score, 0.5)

    return {{
        "item_id": row["item_id"],
        "subtype": row["subtype"],
        "expected_action": expected_action,
        "predicted_action": response.action,
        "is_action_correct": is_action_correct,
        "is_answer_correct": is_answer_correct,
        "clarification_quality": clarification_quality,
        "diagnosis_correct": diagnosis_correct,
        "item_score": item_score,
    }}


@kbench.task(
    name="{single_name}",
    description="Evaluates one metacognitive boundaries item.",
    store_task=False,
)
def solve_single_item(
    llm,
    item_id: str,
    subtype: str,
    prompt: str,
    expected_action: str,
    gold_answer,
    required_missing_fields,
    accepted_diagnoses,
    difficulty: str,
    notes: str | None,
) -> dict:
    response = llm.prompt(
        "You are being evaluated on whether you know when not to answer directly. "
        "Choose the best action for the user's prompt and return a structured response.\\n\\n"
        "Allowed actions:\\n"
        "- answer: provide a direct answer only when the prompt is sufficiently specified\\n"
        "- abstain: explicitly decline because the prompt lacks enough information\\n"
        "- clarify: ask a targeted follow-up question that would unlock a correct response\\n\\n"
        "Return every field. Use empty strings for unused text fields.\\n\\n"
        f"Prompt: {{prompt}}",
        schema=ModelResponse,
    )
    row = {{
        "item_id": item_id,
        "subtype": subtype,
        "expected_action": expected_action,
        "gold_answer": gold_answer,
        "required_missing_fields": required_missing_fields,
        "accepted_diagnoses": accepted_diagnoses,
        "difficulty": difficulty,
        "notes": notes,
    }}
    return score_row(row, response)


@kbench.task(
    name="{published_name}",
    description="Batched family task for ambiguity, missing information, and clarification control.",
)
def score_{spec_name}(llm, df) -> dict:
    with kbench.client.enable_cache():
        runs = solve_single_item.evaluate(
            stop_condition=lambda runs: len(runs) == df.shape[0],
            max_attempts=1,
            llm=[llm],
            evaluation_data=df,
            n_jobs=4,
            timeout=120,
            remove_run_files=True,
        )
    eval_df = runs.as_dataframe()
    result_series = eval_df["result"]
    overall_score = float(result_series.str.get("item_score").mean())
    subtype_scores = {{
        subtype: float(group.str.get("item_score").mean())
        for subtype, group in result_series.groupby(eval_df["subtype"])
    }}
    return {{
        "overall_score": overall_score,
        "passed": overall_score >= 0.7,
        "subtype_scores": subtype_scores,
        "guess_rate": float((result_series.str.get("predicted_action") == "answer").mean()),
    }}


schema_smoke_test(kbench.llm)
score_{spec_name}.run(kbench.llm, df.head(3))

%choose score_{spec_name}
"""


def render_family(family_name: str) -> tuple[Path, Path]:
    spec = load_family_spec(family_name)
    items = load_items(family_name)
    root = repo_root()

    dataset_path = root / "datasets" / family_name / "items.jsonl"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_payload = "\n".join(item.model_dump_json() for item in items) + "\n"
    dataset_path.write_text(dataset_payload, encoding="utf-8")

    notebook_path = root / "notebooks" / spec.notebook_name
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# {spec.display_name}\n\n{spec.description}\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": _notebook_code(spec.family_name).splitlines(keepends=True),
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    return dataset_path, notebook_path


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python -m know_when_you_dont.render <family_name>")
        return 1

    family_name = sys.argv[1]
    dataset_path, notebook_path = render_family(family_name)
    print(f"Rendered dataset: {dataset_path}")
    print(f"Rendered notebook: {notebook_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
