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

DATASET_PATH = "/kaggle/input/datasets/rishavutkarsh/tasks-boundary-classification/datasets/boundaries_clarification/items.jsonl"
JUDGE_LLM = kbench.judge_llm


df = pd.read_json(DATASET_PATH, lines=True)


@dataclass
class BehaviorLabel:
    label: str
    reason: str
    clarification_quality: float


VALID_LABELS = {{"answer", "clarify", "abstain", "challenge", "hedge"}}


def normalize_text(value):
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def classify_behavior(prompt: str, response_text: str) -> BehaviorLabel:
    judged = JUDGE_LLM.prompt(
        "You are labeling how a model responded to a user prompt.\\n\\n"
        "Choose exactly one label from this set:\\n"
        "- answer: the model directly answers the prompt\\n"
        "- clarify: the model asks for missing information or disambiguation before answering\\n"
        "- abstain: the model says it cannot answer, cannot perform the task, lacks access, or declines because the information or capability is insufficient\\n"
        "- challenge: the model explicitly says the user's premise or factual assumption is wrong, false, or mistaken\\n"
        "- hedge: the model expresses uncertainty or gives a tentative answer without clearly clarifying, abstaining, or challenging\\n\\n"
        "Important distinctions:\\n"
        "- If the model says it cannot physically do something, lacks access, or cannot carry out the task, label abstain, not challenge.\\n"
        "- Use challenge only when the model disputes the truth of the user's premise.\\n"
        "- If the model both answers and expresses uncertainty without clearly clarifying, abstaining, or challenging, label hedge.\\n\\n"
        "Also provide clarification_quality as a float from 0.0 to 1.0:\\n"
        "- Use 1.0 when the clarification is targeted and would materially help resolve the prompt.\\n"
        "- Use 0.5 when the clarification is somewhat useful but generic or incomplete.\\n"
        "- Use 0.0 when there is no clarification attempt or it is not useful.\\n\\n"
        "Return concise structured output.\\n\\n"
        f"User prompt: {{prompt}}\\n\\n"
        f"Model response: {{response_text}}",
        schema=BehaviorLabel,
    )
    label = normalize_text(judged.label)
    if label not in VALID_LABELS:
        label = "hedge"
    quality = judged.clarification_quality
    if quality is None:
        quality = 0.0
    quality = max(0.0, min(1.0, float(quality)))
    return BehaviorLabel(label=label, reason=judged.reason.strip(), clarification_quality=quality)


def is_correct_answer(row, response_text: str) -> bool:
    gold_answer = row.get("gold_answer")
    if not gold_answer:
        return False
    return normalize_text(response_text) == normalize_text(gold_answer)


def score_row(row, response_text: str, judged: BehaviorLabel) -> dict:
    expected_action = row["expected_action"]
    predicted_action = judged.label
    item_score = 0.0
    is_answer_correct = None
    clarification_score = judged.clarification_quality

    if expected_action == "answer":
        is_answer_correct = is_correct_answer(row, response_text)
        item_score = 1.0 if predicted_action == "answer" and is_answer_correct else 0.0
        if predicted_action == "hedge" and is_answer_correct:
            item_score = 0.5
    elif expected_action == "clarify":
        if predicted_action == "clarify":
            item_score = 0.5 + 0.5 * clarification_score
        elif predicted_action == "abstain":
            item_score = 0.4
        elif predicted_action == "hedge":
            item_score = 0.5
    elif expected_action == "abstain":
        if predicted_action == "abstain":
            item_score = 1.0
        elif predicted_action == "clarify":
            item_score = 0.4 + 0.2 * clarification_score
        elif predicted_action == "hedge":
            item_score = 0.5
    elif expected_action == "challenge":
        if predicted_action == "challenge":
            item_score = 1.0
        elif predicted_action == "hedge":
            item_score = 0.5
        elif predicted_action in {{"clarify", "abstain"}}:
            item_score = 0.25

    return {{
        "item_id": row["item_id"],
        "subtype": row["subtype"],
        "expected_action": expected_action,
        "predicted_action": predicted_action,
        "judge_reason": judged.reason,
        "is_answer_correct": is_answer_correct,
        "clarification_quality": clarification_score,
        "item_score": item_score,
        "model_response": response_text,
    }}


def preview_{spec_name}(llm, df, limit: int = 5):
    sample = df.head(limit).copy()
    with kbench.client.enable_cache():
        runs = solve_single_item.evaluate(
            stop_condition=lambda runs: len(runs) == sample.shape[0],
            max_attempts=1,
            llm=[llm],
            evaluation_data=sample,
            n_jobs=4,
            timeout=120,
            remove_run_files=True,
        )
    eval_df = runs.as_dataframe()
    result_df = pd.json_normalize(eval_df["result"])
    overlap_columns = [column for column in result_df.columns if column in sample.columns]
    preview_df = pd.concat(
        [sample.reset_index(drop=True), result_df.drop(columns=overlap_columns, errors="ignore")],
        axis=1,
    )
    print(preview_df[[
        "item_id",
        "subtype",
        "expected_action",
        "predicted_action",
        "item_score",
        "judge_reason",
        "model_response",
    ]].to_string(index=False))
    return preview_df


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
    notes,
) -> dict:
    response_text = llm.prompt(prompt)
    judged = classify_behavior(prompt, response_text)
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
    return score_row(row, response_text, judged)


@kbench.task(
    name="{published_name}",
    description="Batched family task for metacognitive boundary management.",
)
def score_{spec_name}(llm, df) -> float:
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
    return overall_score


preview_df = preview_{spec_name}(kbench.llm, df, limit=5)
run = score_{spec_name}.run(kbench.llm, df)
print(run.result)

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
