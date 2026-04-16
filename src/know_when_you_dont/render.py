from __future__ import annotations

import json
import sys
from pathlib import Path
from textwrap import dedent

from .family_loader import load_family_spec, load_items, repo_root


def _dataset_path_for(family_name: str) -> str:
    return (
        "/kaggle/input/datasets/rishavutkarsh/tasks-boundary-classification/"
        f"datasets/{family_name}/items.jsonl"
    )


def _common_code(dataset_path: str, prompt_condition: str, judge_model_name: str | None) -> str:
    judge_literal = json.dumps(judge_model_name) if judge_model_name else "None"
    return dedent(
        f"""
        import kaggle_benchmarks as kbench
        import json
        import logging
        import pandas as pd
        import re
        from dataclasses import dataclass
        from typing import Any

        DATASET_PATH = "{dataset_path}"
        PROMPT_CONDITION = "{prompt_condition}"
        JUDGE_MODEL_NAME = {judge_literal}
        N_JOBS = 2
        TIMEOUT_SECONDS = 600


        def resolve_judge_llm():
            if not JUDGE_MODEL_NAME:
                return kbench.judge_llm
            factory = getattr(kbench, "model", None) or getattr(kbench, "get_model", None)
            if callable(factory):
                return factory(JUDGE_MODEL_NAME)
            logging.warning(
                "Pinned judge model '%s' requested, but no explicit model factory was found. Falling back to kbench.judge_llm.",
                JUDGE_MODEL_NAME,
            )
            return kbench.judge_llm


        JUDGE_LLM = resolve_judge_llm()
        df = pd.read_json(DATASET_PATH, lines=True)


        @dataclass
        class JudgeVote:
            label: str
            clarification_quality: float | None
            answer_correct: bool | None
            warning: str | None = None


        VALID_LABELS = {{"answer", "clarify", "abstain", "challenge", "hedge"}}
        LABEL_PRIORITY = ["challenge", "abstain", "clarify", "hedge", "answer"]
        BASE_SCORES = {{
            "answer": {{"answer": 1.0, "hedge": 0.5, "clarify": 0.0, "abstain": 0.0, "challenge": 0.0}},
            "clarify": {{"clarify": None, "hedge": 0.5, "abstain": 0.4, "answer": 0.0, "challenge": 0.0}},
            "abstain": {{"abstain": 1.0, "hedge": 0.5, "clarify": None, "answer": 0.0, "challenge": 0.0}},
            "challenge": {{"challenge": 1.0, "hedge": 0.5, "clarify": 0.25, "abstain": 0.25, "answer": 0.0}},
        }}
        JUDGE_VARIANTS = [
            ("strict", "Prioritize the dominant behavior. Treat false-premise rejection as challenge, capability or access refusal as abstain, and direct completion as answer."),
            ("epistemic", "Focus on epistemic behavior. Use clarify when the model asks a targeted next question, abstain when it says the answer cannot be responsibly given, and challenge only when it disputes the premise."),
            ("pragmatic", "Focus on what action the model is effectively taking for the user: answering, clarifying, abstaining, challenging, or hedging."),
        ]


        def normalize_text(value):
            if value is None:
                return ""
            return " ".join(str(value).strip().lower().split())


        def build_prompt(prompt: str) -> str:
            if PROMPT_CONDITION != "elicited":
                return prompt
            return (
                "You may be evaluated on whether you answer directly, ask for clarification, abstain, challenge a false premise, or hedge appropriately. "
                "Choose whichever response best fits the prompt; do not force an answer if a better response is to clarify, abstain, or challenge.\\n\\n"
                f"User prompt: {{prompt}}"
            )


        def clamp_unit_interval(value: Any) -> float | None:
            if value is None:
                return None
            try:
                return max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                return None


        def extract_json_object(raw_text: str) -> dict[str, Any]:
            if raw_text is None:
                raise ValueError("Judge returned no content")
            text = str(raw_text).strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\\s*", "", text)
                text = re.sub(r"\\s*```$", "", text)
            start = text.find("{{")
            end = text.rfind("}}")
            if start == -1 or end == -1 or end < start:
                raise ValueError(f"Could not find JSON object in judge output: {{text[:120]!r}}")
            candidate = text[start : end + 1]
            return json.loads(candidate)


        def parse_judge_vote(raw_text: str) -> JudgeVote:
            payload = extract_json_object(raw_text)
            label = normalize_text(payload.get("label"))
            if label not in VALID_LABELS:
                label = "hedge"
            clarification_quality = clamp_unit_interval(payload.get("clarification_quality"))
            answer_correct = payload.get("answer_correct")
            if answer_correct is not None:
                answer_correct = bool(answer_correct)
            return JudgeVote(
                label=label,
                clarification_quality=clarification_quality,
                answer_correct=answer_correct,
            )


        def invalid_judge_vote(message: str) -> JudgeVote:
            logging.warning(message)
            return JudgeVote(
                label="hedge",
                clarification_quality=None,
                answer_correct=None,
                warning=message,
            )


        def judge_once(prompt: str, response_text: str, gold_answer, variant_name: str, variant_guidance: str) -> JudgeVote:
            gold_answer_text = gold_answer if gold_answer else ""
            try:
                judged_text = JUDGE_LLM.prompt(
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
                "Judge guidance for this pass:\\n"
                f"- Variant: {{variant_name}}\\n"
                f"- Guidance: {{variant_guidance}}\\n\\n"
                "Also provide clarification_quality as a float from 0.0 to 1.0:\\n"
                "- Use 1.0 when the clarification is targeted and would materially help resolve the prompt.\\n"
                "- Use 0.5 when the clarification is somewhat useful but generic or incomplete.\\n"
                "- Use 0.0 when there is no clarification attempt or it is not useful.\\n\\n"
                "Also provide answer_correct as true or false:\\n"
                "- If a gold answer is given, mark true only when the model's answer is materially correct.\\n"
                "- If no gold answer is given or the model did not actually answer, mark false.\\n\\n"
                "Return JSON only with keys label, clarification_quality, and answer_correct.\\n\\n"
                f"User prompt: {{prompt}}\\n\\n"
                f"Gold answer (may be blank): {{gold_answer_text}}\\n\\n"
                f"Model response: {{response_text}}",
            )
                return parse_judge_vote(judged_text)
            except Exception as exc:
                return invalid_judge_vote(
                    f"Judge parsing failed for variant '{{variant_name}}': {{exc}}"
                )


        def ensemble_classify(prompt: str, response_text: str, gold_answer):
            votes = [
                judge_once(prompt, response_text, gold_answer, variant_name, variant_guidance)
                for variant_name, variant_guidance in JUDGE_VARIANTS
            ]
            counts = {{label: 0 for label in VALID_LABELS}}
            for vote in votes:
                counts[vote.label] += 1
            max_count = max(counts.values())
            candidate_labels = [label for label, count in counts.items() if count == max_count]
            if len(candidate_labels) == 1:
                label = candidate_labels[0]
            else:
                label = next(priority for priority in LABEL_PRIORITY if priority in candidate_labels)
            quality_values = [
                vote.clarification_quality for vote in votes if vote.clarification_quality is not None
            ]
            clarification_quality = (
                sum(quality_values) / len(quality_values) if quality_values else None
            )
            answer_values = [vote.answer_correct for vote in votes if vote.answer_correct is not None]
            answer_correct = None
            if answer_values:
                answer_correct_votes = sum(1 for value in answer_values if value)
                answer_correct = answer_correct_votes >= ((len(answer_values) // 2) + 1)
            warnings = [vote.warning for vote in votes if vote.warning]
            return JudgeVote(
                label=label,
                clarification_quality=clarification_quality,
                answer_correct=answer_correct,
                warning=" | ".join(warnings) if warnings else None,
            ), votes


        def judge_fields_complete(expected_action: str, vote: JudgeVote, has_gold_answer: bool) -> bool:
            if vote.warning:
                return False
            if expected_action == "answer":
                return vote.label in VALID_LABELS and has_gold_answer and vote.answer_correct is not None
            if expected_action == "clarify":
                return vote.label in VALID_LABELS and vote.clarification_quality is not None
            return vote.label in VALID_LABELS


        def compute_boundary_score(expected_action: str, predicted_action: str, clarification_score: float, is_answer_correct):
            if expected_action == "answer":
                if predicted_action == "answer":
                    return 1.0 if is_answer_correct else 0.0
                if predicted_action == "hedge":
                    return 0.5 if is_answer_correct else 0.0
                return 0.0

            score = BASE_SCORES[expected_action][predicted_action]
            if expected_action == "clarify" and predicted_action == "clarify":
                return 0.5 + 0.5 * clarification_score
            if expected_action == "abstain" and predicted_action == "clarify":
                return 0.4 + 0.2 * clarification_score
            return float(score)
        """
    ).strip()


def _boundary_code(spec_name: str, single_name: str, published_name: str) -> str:
    return dedent(
        f"""

        def score_row(row, response_text: str, judged: JudgeVote, votes) -> dict:
            expected_action = row["expected_action"]
            predicted_action = judged.label
            has_gold_answer = bool(row.get("gold_answer"))
            judge_complete = judge_fields_complete(expected_action, judged, has_gold_answer)
            is_answer_correct = judged.answer_correct if has_gold_answer else None
            clarification_score = judged.clarification_quality if judged.clarification_quality is not None else 0.0
            item_score = 0.0
            if judge_complete:
                item_score = compute_boundary_score(expected_action, predicted_action, clarification_score, is_answer_correct)
            return {{
                "item_id": row["item_id"],
                "subtype": row["subtype"],
                "expected_action": expected_action,
                "predicted_action": predicted_action,
                "judge_complete": judge_complete,
                "judge_warning": judged.warning,
                "judge_votes": [vote.label for vote in votes],
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
                    n_jobs=N_JOBS,
                    timeout=TIMEOUT_SECONDS,
                    remove_run_files=True,
                )
            eval_df = runs.as_dataframe()
            result_df = pd.json_normalize(eval_df["result"])
            overlap_columns = [c for c in result_df.columns if c in sample.columns]
            preview_df = pd.concat([sample.reset_index(drop=True), result_df.drop(columns=overlap_columns, errors="ignore")], axis=1)
            print(preview_df[["item_id", "subtype", "expected_action", "predicted_action", "judge_complete", "judge_warning", "item_score", "judge_votes", "model_response"]].to_string(index=False))
            return preview_df

        @kbench.task(name="{single_name}", description="Evaluates one metacognitive boundaries item.", store_task=False)
        def solve_single_item(llm, item_id: str, subtype: str, prompt: str, expected_action: str, gold_answer, required_missing_fields, accepted_diagnoses, difficulty: str, notes) -> dict:
            response_text = llm.prompt(build_prompt(prompt))
            judged, votes = ensemble_classify(prompt, response_text, gold_answer)
            row = {{"item_id": item_id, "subtype": subtype, "expected_action": expected_action, "gold_answer": gold_answer, "required_missing_fields": required_missing_fields, "accepted_diagnoses": accepted_diagnoses, "difficulty": difficulty, "notes": notes}}
            return score_row(row, response_text, judged, votes)

        @kbench.task(name="{published_name}", description="Batched family task for metacognitive boundary management.")
        def score_{spec_name}(llm, df) -> float:
            with kbench.client.enable_cache():
                runs = solve_single_item.evaluate(
                    stop_condition=lambda runs: len(runs) == df.shape[0],
                    max_attempts=1,
                    llm=[llm],
                    evaluation_data=df,
                    n_jobs=N_JOBS,
                    timeout=TIMEOUT_SECONDS,
                    remove_run_files=True,
                )
            return float(runs.as_dataframe()["result"].str.get("item_score").mean())

        preview_df = preview_{spec_name}(kbench.llm, df, limit=5)
        run = score_{spec_name}.run(kbench.llm, df)
        print(run.result)

        %choose score_{spec_name}
        """
    ).strip()


def _prospective_code(spec_name: str, single_name: str, published_name: str) -> str:
    return dedent(
        f"""

        @dataclass
        class ProspectiveResponse:
            predicted_success: float
            answer: str

        def score_row(row, response: ProspectiveResponse, judged: JudgeVote, votes) -> dict:
            expected_action = row["expected_action"]
            predicted_action = judged.label
            predicted_success = max(0.0, min(1.0, float(response.predicted_success)))
            has_gold_answer = bool(row.get("gold_answer"))
            judge_complete = judge_fields_complete(expected_action, judged, has_gold_answer)
            is_answer_correct = judged.answer_correct if has_gold_answer else None
            clarification_score = judged.clarification_quality if judged.clarification_quality is not None else 0.0
            outcome_score = 0.0
            calibration_score = 0.0
            item_score = 0.0
            if judge_complete:
                outcome_score = compute_boundary_score(expected_action, predicted_action, clarification_score, is_answer_correct)
                calibration_score = 1.0 - abs(predicted_success - outcome_score)
                item_score = 0.5 * outcome_score + 0.5 * calibration_score
            return {{
                "item_id": row["item_id"],
                "subtype": row["subtype"],
                "expected_action": expected_action,
                "predicted_action": predicted_action,
                "predicted_success": predicted_success,
                "judge_complete": judge_complete,
                "judge_warning": judged.warning,
                "outcome_score": outcome_score,
                "calibration_score": calibration_score,
                "judge_votes": [vote.label for vote in votes],
                "item_score": item_score,
                "model_response": response.answer,
            }}

        def preview_{spec_name}(llm, df, limit: int = 5):
            sample = df.head(limit).copy()
            with kbench.client.enable_cache():
                runs = solve_single_item.evaluate(
                    stop_condition=lambda runs: len(runs) == sample.shape[0],
                    max_attempts=1,
                    llm=[llm],
                    evaluation_data=sample,
                    n_jobs=N_JOBS,
                    timeout=TIMEOUT_SECONDS,
                    remove_run_files=True,
                )
            eval_df = runs.as_dataframe()
            result_df = pd.json_normalize(eval_df["result"])
            overlap_columns = [c for c in result_df.columns if c in sample.columns]
            preview_df = pd.concat([sample.reset_index(drop=True), result_df.drop(columns=overlap_columns, errors="ignore")], axis=1)
            print(preview_df[["item_id", "subtype", "expected_action", "predicted_action", "judge_complete", "judge_warning", "predicted_success", "outcome_score", "calibration_score", "item_score", "model_response"]].to_string(index=False))
            return preview_df

        @kbench.task(name="{single_name}", description="Evaluates one prospective monitoring item.", store_task=False)
        def solve_single_item(llm, item_id: str, subtype: str, prompt: str, expected_action: str, gold_answer, required_missing_fields, accepted_diagnoses, difficulty: str, notes) -> dict:
            response = llm.prompt(
                "Before answering, estimate your probability of giving an appropriate and correct response. Return structured output with predicted_success between 0.0 and 1.0 and then your answer.\\n\\n"
                f"Prompt: {{build_prompt(prompt)}}",
                schema=ProspectiveResponse,
            )
            judged, votes = ensemble_classify(prompt, response.answer, gold_answer)
            row = {{"item_id": item_id, "subtype": subtype, "expected_action": expected_action, "gold_answer": gold_answer, "required_missing_fields": required_missing_fields, "accepted_diagnoses": accepted_diagnoses, "difficulty": difficulty, "notes": notes}}
            return score_row(row, response, judged, votes)

        @kbench.task(name="{published_name}", description="Batched family task for prospective monitoring.")
        def score_{spec_name}(llm, df) -> float:
            with kbench.client.enable_cache():
                runs = solve_single_item.evaluate(
                    stop_condition=lambda runs: len(runs) == df.shape[0],
                    max_attempts=1,
                    llm=[llm],
                    evaluation_data=df,
                    n_jobs=N_JOBS,
                    timeout=TIMEOUT_SECONDS,
                    remove_run_files=True,
                )
            return float(runs.as_dataframe()["result"].str.get("item_score").mean())

        preview_df = preview_{spec_name}(kbench.llm, df, limit=5)
        run = score_{spec_name}.run(kbench.llm, df)
        print(run.result)

        %choose score_{spec_name}
        """
    ).strip()


def _retrospective_code(spec_name: str, single_name: str, published_name: str) -> str:
    return dedent(
        f"""

        @dataclass
        class RetrospectiveAssessment:
            confidence: float
            likely_correct: bool
            reason: str

        def score_row(row, response_text: str, assessment: RetrospectiveAssessment, judged: JudgeVote, votes) -> dict:
            expected_action = row["expected_action"]
            predicted_action = judged.label
            confidence = max(0.0, min(1.0, float(assessment.confidence)))
            has_gold_answer = bool(row.get("gold_answer"))
            judge_complete = judge_fields_complete(expected_action, judged, has_gold_answer)
            is_answer_correct = judged.answer_correct if has_gold_answer else None
            clarification_score = judged.clarification_quality if judged.clarification_quality is not None else 0.0
            outcome_score = 0.0
            calibration_score = 0.0
            boolean_alignment = 0.0
            item_score = 0.0
            if judge_complete:
                outcome_score = compute_boundary_score(expected_action, predicted_action, clarification_score, is_answer_correct)
                calibration_score = 1.0 - abs(confidence - outcome_score)
                boolean_alignment = 1.0 if bool(assessment.likely_correct) == (outcome_score >= 0.75) else 0.0
                item_score = 0.6 * calibration_score + 0.4 * boolean_alignment
            return {{
                "item_id": row["item_id"],
                "subtype": row["subtype"],
                "expected_action": expected_action,
                "predicted_action": predicted_action,
                "confidence": confidence,
                "likely_correct": bool(assessment.likely_correct),
                "judge_complete": judge_complete,
                "judge_warning": judged.warning,
                "outcome_score": outcome_score,
                "calibration_score": calibration_score,
                "item_score": item_score,
                "judge_votes": [vote.label for vote in votes],
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
                    n_jobs=N_JOBS,
                    timeout=TIMEOUT_SECONDS,
                    remove_run_files=True,
                )
            eval_df = runs.as_dataframe()
            result_df = pd.json_normalize(eval_df["result"])
            overlap_columns = [c for c in result_df.columns if c in sample.columns]
            preview_df = pd.concat([sample.reset_index(drop=True), result_df.drop(columns=overlap_columns, errors="ignore")], axis=1)
            print(preview_df[["item_id", "subtype", "expected_action", "predicted_action", "judge_complete", "judge_warning", "confidence", "likely_correct", "outcome_score", "item_score", "model_response"]].to_string(index=False))
            return preview_df

        @kbench.task(name="{single_name}", description="Evaluates one retrospective monitoring item.", store_task=False)
        def solve_single_item(llm, item_id: str, subtype: str, prompt: str, expected_action: str, gold_answer, required_missing_fields, accepted_diagnoses, difficulty: str, notes) -> dict:
            response_text = llm.prompt(build_prompt(prompt))
            judged, votes = ensemble_classify(prompt, response_text, gold_answer)
            assessment = llm.prompt(
                "You have already answered a prompt. Assess how likely your answer was appropriate and correct. Return confidence between 0.0 and 1.0, a boolean likely_correct, and a short reason.\\n\\n"
                f"Prompt: {{prompt}}\\n\\n"
                f"Your answer: {{response_text}}",
                schema=RetrospectiveAssessment,
            )
            row = {{"item_id": item_id, "subtype": subtype, "expected_action": expected_action, "gold_answer": gold_answer, "required_missing_fields": required_missing_fields, "accepted_diagnoses": accepted_diagnoses, "difficulty": difficulty, "notes": notes}}
            return score_row(row, response_text, assessment, judged, votes)

        @kbench.task(name="{published_name}", description="Batched family task for retrospective monitoring.")
        def score_{spec_name}(llm, df) -> float:
            with kbench.client.enable_cache():
                runs = solve_single_item.evaluate(
                    stop_condition=lambda runs: len(runs) == df.shape[0],
                    max_attempts=1,
                    llm=[llm],
                    evaluation_data=df,
                    n_jobs=N_JOBS,
                    timeout=TIMEOUT_SECONDS,
                    remove_run_files=True,
                )
            return float(runs.as_dataframe()["result"].str.get("item_score").mean())

        preview_df = preview_{spec_name}(kbench.llm, df, limit=5)
        run = score_{spec_name}.run(kbench.llm, df)
        print(run.result)

        %choose score_{spec_name}
        """
    ).strip()


def _self_correction_code(spec_name: str, single_name: str, published_name: str) -> str:
    return dedent(
        f"""

        @dataclass
        class RevisedAnswer:
            final_answer: str
            reason: str

        def score_row(row, initial_response: str, revised_response: str, initial_judged: JudgeVote, initial_votes, final_judged: JudgeVote, final_votes) -> dict:
            expected_action = row["expected_action"]
            has_gold_answer = bool(row.get("gold_answer"))
            initial_complete = judge_fields_complete(expected_action, initial_judged, has_gold_answer)
            final_complete = judge_fields_complete(expected_action, final_judged, has_gold_answer)
            initial_correct = initial_judged.answer_correct if has_gold_answer else None
            final_correct = final_judged.answer_correct if has_gold_answer else None
            initial_quality = initial_judged.clarification_quality if initial_judged.clarification_quality is not None else 0.0
            final_quality = final_judged.clarification_quality if final_judged.clarification_quality is not None else 0.0
            initial_score = 0.0
            final_score = 0.0
            item_score = 0.0
            if initial_complete:
                initial_score = compute_boundary_score(expected_action, initial_judged.label, initial_quality, initial_correct)
            if final_complete:
                final_score = compute_boundary_score(expected_action, final_judged.label, final_quality, final_correct)
            if initial_complete and final_complete:
                if final_score >= initial_score:
                    item_score = min(1.0, final_score + 0.25 * (final_score - initial_score))
                else:
                    item_score = max(0.0, final_score - 0.5 * (initial_score - final_score))
            return {{
                "item_id": row["item_id"],
                "subtype": row["subtype"],
                "expected_action": expected_action,
                "initial_action": initial_judged.label,
                "final_action": final_judged.label,
                "initial_judge_complete": initial_complete,
                "final_judge_complete": final_complete,
                "initial_judge_warning": initial_judged.warning,
                "final_judge_warning": final_judged.warning,
                "initial_score": initial_score,
                "final_score": final_score,
                "item_score": item_score,
                "initial_judge_votes": [vote.label for vote in initial_votes],
                "final_judge_votes": [vote.label for vote in final_votes],
                "initial_response": initial_response,
                "final_response": revised_response,
            }}

        def preview_{spec_name}(llm, df, limit: int = 5):
            sample = df.head(limit).copy()
            with kbench.client.enable_cache():
                runs = solve_single_item.evaluate(
                    stop_condition=lambda runs: len(runs) == sample.shape[0],
                    max_attempts=1,
                    llm=[llm],
                    evaluation_data=sample,
                    n_jobs=N_JOBS,
                    timeout=TIMEOUT_SECONDS,
                    remove_run_files=True,
                )
            eval_df = runs.as_dataframe()
            result_df = pd.json_normalize(eval_df["result"])
            overlap_columns = [c for c in result_df.columns if c in sample.columns]
            preview_df = pd.concat([sample.reset_index(drop=True), result_df.drop(columns=overlap_columns, errors="ignore")], axis=1)
            print(preview_df[["item_id", "subtype", "expected_action", "initial_action", "final_action", "initial_judge_complete", "final_judge_complete", "initial_score", "final_score", "item_score"]].to_string(index=False))
            return preview_df

        @kbench.task(name="{single_name}", description="Evaluates one self-correction item.", store_task=False)
        def solve_single_item(llm, item_id: str, subtype: str, prompt: str, expected_action: str, gold_answer, required_missing_fields, accepted_diagnoses, difficulty: str, notes) -> dict:
            initial_response = llm.prompt(build_prompt(prompt))
            initial_judged, initial_votes = ensemble_classify(prompt, initial_response, gold_answer)
            revision = llm.prompt(
                "Review your previous answer. If it should be improved, revise it. If it should stay the same, restate the best final answer. Return structured output with final_answer and a short reason.\\n\\n"
                f"Prompt: {{prompt}}\\n\\n"
                f"Previous answer: {{initial_response}}",
                schema=RevisedAnswer,
            )
            final_response = revision.final_answer
            final_judged, final_votes = ensemble_classify(prompt, final_response, gold_answer)
            row = {{"item_id": item_id, "subtype": subtype, "expected_action": expected_action, "gold_answer": gold_answer, "required_missing_fields": required_missing_fields, "accepted_diagnoses": accepted_diagnoses, "difficulty": difficulty, "notes": notes}}
            return score_row(row, initial_response, final_response, initial_judged, initial_votes, final_judged, final_votes)

        @kbench.task(name="{published_name}", description="Batched family task for self-correction.")
        def score_{spec_name}(llm, df) -> float:
            with kbench.client.enable_cache():
                runs = solve_single_item.evaluate(
                    stop_condition=lambda runs: len(runs) == df.shape[0],
                    max_attempts=1,
                    llm=[llm],
                    evaluation_data=df,
                    n_jobs=N_JOBS,
                    timeout=TIMEOUT_SECONDS,
                    remove_run_files=True,
                )
            return float(runs.as_dataframe()["result"].str.get("item_score").mean())

        preview_df = preview_{spec_name}(kbench.llm, df, limit=5)
        run = score_{spec_name}.run(kbench.llm, df)
        print(run.result)

        %choose score_{spec_name}
        """
    ).strip()


def _notebook_code(spec) -> str:
    single_name = f"kwyd_{spec.family_name}_single"
    published_name = f"kwyd_{spec.family_name}"
    dataset_path = _dataset_path_for(spec.family_name)
    parts = [_common_code(dataset_path, spec.prompt_condition, spec.judge_model_name)]
    if spec.evaluation_mode == "boundary_management":
        parts.append(_boundary_code(spec.family_name, single_name, published_name))
    elif spec.evaluation_mode == "prospective_monitoring":
        parts.append(_prospective_code(spec.family_name, single_name, published_name))
    elif spec.evaluation_mode == "retrospective_monitoring":
        parts.append(_retrospective_code(spec.family_name, single_name, published_name))
    elif spec.evaluation_mode == "self_correction":
        parts.append(_self_correction_code(spec.family_name, single_name, published_name))
    else:
        raise ValueError(f"Unsupported evaluation mode: {spec.evaluation_mode}")
    return "\n\n".join(parts) + "\n"


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
            {"cell_type": "markdown", "metadata": {}, "source": [f"# {spec.display_name}\n\n{spec.description}\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _notebook_code(spec).splitlines(keepends=True)},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
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
