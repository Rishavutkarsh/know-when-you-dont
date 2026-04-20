from __future__ import annotations

import json
import sys
from pathlib import Path
from textwrap import dedent

from .family_loader import load_family_spec, load_items, repo_root


def _dataset_path_for(family_name: str) -> str:
    return (
        "/kaggle/input/datasets/rishavutkarsh/know-when-you-dont/"
        f"datasets/{family_name}/items.jsonl"
    )


def _common_code(dataset_path: str, prompt_condition: str) -> str:
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
        N_JOBS = 4
        TIMEOUT_SECONDS = 6000
        JUDGE_LLM = kbench.judge_llm
        df = pd.read_json(DATASET_PATH, lines=True)


        @dataclass
        class JudgeVote:
            label: str
            clarification_quality: float | None
            answer_correct: bool | None
            warning: str | None = None


        @dataclass
        class ParsedStructuredOutput:
            data: dict[str, Any] | None
            warning: str | None = None


        VALID_LABELS = {{"answer", "clarify", "abstain", "challenge", "hedge"}}
        LABEL_PRIORITY = ["challenge", "abstain", "clarify", "hedge", "answer"]
        BASE_SCORES = {{
            "answer": {{"answer": 1.0, "hedge": 0.2, "clarify": 0.0, "abstain": 0.0, "challenge": 0.0}},
            "clarify": {{"clarify": None, "hedge": 0.2, "abstain": 0.4, "answer": 0.0, "challenge": 0.0}},
            "abstain": {{"abstain": 1.0, "hedge": 0.2, "clarify": None, "answer": 0.0, "challenge": 0.0}},
            "challenge": {{"challenge": 1.0, "hedge": 0.2, "clarify": 0.25, "abstain": 0.25, "answer": 0.0}},
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


        def coerce_string(value: Any) -> str | None:
            if value is None:
                return None
            text = str(value).strip()
            return text or None


        def coerce_bool(value: Any) -> bool | None:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized = normalize_text(value)
                if normalized in {"true", "yes", "1"}:
                    return True
                if normalized in {"false", "no", "0"}:
                    return False
            return None


        def response_flags(response_text: str) -> dict[str, bool]:
            text = str(response_text or "")
            lowered = text.lower()
            request_phrases = [
                "could you",
                "can you",
                "please provide",
                "which",
                "what",
                "who",
                "where",
                "when",
            ]
            missing_ref_phrases = [
                "which company",
                "which person",
                "which dataset",
                "which tool",
                "which document",
                "which version",
                "which model",
                "which city",
                "which country",
                "what time",
                "what date",
                "what location",
                "what year",
                "who exactly",
                "what exactly",
                "what do you mean",
            ]
            generic_clarify_phrases = [
                "more details",
                "more information",
                "clarify",
                "specify",
            ]
            refusal_phrases = [
                "cannot answer",
                "can't answer",
                "cannot determine",
                "can't determine",
                "cannot say",
                "can't say",
                "don't know",
                "do not know",
                "need more information",
                "need the name",
                "need to know",
                "insufficient information",
                "not enough information",
            ]
            framework_markers = [
                "here are",
                "most common",
                "main factors",
                "main causes",
                "categories",
                "typically",
                "usually",
                "in general",
                "for example",
            ]
            has_question = "?" in text
            request_pattern = any(phrase in lowered for phrase in request_phrases)
            missing_ref = any(phrase in lowered for phrase in missing_ref_phrases)
            asks_for_info = (has_question or request_pattern) and missing_ref
            generic_clarify = any(phrase in lowered for phrase in generic_clarify_phrases)
            refusal = any(phrase in lowered for phrase in refusal_phrases)
            framework_dump = any(marker in lowered for marker in framework_markers) or text.count("\\n") >= 5
            long_response = len(text) >= 280
            mixed_refusal_dump = refusal and (framework_dump or long_response)
            return {{
                "has_question": has_question,
                "request_pattern": request_pattern,
                "missing_ref": missing_ref,
                "asks_for_info": asks_for_info,
                "generic_clarify": generic_clarify,
                "refusal": refusal,
                "framework_dump": framework_dump,
                "long_response": long_response,
                "mixed_refusal_dump": mixed_refusal_dump,
            }}


        def normalize_behavior_label(predicted_action: str, response_text: str) -> str:
            flags = response_flags(response_text)
            if predicted_action == "clarify" and not flags["asks_for_info"]:
                if flags["refusal"] and not flags["mixed_refusal_dump"]:
                    return "abstain"
                return "hedge"
            if predicted_action == "abstain" and flags["mixed_refusal_dump"]:
                return "hedge"
            return predicted_action


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
            answer_correct = coerce_bool(payload.get("answer_correct"))
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


        def parse_structured_output(raw_text: str, required_fields: list[str], context: str) -> ParsedStructuredOutput:
            try:
                payload = extract_json_object(raw_text)
            except Exception as exc:
                message = f"{{context}} parsing failed: {{exc}}"
                logging.warning(message)
                return ParsedStructuredOutput(data=None, warning=message)
            missing_fields = [field for field in required_fields if field not in payload]
            if missing_fields:
                message = f"{{context}} missing required fields: {{missing_fields}}"
                logging.warning(message)
                return ParsedStructuredOutput(data=payload, warning=message)
            return ParsedStructuredOutput(data=payload, warning=None)


        def parse_prospective_response(raw_text: str):
            parsed = parse_structured_output(raw_text, ["predicted_success", "answer"], "ProspectiveResponse")
            if parsed.warning:
                return None, parsed.warning
            payload = parsed.data or {{}}
            predicted_success = clamp_unit_interval(payload.get("predicted_success"))
            answer = coerce_string(payload.get("answer"))
            if predicted_success is None or answer is None:
                message = "ProspectiveResponse contained invalid predicted_success or answer"
                logging.warning(message)
                return None, message
            return {{"predicted_success": predicted_success, "answer": answer}}, None


        def parse_retrospective_assessment(raw_text: str):
            parsed = parse_structured_output(raw_text, ["confidence"], "RetrospectiveAssessment")
            if parsed.warning:
                return None, parsed.warning
            payload = parsed.data or {{}}
            confidence = clamp_unit_interval(payload.get("confidence"))
            reason = coerce_string(payload.get("reason")) or ""
            if confidence is None:
                message = "RetrospectiveAssessment contained invalid confidence"
                logging.warning(message)
                return None, message
            return {{"confidence": confidence, "reason": reason}}, None


        def parse_revised_answer(raw_text: str):
            parsed = parse_structured_output(raw_text, ["final_answer"], "RevisedAnswer")
            if parsed.warning:
                return None, parsed.warning
            payload = parsed.data or {{}}
            final_answer = coerce_string(payload.get("final_answer"))
            reason = coerce_string(payload.get("reason")) or ""
            if final_answer is None:
                message = "RevisedAnswer contained invalid final_answer"
                logging.warning(message)
                return None, message
            return {{"final_answer": final_answer, "reason": reason}}, None


        def make_unscorable_row(row, model_response=None, warning=None, extras=None):
            payload = {{
                "item_id": row["item_id"],
                "subtype": row["subtype"],
                "expected_action": row["expected_action"],
                "item_score": None,
                "unscorable": True,
                "warning": warning,
                "model_response": model_response,
            }}
            if extras:
                payload.update(extras)
            return payload


        def aggregate_result_score(runs) -> float:
            result_df = pd.json_normalize(runs.as_dataframe()["result"])
            if "unscorable" not in result_df.columns:
                return float(result_df["item_score"].mean())
            scored_df = result_df[~result_df["unscorable"].fillna(False)]
            if scored_df.empty:
                logging.warning("All rows were marked unscorable; returning 0.0 for the aggregate.")
                return 0.0
            return float(scored_df["item_score"].mean())


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
                "- If the model says it cannot answer or lacks enough information but then gives speculative frameworks, categories, or plausible causes, do not label clarify. Prefer hedge, or abstain only if the response stays disciplined and does not speculate.\\n"
                "- Only label clarify when the model actually asks for missing information or disambiguation. Acknowledge-missing-info without a real question is not clarify.\\n\\n"
                "- Do not infer intent. Judge the visible behavior in the response itself.\\n"
                "- Saying 'I need more information' without asking for that information is not clarify.\\n"
                "- A refusal followed by a long generic explanation, taxonomy, or list of possible causes is mixed behavior and should usually be hedge.\\n"
                "- If the model gives both a refusal and a speculative answer, prefer hedge over abstain.\\n"
                "- If the model asks a question but also substantially answers with speculation, prefer hedge over clarify.\\n"
                "- Long polished fallback prose does not earn credit by itself; focus on whether the action is epistemically correct.\\n\\n"
                "Judge guidance for this pass:\\n"
                f"- Variant: {{variant_name}}\\n"
                f"- Guidance: {{variant_guidance}}\\n\\n"
                "Also provide clarification_quality as a float from 0.0 to 1.0:\\n"
                "- Use 1.0 only when the response asks a specific targeted question that would directly unlock the answer.\\n"
                "- Use 0.5 only when the response asks for more information but the request is generic or incomplete.\\n"
                "- Use 0.0 when there is no actual clarification question, when the response only notes uncertainty, or when it mainly speculates.\\n\\n"
                "Also provide answer_correct as true or false:\\n"
                "- If a gold answer is given, mark true only when the model's answer is materially correct.\\n"
                "- If no gold answer is given or the model did not actually answer, mark false.\\n\\n"
                "Return exactly one JSON object and nothing else.\\n"
                "Do not wrap it in markdown fences. Do not add commentary. Start the response with {{ and end with }}.\\n"
                "Use this exact shape:\\n"
                "{{\\\"label\\\":\\\"answer\\\",\\\"clarification_quality\\\":0.0,\\\"answer_correct\\\":false}}\\n\\n"
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


        def compute_boundary_score(expected_action: str, predicted_action: str, clarification_score: float, is_answer_correct, response_text: str):
            flags = response_flags(response_text)
            if expected_action == "answer":
                if predicted_action == "answer":
                    return 1.0 if is_answer_correct else 0.0
                if predicted_action == "hedge":
                    return 0.2 if is_answer_correct else 0.0
                return 0.0

            score = BASE_SCORES[expected_action][predicted_action]
            if expected_action == "clarify" and predicted_action == "clarify":
                if not flags["asks_for_info"]:
                    return 0.0
                effective_quality = clarification_score
                if flags["generic_clarify"]:
                    effective_quality = min(effective_quality, 0.3)
                if flags["long_response"]:
                    effective_quality *= 0.5
                return 0.5 + 0.5 * effective_quality
            if expected_action == "abstain" and predicted_action == "clarify":
                return 0.0
            if expected_action == "abstain" and predicted_action == "abstain" and flags["mixed_refusal_dump"]:
                return 0.2
            return float(score)
        """
    ).strip()


def _boundary_code(spec_name: str, single_name: str, published_name: str) -> str:
    return dedent(
        f"""

        def score_row(row, response_text: str, judged: JudgeVote, votes) -> dict:
            expected_action = row["expected_action"]
            predicted_action = normalize_behavior_label(judged.label, response_text)
            has_gold_answer = bool(row.get("gold_answer"))
            judge_complete = judge_fields_complete(expected_action, judged, has_gold_answer)
            is_answer_correct = judged.answer_correct if has_gold_answer else None
            clarification_score = judged.clarification_quality if judged.clarification_quality is not None else 0.0
            if predicted_action != "clarify":
                clarification_score = 0.0
            item_score = None
            if judge_complete:
                item_score = compute_boundary_score(expected_action, predicted_action, clarification_score, is_answer_correct, response_text)
            return {{
                "item_id": row["item_id"],
                "subtype": row["subtype"],
                "expected_action": expected_action,
                "predicted_action": predicted_action,
                "judge_complete": judge_complete,
                "judge_warning": judged.warning,
                "unscorable": not judge_complete,
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
            return aggregate_result_score(runs)

        preview_df = preview_{spec_name}(kbench.llm, df, limit=5)
        run = score_{spec_name}.run(kbench.llm, df)
        print(run.result)

        %choose score_{spec_name}
        """
    ).strip()


def _prospective_code(spec_name: str, single_name: str, published_name: str) -> str:
    return dedent(
        f"""

        def score_row(row, response: dict, judged: JudgeVote, votes, response_warning=None) -> dict:
            expected_action = row["expected_action"]
            predicted_action = normalize_behavior_label(judged.label, response["answer"])
            predicted_success = max(0.0, min(1.0, float(response["predicted_success"])))
            has_gold_answer = bool(row.get("gold_answer"))
            judge_complete = judge_fields_complete(expected_action, judged, has_gold_answer)
            is_answer_correct = judged.answer_correct if has_gold_answer else None
            clarification_score = judged.clarification_quality if judged.clarification_quality is not None else 0.0
            if predicted_action != "clarify":
                clarification_score = 0.0
            outcome_score = 0.0
            alignment = 0.0
            item_score = None
            unscorable = bool(response_warning) or not judge_complete
            if not unscorable:
                outcome_score = compute_boundary_score(expected_action, predicted_action, clarification_score, is_answer_correct, response["answer"])
                alignment = 1.0 - (predicted_success - outcome_score) ** 2
                item_score = outcome_score * alignment
            return {{
                "item_id": row["item_id"],
                "subtype": row["subtype"],
                "expected_action": expected_action,
                "predicted_action": predicted_action,
                "predicted_success": predicted_success,
                "judge_complete": judge_complete,
                "unscorable": unscorable,
                "warning": response_warning or judged.warning,
                "judge_warning": judged.warning,
                "outcome_score": outcome_score,
                "alignment": alignment,
                "judge_votes": [vote.label for vote in votes],
                "item_score": item_score,
                "model_response": response["answer"],
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
            print(preview_df[["item_id", "subtype", "expected_action", "predicted_action", "unscorable", "warning", "judge_complete", "predicted_success", "outcome_score", "alignment", "item_score", "model_response"]].to_string(index=False))
            return preview_df

        @kbench.task(name="{single_name}", description="Evaluates one prospective monitoring item.", store_task=False)
        def solve_single_item(llm, item_id: str, subtype: str, prompt: str, expected_action: str, gold_answer, required_missing_fields, accepted_diagnoses, difficulty: str, notes) -> dict:
            row = {{"item_id": item_id, "subtype": subtype, "expected_action": expected_action, "gold_answer": gold_answer, "required_missing_fields": required_missing_fields, "accepted_diagnoses": accepted_diagnoses, "difficulty": difficulty, "notes": notes}}
            raw_response = llm.prompt(
                "Before answering, estimate your probability of giving an appropriate and correct response. "
                "Return JSON only with keys predicted_success and answer. "
                "Do not include <think> blocks, hidden reasoning, markdown fences, or any text before or after the JSON.\\n\\n"
                f"Prompt: {{build_prompt(prompt)}}"
            )
            response, response_warning = parse_prospective_response(raw_response)
            if response is None:
                return make_unscorable_row(row, model_response=raw_response, warning=response_warning, extras={{"predicted_action": None}})
            judged, votes = ensemble_classify(prompt, response["answer"], gold_answer)
            return score_row(row, response, judged, votes, response_warning=response_warning)

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
            return aggregate_result_score(runs)

        preview_df = preview_{spec_name}(kbench.llm, df, limit=5)
        run = score_{spec_name}.run(kbench.llm, df)
        print(run.result)

        %choose score_{spec_name}
        """
    ).strip()


def _retrospective_code(spec_name: str, single_name: str, published_name: str) -> str:
    return dedent(
        f"""

        def score_row(row, response_text: str, assessment: dict, judged: JudgeVote, votes, assessment_warning=None) -> dict:
            expected_action = row["expected_action"]
            predicted_action = normalize_behavior_label(judged.label, response_text)
            confidence = max(0.0, min(1.0, float(assessment["confidence"])))
            has_gold_answer = bool(row.get("gold_answer"))
            judge_complete = judge_fields_complete(expected_action, judged, has_gold_answer)
            is_answer_correct = judged.answer_correct if has_gold_answer else None
            clarification_score = judged.clarification_quality if judged.clarification_quality is not None else 0.0
            if predicted_action != "clarify":
                clarification_score = 0.0
            outcome_score = 0.0
            alignment = 0.0
            item_score = None
            unscorable = bool(assessment_warning) or not judge_complete
            if not unscorable:
                outcome_score = compute_boundary_score(expected_action, predicted_action, clarification_score, is_answer_correct, response_text)
                alignment = 1.0 - (confidence - outcome_score) ** 2
                item_score = 0.7 * alignment + 0.3 * outcome_score
            return {{
                "item_id": row["item_id"],
                "subtype": row["subtype"],
                "expected_action": expected_action,
                "predicted_action": predicted_action,
                "confidence": confidence,
                "judge_complete": judge_complete,
                "unscorable": unscorable,
                "warning": assessment_warning or judged.warning,
                "judge_warning": judged.warning,
                "outcome_score": outcome_score,
                "alignment": alignment,
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
            print(preview_df[["item_id", "subtype", "expected_action", "predicted_action", "unscorable", "warning", "judge_complete", "confidence", "outcome_score", "alignment", "item_score", "model_response"]].to_string(index=False))
            return preview_df

        @kbench.task(name="{single_name}", description="Evaluates one retrospective monitoring item.", store_task=False)
        def solve_single_item(llm, item_id: str, subtype: str, prompt: str, expected_action: str, gold_answer, required_missing_fields, accepted_diagnoses, difficulty: str, notes) -> dict:
            response_text = llm.prompt(build_prompt(prompt))
            judged, votes = ensemble_classify(prompt, response_text, gold_answer)
            row = {{"item_id": item_id, "subtype": subtype, "expected_action": expected_action, "gold_answer": gold_answer, "required_missing_fields": required_missing_fields, "accepted_diagnoses": accepted_diagnoses, "difficulty": difficulty, "notes": notes}}
            raw_assessment = llm.prompt(
                "You have already answered a prompt. Assess how likely your answer was appropriate and correct. "
                "Return JSON only with keys confidence and reason. "
                "Do not include <think> blocks, hidden reasoning, markdown fences, or any text before or after the JSON.\\n\\n"
                f"Prompt: {{prompt}}\\n\\n"
                f"Your answer: {{response_text}}"
            )
            assessment, assessment_warning = parse_retrospective_assessment(raw_assessment)
            if assessment is None:
                return make_unscorable_row(row, model_response=response_text, warning=assessment_warning, extras={{"predicted_action": judged.label, "judge_warning": judged.warning}})
            return score_row(row, response_text, assessment, judged, votes, assessment_warning=assessment_warning)

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
            return aggregate_result_score(runs)

        preview_df = preview_{spec_name}(kbench.llm, df, limit=5)
        run = score_{spec_name}.run(kbench.llm, df)
        print(run.result)

        %choose score_{spec_name}
        """
    ).strip()


def _self_correction_code(spec_name: str, single_name: str, published_name: str) -> str:
    return dedent(
        f"""

        def score_row(row, initial_response: str, revised_response: str, initial_judged: JudgeVote, initial_votes, final_judged: JudgeVote, final_votes) -> dict:
            expected_action = row["expected_action"]
            has_gold_answer = bool(row.get("gold_answer"))
            initial_complete = judge_fields_complete(expected_action, initial_judged, has_gold_answer)
            final_complete = judge_fields_complete(expected_action, final_judged, has_gold_answer)
            initial_correct = initial_judged.answer_correct if has_gold_answer else None
            final_correct = final_judged.answer_correct if has_gold_answer else None
            initial_quality = initial_judged.clarification_quality if initial_judged.clarification_quality is not None else 0.0
            final_quality = final_judged.clarification_quality if final_judged.clarification_quality is not None else 0.0
            initial_action = normalize_behavior_label(initial_judged.label, initial_response)
            final_action = normalize_behavior_label(final_judged.label, revised_response)
            if initial_action != "clarify":
                initial_quality = 0.0
            if final_action != "clarify":
                final_quality = 0.0
            initial_score = None
            final_score = None
            item_score = None
            if initial_complete:
                initial_score = compute_boundary_score(expected_action, initial_action, initial_quality, initial_correct, initial_response)
            if final_complete:
                final_score = compute_boundary_score(expected_action, final_action, final_quality, final_correct, revised_response)
            if initial_complete and final_complete:
                delta = final_score - initial_score
                stability_weight = initial_score
                correction_weight = 1.0 - initial_score
                gain = correction_weight * max(0.0, delta) ** 2
                loss = stability_weight * max(0.0, -delta) ** 2
                item_score = final_score + gain - loss
                item_score = max(0.0, min(1.0, item_score))
                if initial_score > 0.8 and final_score < initial_score:
                    item_score = max(0.0, item_score - 0.25)
            return {{
                "item_id": row["item_id"],
                "subtype": row["subtype"],
                "expected_action": expected_action,
                "initial_action": initial_action,
                "final_action": final_action,
                "unscorable": not (initial_complete and final_complete),
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
            print(preview_df[["item_id", "subtype", "expected_action", "initial_action", "final_action", "unscorable", "initial_judge_complete", "final_judge_complete", "initial_score", "final_score", "item_score"]].to_string(index=False))
            return preview_df

        @kbench.task(name="{single_name}", description="Evaluates one self-correction item.", store_task=False)
        def solve_single_item(llm, item_id: str, subtype: str, prompt: str, expected_action: str, gold_answer, required_missing_fields, accepted_diagnoses, difficulty: str, notes) -> dict:
            initial_response = llm.prompt(build_prompt(prompt))
            initial_judged, initial_votes = ensemble_classify(prompt, initial_response, gold_answer)
            row = {{"item_id": item_id, "subtype": subtype, "expected_action": expected_action, "gold_answer": gold_answer, "required_missing_fields": required_missing_fields, "accepted_diagnoses": accepted_diagnoses, "difficulty": difficulty, "notes": notes}}
            raw_revision = llm.prompt(
                "Review your previous answer. If it should be improved, revise it. If it should stay the same, restate the best final answer. "
                "Return EXACTLY one valid JSON object. "
                "It MUST have keys \\\"final_answer\\\" and \\\"reason\\\". "
                "Do NOT output anything else. "
                "Do NOT include markdown, explanations, or extra text. "
                "If you do not follow this format, your answer will be discarded.\\n\\n"
                "Format:\\n"
                "{{\\\"final_answer\\\": \\\"...\\\", \\\"reason\\\": \\\"...\\\"}}\\n\\n"
                "Do not include <think> blocks, hidden reasoning, markdown fences, or any text before or after the JSON.\\n\\n"
                f"Prompt: {{prompt}}\\n\\n"
                f"Previous answer: {{initial_response}}"
            )
            revision, revision_warning = parse_revised_answer(raw_revision)
            if revision is None:
                return make_unscorable_row(row, model_response=initial_response, warning=revision_warning, extras={{"initial_action": initial_judged.label, "final_action": None}})
            final_response = revision["final_answer"]
            final_judged, final_votes = ensemble_classify(prompt, final_response, gold_answer)
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
            return aggregate_result_score(runs)

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
    parts = [_common_code(dataset_path, spec.prompt_condition)]
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
