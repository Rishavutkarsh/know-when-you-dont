from __future__ import annotations

from .schemas import ModelResponse, ResponseAction, RowEvaluationResult, TaskItem


def _diagnosis_matches(item: TaskItem, response: ModelResponse) -> bool | None:
    if not item.accepted_diagnoses:
        return None
    if not response.diagnosis:
        return False
    return response.diagnosis.strip().lower() in {
        diagnosis.strip().lower() for diagnosis in item.accepted_diagnoses
    }


def _clarification_quality(item: TaskItem, response: ModelResponse) -> float:
    if response.action != ResponseAction.CLARIFY or not response.clarification_question:
        return 0.0
    question = response.clarification_question.lower()
    hits = sum(1 for field in item.required_missing_fields if field.lower() in question)
    if not item.required_missing_fields:
        return 1.0
    return hits / len(item.required_missing_fields)


def evaluate_response(item: TaskItem, response: ModelResponse) -> RowEvaluationResult:
    diagnosis_correct = _diagnosis_matches(item, response)
    clarification_quality = _clarification_quality(item, response)
    is_action_correct = response.action == item.expected_action

    is_answer_correct: bool | None = None
    score = 0.0

    if item.expected_action == ResponseAction.ANSWER:
        is_answer_correct = (
            bool(response.answer)
            and bool(item.gold_answer)
            and response.answer.strip().lower() == item.gold_answer.strip().lower()
        )
        score = 1.0 if is_answer_correct else 0.0
    elif item.expected_action == ResponseAction.ABSTAIN:
        score = 1.0 if response.action == ResponseAction.ABSTAIN else 0.0
        if response.action == ResponseAction.CLARIFY:
            score = 0.5
        if response.action == ResponseAction.HEDGE:
            score = 0.5
        if diagnosis_correct is False:
            score = min(score, 0.5)
    elif item.expected_action == ResponseAction.CLARIFY:
        if response.action == ResponseAction.CLARIFY:
            score = 0.5 + 0.5 * clarification_quality
        elif response.action == ResponseAction.ABSTAIN:
            score = 0.25
        elif response.action == ResponseAction.HEDGE:
            score = 0.5
        else:
            score = 0.0
        if diagnosis_correct is False:
            score = min(score, 0.5)
    elif item.expected_action == ResponseAction.CHALLENGE:
        if response.action == ResponseAction.CHALLENGE:
            score = 1.0
        elif response.action == ResponseAction.HEDGE:
            score = 0.5
        elif response.action in (ResponseAction.CLARIFY, ResponseAction.ABSTAIN):
            score = 0.25
        else:
            score = 0.0
        if diagnosis_correct is False:
            score = min(score, 0.5)

    return RowEvaluationResult(
        item_id=item.item_id,
        subtype=item.subtype,
        expected_action=item.expected_action,
        predicted_action=response.action,
        is_action_correct=is_action_correct,
        is_answer_correct=is_answer_correct,
        clarification_quality=clarification_quality,
        diagnosis_correct=diagnosis_correct,
        item_score=score,
    )
