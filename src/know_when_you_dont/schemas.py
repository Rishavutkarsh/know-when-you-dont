from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ResponseAction(str, Enum):
    ANSWER = "answer"
    ABSTAIN = "abstain"
    CLARIFY = "clarify"
    CHALLENGE = "challenge"
    HEDGE = "hedge"


class TaskSubtype(str, Enum):
    AMBIGUITY = "ambiguity"
    MISSING_INFORMATION = "missing_information"
    CLARIFICATION_NEEDED = "clarification_needed"
    FALSE_PREMISE = "false_premise"
    SUFFICIENT_REQUEST = "sufficient_request"
    CLARIFY_AMBIGUITY = "clarify_ambiguity"
    CLARIFY_MISSING_DETAIL = "clarify_missing_detail"
    ABSTAIN_UNDERDETERMINED = "abstain_underdetermined"
    CHALLENGE_FALSE_PREMISE = "challenge_false_premise"
    ANSWER_SAFE_CONTROL = "answer_safe_control"


class ModelResponse(BaseModel):
    action: ResponseAction
    answer: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    clarification_question: str | None = None
    diagnosis: str | None = None

    @model_validator(mode="after")
    def validate_action_fields(self) -> "ModelResponse":
        if self.action == ResponseAction.ANSWER and not self.answer:
            raise ValueError("answer is required when action='answer'")
        if self.action == ResponseAction.CLARIFY and not self.clarification_question:
            raise ValueError(
                "clarification_question is required when action='clarify'"
            )
        return self


class TaskItem(BaseModel):
    item_id: str
    subtype: TaskSubtype
    prompt: str
    expected_action: ResponseAction
    gold_answer: str | None = None
    required_missing_fields: list[str] = Field(default_factory=list)
    accepted_diagnoses: list[str] = Field(default_factory=list)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    notes: str | None = None

    @field_validator("item_id")
    @classmethod
    def item_id_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("item_id must not be blank")
        return value

    @model_validator(mode="after")
    def validate_item_contract(self) -> "TaskItem":
        if self.expected_action == ResponseAction.ANSWER and not self.gold_answer:
            raise ValueError("gold_answer is required when expected_action='answer'")
        if (
            self.subtype == TaskSubtype.CLARIFICATION_NEEDED
            and not self.required_missing_fields
        ):
            raise ValueError(
                "clarification_needed items must define required_missing_fields"
            )
        return self


class TaskFamilySpec(BaseModel):
    family_name: str
    display_name: str
    description: str
    published_task_name: str
    notebook_name: str
    subtypes: list[TaskSubtype]
    primary_metric: str


class RowEvaluationResult(BaseModel):
    item_id: str
    subtype: TaskSubtype
    expected_action: ResponseAction
    predicted_action: ResponseAction
    is_action_correct: bool
    is_answer_correct: bool | None = None
    clarification_quality: float = Field(ge=0.0, le=1.0, default=0.0)
    diagnosis_correct: bool | None = None
    item_score: float = Field(ge=0.0, le=1.0)
