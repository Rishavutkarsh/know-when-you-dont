"""Canonical task-family code for the first Kaggle-published task.

This module mirrors the logic that gets rendered into the Kaggle notebook.
It is intended for local readability and future code generation, not direct
local execution against Kaggle models.
"""

from __future__ import annotations

from pydantic import BaseModel

from know_when_you_dont.scoring import evaluate_response
from know_when_you_dont.schemas import ModelResponse, TaskItem


class KaggleModelResponse(BaseModel):
    action: str
    answer: str | None = None
    confidence: float
    clarification_question: str | None = None
    diagnosis: str | None = None


def score_single_item(item: TaskItem, response: KaggleModelResponse) -> dict:
    normalized = ModelResponse.model_validate(response.model_dump())
    return evaluate_response(item, normalized).model_dump()

