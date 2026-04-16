from __future__ import annotations

from pathlib import Path

import yaml

from .schemas import TaskFamilySpec, TaskItem


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def family_dir(family_name: str) -> Path:
    return repo_root() / "task_families" / family_name


def load_family_spec(family_name: str) -> TaskFamilySpec:
    path = family_dir(family_name) / "family.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TaskFamilySpec.model_validate(data)


def load_items(family_name: str) -> list[TaskItem]:
    family_path = family_dir(family_name)
    current_path = family_path / "items_current.yaml"
    default_path = family_path / "items.yaml"
    path = current_path if current_path.exists() else default_path
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [TaskItem.model_validate(item) for item in data["items"]]
