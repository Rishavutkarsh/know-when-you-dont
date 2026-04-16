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
    spec = load_family_spec(family_name)
    source_family = spec.item_source_family or family_name
    source_path = family_dir(source_family)
    if spec.item_source_file:
        path = source_path / spec.item_source_file
    else:
        current_path = source_path / "items_current.yaml"
        default_path = source_path / "items.yaml"
        path = current_path if current_path.exists() else default_path
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [TaskItem.model_validate(item) for item in data["items"]]
