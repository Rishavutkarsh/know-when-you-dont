from __future__ import annotations

import sys
from collections import Counter

from .family_loader import load_family_spec, load_items


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python -m know_when_you_dont.preview <family_name>")
        return 1

    family_name = sys.argv[1]
    spec = load_family_spec(family_name)
    items = load_items(family_name)
    counts = Counter(item.subtype.value for item in items)

    print(spec.display_name)
    print(spec.description)
    print(f"Published task: {spec.published_task_name}")
    print(f"Notebook: {spec.notebook_name}")
    print(f"Items: {len(items)}")
    for subtype, count in sorted(counts.items()):
        print(f"- {subtype}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

