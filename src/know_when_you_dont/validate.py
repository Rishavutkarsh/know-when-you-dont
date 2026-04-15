from __future__ import annotations

import sys

from .family_loader import load_family_spec, load_items


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python -m know_when_you_dont.validate <family_name>")
        return 1

    family_name = sys.argv[1]
    spec = load_family_spec(family_name)
    items = load_items(family_name)

    allowed_subtypes = set(spec.subtypes)
    item_ids: set[str] = set()
    for item in items:
        if item.subtype not in allowed_subtypes:
            raise ValueError(
                f"Item {item.item_id} uses unsupported subtype {item.subtype.value}"
            )
        if item.item_id in item_ids:
            raise ValueError(f"Duplicate item_id found: {item.item_id}")
        item_ids.add(item.item_id)

    print(
        f"Validated family '{spec.family_name}' with {len(items)} items and "
        f"{len(spec.subtypes)} supported subtypes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

