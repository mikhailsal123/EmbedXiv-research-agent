#!/usr/bin/env python3
"""Export kept suggestion cards to a human-scoring CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from judge_candidates import kept_candidates
from suggestion_cards import RELATION_LABELS, build_suggestion_cards


RUBRIC_FIELDS = [
    "run_label",
    "arxiv_id",
    "title",
    "node_path",
    "node_kind",
    "relation",
    "why",
    "specific_relevance_1_5",
    "actionability_1_5",
    "non_redundancy_1_5",
    "notes",
]


def _node_path(card: dict[str, Any]) -> str:
    attachment = card.get("attachment") or {}
    return str(attachment.get("node_path") or card.get("node_path") or "")


def _node_kind(card: dict[str, Any]) -> str:
    attachment = card.get("attachment") or {}
    return str(attachment.get("kind") or card.get("node_kind") or "")


def export_rubric_rows(payload: dict[str, Any]) -> list[dict[str, str]]:
    run_meta = payload.get("run") or {}
    run_label = str(run_meta.get("label") or payload.get("run_label") or "default")
    candidates = list(payload.get("candidates") or [])
    problems = payload.get("problems") or []
    cards = build_suggestion_cards(candidates, problems=problems, kept_only=True)

    rows: list[dict[str, str]] = []
    for card in cards:
        relation = str(card.get("relation") or "")
        rows.append(
            {
                "run_label": run_label,
                "arxiv_id": str(card.get("arxiv_id") or ""),
                "title": str(card.get("title") or ""),
                "node_path": _node_path(card),
                "node_kind": _node_kind(card),
                "relation": RELATION_LABELS.get(relation, relation),
                "why": str(card.get("why") or ""),
                "specific_relevance_1_5": "",
                "actionability_1_5": "",
                "non_redundancy_1_5": "",
                "notes": "",
            }
        )
    return rows


def write_rubric_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUBRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export kept cards from results JSON to a rubric CSV."
    )
    parser.add_argument("results_json", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output CSV path (default: beside JSON as *_rubric.csv)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append rows to an existing rubric CSV instead of overwriting",
    )
    args = parser.parse_args(argv)

    payload = json.loads(args.results_json.read_text())
    rows = export_rubric_rows(payload)
    if not rows:
        kept = kept_candidates(payload.get("candidates") or [])
        print(
            f"No kept cards found in {args.results_json} "
            f"({len(kept)} kept candidates).",
            file=sys.stderr,
        )
        return 1

    output = args.output or args.results_json.with_name(
        args.results_json.stem + "_rubric.csv"
    )
    if args.append and output.exists():
        existing = list(csv.DictReader(output.open(encoding="utf-8")))
        write_rubric_csv(existing + rows, output)
    else:
        write_rubric_csv(rows, output)

    print(f"Wrote {len(rows)} rubric row(s) → {output}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main())
