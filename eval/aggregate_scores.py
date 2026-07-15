#!/usr/bin/env python3
"""Aggregate human rubric scores from a filled CSV."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCORE_COLUMNS = (
    "specific_relevance_1_5",
    "actionability_1_5",
    "non_redundancy_1_5",
)
LEGACY_RELEVANCE_COLUMN = "node_fit_1_5"


def _relevance_value(row: dict[str, str]) -> str:
    return row.get("specific_relevance_1_5") or row.get(LEGACY_RELEVANCE_COLUMN) or ""


def _parse_score(value: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    score = float(text)
    if score < 1 or score > 5:
        raise ValueError(f"Score out of range 1–5: {score!r}")
    return score


def aggregate_rubric(path: Path) -> dict[str, object]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    scored_rows: list[dict[str, object]] = []
    by_run: dict[str, list[dict[str, object]]] = defaultdict(list)

    for row in rows:
        scores = {
            "specific_relevance_1_5": _parse_score(_relevance_value(row)),
            "actionability_1_5": _parse_score(row.get("actionability_1_5", "")),
            "non_redundancy_1_5": _parse_score(row.get("non_redundancy_1_5", "")),
        }
        if any(value is None for value in scores.values()):
            continue
        overall = statistics.mean(scores.values())
        high_quality = (
            scores["specific_relevance_1_5"] >= 4.0
            and scores["actionability_1_5"] >= 4.0
        )
        item = {
            "run_label": row.get("run_label", ""),
            "arxiv_id": row.get("arxiv_id", ""),
            "node_path": row.get("node_path", ""),
            "scores": scores,
            "overall": overall,
            "high_quality": high_quality,
            "redundant": scores["non_redundancy_1_5"] <= 2,
        }
        scored_rows.append(item)
        by_run[str(row.get("run_label") or "default")].append(item)

    if not scored_rows:
        raise ValueError(f"No fully scored rows found in {path}")

    high_quality_rate = sum(1 for row in scored_rows if row["high_quality"]) / len(
        scored_rows
    )
    redundancy_rate = sum(1 for row in scored_rows if row["redundant"]) / len(
        scored_rows
    )

    column_means = {
        column: statistics.mean(row["scores"][column] for row in scored_rows)
        for column in SCORE_COLUMNS
    }

    per_run = {}
    for label, items in by_run.items():
        per_run[label] = {
            "n": len(items),
            "high_quality_rate": sum(1 for row in items if row["high_quality"])
            / len(items),
            "mean_overall": statistics.mean(row["overall"] for row in items),
        }

    return {
        "n": len(scored_rows),
        "high_quality_rate": high_quality_rate,
        "redundancy_rate": redundancy_rate,
        "column_means": column_means,
        "mean_overall": statistics.mean(row["overall"] for row in scored_rows),
        "per_run": per_run,
    }


def print_summary(stats: dict[str, object]) -> None:
    means = stats["column_means"]
    print("\n=== Human rubric summary ===")
    print(f"Rated suggestions (n): {stats['n']}")
    print(
        "High-quality rate (relevance ≥ 4 and actionability ≥ 4): "
        f"{stats['high_quality_rate'] * 100:.0f}%"
    )
    print(f"Redundancy rate (non-redundancy ≤ 2/5): {stats['redundancy_rate'] * 100:.0f}%")
    print()
    print("| Criterion | Mean (1–5) |")
    print("| --- | ---: |")
    print(f"| Specific relevance | {means['specific_relevance_1_5']:.2f} |")
    print(f"| Actionability | {means['actionability_1_5']:.2f} |")
    print(f"| Non-redundancy | {means['non_redundancy_1_5']:.2f} |")
    print(f"| Overall (mean of three) | {stats['mean_overall']:.2f} |")

    per_run = stats["per_run"]
    if len(per_run) > 1:
        print()
        print("| Run | n | High-quality rate | Mean overall |")
        print("| --- | ---: | ---: | ---: |")
        for label, row in sorted(per_run.items()):
            print(
                f"| {label} | {row['n']} | {row['high_quality_rate'] * 100:.0f}% | "
                f"{row['mean_overall']:.2f} |"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate human rubric scores from a filled CSV."
    )
    parser.add_argument("rubric_csv", type=Path)
    args = parser.parse_args(argv)

    stats = aggregate_rubric(args.rubric_csv)
    print_summary(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
