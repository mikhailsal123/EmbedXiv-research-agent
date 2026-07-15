#!/usr/bin/env python3
"""Print funnel tables and ablation comparisons from pipeline JSON dumps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.metrics import FunnelMetrics, load_funnel


def _pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.0f}%"


def _seconds(value: float | None) -> str:
    if value is None:
        return "—"
    minutes, secs = divmod(int(value), 60)
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def print_funnel(metrics: FunnelMetrics) -> None:
    print(f"\n=== {metrics.run_label} ({Path(metrics.source).name}) ===")
    print(f"Runtime: {_seconds(metrics.elapsed_seconds)}")
    print()
    print("| Stage | Count | Pass rate |")
    print("| --- | ---: | ---: |")
    print(f"| Retrieved candidates | {metrics.total_candidates} | — |")
    print(
        f"| Abstract screen → read_full | {metrics.screen_read_full} | "
        f"{_pct(metrics.screen_pass_rate)} |"
    )
    print(f"| Abstract screen → drop | {metrics.screen_drop} | — |")
    print(
        f"| Full-text judge → keep (pre-cap) | {metrics.pre_cap_kept} | "
        f"{_pct(metrics.full_text_keep_rate)} |"
    )
    print(f"| Full-text judge → drop | {metrics.full_text_drop} | — |")
    print(
        f"| Per-node cap → final kept | {metrics.final_kept} | "
        f"{_pct(metrics.cap_pass_rate)} |"
    )
    print(f"| Cap demotions | {metrics.cap_drop} | — |")
    print()
    print(
        f"Source contamination: {'YES (fix exclusion)' if metrics.source_contamination else 'no'}"
    )
    print(f"Kept from agentic refinement only: {metrics.refinement_kept}")


def print_comparison(rows: list[FunnelMetrics]) -> None:
    baseline = rows[0]
    print("\n=== Ablation comparison (final kept) ===")
    print("| Variant | Final kept | Δ vs baseline | Runtime |")
    print("| --- | ---: | ---: | ---: |")
    for row in rows:
        delta = row.final_kept - baseline.final_kept
        delta_text = "—" if row is baseline else f"{delta:+d}"
        print(
            f"| {row.run_label} | {row.final_kept} | {delta_text} | "
            f"{_seconds(row.elapsed_seconds)} |"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize EmbedXiv funnel metrics from results JSON."
    )
    parser.add_argument(
        "results_json",
        nargs="+",
        type=Path,
        help="One or more full_run_results.json files",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print ablation table when multiple runs are provided",
    )
    args = parser.parse_args(argv)

    metrics = [load_funnel(path) for path in args.results_json]
    for item in metrics:
        print_funnel(item)

    if args.compare and len(metrics) > 1:
        ordered = sorted(
            metrics,
            key=lambda item: (
                0 if item.run_label in ("full", "default") else 1,
                item.run_label,
            ),
        )
        print_comparison(ordered)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
