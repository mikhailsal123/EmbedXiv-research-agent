"""Funnel and sanity metrics from EmbedXiv pipeline JSON dumps."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FunnelMetrics:
    source: str
    run_label: str
    total_candidates: int
    screen_read_full: int
    screen_drop: int
    full_text_keep: int
    full_text_drop: int
    cap_drop: int
    final_kept: int
    source_contamination: bool
    refinement_kept: int
    elapsed_seconds: float | None

    @property
    def pre_cap_kept(self) -> int:
        return self.full_text_keep + self.cap_drop

    @property
    def screen_pass_rate(self) -> float | None:
        if self.total_candidates == 0:
            return None
        return self.screen_read_full / self.total_candidates

    @property
    def full_text_keep_rate(self) -> float | None:
        screened = self.screen_read_full
        if screened == 0:
            return None
        return self.pre_cap_kept / screened

    @property
    def cap_pass_rate(self) -> float | None:
        if self.pre_cap_kept == 0:
            return None
        return self.final_kept / self.pre_cap_kept


def _judgment(candidate: dict[str, Any]) -> dict[str, Any]:
    return candidate.get("judgment") or {}


def _screen(candidate: dict[str, Any]) -> dict[str, Any]:
    return candidate.get("screen") or {}


def compute_funnel(payload: dict[str, Any]) -> FunnelMetrics:
    candidates = list(payload.get("candidates") or [])
    source_ids = {
        str(arxiv_id).strip()
        for arxiv_id in payload.get("source_arxiv_ids") or []
        if str(arxiv_id).strip()
    }

    screen_read_full = 0
    screen_drop = 0
    full_text_keep = 0
    full_text_drop = 0
    cap_drop = 0
    final_kept = 0
    refinement_kept = 0

    for candidate in candidates:
        screen = _screen(candidate)
        judgment = _judgment(candidate)
        stage = str(judgment.get("stage") or "")
        decision = str(judgment.get("decision") or "")

        if screen.get("decision") == "read_full":
            screen_read_full += 1
        elif screen.get("decision") == "drop" or (
            stage == "screen" and decision == "drop"
        ):
            screen_drop += 1

        if stage == "full_text":
            if decision == "keep":
                full_text_keep += 1
            elif decision == "drop":
                full_text_drop += 1
        elif stage == "cap" and decision == "drop":
            cap_drop += 1

        if decision == "keep":
            final_kept += 1
            if candidate.get("retrieval_source") == "search_refinement":
                refinement_kept += 1

    kept_ids = {
        str(candidate.get("arxiv_id", "")).strip()
        for candidate in candidates
        if _judgment(candidate).get("decision") == "keep"
    }
    contamination = bool(source_ids & kept_ids)

    run_meta = payload.get("run") or {}
    return FunnelMetrics(
        source=str(payload.get("source") or ""),
        run_label=str(run_meta.get("label") or payload.get("run_label") or "default"),
        total_candidates=len(candidates),
        screen_read_full=screen_read_full,
        screen_drop=screen_drop,
        full_text_keep=full_text_keep,
        full_text_drop=full_text_drop,
        cap_drop=cap_drop,
        final_kept=final_kept,
        source_contamination=contamination,
        refinement_kept=refinement_kept,
        elapsed_seconds=run_meta.get("elapsed_seconds"),
    )


def load_run(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_funnel(path: Path) -> FunnelMetrics:
    return compute_funnel(load_run(path))
