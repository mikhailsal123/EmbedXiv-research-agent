"""Two-stage Qwen judge for FAISS candidates.

1) Cheap screen on title+abstract (batched): drop obvious junk, or mark read_full.
2) For survivors: fetch the arXiv PDF, then one LLM call per paper with full text
   → keep/drop + relation + why.
"""

from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Literal

import requests
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, field_validator

from extract_claims import ResearchProblem, get_client, load_local_env

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None


load_local_env()

DEFAULT_JUDGE_MODEL = os.getenv("JUDGE_MODEL", os.getenv("EXTRACTION_MODEL", "qwen3:32b"))
DEFAULT_SCREEN_BATCH_SIZE = int(os.getenv("JUDGE_SCREEN_BATCH_SIZE", "8"))
DEFAULT_PDF_MAX_CHARS = int(os.getenv("JUDGE_PDF_MAX_CHARS", "60000"))
DEFAULT_ARXIV_DELAY = float(os.getenv("ARXIV_REQUEST_DELAY", "1.0"))
ARXIV_USER_AGENT = os.getenv(
    "ARXIV_USER_AGENT",
    "EmbedXivResearchAgent/0.1 (mailto:local-dev@example.com)",
)


def _progress(
    iterable: Iterable[Any] | None = None,
    *,
    total: int | None = None,
    desc: str = "",
    unit: str = "it",
):
    """tqdm progress bar when available; otherwise a plain iterable."""
    if _tqdm is not None:
        kwargs = {
            "total": total,
            "desc": desc,
            "unit": unit,
            "file": sys.stdout,
            "mininterval": 0.5,
            "dynamic_ncols": True,
        }
        if iterable is None:
            return _tqdm(**kwargs)
        return _tqdm(iterable, **kwargs)
    if iterable is None:
        class _NoBar:
            def update(self, n: int = 1) -> None:
                return None

            def set_postfix_str(self, _: str, refresh: bool = True) -> None:
                return None

            def close(self) -> None:
                return None

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

        return _NoBar()
    return iterable


Relation = Literal[
    "same_problem",
    "claim_support",
    "claim_extension",
    "claim_qualification",
    "claim_contradiction",
    "implementation_alternative",
    "irrelevant",
]
Decision = Literal["keep", "drop"]
ScreenDecision = Literal["drop", "read_full"]
PrimaryLevel = Literal["problem", "claim", "implementation"]


class SchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ScreenJudgment(SchemaModel):
    arxiv_id: str = Field(min_length=1)
    decision: ScreenDecision
    why: str = Field(min_length=1, max_length=300)

    @field_validator("arxiv_id")
    @classmethod
    def normalize_id(cls, value: str) -> str:
        return value.strip()


class ScreenBatchResult(SchemaModel):
    judgments: list[ScreenJudgment] = Field(min_length=1)


class CandidateJudgment(SchemaModel):
    arxiv_id: str = Field(min_length=1)
    decision: Decision
    relation: Relation
    why: str = Field(min_length=1, max_length=400)
    primary_level: PrimaryLevel = "problem"
    stage: Literal["screen", "full_text"] = "full_text"

    @field_validator("arxiv_id")
    @classmethod
    def normalize_id(cls, value: str) -> str:
        return value.strip()


SCREEN_SYSTEM_PROMPT = """You are a fast triage judge for EmbedXiv related-work retrieval.

You see only title + abstract (plus which search queries matched). Do NOT invent
details that are not in the abstract.

For EACH candidate choose:
- drop: clearly irrelevant, wrong area, or only shared buzzwords
- read_full: plausible enough that reading the PDF is worth it

Be aggressive about dropping weak matches. Prefer read_full when the abstract
credibly engages the same problem, claim, or mechanism role.

Return one judgment per provided arxiv_id. Do not invent or omit ids.
"""


FULL_SYSTEM_PROMPT = """You are a careful research literature judge for EmbedXiv.

You are given:
1. The source paper's extracted structure: problem → claims → implementation details.
2. ONE arXiv candidate: metadata, which queries matched it, and its full paper text
   (possibly truncated).

Decide:
- decision: keep or drop
- relation (single best):
  same_problem | claim_support | claim_extension | claim_qualification |
  claim_contradiction | implementation_alternative | irrelevant
- primary_level: problem | claim | implementation
- why: one concrete sentence

Rules:
- Use the full text; do not rely on the abstract alone when they conflict.
- irrelevant must use decision=drop.
- Prefer drop when the connection is superficial.
"""


def _truncate(text: str, limit: int = 1200) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _extraction_context(problems: list[ResearchProblem]) -> str:
    lines: list[str] = ["SOURCE EXTRACTION"]
    for pi, problem in enumerate(problems):
        lines.append(f"Problem[{pi}]: {problem.problem}")
        lines.append(f"  domain: {problem.domain}")
        lines.append(f"  keywords: {', '.join(problem.keywords)}")
        for ci, claim in enumerate(problem.claims):
            lines.append(f"  Claim[{pi}.{ci}]: {claim.claim}")
            lines.append(f"    functional_role: {claim.functional_role}")
            for di, detail in enumerate(claim.implementation_details):
                lines.append(f"    Detail[{pi}.{ci}.{di}]: {detail.detail}")
                lines.append(f"      functional_role: {detail.functional_role}")
    return "\n".join(lines)


def _match_summary(candidate: dict[str, Any]) -> str:
    matches = candidate.get("matched_queries") or []
    bits = []
    for match in matches[:6]:
        level = match.get("level", "?")
        qtype = match.get("query_type", "?")
        source = _truncate(str(match.get("source_text", "")), 160)
        bits.append(f"{level}/{qtype}: {source}")
    return "; ".join(bits) if bits else "(none)"


def _candidate_abstract_block(candidate: dict[str, Any]) -> str:
    return (
        f"arxiv_id: {candidate.get('arxiv_id', '')}\n"
        f"title: {_truncate(str(candidate.get('title', '')), 300)}\n"
        f"abstract: {_truncate(str(candidate.get('abstract', '')), 1200)}\n"
        f"best_distance: {candidate.get('best_distance', '')}\n"
        f"matched: {_match_summary(candidate)}"
    )


def arxiv_pdf_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def fetch_arxiv_pdf_text(
    arxiv_id: str,
    *,
    session: requests.Session | None = None,
    max_chars: int = DEFAULT_PDF_MAX_CHARS,
    timeout: float = 60.0,
) -> str:
    """Download an arXiv PDF and extract text (truncated)."""
    http = session or requests.Session()
    response = http.get(
        arxiv_pdf_url(arxiv_id),
        headers={"User-Agent": ARXIV_USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    # write to a temp-like buffer path for pypdf via BytesIO workaround:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(response.content))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    text = text.strip()
    if not text:
        raise RuntimeError(f"No extractable text in PDF for {arxiv_id}")
    return text[:max_chars]


def screen_batch(
    problems: list[ResearchProblem],
    batch: list[dict[str, Any]],
    *,
    client: OpenAI | None = None,
    model: str = DEFAULT_JUDGE_MODEL,
) -> list[ScreenJudgment]:
    if not problems:
        raise ValueError("problems must not be empty")
    if not batch:
        return []

    user_parts = [_extraction_context(problems), "", "CANDIDATES"]
    for index, candidate in enumerate(batch):
        user_parts.append(f"--- candidate {index + 1} ---")
        user_parts.append(_candidate_abstract_block(candidate))
    user_parts.append("")
    user_parts.append("Return one screen judgment per arxiv_id above.")

    completion = (client or get_client()).beta.chat.completions.parse(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SCREEN_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(user_parts)},
        ],
        response_format=ScreenBatchResult,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        raise RuntimeError("The screen model returned no structured result")

    by_id = {item.arxiv_id: item for item in result.judgments}
    aligned: list[ScreenJudgment] = []
    for candidate in batch:
        arxiv_id = str(candidate.get("arxiv_id", "")).strip()
        if not arxiv_id:
            continue
        judgment = by_id.get(arxiv_id)
        if judgment is None:
            aligned.append(
                ScreenJudgment(
                    arxiv_id=arxiv_id,
                    decision="drop",
                    why="Model omitted this candidate; treated as drop.",
                )
            )
        else:
            aligned.append(judgment)
    return aligned


def judge_full_paper(
    problems: list[ResearchProblem],
    candidate: dict[str, Any],
    paper_text: str,
    *,
    client: OpenAI | None = None,
    model: str = DEFAULT_JUDGE_MODEL,
) -> CandidateJudgment:
    arxiv_id = str(candidate.get("arxiv_id", "")).strip()
    if not arxiv_id:
        raise ValueError("candidate is missing arxiv_id")
    if not paper_text.strip():
        raise ValueError("paper_text must not be empty")

    user = "\n".join(
        [
            _extraction_context(problems),
            "",
            "CANDIDATE METADATA",
            _candidate_abstract_block(candidate),
            "",
            "CANDIDATE FULL TEXT",
            paper_text,
            "",
            f"Return a single judgment for arxiv_id={arxiv_id}.",
        ]
    )
    completion = (client or get_client()).beta.chat.completions.parse(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": FULL_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        response_format=CandidateJudgment,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        raise RuntimeError("The full-text judge returned no structured result")
    if result.arxiv_id != arxiv_id:
        result = result.model_copy(update={"arxiv_id": arxiv_id})
    if result.relation == "irrelevant":
        result = result.model_copy(update={"decision": "drop"})
    return result.model_copy(update={"stage": "full_text"})


def judge_candidates(
    problems: list[ResearchProblem],
    candidates: list[dict[str, Any]],
    *,
    client: OpenAI | None = None,
    model: str = DEFAULT_JUDGE_MODEL,
    batch_size: int = DEFAULT_SCREEN_BATCH_SIZE,
    fetch_pdfs: bool = True,
    session: requests.Session | None = None,
    request_delay: float = DEFAULT_ARXIV_DELAY,
    pdf_max_chars: int = DEFAULT_PDF_MAX_CHARS,
    pdf_text_by_id: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Screen on abstracts, then full-text judge survivors one paper at a time."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not problems:
        raise ValueError("problems must not be empty")

    screens: dict[str, ScreenJudgment] = {}
    total = len(candidates)
    batch_starts = list(range(0, total, batch_size))
    for start in _progress(
        batch_starts,
        total=len(batch_starts),
        desc="Screen",
        unit="batch",
    ):
        batch = candidates[start : start + batch_size]
        for judgment in screen_batch(problems, batch, client=client, model=model):
            screens[judgment.arxiv_id] = judgment

    read_full_count = 0
    for candidate in candidates:
        screen = screens.get(str(candidate.get("arxiv_id", "")).strip())
        if screen is not None and screen.decision == "read_full":
            read_full_count += 1
    print(
        f"Screen done: {read_full_count} read_full, "
        f"{total - read_full_count} dropped.",
        flush=True,
    )

    http = session or requests.Session()
    pdf_cache = dict(pdf_text_by_id or {})
    output: list[dict[str, Any]] = []
    full_bar = _progress(total=read_full_count, desc="Full-text", unit="paper")
    try:
        for candidate in candidates:
            item = dict(candidate)
            arxiv_id = str(item.get("arxiv_id", "")).strip()
            screen = screens.get(arxiv_id)
            if screen is None:
                screen = ScreenJudgment(
                    arxiv_id=arxiv_id or "unknown",
                    decision="drop",
                    why="Missing screen judgment.",
                )
            item["screen"] = screen.model_dump()

            if screen.decision == "drop":
                item["judgment"] = CandidateJudgment(
                    arxiv_id=screen.arxiv_id,
                    decision="drop",
                    relation="irrelevant",
                    why=screen.why,
                    primary_level="problem",
                    stage="screen",
                ).model_dump()
                output.append(item)
                continue

            if not fetch_pdfs and arxiv_id not in pdf_cache:
                item["judgment"] = CandidateJudgment(
                    arxiv_id=arxiv_id,
                    decision="drop",
                    relation="irrelevant",
                    why="Marked read_full but PDF fetching is disabled.",
                    primary_level="problem",
                    stage="screen",
                ).model_dump()
                output.append(item)
                continue

            if hasattr(full_bar, "set_postfix_str"):
                full_bar.set_postfix_str(arxiv_id)
            try:
                if arxiv_id not in pdf_cache:
                    if request_delay > 0:
                        time.sleep(request_delay)
                    pdf_cache[arxiv_id] = fetch_arxiv_pdf_text(
                        arxiv_id, session=http, max_chars=pdf_max_chars
                    )
                paper_text = pdf_cache[arxiv_id]
                judgment = judge_full_paper(
                    problems, item, paper_text, client=client, model=model
                )
            except Exception as exc:
                judgment = CandidateJudgment(
                    arxiv_id=arxiv_id,
                    decision="drop",
                    relation="irrelevant",
                    why=f"Full-text judge failed: {exc}",
                    primary_level="problem",
                    stage="full_text",
                )
            item["judgment"] = judgment.model_dump()
            output.append(item)
            full_bar.update(1)
    finally:
        full_bar.close()

    return output


def kept_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        candidate
        for candidate in candidates
        if (candidate.get("judgment") or {}).get("decision") == "keep"
    ]


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Two-stage judge: abstract screen, then full-text per paper."
    )
    parser.add_argument("results_json", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_SCREEN_BATCH_SIZE)
    parser.add_argument("--kept-only", action="store_true")
    parser.add_argument("--no-pdf", action="store_true", help="Screen only; no PDF fetch")
    args = parser.parse_args()

    payload = json.loads(args.results_json.read_text())
    problems = [
        ResearchProblem.model_validate(problem)
        for problem in payload.get("problems", [])
    ]
    candidates = list(payload.get("candidates", []))
    judged = judge_candidates(
        problems,
        candidates,
        batch_size=args.batch_size,
        fetch_pdfs=not args.no_pdf,
    )
    if args.kept_only:
        judged = kept_candidates(judged)

    output_path = args.output or args.results_json.with_name(
        args.results_json.stem + "_judged.json"
    )
    kept = sum(
        1
        for candidate in judged
        if (candidate.get("judgment") or {}).get("decision") == "keep"
    )
    payload = {**payload, "candidates": judged, "kept_count": kept}
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    screened_in = sum(
        1
        for candidate in judged
        if (candidate.get("screen") or {}).get("decision") == "read_full"
    )
    print(
        f"Judged {len(judged)} candidates "
        f"({screened_in} read_full, {kept} kept) → {output_path}"
    )


if __name__ == "__main__":
    main()
