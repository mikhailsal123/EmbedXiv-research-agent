"""Two-stage Qwen judge for vector-search candidates.

1) Cheap screen on title+abstract grouped by retrieval query: drop obvious junk,
   or mark read_full.
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
DEFAULT_PDF_MAX_CHARS = int(os.getenv("JUDGE_PDF_MAX_CHARS", "60000"))
DEFAULT_ARXIV_DELAY = float(os.getenv("ARXIV_REQUEST_DELAY", "1.0"))
DEFAULT_MAX_KEPT_PER_NODE = int(os.getenv("JUDGE_MAX_KEPT_PER_NODE", "5"))
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


class ScreenQueryResult(SchemaModel):
    judgments: list[ScreenJudgment] = Field(min_length=1)


class NodeTopSelection(SchemaModel):
    selected_arxiv_ids: list[str] = Field(default_factory=list, max_length=5)


class CandidateJudgment(SchemaModel):
    arxiv_id: str = Field(min_length=1)
    decision: Decision
    relation: Relation
    why: str = Field(min_length=1, max_length=400)
    primary_level: PrimaryLevel = "problem"
    stage: Literal["screen", "full_text", "cap"] = "full_text"

    @field_validator("arxiv_id")
    @classmethod
    def normalize_id(cls, value: str) -> str:
        return value.strip()


SCREEN_SYSTEM_PROMPT = """You run a first-pass filter on candidate papers to find
research that could help improve, refine, support, extend, qualify, or challenge
the author's work—not merely papers in the same field.

A source paper has been decomposed into: a research problem, conceptual claims
that address it, and concrete implementation details under each claim. Vector
search returned candidates for ONE source query at a time. You see the source
query, its source node and functional role when available, and the full list of
candidates returned for that query with rank/distance.

Your job: decide whether the full PDF might contain ideas, evidence, methods, or
critiques the author could actually use when revising or strengthening their
paper.

Be selective. Most vector-search hits are topical neighbors, not usable research
for this author. Prefer drop unless the abstract clearly suggests a concrete
payoff.

For EACH candidate choose exactly one:
- drop: wrong area, only shared buzzwords/keywords, vague topical overlap, or no
  credible way this paper could inform the source problem, claims, or mechanisms
- read_full: the abstract suggests a specific, usable connection—same gap with a
  distinctive take, evidence for/against a named claim, or an alternative
  mechanism for a similar role that the author could learn from

Rules:
- Do not invent details absent from the title/abstract.
- Prefer drop when the match looks lexical or topical only (same field, same
  architecture family, shared dataset/benchmark, different question).
- Prefer drop when you cannot name what the author would take from this paper.
- Prefer read_full only when the abstract suggests usable insight beyond "also
  does attention / CNN / transformers / vision."
- Return one judgment per provided arxiv_id. Do not invent or omit ids.
"""


FULL_SYSTEM_PROMPT = """You judge whether ONE candidate paper is useful for the
author's research—not just related by topic, but something they could apply to
improve, refine, support, extend, qualify, or challenge their paper.

You are given:
1. The source paper's extracted structure: problem → claims → implementation
   details (with functional roles).
2. One arXiv candidate: metadata, matched queries, and its full text (may be
   truncated).

Keep should be rare. Keep only if you can name a concrete takeaway for the
author (a method to try, evidence to cite, caveat to address, competing
approach to compare, or mechanism to borrow). If the paper is merely in the
same neighborhood (attention, CNNs, vision, similar datasets) without that
payoff, drop it.

Decide:
- decision: keep or drop
- relation (single best label):
  same_problem — addresses the same research gap; helps situate or sharpen the
    source problem
  claim_support — provides evidence or argument that backs a source claim
  claim_extension — builds on or generalizes a source claim the author could adopt
  claim_qualification — limits or conditions a source claim the author should heed
  claim_contradiction — challenges a source claim the author must address
  implementation_alternative — different mechanism for a similar functional role
    the author could compare or borrow from
  irrelevant — no meaningful way to use this paper
- primary_level: problem | claim | implementation (where the usefulness is strongest)
- why: one concrete sentence on what the author could take from this paper

Rules:
- Use the full text; if abstract and body conflict, trust the body.
- For claim_* relations: the paper must engage that specific claim, not just the
  broad problem area.
- For implementation_alternative: the candidate must serve a similar functional
  role with a meaningfully different mechanism—not a near-duplicate or rename.
- Keep only when the author would change how they write, experiment, or argue
  after reading this paper.
- Drop when the overlap is only shared field, architecture family, dataset, or
  keywords.
- Drop near-duplicates of the source idea that add little the author does not
  already claim.
- relation=irrelevant requires decision=drop.
- Prefer drop on borderline matches.
"""


SELECT_SYSTEM_PROMPT = """You choose the most useful papers for ONE source node in
the author's paper hierarchy (problem, claim, or implementation detail).

You are given papers that already passed full-text judging for this node. Pick the
0-5 papers the author should actually read. Prefer distinct, high-value takeaways
over redundant near-duplicates.

Rules:
- Return at most 5 arxiv_ids.
- It is valid to return fewer than 5, or none, if the survivors are weak or
  redundant.
- Do not invent ids.
- Prefer papers with concrete, non-overlapping usefulness for this node.
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


def _screen_query_block(match: dict[str, Any]) -> str:
    if match.get("query_type") == "unmatched":
        return "SOURCE QUERY\n(no matched query provenance)"
    lines = [
        "SOURCE QUERY",
        f"level: {match.get('level', '')}",
        f"query_type: {match.get('query_type', '')}",
        f"query: {_truncate(str(match.get('query', '')), 500)}",
        f"source_text: {_truncate(str(match.get('source_text', '')), 500)}",
    ]
    if match.get("functional_role"):
        lines.append(
            f"functional_role: {_truncate(str(match.get('functional_role')), 500)}"
        )
    for key in (
        "problem_index",
        "claim_index",
        "detail_index",
        "target_id",
        "refinement_round",
    ):
        if match.get(key) is not None:
            lines.append(f"{key}: {match.get(key)}")
    return "\n".join(lines)


def _candidate_screen_block(
    candidate: dict[str, Any],
    match: dict[str, Any],
) -> str:
    distance = match.get("distance", candidate.get("best_distance", ""))
    rank = match.get("rank", candidate.get("best_rank", ""))
    return (
        f"arxiv_id: {candidate.get('arxiv_id', '')}\n"
        f"title: {_truncate(str(candidate.get('title', '')), 300)}\n"
        f"abstract: {_truncate(str(candidate.get('abstract', '')), 1200)}\n"
        f"query_rank: {rank}\n"
        f"query_distance: {distance}\n"
        f"best_distance: {candidate.get('best_distance', '')}\n"
        f"all_matched_queries: {_match_summary(candidate)}"
    )


def _query_group_key(match: dict[str, Any]) -> tuple:
    return (
        match.get("level"),
        match.get("query_type"),
        match.get("query"),
        match.get("source_text"),
        match.get("functional_role"),
        match.get("problem_index"),
        match.get("claim_index"),
        match.get("detail_index"),
        match.get("target_id"),
        match.get("refinement_round"),
    )


def _screen_query_groups(
    candidates: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], list[tuple[dict[str, Any], dict[str, Any]]]]]:
    """Group candidates by the source query that retrieved them."""
    groups: dict[
        tuple,
        tuple[dict[str, Any], list[tuple[dict[str, Any], dict[str, Any]]]],
    ] = {}
    unmatched = {
        "level": "unknown",
        "query_type": "unmatched",
        "query": "",
        "source_text": "",
    }
    for candidate in candidates:
        matches = [
            match
            for match in candidate.get("matched_queries") or []
            if isinstance(match, dict)
        ]
        if not matches:
            matches = [unmatched]
        for match in matches:
            key = _query_group_key(match)
            if key not in groups:
                groups[key] = (match, [])
            groups[key][1].append((candidate, match))

    output = []
    for match, group_candidates in groups.values():
        deduped = {}
        for candidate, candidate_match in group_candidates:
            arxiv_id = str(candidate.get("arxiv_id", "")).strip()
            if arxiv_id and arxiv_id not in deduped:
                deduped[arxiv_id] = (candidate, candidate_match)
        output.append(
            (
                match,
                sorted(
                    deduped.values(),
                    key=lambda item: (
                        item[1].get("rank", item[0].get("best_rank", 10_000)),
                        item[1].get("distance", item[0].get("best_distance", 1e9)),
                    ),
                ),
            )
        )
    return output


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


def screen_query_candidates(
    problems: list[ResearchProblem],
    query_match: dict[str, Any],
    candidate_matches: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    client: OpenAI | None = None,
    model: str = DEFAULT_JUDGE_MODEL,
) -> list[ScreenJudgment]:
    if not problems:
        raise ValueError("problems must not be empty")
    if not candidate_matches:
        return []

    user_parts = [
        _extraction_context(problems),
        "",
        _screen_query_block(query_match),
        "",
        "CANDIDATES RETURNED FOR THIS QUERY",
    ]
    for index, (candidate, match) in enumerate(candidate_matches):
        user_parts.append(f"--- candidate {index + 1} ---")
        user_parts.append(_candidate_screen_block(candidate, match))
    user_parts.append("")
    user_parts.append(
        "Return one screen judgment per arxiv_id above, judging this candidate "
        "in the context of the source query and its full candidate list."
    )

    completion = (client or get_client()).beta.chat.completions.parse(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SCREEN_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(user_parts)},
        ],
        response_format=ScreenQueryResult,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        raise RuntimeError("The screen model returned no structured result")

    by_id = {item.arxiv_id: item for item in result.judgments}
    aligned: list[ScreenJudgment] = []
    for candidate, _match in candidate_matches:
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


def _merge_screen_judgment(
    screens: dict[str, ScreenJudgment],
    judgment: ScreenJudgment,
) -> None:
    """Keep read_full if any retrieval query makes the candidate worth reading."""
    existing = screens.get(judgment.arxiv_id)
    if existing is None:
        screens[judgment.arxiv_id] = judgment
        return
    if existing.decision == "read_full":
        return
    if judgment.decision == "read_full":
        screens[judgment.arxiv_id] = judgment


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
    fetch_pdfs: bool = True,
    session: requests.Session | None = None,
    request_delay: float = DEFAULT_ARXIV_DELAY,
    pdf_max_chars: int = DEFAULT_PDF_MAX_CHARS,
    pdf_text_by_id: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Screen on abstracts, then full-text judge survivors one paper at a time."""
    if not problems:
        raise ValueError("problems must not be empty")

    screens: dict[str, ScreenJudgment] = {}
    total = len(candidates)
    query_groups = _screen_query_groups(candidates)
    for query_match, candidate_matches in _progress(
        query_groups,
        total=len(query_groups),
        desc="Screen",
        unit="query",
    ):
        for judgment in screen_query_candidates(
            problems,
            query_match,
            candidate_matches,
            client=client,
            model=model,
        ):
            _merge_screen_judgment(screens, judgment)

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


def _match_distance(match: dict[str, Any]) -> float:
    try:
        return float(match.get("distance"))
    except (TypeError, ValueError):
        return 1e9


def _safe_index(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _node_descriptor(
    problems: list[ResearchProblem],
    attachment: dict[str, Any],
) -> str:
    pi = attachment["problem_index"]
    ci = attachment.get("claim_index")
    di = attachment.get("detail_index")
    kind = attachment["node_kind"]
    problem = problems[pi]
    if kind == "problem":
        return (
            f"Problem[{pi}]\n"
            f"text: {problem.problem}\n"
            f"domain: {problem.domain}"
        )
    claim = problem.claims[ci]
    if kind == "claim":
        return (
            f"Claim[{pi}.{ci}]\n"
            f"text: {claim.claim}\n"
            f"functional_role: {claim.functional_role}"
        )
    detail = claim.implementation_details[di]
    return (
        f"Implementation detail[{pi}.{ci}.{di}]\n"
        f"text: {detail.detail}\n"
        f"functional_role: {detail.functional_role}"
    )


def _attachment_options_from_matches(
    candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    for match in candidate.get("matched_queries") or []:
        if not isinstance(match, dict):
            continue
        level = match.get("level")
        if level not in ("problem", "claim", "implementation"):
            continue
        pi = _safe_index(match.get("problem_index"), 0)
        ci = match.get("claim_index")
        di = match.get("detail_index")
        distance = _match_distance(match)
        if level == "problem":
            options.append(
                {
                    "node_kind": "problem",
                    "node_path": str(pi),
                    "problem_index": pi,
                    "claim_index": None,
                    "detail_index": None,
                    "distance": distance,
                }
            )
        elif level == "claim" and ci is not None:
            ci = _safe_index(ci)
            options.append(
                {
                    "node_kind": "claim",
                    "node_path": f"{pi}.{ci}",
                    "problem_index": pi,
                    "claim_index": ci,
                    "detail_index": None,
                    "distance": distance,
                }
            )
        elif (
            level == "implementation"
            and ci is not None
            and di is not None
        ):
            ci = _safe_index(ci)
            di = _safe_index(di)
            options.append(
                {
                    "node_kind": "implementation",
                    "node_path": f"{pi}.{ci}.{di}",
                    "problem_index": pi,
                    "claim_index": ci,
                    "detail_index": di,
                    "distance": distance,
                }
            )
    return options


def _pick_strongest_attachment(
    candidate: dict[str, Any],
    problems: list[ResearchProblem],
) -> dict[str, Any]:
    judgment = candidate.get("judgment") or {}
    primary_level = judgment.get("primary_level") or "problem"
    options = _attachment_options_from_matches(candidate)
    if not options:
        return {
            "node_kind": primary_level
            if primary_level in ("problem", "claim", "implementation")
            else "problem",
            "node_path": "0",
            "problem_index": 0,
            "claim_index": None,
            "detail_index": None,
            "distance": _match_distance({}),
        }

    def sort_key(option: dict[str, Any]) -> tuple:
        level_match = 0 if option["node_kind"] == primary_level else 1
        specificity = {
            "implementation": 0,
            "claim": 1,
            "problem": 2,
        }[option["node_kind"]]
        return (level_match, option["distance"], specificity)

    return min(options, key=sort_key)


def _selection_candidate_block(candidate: dict[str, Any]) -> str:
    judgment = candidate.get("judgment") or {}
    return "\n".join(
        [
            f"arxiv_id: {candidate.get('arxiv_id', '')}",
            f"title: {_truncate(str(candidate.get('title', '')), 300)}",
            f"abstract: {_truncate(str(candidate.get('abstract', '')), 500)}",
            f"relation: {judgment.get('relation', '')}",
            f"why: {_truncate(str(judgment.get('why', '')), 300)}",
            f"best_distance: {candidate.get('best_distance', '')}",
        ]
    )


def select_top_candidates_for_node(
    problems: list[ResearchProblem],
    attachment: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    client: OpenAI | None = None,
    model: str = DEFAULT_JUDGE_MODEL,
    max_per_node: int = DEFAULT_MAX_KEPT_PER_NODE,
) -> list[str]:
    if not candidates:
        return []
    if len(candidates) <= max_per_node:
        return [str(candidate.get("arxiv_id", "")).strip() for candidate in candidates]

    allowed = {
        str(candidate.get("arxiv_id", "")).strip(): candidate
        for candidate in candidates
        if str(candidate.get("arxiv_id", "")).strip()
    }
    user_parts = [
        _extraction_context(problems),
        "",
        "SOURCE NODE",
        _node_descriptor(problems, attachment),
        "",
        "KEPT CANDIDATES FOR THIS NODE",
    ]
    for index, candidate in enumerate(candidates, start=1):
        user_parts.append(f"--- candidate {index} ---")
        user_parts.append(_selection_candidate_block(candidate))
    user_parts.append("")
    user_parts.append(
        f"Return at most {max_per_node} arxiv_ids for this node. Prefer distinct,"
        " high-value papers and omit redundant near-duplicates."
    )

    try:
        completion = (client or get_client()).beta.chat.completions.parse(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": SELECT_SYSTEM_PROMPT},
                {"role": "user", "content": "\n".join(user_parts)},
            ],
            response_format=NodeTopSelection,
        )
        result = completion.choices[0].message.parsed
        if result is None:
            raise RuntimeError("The cap selector returned no structured result")
        selected = [
            arxiv_id
            for arxiv_id in result.selected_arxiv_ids
            if arxiv_id in allowed
        ]
        if selected:
            return selected[:max_per_node]
    except Exception:
        pass

    ranked = sorted(
        candidates,
        key=lambda candidate: (
            candidate.get("best_distance", 1e9),
            candidate.get("best_rank", 10_000),
            str(candidate.get("arxiv_id", "")),
        ),
    )
    return [
        str(candidate.get("arxiv_id", "")).strip()
        for candidate in ranked[:max_per_node]
        if str(candidate.get("arxiv_id", "")).strip()
    ]


def cap_kept_candidates_per_node(
    problems: list[ResearchProblem],
    candidates: list[dict[str, Any]],
    *,
    client: OpenAI | None = None,
    model: str = DEFAULT_JUDGE_MODEL,
    max_per_node: int = DEFAULT_MAX_KEPT_PER_NODE,
) -> list[dict[str, Any]]:
    """Keep at most max_per_node distinct kept papers per source node."""
    if max_per_node < 0:
        raise ValueError("max_per_node must be >= 0")
    if not problems:
        raise ValueError("problems must not be empty")

    kept_by_id = {
        str(candidate.get("arxiv_id", "")).strip(): candidate
        for candidate in candidates
        if str(candidate.get("arxiv_id", "")).strip()
        and (candidate.get("judgment") or {}).get("decision") == "keep"
    }
    if not kept_by_id:
        return candidates

    groups: dict[str, list[dict[str, Any]]] = {}
    attachments: dict[str, dict[str, Any]] = {}
    for candidate in kept_by_id.values():
        attachment = _pick_strongest_attachment(candidate, problems)
        node_path = attachment["node_path"]
        attachments[str(candidate.get("arxiv_id", "")).strip()] = attachment
        groups.setdefault(node_path, []).append(candidate)

    selected_ids: set[str] = set()
    for node_path, group in groups.items():
        attachment = group[0]
        attachment = attachments[str(group[0].get("arxiv_id", "")).strip()]
        chosen = select_top_candidates_for_node(
            problems,
            attachment,
            group,
            client=client,
            model=model,
            max_per_node=max_per_node,
        )
        selected_ids.update(chosen)

    output: list[dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        arxiv_id = str(item.get("arxiv_id", "")).strip()
        judgment = dict(item.get("judgment") or {})
        if judgment.get("decision") != "keep":
            output.append(item)
            continue
        if arxiv_id in selected_ids:
            item["attachment"] = attachments.get(arxiv_id)
            output.append(item)
            continue
        judgment.update(
            {
                "decision": "drop",
                "stage": "cap",
                "why": (
                    f"Not among the top {max_per_node} distinct suggestions for "
                    f"node {attachments.get(arxiv_id, {}).get('node_path', '?')}."
                ),
            }
        )
        item["judgment"] = judgment
        output.append(item)
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
        fetch_pdfs=not args.no_pdf,
    )
    judged = cap_kept_candidates_per_node(problems, judged)
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
