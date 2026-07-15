"""Bounded agentic search refinement for replacement-style paper candidates.

This module refines retrieval before judging. It looks for papers proposing
different mechanisms for the same role as a source claim or implementation
detail, inspects title/abstract search results, and retries with rewritten
queries when the result set looks too generic or too close to the source.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, field_validator

from extract_claims import ResearchProblem, get_client
from search_candidates import VectorIndex, canonical_arxiv_id


DEFAULT_REFINEMENT_MAX_ROUNDS = 2
DEFAULT_REFINEMENT_MAX_TARGETS = 12
DEFAULT_REFINEMENT_QUERIES_PER_TARGET = 3
DEFAULT_REFINEMENT_FOLLOWUPS_PER_TARGET = 2
DEFAULT_REFINEMENT_LIMIT = 8


class SchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RefinementSearchQuery(SchemaModel):
    target_id: str = Field(min_length=1)
    query: str = Field(min_length=3, max_length=220)
    rationale: str = Field(min_length=1, max_length=300)

    @field_validator("query")
    @classmethod
    def query_must_be_search_text(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("query must not be empty")
        return value


class SearchRefinementPlan(SchemaModel):
    queries: list[RefinementSearchQuery] = Field(min_length=1, max_length=80)


class SearchRefinementAssessment(SchemaModel):
    target_id: str = Field(min_length=1)
    found_plausible_substitutes: bool
    diagnosis: str = Field(min_length=1, max_length=500)
    followup_queries: list[str] = Field(default_factory=list, max_length=4)


@dataclass(frozen=True)
class RefinementTarget:
    target_id: str
    level: Literal["claim", "implementation"]
    source_choice: str
    functional_role: str
    problem: str
    domain: str
    problem_index: int
    claim_index: int
    detail_index: int | None = None


@dataclass(frozen=True)
class RefinementQuery:
    target_id: str
    query: str
    rationale: str
    round_index: int


PLAN_SYSTEM_PROMPT = """You generate search queries to refine retrieval toward
replacement-style candidates for a research paper's choices.

The source has claims and implementation details. For each target, generate
queries for mechanisms, architectures, methods, or conceptual approaches that
could serve the SAME functional role while being meaningfully different from
the source choice.

Do not search for generic related work, support papers, surveys, benchmarks, or
near-duplicates. Prefer concrete method families and substitute mechanisms.
"""


ASSESS_SYSTEM_PROMPT = """You inspect vector-search results during retrieval
refinement.

Decide whether the result set contains plausible substitutes: papers that offer
a meaningfully different mechanism for the same functional role as the source
choice. If results are generic, near-duplicates, or solve a different problem,
diagnose the failure and propose tighter follow-up queries.

Keep the loop bounded: propose only high-value follow-up queries.
"""


def build_refinement_targets(
    problems: list[ResearchProblem],
    *,
    max_targets: int = DEFAULT_REFINEMENT_MAX_TARGETS,
) -> list[RefinementTarget]:
    targets: list[RefinementTarget] = []
    for problem_index, problem in enumerate(problems):
        for claim_index, claim in enumerate(problem.claims):
            targets.append(
                RefinementTarget(
                    target_id=f"p{problem_index}.c{claim_index}",
                    level="claim",
                    source_choice=claim.claim,
                    functional_role=claim.functional_role,
                    problem=problem.problem,
                    domain=problem.domain,
                    problem_index=problem_index,
                    claim_index=claim_index,
                )
            )
            for detail_index, detail in enumerate(claim.implementation_details):
                targets.append(
                    RefinementTarget(
                        target_id=(
                            f"p{problem_index}.c{claim_index}.d{detail_index}"
                        ),
                        level="implementation",
                        source_choice=detail.detail,
                        functional_role=detail.functional_role,
                        problem=problem.problem,
                        domain=problem.domain,
                        problem_index=problem_index,
                        claim_index=claim_index,
                        detail_index=detail_index,
                    )
                )
            if len(targets) >= max_targets:
                return targets[:max_targets]
    return targets[:max_targets]


def _target_block(targets: list[RefinementTarget]) -> str:
    lines = ["TARGETS"]
    for target in targets:
        lines.extend(
            [
                f"target_id: {target.target_id}",
                f"level: {target.level}",
                f"problem: {target.problem}",
                f"domain: {target.domain}",
                f"source_choice: {target.source_choice}",
                f"functional_role: {target.functional_role}",
                "",
            ]
        )
    return "\n".join(lines)


def plan_refinement_queries(
    targets: list[RefinementTarget],
    *,
    client: OpenAI | None = None,
    model: str | None = None,
    queries_per_target: int = DEFAULT_REFINEMENT_QUERIES_PER_TARGET,
) -> list[RefinementQuery]:
    if not targets:
        return []
    target_ids = {target.target_id for target in targets}
    completion = (client or get_client()).beta.chat.completions.parse(
        model=model or "qwen3:32b",
        temperature=0,
        messages=[
            {"role": "system", "content": PLAN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "\n".join(
                    [
                        _target_block(targets),
                        (
                            f"Generate at most {queries_per_target} search "
                            "queries per target_id. Return only listed target ids."
                        ),
                    ]
                ),
            },
        ],
        response_format=SearchRefinementPlan,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        raise RuntimeError("The search refiner returned no structured plan")

    counts: dict[str, int] = {}
    queries: list[RefinementQuery] = []
    seen: set[tuple[str, str]] = set()
    for item in result.queries:
        if item.target_id not in target_ids:
            continue
        key = (item.target_id, item.query.casefold())
        if key in seen:
            continue
        if counts.get(item.target_id, 0) >= queries_per_target:
            continue
        seen.add(key)
        counts[item.target_id] = counts.get(item.target_id, 0) + 1
        queries.append(
            RefinementQuery(
                target_id=item.target_id,
                query=item.query,
                rationale=item.rationale,
                round_index=0,
            )
        )
    return queries


def _target_metadata(target: RefinementTarget, query: RefinementQuery) -> dict[str, Any]:
    metadata = {
        "level": target.level,
        "query_type": "refined_substitute",
        "query": query.query,
        "source_text": target.source_choice,
        "functional_role": target.functional_role,
        "problem_index": target.problem_index,
        "claim_index": target.claim_index,
        "detail_index": target.detail_index,
        "target_id": target.target_id,
        "refinement_round": query.round_index,
        "rationale": query.rationale,
    }
    return metadata


def _merge_refined_candidate(
    candidates_by_id: dict[str, dict[str, Any]],
    result: dict[str, Any],
    *,
    metadata: dict[str, Any],
) -> None:
    arxiv_id = canonical_arxiv_id(str(result.get("arxiv_id", "")))
    if not arxiv_id:
        return
    if arxiv_id not in candidates_by_id:
        candidate = dict(result)
        candidate["arxiv_id"] = arxiv_id
        candidate["best_distance"] = float(result["distance"])
        candidate["best_rank"] = int(result["rank"])
        candidate["matched_queries"] = []
        candidate["retrieval_source"] = "search_refinement"
        candidates_by_id[arxiv_id] = candidate

    candidate = candidates_by_id[arxiv_id]
    distance = float(result["distance"])
    rank = int(result["rank"])
    if distance < candidate["best_distance"]:
        candidate["best_distance"] = distance
        candidate["best_rank"] = rank
    match = {**metadata, "distance": distance, "rank": rank}
    if match not in candidate["matched_queries"]:
        candidate["matched_queries"].append(match)


def search_refinement_queries(
    queries: list[RefinementQuery],
    targets_by_id: dict[str, RefinementTarget],
    index: VectorIndex,
    *,
    limit: int = DEFAULT_REFINEMENT_LIMIT,
    exclude_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    if not queries:
        return []
    if limit < 1:
        raise ValueError("limit must be positive")
    blocked = {
        canonical_arxiv_id(arxiv_id)
        for arxiv_id in (exclude_ids or set())
        if canonical_arxiv_id(arxiv_id)
    }
    result_sets = index.search([query.query for query in queries], k=limit)
    if len(result_sets) != len(queries):
        raise ValueError("Vector index returned a different number of result sets")

    candidates_by_id: dict[str, dict[str, Any]] = {}
    for query, results in zip(queries, result_sets):
        target = targets_by_id.get(query.target_id)
        if target is None:
            continue
        metadata = _target_metadata(target, query)
        for result in results:
            arxiv_id = canonical_arxiv_id(str(result.get("arxiv_id", "")))
            if not arxiv_id or arxiv_id in blocked:
                continue
            _merge_refined_candidate(candidates_by_id, result, metadata=metadata)
    return sorted(
        candidates_by_id.values(),
        key=lambda candidate: (candidate["best_distance"], candidate["best_rank"]),
    )


def _candidate_block(candidates: list[dict[str, Any]], *, limit: int = 8) -> str:
    lines = ["SEARCH RESULTS"]
    for index, candidate in enumerate(candidates[:limit], start=1):
        lines.extend(
            [
                f"result {index}",
                f"arxiv_id: {candidate.get('arxiv_id', '')}",
                f"title: {candidate.get('title', '')}",
                f"abstract: {' '.join(str(candidate.get('abstract', '')).split())[:900]}",
                "",
            ]
        )
    if not candidates:
        lines.append("(no results)")
    return "\n".join(lines)


def assess_attempt(
    target: RefinementTarget,
    queries: list[RefinementQuery],
    candidates: list[dict[str, Any]],
    *,
    client: OpenAI | None = None,
    model: str | None = None,
    max_followups: int = DEFAULT_REFINEMENT_FOLLOWUPS_PER_TARGET,
) -> SearchRefinementAssessment:
    completion = (client or get_client()).beta.chat.completions.parse(
        model=model or "qwen3:32b",
        temperature=0,
        messages=[
            {"role": "system", "content": ASSESS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "\n".join(
                    [
                        "SOURCE TARGET",
                        f"target_id: {target.target_id}",
                        f"level: {target.level}",
                        f"problem: {target.problem}",
                        f"domain: {target.domain}",
                        f"source_choice: {target.source_choice}",
                        f"functional_role: {target.functional_role}",
                        "",
                        "QUERIES TRIED",
                        "\n".join(f"- {query.query}" for query in queries),
                        "",
                        _candidate_block(candidates),
                        "",
                        (
                            f"If the results do not contain plausible substitutes, "
                            f"return at most {max_followups} follow-up queries."
                        ),
                    ]
                ),
            },
        ],
        response_format=SearchRefinementAssessment,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        raise RuntimeError("The search refiner returned no structured assessment")
    if result.target_id != target.target_id:
        result = result.model_copy(update={"target_id": target.target_id})
    return result.model_copy(
        update={"followup_queries": result.followup_queries[:max_followups]}
    )


def _candidates_for_target(
    candidates: list[dict[str, Any]], target_id: str
) -> list[dict[str, Any]]:
    output = []
    for candidate in candidates:
        for match in candidate.get("matched_queries") or []:
            if match.get("target_id") == target_id:
                output.append(candidate)
                break
    return sorted(
        output,
        key=lambda candidate: (candidate["best_distance"], candidate["best_rank"]),
    )


def run_search_refinement(
    problems: list[ResearchProblem],
    index: VectorIndex,
    *,
    client: OpenAI | None = None,
    model: str | None = None,
    limit: int = DEFAULT_REFINEMENT_LIMIT,
    max_rounds: int = DEFAULT_REFINEMENT_MAX_ROUNDS,
    max_targets: int = DEFAULT_REFINEMENT_MAX_TARGETS,
    queries_per_target: int = DEFAULT_REFINEMENT_QUERIES_PER_TARGET,
    max_followups_per_target: int = DEFAULT_REFINEMENT_FOLLOWUPS_PER_TARGET,
    exclude_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run a bounded search/inspect/rewrite loop.

    Returns candidate papers plus a serializable trace. The hard caps make the
    loop finite: at most max_rounds and at most max_followups_per_target new
    queries for each target in each retry round.
    """
    if max_rounds < 1:
        raise ValueError("max_rounds must be >= 1")
    targets = build_refinement_targets(problems, max_targets=max_targets)
    targets_by_id = {target.target_id: target for target in targets}
    planned = plan_refinement_queries(
        targets,
        client=client,
        model=model,
        queries_per_target=queries_per_target,
    )

    active_by_target: dict[str, list[RefinementQuery]] = {}
    seen_queries: dict[str, set[str]] = {}
    for query in planned:
        active_by_target.setdefault(query.target_id, []).append(query)
        seen_queries.setdefault(query.target_id, set()).add(query.query.casefold())

    candidates_by_id: dict[str, dict[str, Any]] = {}
    trace: list[dict[str, Any]] = []
    blocked = set(exclude_ids or set())
    for round_index in range(max_rounds):
        round_queries = [
            query
            for queries in active_by_target.values()
            for query in queries
            if query.round_index == round_index
        ]
        if not round_queries:
            break
        round_candidates = search_refinement_queries(
            round_queries,
            targets_by_id,
            index,
            limit=limit,
            exclude_ids=blocked,
        )
        for candidate in round_candidates:
            existing = candidates_by_id.get(candidate["arxiv_id"])
            if existing is None:
                candidates_by_id[candidate["arxiv_id"]] = candidate
                continue
            existing["best_distance"] = min(
                existing["best_distance"], candidate["best_distance"]
            )
            existing["best_rank"] = min(existing["best_rank"], candidate["best_rank"])
            for match in candidate.get("matched_queries") or []:
                if match not in existing["matched_queries"]:
                    existing["matched_queries"].append(match)

        next_active: dict[str, list[RefinementQuery]] = {}
        for target_id, target in targets_by_id.items():
            queries_for_target = [
                query for query in round_queries if query.target_id == target_id
            ]
            if not queries_for_target:
                continue
            candidates_for_target = _candidates_for_target(
                list(candidates_by_id.values()), target_id
            )
            assessment = assess_attempt(
                target,
                queries_for_target,
                candidates_for_target,
                client=client,
                model=model,
                max_followups=max_followups_per_target,
            )
            trace.append(
                {
                    **assessment.model_dump(),
                    "round": round_index,
                    "queries": [asdict(query) for query in queries_for_target],
                    "candidate_count": len(candidates_for_target),
                }
            )
            if assessment.found_plausible_substitutes:
                continue
            if round_index + 1 >= max_rounds:
                continue
            # Retry only the targets that failed assessment, and suppress
            # duplicate query text so the loop cannot churn on one rewrite.
            followups = []
            for text in assessment.followup_queries[:max_followups_per_target]:
                normalized = " ".join(text.split())
                if not normalized:
                    continue
                seen = seen_queries.setdefault(target_id, set())
                if normalized.casefold() in seen:
                    continue
                seen.add(normalized.casefold())
                followups.append(
                    RefinementQuery(
                        target_id=target_id,
                        query=normalized,
                        rationale=assessment.diagnosis,
                        round_index=round_index + 1,
                    )
                )
            if followups:
                next_active[target_id] = followups
        active_by_target = next_active

    candidates = sorted(
        candidates_by_id.values(),
        key=lambda candidate: (candidate["best_distance"], candidate["best_rank"]),
    )
    return candidates, trace
