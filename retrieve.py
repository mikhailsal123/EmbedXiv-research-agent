"""Hierarchy-aware arXiv vector retrieval and optional S2 enrichment."""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol, Sequence

import requests

from arxiv_index import ArxivIndex, canonical_arxiv_id
from extract_claims import ExtractionResult, ResearchProblem, load_local_env


S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_FIELDS = ",".join(
    (
        "paperId",
        "externalIds",
        "url",
        "year",
        "publicationDate",
        "venue",
        "publicationVenue",
        "publicationTypes",
        "journal",
        "citationCount",
        "influentialCitationCount",
        "referenceCount",
        "s2FieldsOfStudy",
        "isOpenAccess",
        "openAccessPdf",
    )
)
TOP_K_PER_QUERY = 20
DEFAULT_REQUEST_DELAY = 1.1
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class VectorIndex(Protocol):
    connection: sqlite3.Connection

    def search(self, query_texts: Sequence[str], k: int) -> list[list[dict]]:
        ...


@dataclass(frozen=True)
class SearchQuery:
    level: Literal["problem", "claim", "implementation"]
    query_type: Literal["direct", "functional", "alternative"]
    query: str
    source_text: str
    problem_index: int
    claim_index: int | None = None
    detail_index: int | None = None


def _join_context(*parts: str) -> str:
    return "\n".join(part.strip() for part in parts if part.strip())


def build_queries(
    problem: ResearchProblem, problem_index: int = 0
) -> list[SearchQuery]:
    """Build full contextual texts for asymmetric SPECTER2 query encoding."""
    problem_context = _join_context(
        f"Problem: {problem.problem}",
        f"Domain: {problem.domain}",
        f"Keywords: {', '.join(problem.keywords)}",
    )
    queries = [
        SearchQuery(
            level="problem",
            query_type="direct",
            query=problem_context,
            source_text=problem.problem,
            problem_index=problem_index,
        )
    ]

    for claim_index, claim in enumerate(problem.claims):
        queries.extend(
            [
                SearchQuery(
                    level="claim",
                    query_type="direct",
                    query=_join_context(
                        f"Problem: {problem.problem}",
                        f"Claim: {claim.claim}",
                    ),
                    source_text=claim.claim,
                    problem_index=problem_index,
                    claim_index=claim_index,
                ),
                SearchQuery(
                    level="claim",
                    query_type="functional",
                    query=_join_context(
                        f"Problem: {problem.problem}",
                        f"Research goal: {claim.functional_role}",
                    ),
                    source_text=claim.functional_role,
                    problem_index=problem_index,
                    claim_index=claim_index,
                ),
            ]
        )

        for detail_index, detail in enumerate(claim.implementation_details):
            queries.extend(
                [
                    SearchQuery(
                        level="implementation",
                        query_type="direct",
                        query=_join_context(
                            f"Claim: {claim.claim}",
                            f"Implementation detail: {detail.detail}",
                            f"Role: {detail.functional_role}",
                        ),
                        source_text=detail.detail,
                        problem_index=problem_index,
                        claim_index=claim_index,
                        detail_index=detail_index,
                    ),
                    SearchQuery(
                        level="implementation",
                        query_type="alternative",
                        query=_join_context(
                            f"Claim: {claim.claim}",
                            f"Desired implementation role: {detail.functional_role}",
                        ),
                        source_text=detail.functional_role,
                        problem_index=problem_index,
                        claim_index=claim_index,
                        detail_index=detail_index,
                    ),
                ]
            )

    return queries


def retrieve_candidates(
    problem: ResearchProblem,
    index: VectorIndex,
    *,
    problem_index: int = 0,
    limit: int = TOP_K_PER_QUERY,
) -> list[dict]:
    """Vector-search every hierarchy query and preserve match provenance."""
    queries = build_queries(problem, problem_index)
    result_sets = index.search([query.query for query in queries], k=limit)
    if len(result_sets) != len(queries):
        raise ValueError("Vector index returned a different number of result sets")

    candidates_by_id: dict[str, dict] = {}
    for search_query, results in zip(queries, result_sets):
        query_metadata = asdict(search_query)
        for result in results:
            arxiv_id = canonical_arxiv_id(result.get("arxiv_id", ""))
            if not arxiv_id:
                continue

            if arxiv_id not in candidates_by_id:
                candidate = dict(result)
                candidate["arxiv_id"] = arxiv_id
                candidate["best_distance"] = float(result["distance"])
                candidate["best_rank"] = int(result["rank"])
                candidate["matched_queries"] = []
                candidates_by_id[arxiv_id] = candidate

            candidate = candidates_by_id[arxiv_id]
            distance = float(result["distance"])
            rank = int(result["rank"])
            if distance < candidate["best_distance"]:
                candidate["best_distance"] = distance
                candidate["best_rank"] = rank

            match = {
                **query_metadata,
                "distance": distance,
                "rank": rank,
            }
            if match not in candidate["matched_queries"]:
                candidate["matched_queries"].append(match)

    return sorted(
        candidates_by_id.values(),
        key=lambda candidate: (candidate["best_distance"], candidate["best_rank"]),
    )


def retrieve_all_candidates(
    problems: list[ResearchProblem],
    index: VectorIndex,
    *,
    limit: int = TOP_K_PER_QUERY,
    enrich_s2: bool = True,
    api_key: str | None = None,
    session: requests.Session | None = None,
    request_delay: float = DEFAULT_REQUEST_DELAY,
) -> list[dict]:
    """Retrieve all hierarchy candidates, deduplicate, then optionally enrich."""
    candidates_by_id: dict[str, dict] = {}
    for problem_index, problem in enumerate(problems):
        for candidate in retrieve_candidates(
            problem,
            index,
            problem_index=problem_index,
            limit=limit,
        ):
            arxiv_id = candidate["arxiv_id"]
            if arxiv_id not in candidates_by_id:
                candidates_by_id[arxiv_id] = candidate
                continue

            existing = candidates_by_id[arxiv_id]
            existing["best_distance"] = min(
                existing["best_distance"], candidate["best_distance"]
            )
            existing["best_rank"] = min(
                existing["best_rank"], candidate["best_rank"]
            )
            for match in candidate["matched_queries"]:
                if match not in existing["matched_queries"]:
                    existing["matched_queries"].append(match)

    candidates = sorted(
        candidates_by_id.values(),
        key=lambda candidate: (candidate["best_distance"], candidate["best_rank"]),
    )
    if enrich_s2:
        enrich_semantic_scholar(
            candidates,
            api_key=api_key,
            session=session,
            cache_connection=getattr(index, "connection", None),
            request_delay=request_delay,
        )
    return candidates


def _cached_s2(
    connection: sqlite3.Connection | None, arxiv_id: str
) -> tuple[str, dict | None] | None:
    if connection is None:
        return None
    row = connection.execute(
        "SELECT status, payload FROM s2_cache WHERE arxiv_id = ?",
        (arxiv_id,),
    ).fetchone()
    if row is None:
        return None
    payload = json.loads(row["payload"]) if row["payload"] else None
    return row["status"], payload


def _cache_s2(
    connection: sqlite3.Connection | None,
    arxiv_id: str,
    status: str,
    payload: dict | None,
) -> None:
    if connection is None:
        return
    with connection:
        connection.execute(
            """
            INSERT INTO s2_cache (arxiv_id, status, payload, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(arxiv_id) DO UPDATE SET
                status=excluded.status,
                payload=excluded.payload,
                updated_at=CURRENT_TIMESTAMP
            """,
            (arxiv_id, status, json.dumps(payload) if payload else None),
        )


def _post_s2_batch(
    ids: list[str],
    *,
    api_key: str,
    session: requests.Session | None,
    max_retries: int,
) -> list[dict | None]:
    http = session or requests
    headers = {"x-api-key": api_key}
    for attempt in range(max_retries + 1):
        response = http.post(
            S2_BATCH_URL,
            params={"fields": S2_FIELDS},
            headers=headers,
            json={"ids": [f"ARXIV:{arxiv_id}" for arxiv_id in ids]},
            timeout=60,
        )
        if response.status_code not in RETRYABLE_STATUSES:
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list) or len(payload) != len(ids):
                raise ValueError("Semantic Scholar returned an invalid batch")
            return payload
        if attempt == max_retries:
            response.raise_for_status()
        retry_after = response.headers.get("Retry-After")
        try:
            delay = float(retry_after) if retry_after else 2**attempt
        except ValueError:
            delay = 2**attempt
        time.sleep(min(max(delay + random.uniform(0, 0.25), 0), 30))
    raise RuntimeError("Semantic Scholar retry loop ended unexpectedly")


def _merge_s2(candidate: dict, status: str, payload: dict | None) -> None:
    candidate["semantic_scholar"] = {"status": status}
    if payload is None:
        return
    candidate["semantic_scholar"]["paper"] = payload
    for field in (
        "paperId",
        "externalIds",
        "year",
        "publicationDate",
        "venue",
        "citationCount",
        "influentialCitationCount",
        "referenceCount",
        "s2FieldsOfStudy",
        "isOpenAccess",
        "openAccessPdf",
    ):
        value = payload.get(field)
        if value is not None and not candidate.get(field):
            candidate[field] = value


def enrich_semantic_scholar(
    candidates: list[dict],
    *,
    api_key: str | None = None,
    session: requests.Session | None = None,
    cache_connection: sqlite3.Connection | None = None,
    batch_size: int = 500,
    max_retries: int = 3,
    request_delay: float = DEFAULT_REQUEST_DELAY,
) -> list[dict]:
    """Best-effort S2 batch enrichment; arXiv results remain authoritative."""
    if not 1 <= batch_size <= 500:
        raise ValueError("batch_size must be between 1 and 500")
    resolved_key = api_key if api_key is not None else os.getenv("S2_API_KEY")
    if not resolved_key:
        for candidate in candidates:
            _merge_s2(candidate, "disabled", None)
        return candidates

    by_id = {candidate["arxiv_id"]: candidate for candidate in candidates}
    uncached = []
    for arxiv_id, candidate in by_id.items():
        cached = _cached_s2(cache_connection, arxiv_id)
        if cached is None:
            uncached.append(arxiv_id)
        else:
            _merge_s2(candidate, *cached)

    for start in range(0, len(uncached), batch_size):
        chunk = uncached[start : start + batch_size]
        try:
            payloads = _post_s2_batch(
                chunk,
                api_key=resolved_key,
                session=session,
                max_retries=max_retries,
            )
        except (requests.RequestException, RuntimeError, ValueError):
            for arxiv_id in chunk:
                _merge_s2(by_id[arxiv_id], "unavailable", None)
            continue

        for arxiv_id, payload in zip(chunk, payloads):
            status = "ok" if payload is not None else "not_found"
            _cache_s2(cache_connection, arxiv_id, status, payload)
            _merge_s2(by_id[arxiv_id], status, payload)
        if request_delay > 0 and start + batch_size < len(uncached):
            time.sleep(request_delay)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve papers from a local arXiv vector index."
    )
    parser.add_argument("extraction_json", help="JSON produced by extract_claims.py")
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    parser.add_argument("-o", "--output", default="retrieval_results.json")
    parser.add_argument("--limit", type=int, default=TOP_K_PER_QUERY)
    parser.add_argument("--no-s2", action="store_true")
    parser.add_argument("--request-delay", type=float, default=DEFAULT_REQUEST_DELAY)
    args = parser.parse_args()

    load_local_env()
    extraction = ExtractionResult.model_validate_json(
        Path(args.extraction_json).read_text()
    )
    with ArxivIndex(args.index_dir, device=args.device) as index:
        candidates = retrieve_all_candidates(
            extraction.problems,
            index,
            limit=args.limit,
            enrich_s2=not args.no_s2,
            request_delay=args.request_delay,
        )
    output = {
        "problems": [problem.model_dump() for problem in extraction.problems],
        "candidates": candidates,
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved {len(candidates)} unique candidates to {args.output}")


if __name__ == "__main__":
    main()
