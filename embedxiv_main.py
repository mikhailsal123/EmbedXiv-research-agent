"""Extract → vector search → judge → S2 graph recommend → judge again.

Glue CLI for extract_claims.py + search_candidates.py + judge_candidates.py.
Search uses Postgres/pgvector when DATABASE_URL is set, else FAISS fallback.
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path

from extract_claims import extract_claims, read_pdf_text
from judge_candidates import kept_candidates, judge_candidates
from search_refinement import (
    DEFAULT_REFINEMENT_FOLLOWUPS_PER_TARGET,
    DEFAULT_REFINEMENT_LIMIT,
    DEFAULT_REFINEMENT_MAX_ROUNDS,
    DEFAULT_REFINEMENT_MAX_TARGETS,
    DEFAULT_REFINEMENT_QUERIES_PER_TARGET,
    run_search_refinement,
)
from search_candidates import (
    DEFAULT_REQUEST_DELAY,
    DEFAULT_S2_RECOMMEND_LIMIT,
    TOP_K_PER_QUERY,
    canonical_arxiv_id,
    detect_source_arxiv_id,
    open_index,
    recommend_semantic_scholar,
    search_all_candidates,
)
from suggestion_cards import write_suggestion_outputs


DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_DIR / "full_run_results.json"


def _merge_candidates(
    base: list[dict],
    additions: list[dict],
) -> list[dict]:
    candidates_by_id = {candidate["arxiv_id"]: dict(candidate) for candidate in base}
    for candidate in additions:
        arxiv_id = candidate["arxiv_id"]
        existing = candidates_by_id.get(arxiv_id)
        if existing is None:
            candidates_by_id[arxiv_id] = candidate
            continue
        existing["best_distance"] = min(
            existing["best_distance"], candidate["best_distance"]
        )
        existing["best_rank"] = min(existing["best_rank"], candidate["best_rank"])
        for match in candidate.get("matched_queries") or []:
            if match not in existing["matched_queries"]:
                existing["matched_queries"].append(match)
    return sorted(
        candidates_by_id.values(),
        key=lambda candidate: (candidate["best_distance"], candidate["best_rank"]),
    )


def _print_judge_summary(label: str, candidates: list) -> None:
    screen_drop = 0
    read_full = 0
    full_keep = 0
    full_drop = 0
    for candidate in candidates:
        screen = candidate.get("screen") or {}
        judgment = candidate.get("judgment") or {}
        if screen.get("decision") == "drop":
            screen_drop += 1
        elif screen.get("decision") == "read_full":
            read_full += 1
        if judgment.get("stage") == "full_text":
            if judgment.get("decision") == "keep":
                full_keep += 1
            else:
                full_drop += 1
    kept = sum(
        1
        for candidate in candidates
        if (candidate.get("judgment") or {}).get("decision") == "keep"
    )
    print(
        f"{label}: screen drop {screen_drop}, read_full {read_full}; "
        f"full-text keep {full_keep}, drop {full_drop}; "
        f"kept {kept}/{len(candidates)}.",
        flush=True,
    )


def read_input(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return read_pdf_text(str(path))
    return path.read_text()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a research structure and search related arXiv papers."
    )
    parser.add_argument("input", help="PDF or UTF-8 text file")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    parser.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--limit", type=int, default=TOP_K_PER_QUERY)
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip Qwen judging (also skips S2 recommendation expansion)",
    )
    parser.add_argument(
        "--no-s2",
        action="store_true",
        help="Disable Semantic Scholar graph recommendations from kept papers",
    )
    parser.add_argument(
        "--no-search-refinement",
        action="store_true",
        help="Disable bounded agentic query refinement for substitute mechanisms",
    )
    parser.add_argument(
        "--search-refinement-rounds",
        type=int,
        default=DEFAULT_REFINEMENT_MAX_ROUNDS,
        help="Maximum search refinement rounds per source node (default 2)",
    )
    parser.add_argument(
        "--search-refinement-targets",
        type=int,
        default=DEFAULT_REFINEMENT_MAX_TARGETS,
        help="Maximum claim/detail nodes search refinement may inspect",
    )
    parser.add_argument(
        "--search-refinement-limit",
        type=int,
        default=DEFAULT_REFINEMENT_LIMIT,
        help="Vector hits per refined query",
    )
    parser.add_argument(
        "--search-refinement-queries",
        type=int,
        default=DEFAULT_REFINEMENT_QUERIES_PER_TARGET,
        help="Initial refined queries per source node",
    )
    parser.add_argument(
        "--search-refinement-followups",
        type=int,
        default=DEFAULT_REFINEMENT_FOLLOWUPS_PER_TARGET,
        help="Follow-up queries per failed source node per round",
    )
    parser.add_argument(
        "--s2-recommend-limit",
        type=int,
        default=DEFAULT_S2_RECOMMEND_LIMIT,
        help="Recommendations to pull per kept seed paper (default 5)",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY,
    )
    parser.add_argument(
        "--no-cards",
        action="store_true",
        help="Skip writing suggestion card HTML/Markdown outputs",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the generated HTML in the default browser",
    )
    parser.add_argument(
        "--source-arxiv-id",
        action="append",
        default=[],
        help=(
            "arXiv id of the input paper to exclude from results "
            "(repeatable). Auto-detected from the paper header when omitted."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    paper_text = read_input(input_path)
    exclude_ids = {
        canonical_arxiv_id(arxiv_id)
        for arxiv_id in args.source_arxiv_id
        if canonical_arxiv_id(arxiv_id)
    }
    detected = detect_source_arxiv_id(paper_text)
    if detected:
        exclude_ids.add(detected)
    if exclude_ids:
        print(
            "Excluding source paper id(s): "
            + ", ".join(sorted(exclude_ids)),
            flush=True,
        )

    print("Extracting claims…", flush=True)
    problems = extract_claims(paper_text)
    print(f"Extracted {len(problems)} problem(s).", flush=True)

    print("Loading index and searching…", flush=True)
    search_refinement_trace = []
    with open_index(device=args.device) as index:
        candidates = search_all_candidates(
            problems,
            index,
            limit=args.limit,
            exclude_ids=exclude_ids or None,
            enrich_s2=False,
            request_delay=args.request_delay,
        )
        print(f"Search returned {len(candidates)} unique candidate(s).", flush=True)
        if not args.no_judge and not args.no_search_refinement:
            seen_ids = (exclude_ids or set()) | {
                candidate["arxiv_id"] for candidate in candidates
            }
            print(
                "Running agentic search refinement for substitute mechanisms…",
                flush=True,
            )
            try:
                refined_candidates, search_refinement_trace = (
                    run_search_refinement(
                        problems,
                        index,
                        limit=args.search_refinement_limit,
                        max_rounds=args.search_refinement_rounds,
                        max_targets=args.search_refinement_targets,
                        queries_per_target=args.search_refinement_queries,
                        max_followups_per_target=(
                            args.search_refinement_followups
                        ),
                        exclude_ids=seen_ids,
                    )
                )
            except Exception as exc:
                # Retrieval refinement should improve recall, not block the
                # baseline search/judge path when the model or API is flaky.
                print(f"Agentic search refinement skipped: {exc}", flush=True)
                refined_candidates = []
                search_refinement_trace = [
                    {"error": str(exc), "stage": "search_refinement"}
                ]
            if refined_candidates:
                candidates = _merge_candidates(candidates, refined_candidates)
                print(
                    "Agentic search refinement added "
                    f"{len(refined_candidates)} candidate(s); "
                    f"{len(candidates)} total before judging.",
                    flush=True,
                )
            else:
                print("Agentic search refinement added no candidates.", flush=True)

        if not args.no_judge:
            print("Judging candidates…", flush=True)
            candidates = judge_candidates(
                problems,
                candidates,
            )
            _print_judge_summary("Vector judge", candidates)
            if not args.no_s2:
                seeds = kept_candidates(candidates)
                print(
                    f"Fetching S2 recommendations for {len(seeds)} seed(s)…",
                    flush=True,
                )
                recommendations = recommend_semantic_scholar(
                    seeds,
                    limit_per_seed=args.s2_recommend_limit,
                    exclude_ids=(exclude_ids or set())
                    | {c["arxiv_id"] for c in candidates},
                    request_delay=args.request_delay,
                )
                if recommendations:
                    print(
                        f"Judging {len(recommendations)} S2 recommendation(s)…",
                        flush=True,
                    )
                    judged_recs = judge_candidates(
                        problems,
                        recommendations,
                    )
                    _print_judge_summary("S2 judge", judged_recs)
                    candidates.extend(judged_recs)
                else:
                    print("No S2 recommendations to judge.", flush=True)

    kept = sum(
        1
        for candidate in candidates
        if (candidate.get("judgment") or {}).get("decision") == "keep"
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "source": str(input_path),
        "source_arxiv_ids": sorted(exclude_ids),
        "problems": [problem.model_dump() for problem in problems],
        "candidates": candidates,
        "search_refinement_trace": search_refinement_trace,
        "kept_count": kept if not args.no_judge else None,
    }
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    if args.no_judge:
        print(f"Saved {len(candidates)} unique candidates to {output_path}")
    else:
        print(
            f"Saved {len(candidates)} candidates ({kept} kept) to {output_path}"
        )
        if not args.no_cards:
            written = write_suggestion_outputs(
                candidates,
                output_path,
                source=str(input_path),
                problems=problems,
            )
            print(f"Suggestion cards → {written['html']}")
            if not args.no_open:
                webbrowser.open(written["html"].resolve().as_uri())


if __name__ == "__main__":
    main()
