"""Extract → FAISS search → judge → S2 graph recommend → judge again.

Glue CLI for extract_claims.py + search_candidates.py + judge_candidates.py.
The SPECTER2/FAISS corpus is loaded from Nebius Object Storage.
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path

from extract_claims import extract_claims, read_pdf_text
from judge_candidates import kept_candidates, judge_candidates
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
    parser.add_argument("--judge-batch-size", type=int, default=8)
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

        if not args.no_judge:
            print("Judging candidates…", flush=True)
            candidates = judge_candidates(
                problems,
                candidates,
                batch_size=args.judge_batch_size,
            )
            _print_judge_summary("FAISS judge", candidates)
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
                        batch_size=args.judge_batch_size,
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
