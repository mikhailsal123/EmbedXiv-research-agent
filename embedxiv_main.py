"""Extract problems/claims/details with Qwen, then FAISS-search related arXiv papers.

Glue CLI for extract_claims.py + search_candidates.py. The SPECTER2/FAISS corpus
is loaded from Nebius Object Storage (published by datagen.create_corpus).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from extract_claims import extract_claims, read_pdf_text
from search_candidates import (
    DEFAULT_REQUEST_DELAY,
    TOP_K_PER_QUERY,
    open_index,
    search_all_candidates,
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
    parser.add_argument("-o", "--output", default="research_results.json")
    parser.add_argument("--limit", type=int, default=TOP_K_PER_QUERY)
    parser.add_argument("--no-s2", action="store_true")
    parser.add_argument(
        "--request-delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY,
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    paper_text = read_input(input_path)
    problems = extract_claims(paper_text)
    with open_index(device=args.device) as index:
        candidates = search_all_candidates(
            problems,
            index,
            limit=args.limit,
            enrich_s2=not args.no_s2,
            request_delay=args.request_delay,
        )

    output = {
        "source": str(input_path),
        "problems": [problem.model_dump() for problem in problems],
        "candidates": candidates,
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved {len(candidates)} unique candidates to {args.output}")


if __name__ == "__main__":
    main()
