"""CLI for harvesting arXiv metadata and building a resumable vector index."""

from __future__ import annotations

import argparse
from pathlib import Path

from arxiv_index import (
    Specter2Encoder,
    build_faiss_index,
    connect_database,
    embed_pending,
    harvest_oai,
    iter_snapshot,
    upsert_papers,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a SPECTER2/FAISS arXiv title-and-abstract index."
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/arxiv-index"),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--metadata-jsonl",
        type=Path,
        help="Local arXiv metadata snapshot in JSONL format",
    )
    source.add_argument(
        "--oai-sample",
        type=int,
        metavar="COUNT",
        help="Harvest approximately COUNT real records through arXiv OAI-PMH",
    )
    source.add_argument(
        "--existing-metadata",
        action="store_true",
        help="Skip ingestion and embed metadata already stored in --index-dir",
    )
    parser.add_argument("--from-date", help="OAI lower datestamp, YYYY-MM-DD")
    parser.add_argument("--set", dest="set_spec", help="Optional OAI set")
    parser.add_argument("--request-interval", type=float, default=3.0)
    parser.add_argument(
        "--ingest-limit",
        type=int,
        help="Maximum records read from a local snapshot",
    )
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=50_000)
    parser.add_argument("--max-embed", type=int)
    parser.add_argument(
        "--index-type",
        default="flat",
        help="flat, ivf, ivf-sq8, or a FAISS factory string",
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Only ingest metadata",
    )
    args = parser.parse_args()

    connection = connect_database(args.index_dir)
    try:
        if args.existing_metadata:
            print("Using metadata already stored in the index directory")
        elif args.metadata_jsonl:
            changed = upsert_papers(
                connection,
                iter_snapshot(args.metadata_jsonl, limit=args.ingest_limit),
            )
            print(f"Ingested or updated {changed} snapshot records")
        else:
            harvested = harvest_oai(
                connection,
                max_records=args.oai_sample,
                from_date=args.from_date,
                set_spec=args.set_spec,
                request_interval=args.request_interval,
            )
            print(f"Harvested {harvested} OAI records")

        if args.skip_embedding:
            return

        encoder = Specter2Encoder(device=args.device)
        embedded = embed_pending(
            connection,
            args.index_dir,
            encoder,
            batch_size=args.batch_size,
            shard_size=args.shard_size,
            max_papers=args.max_embed,
        )
        print(f"Embedded {embedded} records")
        manifest = build_faiss_index(
            connection,
            args.index_dir,
            index_type=args.index_type,
        )
        print(
            f"Built {manifest['index_type']} index with "
            f"{manifest['paper_count']} papers in {args.index_dir}"
        )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
