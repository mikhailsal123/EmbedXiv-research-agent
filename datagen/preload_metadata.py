"""CPU preload: Kaggle arXiv metadata → SQLite → Nebius Object Storage.

Run on your laptop (no GPU, no arXiv rate limits):

  python -m datagen.preload_metadata \\
    --metadata-jsonl /path/to/arxiv-metadata-oai-snapshot.json

Download the snapshot from the Cornell Kaggle arXiv metadata dataset first.
By default only computer-science papers (categories containing cs.*) are kept.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator


INDEX_PREFIX = "arxiv-index"


@dataclass(frozen=True)
class ArxivPaper:
    arxiv_id: str
    title: str
    abstract: str
    categories: str = ""
    authors: str = ""
    license: str = ""
    datestamp: str = ""
    deleted: bool = False


def load_local_env(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Set {name} in .env")
    return value


def canonical_arxiv_id(value: str) -> str:
    value = value.strip()
    for prefix in ("oai:arXiv.org:", "arXiv:", "ARXIV:"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
            break
    return re.sub(r"v\d+$", "", value)


def normalize_text(value: str) -> str:
    return " ".join(value.split())


def content_hash(title: str, abstract: str) -> str:
    payload = f"{normalize_text(title)}\0{normalize_text(abstract)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_computer_science(categories: str) -> bool:
    return any(
        token == "cs" or token.startswith("cs.")
        for token in categories.split()
    )


def connect_database(index_dir: Path) -> sqlite3.Connection:
    index_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(index_dir / "metadata.sqlite")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS papers (
            vector_id INTEGER PRIMARY KEY AUTOINCREMENT,
            arxiv_id TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            abstract TEXT NOT NULL,
            categories TEXT NOT NULL DEFAULT '',
            authors TEXT NOT NULL DEFAULT '',
            license TEXT NOT NULL DEFAULT '',
            datestamp TEXT NOT NULL DEFAULT '',
            deleted INTEGER NOT NULL DEFAULT 0,
            content_hash TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS embedding_jobs (
            vector_id INTEGER PRIMARY KEY REFERENCES papers(vector_id),
            content_hash TEXT NOT NULL,
            model_fingerprint TEXT NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            shard_name TEXT,
            row_offset INTEGER,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS harvest_state (
            source TEXT PRIMARY KEY,
            resumption_token TEXT,
            last_datestamp TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS s2_cache (
            arxiv_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            payload TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS index_manifest (
            generation INTEGER PRIMARY KEY AUTOINCREMENT,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    return connection


def upsert_papers(
    connection: sqlite3.Connection, papers: Iterable[ArxivPaper]
) -> int:
    changed = 0
    with connection:
        for paper in papers:
            arxiv_id = canonical_arxiv_id(paper.arxiv_id)
            title = normalize_text(paper.title)
            abstract = normalize_text(paper.abstract)
            digest = content_hash(title, abstract)
            existing = connection.execute(
                "SELECT vector_id, content_hash, deleted FROM papers WHERE arxiv_id = ?",
                (arxiv_id,),
            ).fetchone()
            if existing and existing["content_hash"] == digest and bool(
                existing["deleted"]
            ) == paper.deleted:
                continue

            connection.execute(
                """
                INSERT INTO papers (
                    arxiv_id, title, abstract, categories, authors, license,
                    datestamp, deleted, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(arxiv_id) DO UPDATE SET
                    title=excluded.title,
                    abstract=excluded.abstract,
                    categories=excluded.categories,
                    authors=excluded.authors,
                    license=excluded.license,
                    datestamp=excluded.datestamp,
                    deleted=excluded.deleted,
                    content_hash=excluded.content_hash
                """,
                (
                    arxiv_id,
                    title,
                    abstract,
                    normalize_text(paper.categories),
                    normalize_text(paper.authors),
                    paper.license.strip(),
                    paper.datestamp.strip(),
                    int(paper.deleted),
                    digest,
                ),
            )
            row = connection.execute(
                "SELECT vector_id FROM papers WHERE arxiv_id = ?", (arxiv_id,)
            ).fetchone()
            connection.execute(
                """
                UPDATE embedding_jobs
                SET status='pending', content_hash=?, error=NULL,
                    shard_name=NULL, row_offset=NULL, updated_at=CURRENT_TIMESTAMP
                WHERE vector_id=?
                """,
                (digest, row["vector_id"]),
            )
            changed += 1
    return changed


def iter_snapshot(
    path: Path,
    *,
    cs_only: bool = True,
    limit: int | None = None,
) -> Iterator[ArxivPaper]:
    """Read the Kaggle / OAI-style arXiv metadata JSONL snapshot."""
    yielded = 0
    with path.open() as source:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            categories = item.get("categories", "") or ""
            if cs_only and not is_computer_science(categories):
                continue
            yield ArxivPaper(
                arxiv_id=item["id"],
                title=item.get("title", "") or "",
                abstract=item.get("abstract", "") or "",
                categories=categories,
                authors=item.get("authors", "") or "",
                license=item.get("license") or "",
                datestamp=item.get("update_date", "") or "",
            )
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def s3_client():
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt (boto3) for Object Storage") from exc

    return boto3.client(
        "s3",
        endpoint_url=_require_env("NEBIUS_S3_ENDPOINT_URL"),
        region_name=os.getenv("NEBIUS_S3_REGION", "").strip() or None,
        aws_access_key_id=_require_env("NEBIUS_S3_ACCESS_KEY_ID"),
        aws_secret_access_key=_require_env("NEBIUS_S3_SECRET_ACCESS_KEY"),
    )


def upload_preload(local_dir: Path, *, client=None) -> dict:
    """Upload metadata.sqlite + PRELOAD_READY.json to the Nebius bucket."""
    http = client or s3_client()
    bucket = _require_env("NEBIUS_S3_BUCKET")
    db_path = local_dir / "metadata.sqlite"
    ready_path = local_dir / "PRELOAD_READY.json"
    if not db_path.is_file() or not ready_path.is_file():
        raise RuntimeError("Missing metadata.sqlite or PRELOAD_READY.json")

    # Flush WAL into the main DB file before upload.
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    keys = {
        f"{INDEX_PREFIX}/metadata.sqlite": db_path,
        f"{INDEX_PREFIX}/PRELOAD_READY.json": ready_path,
    }
    for key, path in keys.items():
        http.upload_file(str(path), bucket, key)
        print(f"Uploaded s3://{bucket}/{key}", flush=True)
    return {"bucket": bucket, "keys": list(keys)}


def build_preload(
    metadata_jsonl: Path,
    local_dir: Path,
    *,
    cs_only: bool = True,
    limit: int | None = None,
) -> dict:
    local_dir.mkdir(parents=True, exist_ok=True)
    connection = connect_database(local_dir)
    try:
        changed = upsert_papers(
            connection,
            iter_snapshot(metadata_jsonl, cs_only=cs_only, limit=limit),
        )
        papers = connection.execute(
            "SELECT COUNT(*) FROM papers WHERE deleted = 0"
        ).fetchone()[0]
    finally:
        connection.close()

    ready = {
        "papers": papers,
        "changed": changed,
        "cs_only": cs_only,
        "source": str(metadata_jsonl),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    (local_dir / "PRELOAD_READY.json").write_text(
        json.dumps(ready, indent=2) + "\n"
    )
    return ready


def main() -> None:
    load_local_env()
    parser = argparse.ArgumentParser(
        description="Preload CS arXiv metadata into Nebius Object Storage."
    )
    parser.add_argument(
        "--metadata-jsonl",
        type=Path,
        required=True,
        help="Kaggle arxiv-metadata-oai-snapshot.json (JSONL)",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("data/arxiv-preload"),
        help="Local working directory for the SQLite DB before upload",
    )
    parser.add_argument(
        "--all-categories",
        action="store_true",
        help="Keep every category (default: computer science only)",
    )
    parser.add_argument("--limit", type=int, help="Optional cap for a smoke test")
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Build local SQLite only; do not upload to Object Storage",
    )
    args = parser.parse_args()

    if not args.metadata_jsonl.is_file():
        raise SystemExit(f"Metadata file not found: {args.metadata_jsonl}")

    ready = build_preload(
        args.metadata_jsonl,
        args.local_dir,
        cs_only=not args.all_categories,
        limit=args.limit,
    )
    print(
        f"Preloaded {ready['papers']:,} papers "
        f"(cs_only={ready['cs_only']}) into {args.local_dir}",
        flush=True,
    )
    if args.skip_upload:
        return
    upload_preload(args.local_dir)
    print("Preload uploaded. Run the GPU embed Job next.", flush=True)


if __name__ == "__main__":
    main()
