"""GPU Job: embed preloaded arXiv metadata with SPECTER2 and publish FAISS.

Expects metadata.sqlite already in Object Storage (from datagen.preload_metadata),
mounted at INDEX_OUTPUT_DIR (default /output/arxiv-index).

  python -m datagen.embed_corpus
"""

from __future__ import annotations

import hashlib
import shutil
import json
import os
import random
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Protocol, Sequence

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - dependency should be installed
    tqdm = None  # type: ignore[assignment]


DIMENSION = 768
BASE_MODEL = "allenai/specter2_base"
DOCUMENT_ADAPTER = "allenai/specter2"
QUERY_ADAPTER = "allenai/specter2_adhoc_query"


class Encoder(Protocol):
    dimension: int
    fingerprint: str

    def encode_documents(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        ...

    def encode_queries(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        ...


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


def canonical_arxiv_id(value: str) -> str:
    value = value.strip()
    for prefix in ("oai:arXiv.org:", "arXiv:", "ARXIV:"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
            break
    return re.sub(r"v\d+$", "", value)


def normalize_text(value: str) -> str:
    return " ".join(value.split())


def document_text(title: str, abstract: str, separator: str = "[SEP]") -> str:
    return f"{normalize_text(title)} {separator} {normalize_text(abstract)}".strip()


def content_hash(title: str, abstract: str) -> str:
    payload = f"{normalize_text(title)}\0{normalize_text(abstract)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def _resolve_revision(repo_id: str, revision: str) -> str:
    try:
        from huggingface_hub import model_info

        return model_info(repo_id, revision=revision).sha
    except Exception:
        return revision


class Specter2Encoder:
    """Asymmetric SPECTER2 document and ad-hoc query encoder."""

    dimension = DIMENSION

    def __init__(
        self,
        *,
        device: str | None = None,
        revision: str = "main",
    ) -> None:
        # Standard HTTP is more portable than the optional Xet transfer backend.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        try:
            import torch
            from adapters import AutoAdapterModel
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install requirements.txt to use SPECTER2"
            ) from exc

        base_revision = _resolve_revision(BASE_MODEL, revision)
        document_revision = _resolve_revision(DOCUMENT_ADAPTER, revision)
        query_revision = _resolve_revision(QUERY_ADAPTER, revision)
        self.fingerprint = "|".join(
            (base_revision, document_revision, query_revision)
        )
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, revision=base_revision
        )
        self.model = AutoAdapterModel.from_pretrained(
            BASE_MODEL, revision=base_revision
        )
        self.document_adapter = self.model.load_adapter(
            DOCUMENT_ADAPTER,
            source="hf",
            load_as="specter2_document",
            revision=document_revision,
        )
        self.query_adapter = self.model.load_adapter(
            QUERY_ADAPTER,
            source="hf",
            load_as="specter2_query",
            revision=query_revision,
        )
        self.device = device or self._default_device()
        self.model.to(self.device)
        self.model.eval()

    def _default_device(self) -> str:
        if self.torch.cuda.is_available():
            return "cuda"
        if (
            hasattr(self.torch.backends, "mps")
            and self.torch.backends.mps.is_available()
        ):
            return "mps"
        return "cpu"

    def _encode(
        self, texts: Sequence[str], *, adapter: str, batch_size: int
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        self.model.set_active_adapters(adapter)
        batches = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = self.tokenizer(
                    list(texts[start : start + batch_size]),
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                    return_token_type_ids=False,
                )
                batch = {key: value.to(self.device) for key, value in batch.items()}
                output = self.model(**batch)
                batches.append(
                    output.last_hidden_state[:, 0, :]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
        return np.ascontiguousarray(np.concatenate(batches), dtype=np.float32)

    def encode_documents(
        self, texts: Sequence[str], batch_size: int = 32
    ) -> np.ndarray:
        return self._encode(
            texts, adapter=self.document_adapter, batch_size=batch_size
        )

    def encode_queries(
        self, texts: Sequence[str], batch_size: int = 32
    ) -> np.ndarray:
        return self._encode(texts, adapter=self.query_adapter, batch_size=batch_size)



def _next_shard_number(shard_dir: Path) -> int:
    numbers = []
    for path in shard_dir.glob("embeddings-*.npy"):
        try:
            numbers.append(int(path.stem.rsplit("-", 1)[1]))
        except ValueError:
            continue
    return max(numbers, default=-1) + 1


def _atomic_save_array(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as destination:
        np.save(destination, array)
    os.replace(temporary, path)


def embed_pending(
    connection: sqlite3.Connection,
    index_dir: Path,
    encoder: Encoder,
    *,
    batch_size: int = 32,
    shard_size: int = 50_000,
    max_papers: int | None = None,
) -> int:
    shard_dir = index_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    rows = connection.execute(
        """
        SELECT p.vector_id, p.title, p.abstract, p.content_hash
        FROM papers p
        LEFT JOIN embedding_jobs j ON p.vector_id = j.vector_id
        WHERE p.deleted = 0
          AND (
            j.vector_id IS NULL OR j.status != 'done'
            OR j.content_hash != p.content_hash
            OR j.model_fingerprint != ?
          )
        ORDER BY p.vector_id
        """,
        (encoder.fingerprint,),
    ).fetchall()
    if max_papers is not None:
        rows = rows[:max_papers]

    embedded = 0
    next_shard = _next_shard_number(shard_dir)
    total = len(rows)
    print(
        f"Embedding {total:,} pending papers "
        f"(batch_size={batch_size}, shard_size={shard_size})",
        flush=True,
    )
    if total == 0:
        return 0

    # Update only after a shard is fully encoded + persisted so failures cannot
    # advance the bar past work that will be retried.
    if tqdm is None:
        progress = None
        print(
            "tqdm is not installed; progress bar disabled "
            "(pip install tqdm or rebuild the Job image)",
            flush=True,
        )
    else:
        progress = tqdm(
            total=total,
            unit="paper",
            desc="SPECTER2 embed",
            file=sys.stdout,
            dynamic_ncols=True,
            mininterval=5.0,
            ascii=True,
            leave=True,
        )
    try:
        for shard_start in range(0, len(rows), shard_size):
            shard_rows = rows[shard_start : shard_start + shard_size]
            vectors = []
            try:
                for batch_start in range(0, len(shard_rows), batch_size):
                    batch_rows = shard_rows[batch_start : batch_start + batch_size]
                    texts = [
                        document_text(row["title"], row["abstract"])
                        for row in batch_rows
                    ]
                    vectors.append(
                        encoder.encode_documents(texts, batch_size=batch_size)
                    )
                matrix = np.ascontiguousarray(
                    np.concatenate(vectors), dtype=np.float32
                )
            except Exception as exc:
                with connection:
                    for row in shard_rows:
                        connection.execute(
                            """
                            INSERT INTO embedding_jobs (
                                vector_id, content_hash, model_fingerprint, status,
                                attempts, error
                            ) VALUES (?, ?, ?, 'failed', 1, ?)
                            ON CONFLICT(vector_id) DO UPDATE SET
                                status='failed', attempts=attempts+1,
                                error=excluded.error,
                                updated_at=CURRENT_TIMESTAMP
                            """,
                            (
                                row["vector_id"],
                                row["content_hash"],
                                encoder.fingerprint,
                                str(exc),
                            ),
                        )
                raise

            shard_name = f"embeddings-{next_shard:06d}.npy"
            ids_name = f"vector-ids-{next_shard:06d}.npy"
            _atomic_save_array(shard_dir / shard_name, matrix)
            _atomic_save_array(
                shard_dir / ids_name,
                np.asarray(
                    [row["vector_id"] for row in shard_rows], dtype=np.int64
                ),
            )
            with connection:
                for offset, row in enumerate(shard_rows):
                    connection.execute(
                        """
                        INSERT INTO embedding_jobs (
                            vector_id, content_hash, model_fingerprint, status,
                            attempts, error, shard_name, row_offset
                        ) VALUES (?, ?, ?, 'done', 1, NULL, ?, ?)
                        ON CONFLICT(vector_id) DO UPDATE SET
                            content_hash=excluded.content_hash,
                            model_fingerprint=excluded.model_fingerprint,
                            status='done', attempts=attempts+1, error=NULL,
                            shard_name=excluded.shard_name,
                            row_offset=excluded.row_offset,
                            updated_at=CURRENT_TIMESTAMP
                        """,
                        (
                            row["vector_id"],
                            row["content_hash"],
                            encoder.fingerprint,
                            shard_name,
                            offset,
                        ),
                    )
            embedded += len(shard_rows)
            next_shard += 1
            if progress is not None:
                progress.update(len(shard_rows))
                progress.set_postfix(shard=next_shard - 1, refresh=False)
            else:
                print(
                    f"Embedded {embedded:,}/{total:,} "
                    f"({100.0 * embedded / total:.1f}%)",
                    flush=True,
                )
    finally:
        if progress is not None:
            progress.close()
    return embedded


def _current_embedding_rows(
    connection: sqlite3.Connection,
) -> tuple[list[sqlite3.Row], str]:
    rows = connection.execute(
        """
        SELECT p.vector_id, j.shard_name, j.row_offset, j.model_fingerprint
        FROM papers p
        JOIN embedding_jobs j ON p.vector_id = j.vector_id
        WHERE p.deleted = 0 AND j.status = 'done'
          AND p.content_hash = j.content_hash
        ORDER BY p.vector_id
        """
    ).fetchall()
    if not rows:
        raise RuntimeError("No current embeddings are available")

    fingerprints = {row["model_fingerprint"] for row in rows}
    if len(fingerprints) != 1:
        raise RuntimeError("Current embeddings use multiple model fingerprints")
    return rows, fingerprints.pop()


def _vectors_for_rows(
    rows: Sequence[sqlite3.Row], index_dir: Path
) -> tuple[np.ndarray, np.ndarray]:
    loaded_shards: dict[str, np.ndarray] = {}
    vectors = []
    vector_ids = []
    for row in rows:
        shard_name = row["shard_name"]
        if shard_name not in loaded_shards:
            loaded_shards[shard_name] = np.load(
                index_dir / "shards" / shard_name, mmap_mode="r"
            )
        vectors.append(loaded_shards[shard_name][row["row_offset"]])
        vector_ids.append(row["vector_id"])
    return (
        np.ascontiguousarray(np.stack(vectors), dtype=np.float32),
        np.asarray(vector_ids, dtype=np.int64),
    )


HNSW_INDEX_NAME = "papers_embedding_hnsw"


def database_url() -> str | None:
    value = os.getenv("DATABASE_URL", "").strip()
    return value or None


def require_database_url() -> str:
    url = database_url()
    if not url:
        raise RuntimeError("Set DATABASE_URL in the Job environment")
    return url


def connect_pg():
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError(
            "Install requirements.txt (psycopg) for Postgres publish"
        ) from exc

    return psycopg.connect(require_database_url(), row_factory=dict_row)


def pg_vector_literal(vector: Sequence[float] | np.ndarray) -> str:
    values = np.asarray(vector, dtype=np.float32).tolist()
    return "[" + ",".join(f"{value:.8g}" for value in values) + "]"


def ensure_pg_schema(connection, *, create_index: bool = False) -> None:
    with connection.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS corpus_manifest (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                payload JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS papers (
                vector_id BIGINT PRIMARY KEY,
                arxiv_id TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                abstract TEXT NOT NULL DEFAULT '',
                categories TEXT NOT NULL DEFAULT '',
                authors TEXT NOT NULL DEFAULT '',
                license TEXT NOT NULL DEFAULT '',
                datestamp TEXT NOT NULL DEFAULT '',
                deleted BOOLEAN NOT NULL DEFAULT FALSE,
                content_hash TEXT NOT NULL,
                model_fingerprint TEXT NOT NULL,
                embedding vector({DIMENSION}) NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS papers_arxiv_id_idx ON papers (arxiv_id)"
        )
        if create_index:
            cursor.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {HNSW_INDEX_NAME}
                ON papers USING hnsw (embedding vector_l2_ops)
                WITH (m = 16, ef_construction = 64)
                """
            )
    connection.commit()


def write_pg_manifest(connection, manifest: dict) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO corpus_manifest (id, payload, updated_at)
            VALUES (1, %s::jsonb, NOW())
            ON CONFLICT (id) DO UPDATE SET
                payload = EXCLUDED.payload,
                updated_at = NOW()
            """,
            (json.dumps(manifest),),
        )
    connection.commit()


def read_pg_manifest(connection) -> dict:
    with connection.cursor() as cursor:
        cursor.execute("SELECT payload FROM corpus_manifest WHERE id = 1")
        row = cursor.fetchone()
    if not row:
        raise RuntimeError("Postgres corpus is missing corpus_manifest")
    payload = row["payload"]
    if isinstance(payload, str):
        return json.loads(payload)
    return dict(payload)


def _embedded_paper_rows(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    rows = connection.execute(
        """
        SELECT
            p.vector_id,
            p.arxiv_id,
            p.title,
            p.abstract,
            p.categories,
            p.authors,
            p.license,
            p.datestamp,
            p.deleted,
            p.content_hash,
            j.shard_name,
            j.row_offset,
            j.model_fingerprint
        FROM papers p
        JOIN embedding_jobs j ON p.vector_id = j.vector_id
        WHERE p.deleted = 0
          AND j.status = 'done'
          AND p.content_hash = j.content_hash
        ORDER BY p.vector_id
        """
    ).fetchall()
    if not rows:
        raise RuntimeError("No embedded papers are available to publish")
    fingerprints = {row["model_fingerprint"] for row in rows}
    if len(fingerprints) != 1:
        raise RuntimeError("Embedded papers use multiple model fingerprints")
    return rows


def rebuild_pg_hnsw_index(connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute(f"DROP INDEX IF EXISTS {HNSW_INDEX_NAME}")
        cursor.execute(
            f"""
            CREATE INDEX {HNSW_INDEX_NAME}
            ON papers USING hnsw (embedding vector_l2_ops)
            WITH (m = 16, ef_construction = 64)
            """
        )
    connection.commit()


def publish_postgres(
    sqlite_connection: sqlite3.Connection,
    index_dir: Path,
    manifest: dict,
    *,
    batch_size: int = 5_000,
) -> dict:
    """Upsert embedded papers from workspace shards into Managed Postgres."""
    rows = _embedded_paper_rows(sqlite_connection)
    model_fingerprint = rows[0]["model_fingerprint"]
    pg = connect_pg()
    try:
        ensure_pg_schema(pg, create_index=False)
        published = 0
        sql = """
            INSERT INTO papers (
                vector_id, arxiv_id, title, abstract, categories, authors,
                license, datestamp, deleted, content_hash, model_fingerprint,
                embedding
            ) VALUES (
                %(vector_id)s, %(arxiv_id)s, %(title)s, %(abstract)s,
                %(categories)s, %(authors)s, %(license)s, %(datestamp)s,
                %(deleted)s, %(content_hash)s, %(model_fingerprint)s,
                %(embedding)s::vector
            )
            ON CONFLICT (vector_id) DO UPDATE SET
                arxiv_id = EXCLUDED.arxiv_id,
                title = EXCLUDED.title,
                abstract = EXCLUDED.abstract,
                categories = EXCLUDED.categories,
                authors = EXCLUDED.authors,
                license = EXCLUDED.license,
                datestamp = EXCLUDED.datestamp,
                deleted = EXCLUDED.deleted,
                content_hash = EXCLUDED.content_hash,
                model_fingerprint = EXCLUDED.model_fingerprint,
                embedding = EXCLUDED.embedding,
                updated_at = NOW()
        """
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            vectors, _ = _vectors_for_rows(chunk, index_dir)
            payload = []
            for row, vector in zip(chunk, vectors):
                payload.append(
                    {
                        "vector_id": int(row["vector_id"]),
                        "arxiv_id": row["arxiv_id"],
                        "title": row["title"],
                        "abstract": row["abstract"],
                        "categories": row["categories"],
                        "authors": row["authors"],
                        "license": row["license"],
                        "datestamp": row["datestamp"],
                        "deleted": bool(row["deleted"]),
                        "content_hash": row["content_hash"],
                        "model_fingerprint": model_fingerprint,
                        "embedding": pg_vector_literal(vector),
                    }
                )
            with pg.cursor() as cursor:
                cursor.executemany(sql, payload)
            pg.commit()
            published += len(payload)
            print(
                f"Published {published:,}/{len(rows):,} papers to Postgres",
                flush=True,
            )

        summary = {
            **manifest,
            "paper_count": published,
            "model_fingerprint": model_fingerprint,
            "storage": "postgres_pgvector",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        write_pg_manifest(pg, summary)
        print("Building HNSW index in Postgres…", flush=True)
        rebuild_pg_hnsw_index(pg)
        return summary
    finally:
        pg.close()


def verify_postgres(manifest: dict) -> dict:
    pg = connect_pg()
    try:
        with pg.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS count FROM papers WHERE NOT deleted")
            papers = cursor.fetchone()["count"]
            cursor.execute(
                "SELECT 1 FROM pg_indexes WHERE indexname = %s",
                (HNSW_INDEX_NAME,),
            )
            has_index = cursor.fetchone() is not None
        expected = int(manifest["paper_count"])
        if papers != expected:
            raise RuntimeError(
                f"Postgres paper count mismatch: db={papers}, manifest={expected}"
            )
        if not has_index:
            raise RuntimeError("Postgres corpus is missing the HNSW index")
        return {**manifest, "verified_paper_count": papers}
    finally:
        pg.close()


def build_faiss_index(
    connection: sqlite3.Connection,
    index_dir: Path,
    *,
    index_type: str = "flat",
    training_size: int = 300_000,
    seed: int = 17,
) -> dict:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt to use FAISS") from exc

    rows, fingerprint = _current_embedding_rows(connection)

    normalized_type = index_type.lower()
    if normalized_type == "flat":
        index = faiss.IndexIDMap2(faiss.IndexFlatL2(DIMENSION))
        factory = "IDMap2,Flat"
    else:
        factory = {
            "ivf": "IVF4096,Flat",
            "ivf-sq8": "IVF4096,SQ8",
        }.get(normalized_type, index_type)
        index = faiss.index_factory(DIMENSION, factory, faiss.METRIC_L2)
        sample_size = min(training_size, len(rows))
        if sample_size < 4096 and "IVF4096" in factory.upper():
            raise ValueError(
                "IVF4096 needs at least 4096 training vectors; use flat for samples"
            )
        randomizer = random.Random(seed)
        training_rows = randomizer.sample(rows, sample_size)
        training_vectors, _ = _vectors_for_rows(training_rows, index_dir)
        index.train(training_vectors)

    add_batch_size = 50_000
    for start in range(0, len(rows), add_batch_size):
        vectors, vector_ids = _vectors_for_rows(
            rows[start : start + add_batch_size], index_dir
        )
        if vectors.shape[1] != DIMENSION:
            raise ValueError(
                f"Expected {DIMENSION} dimensions, got {vectors.shape[1]}"
            )
        index.add_with_ids(vectors, vector_ids)

    temporary_index = index_dir / "index.faiss.tmp"
    faiss.write_index(index, str(temporary_index))
    os.replace(temporary_index, index_dir / "index.faiss")
    manifest = {
        "dimension": DIMENSION,
        "metric": "l2",
        "index_type": factory,
        "paper_count": len(rows),
        "model_fingerprint": fingerprint,
        "document_adapter": DOCUMENT_ADAPTER,
        "query_adapter": QUERY_ADAPTER,
    }
    temporary_manifest = index_dir / "manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary_manifest, index_dir / "manifest.json")
    with connection:
        connection.execute(
            "INSERT INTO index_manifest (payload) VALUES (?)",
            (json.dumps(manifest),),
        )
    return manifest



WORK_DIR = Path(os.getenv("INDEX_WORK_DIR", "/workspace/arxiv-index"))
OUTPUT_DIR = Path(os.getenv("INDEX_OUTPUT_DIR", "/output/arxiv-index"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify(index_dir: Path) -> dict:
    manifest_path = index_dir / "manifest.json"
    index_path = index_dir / "index.faiss"
    database_path = index_dir / "metadata.sqlite"
    for path in (manifest_path, index_path, database_path):
        if not path.is_file():
            raise RuntimeError(f"Missing artifact: {path}")

    manifest = json.loads(manifest_path.read_text())
    with sqlite3.connect(database_path) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        papers = connection.execute(
            "SELECT COUNT(*) FROM papers WHERE deleted = 0"
        ).fetchone()[0]
        embeddings = connection.execute(
            """
            SELECT COUNT(*)
            FROM papers p
            JOIN embedding_jobs j ON p.vector_id = j.vector_id
            WHERE p.deleted = 0
              AND j.status = 'done'
              AND p.content_hash = j.content_hash
            """
        ).fetchone()[0]

    import faiss

    indexed = faiss.read_index(str(index_path)).ntotal
    expected = manifest["paper_count"]
    if integrity != "ok":
        raise RuntimeError(f"SQLite integrity check failed: {integrity}")
    if not (papers == embeddings == indexed == expected):
        raise RuntimeError(
            "Count mismatch: "
            f"papers={papers}, embeddings={embeddings}, "
            f"indexed={indexed}, manifest={expected}"
        )
    return {
        **manifest,
        "sqlite_integrity": integrity,
        "verified_paper_count": papers,
    }


def publish(index_dir: Path, summary: dict) -> Path:
    generation = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = OUTPUT_DIR / "generations" / generation
    destination.mkdir(parents=True, exist_ok=False)

    # Only publish what search needs (not intermediate embedding shards).
    publish_names = ("metadata.sqlite", "index.faiss", "manifest.json")
    checksums = {}
    for name in publish_names:
        source = index_dir / name
        if not source.is_file():
            raise RuntimeError(f"Missing artifact to publish: {source}")
        target = destination / name
        shutil.copyfile(source, target)
        source_digest = digest(source)
        if digest(target) != source_digest:
            raise RuntimeError(f"Published checksum mismatch: {name}")
        checksums[name] = source_digest

    (destination / "checksums.sha256.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n"
    )
    success = {
        **summary,
        "generation": generation,
        "artifact_count": len(checksums),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    (destination / "SUCCESS.json").write_text(json.dumps(success, indent=2) + "\n")
    (OUTPUT_DIR / "LATEST.json").write_text(json.dumps(success, indent=2) + "\n")
    return destination



def main() -> None:
    """Read preloaded metadata from the bucket mount, embed on GPU, publish."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    use_postgres = bool(database_url())

    preload_db = OUTPUT_DIR / "metadata.sqlite"
    ready = OUTPUT_DIR / "PRELOAD_READY.json"
    if not preload_db.is_file():
        raise RuntimeError(
            f"Missing {preload_db}. Run datagen.preload_metadata on your laptop first."
        )
    if not ready.is_file():
        raise RuntimeError(f"Missing {ready}. Preload did not finish cleanly.")

    # Copy preload into workspace so shards stay off the bucket until publish.
    work_db = WORK_DIR / "metadata.sqlite"
    if work_db.resolve() != preload_db.resolve():
        shutil.copyfile(preload_db, work_db)

    connection = connect_database(WORK_DIR)
    try:
        papers = connection.execute(
            "SELECT COUNT(*) FROM papers WHERE deleted = 0"
        ).fetchone()[0]
        print(f"Loaded preload with {papers:,} papers", flush=True)

        encoder = Specter2Encoder(device="cuda")
        print("SPECTER2 encoder ready", flush=True)
        embedded = embed_pending(
            connection,
            WORK_DIR,
            encoder,
            batch_size=int(os.getenv("EMBED_BATCH_SIZE", "128")),
            shard_size=int(os.getenv("SHARD_SIZE", "50000")),
        )
        print(f"Embedded {embedded:,} papers", flush=True)

        embedded_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM papers p
            JOIN embedding_jobs j ON p.vector_id = j.vector_id
            WHERE p.deleted = 0
              AND j.status = 'done'
              AND p.content_hash = j.content_hash
            """
        ).fetchone()[0]
        manifest = {
            "dimension": DIMENSION,
            "metric": "l2",
            "index_type": "hnsw" if use_postgres else os.getenv("INDEX_TYPE", "ivf-sq8"),
            "paper_count": embedded_count,
            "model_fingerprint": encoder.fingerprint,
            "document_adapter": DOCUMENT_ADAPTER,
            "query_adapter": QUERY_ADAPTER,
        }

        if use_postgres:
            print("Publishing embeddings to Postgres…", flush=True)
            summary = publish_postgres(
                connection,
                WORK_DIR,
                manifest,
                batch_size=int(os.getenv("PG_BATCH_SIZE", "5000")),
            )
            summary = verify_postgres(summary)
            print(
                f"Published verified Postgres corpus ({summary['paper_count']:,} papers)",
                flush=True,
            )
            return

        print("Building FAISS index…", flush=True)
        build_faiss_index(
            connection,
            WORK_DIR,
            index_type=os.getenv("INDEX_TYPE", "ivf-sq8"),
        )
        print("FAISS index written", flush=True)
    finally:
        connection.close()

    summary = verify(WORK_DIR)
    destination = publish(WORK_DIR, summary)
    print(f"Published verified generation to {destination}", flush=True)


if __name__ == "__main__":
    main()
