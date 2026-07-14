"""Create the arXiv CS SPECTER2 corpus and publish it to Nebius Object Storage.

Dataset generation only: harvest → embed → FAISS → verify → /output.
Search/loading lives in search_candidates.py.

Nebius Job entrypoint:

  python -m datagen.create_corpus
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
import time
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Protocol, Sequence

import numpy as np
import requests


OAI_URL = "https://export.arxiv.org/oai2"
OAI_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "arxiv": "http://arxiv.org/OAI/arXiv/",
}
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


def iter_snapshot(path: Path, limit: int | None = None) -> Iterator[ArxivPaper]:
    """Read the standard arXiv metadata JSONL snapshot."""
    yielded = 0
    with path.open() as source:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            yield ArxivPaper(
                arxiv_id=item["id"],
                title=item.get("title", ""),
                abstract=item.get("abstract", ""),
                categories=item.get("categories", ""),
                authors=item.get("authors", ""),
                license=item.get("license") or "",
                datestamp=item.get("update_date", ""),
            )
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def _record_text(element: ET.Element | None, path: str) -> str:
    if element is None:
        return ""
    child = element.find(path, OAI_NS)
    return child.text or "" if child is not None else ""


def parse_oai_page(xml_payload: bytes) -> tuple[list[ArxivPaper], str | None]:
    root = ET.fromstring(xml_payload)
    error = root.find("oai:error", OAI_NS)
    if error is not None:
        raise RuntimeError(f"arXiv OAI error: {error.get('code')}: {error.text}")

    papers: list[ArxivPaper] = []
    for record in root.findall(".//oai:record", OAI_NS):
        header = record.find("oai:header", OAI_NS)
        if header is None:
            continue
        identifier = _record_text(header, "oai:identifier")
        datestamp = _record_text(header, "oai:datestamp")
        deleted = header.get("status") == "deleted"
        metadata = record.find("oai:metadata/arxiv:arXiv", OAI_NS)
        if deleted:
            papers.append(
                ArxivPaper(
                    arxiv_id=identifier,
                    title="",
                    abstract="",
                    datestamp=datestamp,
                    deleted=True,
                )
            )
            continue
        if metadata is None:
            continue

        author_names = []
        for author in metadata.findall("arxiv:authors/arxiv:author", OAI_NS):
            keyname = _record_text(author, "arxiv:keyname")
            forenames = _record_text(author, "arxiv:forenames")
            author_names.append(normalize_text(f"{forenames} {keyname}"))

        papers.append(
            ArxivPaper(
                arxiv_id=_record_text(metadata, "arxiv:id") or identifier,
                title=_record_text(metadata, "arxiv:title"),
                abstract=_record_text(metadata, "arxiv:abstract"),
                categories=_record_text(metadata, "arxiv:categories"),
                authors=", ".join(name for name in author_names if name),
                license=_record_text(metadata, "arxiv:license"),
                datestamp=datestamp,
            )
        )

    token_element = root.find(".//oai:resumptionToken", OAI_NS)
    token = (
        token_element.text.strip()
        if token_element is not None and token_element.text
        else None
    )
    return papers, token


def harvest_oai(
    connection: sqlite3.Connection,
    *,
    max_records: int | None = None,
    from_date: str | None = None,
    set_spec: str | None = None,
    request_interval: float = 3.0,
    session: requests.Session | None = None,
) -> int:
    """Harvest an idempotent, resumable sample or update from official OAI-PMH."""
    http = session or requests.Session()
    state_key = json.dumps(
        {"from": from_date or "", "set": set_spec or ""}, sort_keys=True
    )
    state = connection.execute(
        "SELECT resumption_token FROM harvest_state WHERE source = ?",
        (state_key,),
    ).fetchone()
    token = state["resumption_token"] if state else None
    harvested = 0

    while True:
        if token:
            params = {"verb": "ListRecords", "resumptionToken": token}
        else:
            params = {"verb": "ListRecords", "metadataPrefix": "arXiv"}
            if from_date:
                params["from"] = from_date
            if set_spec:
                params["set"] = set_spec

        for attempt in range(5):
            response = http.get(
                OAI_URL,
                params=params,
                headers={
                    "User-Agent": os.getenv(
                        "ARXIV_USER_AGENT", "EmbedXivResearchAgent/0.1"
                    )
                },
                timeout=60,
            )
            if response.status_code not in {429, 500, 502, 503, 504}:
                response.raise_for_status()
                break
            if attempt == 4:
                response.raise_for_status()
            retry_after = response.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else 2**attempt
            except ValueError:
                delay = 2**attempt
            time.sleep(min(max(delay, request_interval), 60))
        papers, next_token = parse_oai_page(response.content)
        upsert_papers(connection, papers)
        harvested += len(papers)

        with connection:
            connection.execute(
                """
                INSERT INTO harvest_state (source, resumption_token, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(source) DO UPDATE SET
                    resumption_token=excluded.resumption_token,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (state_key, next_token),
            )

        if not next_token or (max_records is not None and harvested >= max_records):
            break
        token = next_token
        if request_interval > 0:
            time.sleep(request_interval)

    return harvested


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
                vectors.append(encoder.encode_documents(texts, batch_size=batch_size))
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
                            status='failed', attempts=attempts+1, error=excluded.error,
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
            np.asarray([row["vector_id"] for row in shard_rows], dtype=np.int64),
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

    artifacts = [
        path
        for path in sorted(index_dir.rglob("*"))
        if path.is_file()
        and not path.name.endswith((".tmp", ".sqlite-shm", ".sqlite-wal"))
    ]
    checksums = {}
    for source in artifacts:
        relative = source.relative_to(index_dir)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        source_digest = digest(source)
        if digest(target) != source_digest:
            raise RuntimeError(f"Published checksum mismatch: {relative}")
        checksums[str(relative)] = source_digest

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
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    connection = connect_database(WORK_DIR)
    try:
        print("Harvesting the arXiv computer-science corpus", flush=True)
        harvested = harvest_oai(connection, max_records=None, set_spec="cs")

        encoder = Specter2Encoder(device="cuda")
        embedded = embed_pending(
            connection,
            WORK_DIR,
            encoder,
            batch_size=int(os.getenv("EMBED_BATCH_SIZE", "128")),
            shard_size=int(os.getenv("SHARD_SIZE", "50000")),
        )
        print(f"Embedded {embedded:,} papers", flush=True)
        build_faiss_index(
            connection,
            WORK_DIR,
            index_type=os.getenv("INDEX_TYPE", "ivf-sq8"),
        )
    finally:
        connection.close()

    summary = verify(WORK_DIR)
    destination = publish(WORK_DIR, summary)
    print(f"Published verified generation to {destination}", flush=True)


if __name__ == "__main__":
    main()
