"""Map extraction JSON to SPECTER2 queries, FAISS-search arXiv, optionally enrich via S2.

Loads the latest corpus generation from Nebius Object Storage (published by
datagen.create_corpus), encodes queries with SPECTER2, searches FAISS, and
optionally enriches hits via Semantic Scholar.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Literal, Protocol, Sequence

import requests

import numpy as np
import sys
import traceback
import re

# --- Corpus loader (published Object Storage artifacts) ---

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

def canonical_arxiv_id(value: str) -> str:
    value = value.strip()
    for prefix in ("oai:arXiv.org:", "arXiv:", "ARXIV:"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
            break
    return re.sub(r"v\d+$", "", value)

def connect_database(index_dir: Path) -> sqlite3.Connection:
    """Open an already-published corpus SQLite database."""
    connection = sqlite3.connect(index_dir / "metadata.sqlite")
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS s2_cache (
            arxiv_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            payload TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    return connection


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

def _query_worker_main(
    connection: object, device: str | None, revision: str
) -> None:
    try:
        encoder = Specter2Encoder(device=device, revision=revision)
        connection.send(("ready", encoder.dimension, encoder.fingerprint))
        while True:
            request = connection.recv()
            if request is None:
                return
            texts, batch_size = request
            vectors = encoder.encode_queries(texts, batch_size=batch_size)
            connection.send(("ok", vectors))
    except Exception:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


class Specter2QueryWorker:
    """Keep PyTorch outside the FAISS process on macOS."""

    dimension = DIMENSION

    def __init__(
        self, *, device: str | None = None, revision: str = "main"
    ) -> None:
        import multiprocessing

        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        self._connection = parent
        self._process = context.Process(
            target=_query_worker_main,
            args=(child, device, revision),
            daemon=True,
        )
        self._process.start()
        child.close()
        status, *payload = self._connection.recv()
        if status != "ready":
            self.close()
            raise RuntimeError(f"SPECTER2 query worker failed:\n{payload[0]}")
        dimension, self.fingerprint = payload
        if dimension != self.dimension:
            self.close()
            raise ValueError("Unexpected SPECTER2 query dimension")

    def encode_documents(
        self, texts: Sequence[str], batch_size: int
    ) -> np.ndarray:
        raise NotImplementedError("Query worker cannot encode documents")

    def encode_queries(
        self, texts: Sequence[str], batch_size: int
    ) -> np.ndarray:
        self._connection.send((list(texts), batch_size))
        status, payload = self._connection.recv()
        if status != "ok":
            raise RuntimeError(f"SPECTER2 query worker failed:\n{payload}")
        return payload

    def close(self) -> None:
        if getattr(self, "_connection", None) is not None:
            try:
                if self._process.is_alive():
                    self._connection.send(None)
            except (BrokenPipeError, EOFError):
                pass
            self._process.join(timeout=10)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5)
            self._connection.close()
            self._connection = None

class ArxivIndex:
    def __init__(
        self,
        index_dir: Path,
        *,
        encoder: Encoder | None = None,
        device: str | None = None,
        query_batch_size: int = 32,
    ) -> None:
        self._owns_encoder = encoder is None
        if encoder is not None:
            self.encoder = encoder
        elif sys.platform == "darwin":
            self.encoder = Specter2QueryWorker(device=device)
        else:
            self.encoder = Specter2Encoder(device=device)
        self.index_dir = index_dir
        self.connection = connect_database(index_dir)
        self.manifest = json.loads((index_dir / "manifest.json").read_text())
        self.index = None
        self.query_batch_size = query_batch_size
        if self.encoder.dimension != self.manifest["dimension"]:
            raise ValueError("Query encoder and index dimensions differ")
        if self.encoder.fingerprint != self.manifest["model_fingerprint"]:
            raise ValueError("Query encoder and index model revisions differ")

    def _load_index(self) -> None:
        if self.index is not None:
            return
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("Install requirements.txt to use FAISS") from exc
        self.index = faiss.read_index(str(self.index_dir / "index.faiss"))

    def close(self) -> None:
        self.connection.close()
        if self._owns_encoder and hasattr(self.encoder, "close"):
            self.encoder.close()

    def __enter__(self) -> "ArxivIndex":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def search(self, query_texts: Sequence[str], k: int = 20) -> list[list[dict]]:
        if not query_texts:
            return []
        if k < 1:
            raise ValueError("k must be positive")
        vectors = self.encoder.encode_queries(
            query_texts, batch_size=self.query_batch_size
        )
        # Encode first: loading FAISS before PyTorch inference can crash on macOS
        # because both packages bring their own OpenMP runtime.
        self._load_index()
        distances, ids = self.index.search(vectors, k)
        output = []
        for query_distances, query_ids in zip(distances, ids):
            valid_ids = [int(value) for value in query_ids if value >= 0]
            metadata = {}
            if valid_ids:
                placeholders = ",".join("?" for _ in valid_ids)
                rows = self.connection.execute(
                    f"""
                    SELECT vector_id, arxiv_id, title, abstract, categories,
                           authors, license, datestamp
                    FROM papers
                    WHERE vector_id IN ({placeholders}) AND deleted = 0
                    """,
                    valid_ids,
                ).fetchall()
                metadata = {row["vector_id"]: dict(row) for row in rows}

            hits = []
            for rank, (distance, vector_id) in enumerate(
                zip(query_distances, query_ids), start=1
            ):
                vector_id = int(vector_id)
                if vector_id < 0 or vector_id not in metadata:
                    continue
                hit = metadata[vector_id]
                hit.update(
                    {
                        "distance": float(distance),
                        "rank": rank,
                        "url": f"https://arxiv.org/abs/{hit['arxiv_id']}",
                    }
                )
                hits.append(hit)
            output.append(hits)
        return output

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
SEARCH_ARTIFACTS = ("manifest.json", "index.faiss", "metadata.sqlite")


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Set {name} in .env")
    return value


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


def index_prefix() -> str:
    return os.getenv("NEBIUS_INDEX_PREFIX", "arxiv-index").strip().strip("/")


def bucket_name() -> str:
    return _require_env("NEBIUS_S3_BUCKET")


def latest_generation(client=None) -> str:
    """Read LATEST.json from the bucket and return the generation id."""
    http = client or s3_client()
    prefix = index_prefix()
    response = http.get_object(Bucket=bucket_name(), Key=f"{prefix}/LATEST.json")
    payload = json.loads(response["Body"].read())
    generation = payload.get("generation")
    if not generation:
        raise RuntimeError("LATEST.json is missing a generation field")
    return str(generation)


def download_generation(
    destination: Path, *, generation: str | None = None, client=None
) -> Path:
    """Download FAISS search artifacts for one published generation."""
    http = client or s3_client()
    bucket = bucket_name()
    prefix = index_prefix()
    generation = generation or latest_generation(client=http)
    root = destination / generation
    root.mkdir(parents=True, exist_ok=True)
    base = f"{prefix}/generations/{generation}"
    for name in SEARCH_ARTIFACTS:
        http.download_file(bucket, f"{base}/{name}", str(root / name))
    return root


@contextmanager
def open_index(
    *, device: str | None = None, query_batch_size: int = 32
) -> Iterator[ArxivIndex]:
    """Fetch the latest corpus from Object Storage and open it for search."""
    with tempfile.TemporaryDirectory(prefix="embedxiv-index-") as temporary:
        index_dir = download_generation(Path(temporary))
        with ArxivIndex(
            index_dir,
            device=device,
            query_batch_size=query_batch_size,
        ) as index:
            yield index


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


def specter2_query_text(*parts: str) -> str:
    """Join extraction fields into short raw text for the adhoc query adapter."""
    chunks = [part.strip().rstrip(".") for part in parts if part and part.strip()]
    if not chunks:
        return ""
    return ". ".join(chunks) + "."


def build_queries(
    problem: ResearchProblem, problem_index: int = 0
) -> list[SearchQuery]:
    """Map extracted hierarchy nodes to SPECTER2 adhoc query strings.

    The adhoc query adapter expects short raw text, not labeled templates.
    Each query type uses the extracted sentence(s) that best match the search
    intent; see README for the mapping.
    """
    queries = [
        SearchQuery(
            level="problem",
            query_type="direct",
            query=specter2_query_text(
                problem.problem,
                problem.domain,
                ", ".join(problem.keywords),
            ),
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
                    query=specter2_query_text(claim.claim),
                    source_text=claim.claim,
                    problem_index=problem_index,
                    claim_index=claim_index,
                ),
                SearchQuery(
                    level="claim",
                    query_type="functional",
                    query=specter2_query_text(claim.functional_role),
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
                        query=specter2_query_text(detail.detail),
                        source_text=detail.detail,
                        problem_index=problem_index,
                        claim_index=claim_index,
                        detail_index=detail_index,
                    ),
                    SearchQuery(
                        level="implementation",
                        query_type="alternative",
                        query=specter2_query_text(detail.functional_role),
                        source_text=detail.functional_role,
                        problem_index=problem_index,
                        claim_index=claim_index,
                        detail_index=detail_index,
                    ),
                ]
            )

    return queries


def search_candidates(
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


def search_all_candidates(
    problems: list[ResearchProblem],
    index: VectorIndex,
    *,
    limit: int = TOP_K_PER_QUERY,
    enrich_s2: bool = True,
    api_key: str | None = None,
    session: requests.Session | None = None,
    request_delay: float = DEFAULT_REQUEST_DELAY,
) -> list[dict]:
    """Search all hierarchy candidates, deduplicate, then optionally enrich."""
    candidates_by_id: dict[str, dict] = {}
    for problem_index, problem in enumerate(problems):
        for candidate in search_candidates(
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
        description=(
            "Search the Nebius Object Storage SPECTER2 corpus using extraction JSON "
            "(problems/claims/details)."
        )
    )
    parser.add_argument("extraction_json", help="JSON produced by extract_claims.py")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    parser.add_argument("-o", "--output", default="candidate_results.json")
    parser.add_argument("--limit", type=int, default=TOP_K_PER_QUERY)
    parser.add_argument("--no-s2", action="store_true")
    parser.add_argument("--request-delay", type=float, default=DEFAULT_REQUEST_DELAY)
    args = parser.parse_args()

    load_local_env()
    extraction = ExtractionResult.model_validate_json(
        Path(args.extraction_json).read_text()
    )
    with open_index(device=args.device) as index:
        candidates = search_all_candidates(
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
