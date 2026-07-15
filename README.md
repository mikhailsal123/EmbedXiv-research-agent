# EmbedXiv Research Agent

## Purpose

Given a private abstract, preprint, or paper, extract:

- **Problem**: the limitation or research gap.
- **Claims**: independent conceptual ideas addressing that problem.
- **Implementation details**: concrete mechanisms realizing each claim.

Search for useful research at each level:

1. Papers addressing the same problem.
2. Ideas that support, extend, qualify, or challenge each claim.
3. Alternative mechanisms serving each implementation detail's role.

## Implemented pipeline

```text
Paper or abstract
  → Qwen extraction: problem → claims → implementation details
  → hierarchy-aware query texts
  → SPECTER2 ad-hoc query embeddings
  → vector search (Postgres/pgvector when DATABASE_URL is set, else FAISS from Object Storage)
  → Qwen screen (title+abstract): drop junk or mark read_full
  → fetch arXiv PDF for survivors; one Qwen call per paper on full text
  → Semantic Scholar graph recommendations from kept papers
  → same two-stage judge on those recommendations
  → suggestion cards (HTML / Markdown)
```

### Extraction

`extract_claims.py` uses `qwen3:32b` through the Nebius OpenAI-compatible
endpoint. The Pydantic schema constrains counts, required fields, duplicates,
and extra fields.

### Vector queries

Starting from the extraction JSON, `search_candidates.py` turns each hierarchy
node into
one or more short search strings. SPECTER2's adhoc query adapter is trained for
short raw text (Allen AI's example is `"Bidirectional transformers"`), not for
labeled templates like `Problem:` / `Claim:`. So we map the extracted sentences
themselves:

- Problem direct: problem statement + domain + keywords.
- Claim direct: claim sentence.
- Claim functional: claim functional role.
- Detail direct: implementation detail sentence.
- Detail alternative: detail functional role only (same role, different
  mechanism wording).

Each string is encoded with `allenai/specter2_adhoc_query` into a 768-d query
vector. Hits from all queries are merged and deduplicated by arXiv ID, with
provenance of which query matched each paper.

### arXiv index

Each indexed paper is `title [SEP] abstract`, encoded with the document adapter
`allenai/specter2`. Queries use `allenai/specter2_adhoc_query`. Both adapters
share `allenai/specter2_base` and produce compatible 768-d vectors. FAISS finds
nearest neighbors by L2 distance on unnormalized vectors.

The production index is arXiv computer science only (Kaggle metadata rows whose
`categories` include `cs` or `cs.*`), title+abstract only — not full PDFs.

### Semantic Scholar

Semantic Scholar is used for **graph recommendations**, not as the primary search
and not as citation metadata decoration.

After the Qwen judge keeps FAISS candidates, each kept paper seeds a small
`recommendations/v1/papers/forpaper/ARXIV:<id>` request (`--s2-recommend-limit`,
default 5, pool `all-cs`). Recommendations with an arXiv id are merged in and
run through the **same two-stage judge**. Use `--no-s2` to skip.

The older batch metadata enrich helper still exists in `search_candidates.py`
but is not part of the default `embedxiv_main` path.

## Install

```bash
python3 -m pip install -r requirements.txt
```

Required `.env` values:

```text
NEBIUS_ENDPOINT_URL=...
NEBIUS_ENDPOINT_TOKEN=...
NEBIUS_S3_ENDPOINT_URL=https://storage.<region>.nebius.cloud
NEBIUS_S3_REGION=<region>
NEBIUS_S3_ACCESS_KEY_ID=...
NEBIUS_S3_SECRET_ACCESS_KEY=...
NEBIUS_S3_BUCKET=...
DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/DBNAME?sslmode=verify-full
S2_API_KEY=...
```

`DATABASE_URL` points search and the GPU embed Job at Managed Postgres +
pgvector. When it is unset, search falls back to downloading the latest FAISS
generation from Object Storage.

If the database password contains `@`, percent-encode it as `%40` in the URL
(for example `pass@word` → `pass%40word`).

`S2_API_KEY` is optional when search is run with `--no-s2`.
The Object Storage prefix is `arxiv-index` in bucket `embedxiv-storage` (for
example `embedxiv-storage/arxiv-index/metadata.sqlite`).

### Managed Postgres (pgvector)

Search and the GPU embed Job can use **Nebius Managed PostgreSQL** with the
`pgvector` extension instead of downloading a FAISS index to your laptop.

**One-time database setup** (run once per database):

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

**What the embed Job writes** when `DATABASE_URL` is set on the Job:

- `papers` — one row per arXiv paper: metadata (`arxiv_id`, `title`, `abstract`,
  `categories`, `authors`, …) plus a 768-d `embedding` column
- `corpus_manifest` — JSON manifest (paper count, model fingerprint, dimension)
- HNSW index on `papers.embedding` for fast nearest-neighbor search

**What search reads** when `DATABASE_URL` is in your laptop `.env`:

- Same `papers` table and `corpus_manifest` via `PgvectorIndex` in
  `search_candidates.py` (no FAISS download, no separate metadata lookup)

**Routing rule:** if `DATABASE_URL` is set, `datagen/embed_corpus.py` publishes
to Postgres and skips FAISS/Object Storage publish. If it is unset, the Job
builds FAISS and writes under `embedxiv-storage/arxiv-index/generations/`.

## Dataset

The corpus is built from the **Cornell University arXiv Metadata OAI Snapshot**
on Kaggle — one JSON object per line (JSONL) with `id`, `title`, `abstract`,
`categories`, `authors`, and related fields for every arXiv paper in the dump.

- **Source:** [Kaggle — arXiv Dataset (Cornell University)](https://www.kaggle.com/datasets/Cornell-University/arxiv)
- **File needed:** `arxiv-metadata-oai-snapshot.json` (Kaggle ships it inside
  `arxiv-metadata-oai-snapshot.json.zip`, ~1 GB compressed)
- **Local path:** place the extracted file at
  `data/arxiv-metadata-oai-snapshot.json` (`data/` is gitignored)

**Download via Kaggle website:** create a free Kaggle account, open the dataset
page above, download `arxiv-metadata-oai-snapshot.json.zip`, unzip, and move the
`.json` file into `data/`.

**Download via Kaggle CLI** (requires [API credentials](https://www.kaggle.com/docs/api)
in `~/.kaggle/kaggle.json`):

```bash
mkdir -p data
kaggle datasets download -d Cornell-University/arxiv \
  -f arxiv-metadata-oai-snapshot.json.zip -p data --unzip
```

By default `datagen/preload_metadata.py` keeps **computer science** papers only
(rows whose `categories` include `cs` or `cs.*`). The embed Job indexes
**title + abstract** from those rows — not full PDFs. Use `--all-categories` on
preload to keep the full Kaggle dump.

You only need this file on your laptop for the one-time preload step. Day-to-day
search against Postgres or Object Storage does not read the Kaggle snapshot.

## Nebius deployments

Two separate Nebius deployments — same pattern as each other: build an image,
push it, create the resource in the Nebius UI.

### Qwen endpoint (`Dockerfile`)

Always-on Ollama container with `qwen3:32b`. `extract_claims.py` calls it over
the OpenAI-compatible API (`NEBIUS_ENDPOINT_URL` / `NEBIUS_ENDPOINT_TOKEN`).

### Corpus build (`datagen/`)

Two steps — no live arXiv OAI crawl for the first build.

**1. Laptop CPU preload** (`datagen/preload_metadata.py`)

Use the Kaggle snapshot from [Dataset](#dataset) (`data/arxiv-metadata-oai-snapshot.json`).
Build `metadata.sqlite` and upload to Object Storage under `arxiv-index/`:

```bash
python3 -m datagen.preload_metadata \
  --metadata-jsonl data/arxiv-metadata-oai-snapshot.json
```

Requires `boto3` and the `NEBIUS_S3_*` `.env` values. Use `--skip-upload` to
build locally only.

**2. Nebius GPU Job** (`datagen/embed_corpus.py`)

Rebuild and push the image after code changes:

```text
docker build --platform linux/amd64 \
  -f datagen/Dockerfile \
  -t msaleev/embedxiv-specter2:latest .
docker push msaleev/embedxiv-specter2:latest
```

Create a **new** GPU Job (Restart does not reliably pick up a new image digest).
Use 1 GPU, 450+ GiB container disk, and a long timeout (72–168 hours for the
full CS corpus).

**Entrypoint** (Nebius prefills `#!sh`):

```text
#!sh
python -m datagen.embed_corpus
```

**Environment variables** — add on the Job in the UI under *Environment
variables* (or `--env` in the CLI):

| Key | Value |
| --- | --- |
| `DATABASE_URL` | Same connection string as in your `.env` (`@` in password → `%40`) |

This is how the Job knows to publish into Postgres. Without `DATABASE_URL`, it
falls back to building FAISS and writing generations to Object Storage.

Optional tuning (defaults are fine):

| Key | Default |
| --- | --- |
| `EMBED_BATCH_SIZE` | `128` |
| `SHARD_SIZE` | `50000` |
| `PG_BATCH_SIZE` | `5000` |
| `INDEX_OUTPUT_DIR` | `/output/arxiv-index` |

**Mounted volumes** — in the Job UI: *Storage → Mounted volumes → Attach volume*:

| Field | Value |
| --- | --- |
| Bucket | `embedxiv-storage` |
| Mount path | `/output` |
| Mode | `rw` |

Mount the **bucket root** at `/output`, not at `/output/arxiv-index`. The Job
reads preload from `embedxiv-storage/arxiv-index/`, which appears inside the
container as `/output/arxiv-index/`:

- `embedxiv-storage/arxiv-index/metadata.sqlite`
- `embedxiv-storage/arxiv-index/PRELOAD_READY.json`

If you mount the bucket elsewhere (for example `/data`), set
`INDEX_OUTPUT_DIR=/data/arxiv-index`.

**What the Job does** when `DATABASE_URL` is set:

1. Read preload from the mounted bucket
2. Embed title+abstract with SPECTER2 on GPU (tqdm progress in logs)
3. Upsert metadata + vectors into Postgres
4. Build the HNSW index

Success log line: `Published verified Postgres corpus (N papers)`.

**Legacy path** (no `DATABASE_URL` on the Job): embed, build FAISS, publish
`metadata.sqlite`, `index.faiss`, `manifest.json` under
`embedxiv-storage/arxiv-index/generations/<timestamp>/` plus `LATEST.json`.

The image keeps CUDA PyTorch from the base layer (do not reinstall CPU `torch`
from PyPI on top).

## Run search (after the corpus Job has published)

Use an existing extraction:

```bash
python3 search_candidates.py grokking_paper_claims.json \
  --output grokking_vector_results.json
```

Run extraction and search together:

```bash
python3 embedxiv_main.py papers/Grokking_paper.pdf
```

By default this writes all generated files into `output/`:

- `full_run_results.json` — full pipeline dump
- `full_run_results_cards.html` / `.md` — suggestion cards

The HTML opens automatically in the default browser after a successful run.
Use `--no-open` to suppress that behavior. Override the destination with
`-o path/to/custom_results.json`; card sidecars are written beside it.

Add `--no-s2` to disable metadata enrichment.

Each candidate contains:

- arXiv metadata and URL.
- Best vector distance and rank.
- Every problem, claim, or detail query that matched it.
- Optional nested Semantic Scholar metadata and enrichment status.

`grokking_retrieval_results.json` is stale output from the retired Semantic
Scholar keyword retriever.

## Source files

| File | Role |
| --- | --- |
| `extract_claims.py` | Client: call Nebius Qwen → problem/claim/detail JSON |
| `search_candidates.py` | Vector search (Postgres/pgvector or FAISS fallback), optional S2 |
| `judge_candidates.py` | Two-stage judge: abstract screen, then full-text per PDF |
| `suggestion_cards.py` | Ranked suggestion cards (HTML / Markdown) |
| `embedxiv_main.py` | Glue CLI: PDF/text → extract → search → judge → cards |
| `datagen/preload_metadata.py` | Laptop CPU: Kaggle metadata → SQLite → Nebius bucket |
| `datagen/embed_corpus.py` | GPU Job: embed preload → Postgres or FAISS publish |
| `datagen/Dockerfile` | Nebius image for the GPU embed Job |
| `Dockerfile` | Nebius image for the Qwen/Ollama endpoint |

## Remaining stages

Partially implemented. **Qwen judge** (`judge_candidates.py`) is two-stage:
(1) batched title+abstract screen → `drop` or `read_full`; (2) download arXiv
PDF for survivors and judge **one paper per LLM call**. Final `judgment` has
`decision`, `relation`, `why`, `primary_level`, and `stage` (`screen` or
`full_text`). Kept papers then seed a small Semantic Scholar recommendation
neighborhood, which is judged the same way. Kept results are rendered as
suggestion cards (`output/*_cards.html` / `.md`).

Still not implemented:

```text
optional second FAISS retrieval pass (rewrite query / more neighbors)
```

arXiv-only retrieval excludes journal-only and substantial biomedical literature;
another corpus can be added behind the same query interface later.

