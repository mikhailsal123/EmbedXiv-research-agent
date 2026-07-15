# EmbedXiv Research Agent

## Purpose

CS/AI moves too quickly for authors to read every relevant new paper before
submitting their own work. EmbedXiv is a literature-aware paper proofreader: it
reads a draft, searches a CS arXiv corpus, and surfaces papers that may expose
gaps in the draft.

Those gaps can look like:

- a closely related paper the author should position against;
- an alternative architecture or mechanism that may fit the author's goal
  better than the one in the draft;
- a paper that tested, qualified, contradicted, or extended a claim the author
  is making.

Given a private abstract, preprint, or paper, the pipeline extracts:

- **Problem**: the limitation or research gap.
- **Claims**: independent conceptual ideas addressing that problem.
- **Implementation details**: concrete mechanisms realizing each claim.

It then searches for papers that can help proofread those parts of the draft:

1. Papers addressing the same problem.
2. Ideas that support, extend, qualify, or challenge each claim.
3. Alternative mechanisms serving each implementation detail's role.

## Pipeline

```text
Paper or abstract
  → Qwen extraction: problem → claims → implementation details
  → hierarchy-aware query texts
  → SPECTER2 ad-hoc query embeddings
  → vector search (Postgres/pgvector; legacy FAISS fallback when DATABASE_URL is unset)
  → bounded agentic search refinement: inspect weak substitute searches, rewrite, retry
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

### Agentic search refinement

`search_refinement.py` runs during `embedxiv_main.py` after the initial vector
search and before the first judge pass. It uses the active vector backend:
Postgres/pgvector in the current production path, or the legacy FAISS fallback
when `DATABASE_URL` is unset. Its goal is narrow: find papers that reveal a
possible draft gap, especially papers whose mechanism, architecture, or tested
claim could replace, qualify, or strengthen a source claim/detail.

It is agentic because it closes the retrieval loop: it inspects search quality,
diagnoses why results are too generic, too close to the source, or off-role,
then rewrites the query and tries again. The loop is bounded and cannot run
forever:

```text
plan substitute-oriented queries
  → vector search
  → inspect title/abstract results
  → diagnose failures such as "too generic" or "near-duplicate"
  → rewrite only failed searches
  → stop after --search-refinement-rounds
```

Defaults are intentionally small: 2 rounds, 12 source nodes, 3 initial queries
per node, 2 follow-up queries per failed node, and 8 vector hits per query.
Refined candidates are merged into the initial vector results and then all
candidates go through the normal two-stage judge together.

### arXiv index

Each indexed paper is `title [SEP] abstract`, encoded with the document adapter
`allenai/specter2`. Queries use `allenai/specter2_adhoc_query`. Both adapters
share `allenai/specter2_base` and produce compatible 768-d vectors. The current
production path stores vectors in Postgres with pgvector and uses an HNSW index
with L2 distance. The legacy Object Storage path stores the same embeddings in
a FAISS index.

The production index is arXiv computer science only (Kaggle metadata rows whose
`categories` include `cs` or `cs.*`), title+abstract only — not full PDFs.

### Semantic Scholar

Semantic Scholar is used for **graph recommendations**, not as the primary search
and not as citation metadata decoration.

After the Qwen judge keeps initial vector-search candidates, each kept paper seeds a small
`recommendations/v1/papers/forpaper/ARXIV:<id>` request (`--s2-recommend-limit`,
default 5, pool `all-cs`). Recommendations with an arXiv id are merged in and
run through the **same two-stage judge**. Use `--no-s2` to skip.

The older batch metadata enrich helper still exists in `search_candidates.py`
but is not part of the default `embedxiv_main` path.

## Reproduce from scratch

Full order of operations for a clean machine and a fresh Nebius project:

1. Install dependencies and create `.env` — [Reproduce the environment](#reproduce-the-environment).
2. Deploy the Qwen endpoint on Nebius — [Qwen endpoint](#qwen-endpoint-dockerfile). Put its
   URL and token in `.env`.
3. Create the Managed Postgres database, enable pgvector, and set
   `DATABASE_URL` in `.env` — [Managed Postgres (pgvector)](#managed-postgres-pgvector).
4. Download the Kaggle arXiv snapshot into `data/` — [Data](#data).
5. Run the laptop preload to build `metadata.sqlite` and upload it to Object
   Storage — [Corpus build](#corpus-build-datagen), step 1.
6. Build and push the GPU Job image, then run the Nebius GPU Job with
   `DATABASE_URL` set and the bucket mounted at `/output` — [Corpus build](#corpus-build-datagen), step 2.
7. Run the pipeline on the included synthetic example — [Run the pipeline](#run-the-pipeline).

Steps 4–6 are one-time corpus setup. After the Job publishes to Postgres,
day-to-day runs only need steps 1–2 configured.

## Reproduce the environment

Requires Python 3.11+ (developed on 3.12).

```bash
python3 -m pip install -r requirements.txt
cp .env.example .env
```

Required `.env` values:

```text
NEBIUS_ENDPOINT_URL=https://<your-qwen-endpoint>
NEBIUS_ENDPOINT_TOKEN=...
NEBIUS_S3_ENDPOINT_URL=https://storage.<region>.nebius.cloud
NEBIUS_S3_REGION=<region>
NEBIUS_S3_ACCESS_KEY_ID=...
NEBIUS_S3_SECRET_ACCESS_KEY=...
NEBIUS_S3_BUCKET=...
DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/DBNAME?sslmode=verify-full
S2_API_KEY=...
```

The endpoint URL should be the base endpoint URL; the client adds `/v1`.
`.env.example` is safe to commit. Real credentials, local PDFs, generated
outputs, Kaggle credentials, and downloaded datasets are ignored by git and by
the Docker build context.

`DATABASE_URL` points search and the GPU embed Job at Managed Postgres +
pgvector. Keep it set for the current production path. When it is unset, search
falls back to downloading the latest legacy FAISS generation from Object
Storage.

If the database password contains `@`, percent-encode it as `%40` in the URL
(for example `pass@word` → `pass%40word`).

For Nebius Managed Postgres SSL use `sslmode=verify-full`. The GPU Job
automatically downloads the [Nebius MSP CA](https://docs.nebius.com/postgresql/databases/connect)
into `~/.postgresql/root.crt` before connecting. On your laptop you can install
the same CA once, or let the client code download it.

Do **not** set `sslrootcert=system` for Nebius — it often fails certificate
verification. Bare `sslmode=verify-full` without any CA file also fails inside
Job containers (missing `/root/.postgresql/root.crt`).

`S2_API_KEY` is optional when search is run with `--no-s2`.
The Object Storage prefix is `arxiv-index` in bucket `embedxiv-storage` (for
example `embedxiv-storage/arxiv-index/metadata.sqlite`). Override the prefix
with `NEBIUS_INDEX_PREFIX`.

Optional environment variables (all have sensible defaults):

| Key | Default | Purpose |
| --- | --- | --- |
| `EXTRACTION_MODEL` | `qwen3:32b` | Model name for claim extraction |
| `JUDGE_MODEL` | `EXTRACTION_MODEL` value | Model name for the two-stage judge |
| `JUDGE_PDF_MAX_CHARS` | `60000` | Truncation limit for full-text PDF judging |
| `ARXIV_REQUEST_DELAY` | `1.0` | Seconds between arXiv PDF downloads |
| `ARXIV_USER_AGENT` | dev placeholder | Set to your contact info per arXiv etiquette |

### Managed Postgres (pgvector)

Search and the GPU embed Job use **Nebius Managed PostgreSQL** with the
`pgvector` extension. The older FAISS/Object Storage path remains as a fallback
when `DATABASE_URL` is not set.

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

## Data

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

The Kaggle snapshot is public and is not committed to this repository. The
pipeline indexes title and abstract text from the downloaded snapshot; it does
not ingest full PDFs into the corpus. You only need the snapshot locally for the
one-time preload step. Day-to-day search against Postgres or Object Storage does
not read the Kaggle file.

The repository also includes `examples/sample_research_note.txt`, a synthetic
input document that can be used to run the end-to-end pipeline without adding a
private paper or copyrighted PDF to the repo.

## Nebius architecture

The system uses two Nebius resources:

### Qwen endpoint (`Dockerfile`)

Always-on Ollama container with `qwen3:32b`. `extract_claims.py` calls it over
the OpenAI-compatible API (`NEBIUS_ENDPOINT_URL` / `NEBIUS_ENDPOINT_TOKEN`).
Run it as a GPU-backed Serverless Endpoint sized to load `qwen3:32b`; endpoint
GPU choice controls latency and hourly cost.

Build and push the endpoint image from the repository root:

```text
docker build --platform linux/amd64 -t <docker-user>/embedxiv-qwen:latest .
docker push <docker-user>/embedxiv-qwen:latest
```

Reference endpoint configuration:

| Setting | Value |
| --- | --- |
| Image path | `<docker-user>/embedxiv-qwen:latest` |
| Container command | default image entrypoint: `ollama serve` |
| Exposed port | `11434` |
| Model loaded in image | `qwen3:32b` |
| Endpoint type | GPU-backed Nebius Serverless Endpoint |
| Hardware | choose a GPU/memory shape large enough to load and serve `qwen3:32b` |
| Client env vars | `NEBIUS_ENDPOINT_URL`, `NEBIUS_ENDPOINT_TOKEN` |

Do not publish the endpoint URL token. `NEBIUS_ENDPOINT_URL` and
`NEBIUS_ENDPOINT_TOKEN` belong in `.env` or in the deployment UI only.

### Corpus build (`datagen/`)

Two steps — no live arXiv OAI crawl for the first build.

**1. Laptop CPU preload** (`datagen/preload_metadata.py`)

Use the Kaggle snapshot from [Data](#data) (`data/arxiv-metadata-oai-snapshot.json`).
Build `metadata.sqlite` and upload to Object Storage under `arxiv-index/`:

```bash
python3 -m datagen.preload_metadata \
  --metadata-jsonl data/arxiv-metadata-oai-snapshot.json
```

Requires `boto3` and the `NEBIUS_S3_*` `.env` values. Use `--skip-upload` to
build locally only.

**2. Nebius GPU Job** (`datagen/embed_corpus.py`)

Build and push the GPU Job image after code changes:

```text
docker build --platform linux/amd64 \
  -f datagen/Dockerfile \
  -t <docker-user>/embedxiv-specter2:latest .
docker push <docker-user>/embedxiv-specter2:latest
```

Create a **new** GPU Job (Restart does not reliably pick up a new image digest).
Use 1 GPU or more, 350+ GiB container disk for the current CS-only corpus path,
and a timeout above the observed runtime. The current corpus build used for this
project runs in about 2 hours on the reference 1-GPU RTX PRO 6000 Job.

On multi-GPU nodes, the embed Job automatically uses all visible CUDA devices
for SPECTER2 document embedding. Set `EMBED_CUDA_DEVICES=0,1` to restrict the
device list, `EMBED_BATCH_SIZE` to control the total batch split across GPUs,
or `EMBED_DEVICE=cpu` / `EMBED_DEVICE=cuda:0` to force the single-device path.

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
uses the legacy fallback: building FAISS and writing generations to Object
Storage.

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
2. Embed title+abstract with SPECTER2 on GPU
3. Upsert metadata + vectors into Postgres
4. Build the HNSW index

Success log line: `Published verified Postgres corpus (N papers)`.

**Legacy path** (no `DATABASE_URL` on the Job): embed, build FAISS, publish
`metadata.sqlite`, `index.faiss`, `manifest.json` under
`embedxiv-storage/arxiv-index/generations/<timestamp>/` plus `LATEST.json`.

The image keeps CUDA PyTorch from the base layer (do not reinstall CPU `torch`
from PyPI on top).

### Reference Nebius Job configuration

This is the Serverless AI Job configuration used for the corpus embedding run.
Do not copy real database passwords into public docs; keep the real
`DATABASE_URL` in the Nebius UI or local `.env` only.

| Setting | Value |
| --- | --- |
| Image path | `msaleev/embedxiv-specter2:latest` |
| Entrypoint command | `python -m datagen.embed_corpus` |
| Timeout | `3 hours` |
| Container disk | `350 GiB` |
| Mounted bucket | `embedxiv-storage` |
| Mount path | `/output` |
| Environment variable | `DATABASE_URL=postgresql://...?...sslmode=verify-full` |
| GPUs | `1 GPU` |
| GPU platform | `NVIDIA RTX PRO 6000` (`gpu-rtx6000`) |
| CPUs | `24 vCPUs` |
| Memory | `218 GiB` |
| VM type | `Regular` |

The job expects the preload files at `/output/arxiv-index/metadata.sqlite` and
`/output/arxiv-index/PRELOAD_READY.json`. With `DATABASE_URL` set, it publishes
metadata and 768-d SPECTER2 vectors into Managed Postgres/pgvector instead of
publishing a FAISS generation to Object Storage.

### Expected runtime and cost shape

Actual cost depends on the Nebius region, GPU type, endpoint uptime, Managed
Postgres size, Object Storage usage, and current Nebius pricing. Treat the
numbers below as planning estimates and calculate final spend from the selected
resource prices:

- Qwen endpoint: billed while the Serverless Endpoint is running. It handles
  extraction, agentic search-refinement decisions, abstract screening, and
  full-text judging.
- Laptop preload: usually CPU-bound local work over the Kaggle JSONL snapshot;
  the output is `metadata.sqlite` plus `PRELOAD_READY.json` in Object Storage.
- Corpus GPU Job: the current CS corpus build used here takes about 2 hours on
  one NVIDIA RTX PRO 6000 GPU with a 350 GiB container disk. Multi-GPU nodes
  split embedding batches across visible CUDA devices and should reduce wall
  time roughly with usable GPU throughput.
- Per-draft pipeline run: writes `output/full_run_results.json`,
  `output/full_run_results_cards.html`, and
  `output/full_run_results_cards.md`; runtime is dominated by Qwen judge calls
  and the number of candidate PDFs that pass the abstract screen.

For cost reporting, record endpoint uptime, GPU Job wall time, number/type of
GPUs, Postgres size, and Object Storage size. The direct GPU Job estimate is:
`GPU job cost = job hours x GPU count x selected GPU hourly price`, plus the
endpoint, Postgres, and storage line items from the Nebius console.

## Run the pipeline

After the Qwen endpoint is live and the corpus Job has published the Postgres
index, run the included synthetic example:

```bash
python3 embedxiv_main.py examples/sample_research_note.txt --no-open
```

This command runs without code changes once `.env` points at the Nebius
endpoint, Managed Postgres database, and Object Storage bucket described above.

Use an existing extraction:

```bash
python3 search_candidates.py path/to/extraction.json \
  --output output/vector_results.json
```

Run extraction and search together:

```bash
python3 embedxiv_main.py path/to/local_paper.pdf
```

By default this writes all generated files into `output/`:

- `full_run_results.json` — full pipeline dump
- `full_run_results_cards.html` / `.md` — paper-proofreading suggestion cards

The HTML opens automatically in the default browser after a successful run.
Use `--no-open` to suppress that behavior. Override the destination with
`-o path/to/custom_results.json`; card sidecars are written beside it.

Add `--no-s2` to disable Semantic Scholar graph recommendations.
Add `--no-search-refinement` to disable the bounded agentic substitute-search loop.

Agentic search-refinement cost controls:

```bash
python3 embedxiv_main.py examples/sample_research_note.txt \
  --search-refinement-rounds 2 \
  --search-refinement-targets 12 \
  --search-refinement-queries 3 \
  --search-refinement-followups 2 \
  --search-refinement-limit 8
```

Each candidate contains:

- arXiv metadata and URL.
- Best vector distance and rank.
- Every problem, claim, or detail query that matched it.
- The judge's explanation of the draft gap or useful takeaway.
- Optional nested Semantic Scholar metadata and enrichment status.

## Run the tests

The suite is offline — no credentials, database, or GPU needed:

```bash
python3 -m pytest tests/
```

It covers extraction schema validation, query building, the pgvector/FAISS
backend switch, judge parsing, card rendering, and repository hygiene (no
tracked PDFs, datasets, or generated outputs; exactly pinned requirements).

## Source files

| File | Role |
| --- | --- |
| `extract_claims.py` | Client: call Nebius Qwen → problem/claim/detail JSON |
| `search_candidates.py` | Vector search (Postgres/pgvector, with legacy FAISS fallback), optional S2 |
| `search_refinement.py` | Bounded agentic query-rewrite loop for substitute mechanisms |
| `judge_candidates.py` | Two-stage judge: abstract screen, then full-text per PDF |
| `suggestion_cards.py` | Ranked suggestion cards (HTML / Markdown) |
| `embedxiv_main.py` | Glue CLI: PDF/text → extract → search → judge → cards |
| `datagen/preload_metadata.py` | Laptop CPU: Kaggle metadata → SQLite → Nebius bucket |
| `datagen/embed_corpus.py` | GPU Job: embed preload → Postgres publish, or legacy FAISS publish |
| `datagen/Dockerfile` | Nebius image for the GPU embed Job |
| `Dockerfile` | Nebius image for the Qwen/Ollama endpoint |

## Current limitations

The judge is implemented as a two-stage process. First, Qwen screens each
source query with the complete candidate list returned for that query and marks
papers as `drop` or `read_full`. Second, arXiv PDFs are downloaded only for
survivors and judged one paper per model call. Kept papers seed a small
Semantic Scholar recommendation neighborhood, which is judged with the same
two-stage process.

arXiv-only retrieval excludes journal-only and substantial biomedical literature;
another corpus can be added behind the same query interface later.

## Reproducibility and compliance

The repository is self-contained except for credentials and the public arXiv
metadata download:

- `Dockerfile` reproduces the Qwen/Ollama Serverless Endpoint image.
- `datagen/Dockerfile` reproduces the SPECTER2 Serverless Job image.
- `requirements.txt` pins Python dependencies exactly.
- `examples/sample_research_note.txt` provides a synthetic input for an
  end-to-end run.
- The corpus data is publicly available from Kaggle with download instructions
  above.
- `LICENSE` provides the MIT open-source license.
- `.dockerignore` and `.gitignore` exclude credentials, local PDFs, generated
  outputs, downloaded datasets, private keys, certs, and local cache files.

The expected successful run produces:

- `output/full_run_results.json`
- `output/full_run_results_cards.html`
- `output/full_run_results_cards.md`

The test suite includes repository hygiene checks so local papers, generated
indexes, and generated outputs are not accidentally added to the tracked
repository.
