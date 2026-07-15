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
  → FAISS search over corpus loaded from Nebius Object Storage
  → Qwen screen (title+abstract): drop junk or mark read_full
  → fetch arXiv PDF for survivors; one Qwen call per paper on full text
  → Semantic Scholar graph recommendations from kept papers
  → same two-stage judge on those recommendations
  → suggestion cards (HTML / Markdown / JSON)
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
S2_API_KEY=...
```

`S2_API_KEY` is optional when search is run with `--no-s2`.
The Object Storage key prefix is fixed in code as `arxiv-index` (same as
`/output/arxiv-index` when the Job mounts the bucket at `/output`).

## Nebius deployments

Two separate Nebius deployments — same pattern as each other: build an image,
push it, create the resource in the Nebius UI.

### Qwen endpoint (`Dockerfile`)

Always-on Ollama container with `qwen3:32b`. `extract_claims.py` calls it over
the OpenAI-compatible API (`NEBIUS_ENDPOINT_URL` / `NEBIUS_ENDPOINT_TOKEN`).

### Corpus build (`datagen/`)

Two steps — no live arXiv OAI crawl for the first build.

**1. Laptop CPU preload** (`datagen/preload_metadata.py`)

Download the Cornell Kaggle arXiv metadata snapshot
(`arxiv-metadata-oai-snapshot.json`), keep CS papers by default (`cs` /
`cs.*` in `categories`; use `--all-categories` for the full dump), build
`metadata.sqlite`, upload to Object Storage under `arxiv-index/`:

```bash
python3 -m datagen.preload_metadata \
  --metadata-jsonl ./arxiv-metadata-oai-snapshot.json
```

Requires `boto3` and the `NEBIUS_S3_*` `.env` values. Use `--skip-upload` to
build locally only.

**2. Nebius GPU Job** (`datagen/embed_corpus.py`)

Mount bucket `embedxiv-storage` at **`/output`** (read-write). Do **not** mount
it at `/output/arxiv-index`. Preload objects are keys under `arxiv-index/`, so
the Job must see:

- `/output/arxiv-index/metadata.sqlite`
- `/output/arxiv-index/PRELOAD_READY.json`

In the Job UI Entrypoint (Nebius prefills `#!sh`), use:

```text
#!sh
python -m datagen.embed_corpus
```

The Job embeds title+abstract with SPECTER2, builds FAISS, and publishes
`metadata.sqlite`, `index.faiss`, `manifest.json` under
`arxiv-index/generations/<timestamp>/` plus `LATEST.json`.

```text
docker build --platform linux/amd64 \
  -f datagen/Dockerfile \
  -t msaleev/embedxiv-specter2:latest .
docker push msaleev/embedxiv-specter2:latest
```

Create a **new** GPU Job from that image after every code/image change (Restart
does not pick up a new image digest reliably). Use 1 GPU. The image keeps CUDA
PyTorch from the base layer (do not reinstall CPU `torch` from PyPI on top).

If the mount path differs, set `INDEX_OUTPUT_DIR` to the directory that contains
both preload files.

Search never uses a project-local index directory. `search_candidates.py` reads
`LATEST.json` from the bucket over the S3 API and downloads that generation's
FAISS artifacts for the query.

## Run search (after the corpus Job has published)

Use an existing extraction:

```bash
python3 search_candidates.py grokking_paper_claims.json \
  --output grokking_vector_results.json
```

Run extraction and search together:

```bash
python3 embedxiv_main.py papers/Grokking_paper.pdf \
  --output grokking_vector_results.json
```

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
| `search_candidates.py` | Load corpus from Object Storage, SPECTER2/FAISS search, optional S2 |
| `judge_candidates.py` | Two-stage judge: abstract screen, then full-text per PDF |
| `suggestion_cards.py` | Ranked suggestion cards (HTML / Markdown / JSON) |
| `embedxiv_main.py` | Glue CLI: PDF/text → extract → search → judge → cards |
| `datagen/preload_metadata.py` | Laptop CPU: Kaggle metadata → SQLite → Nebius bucket |
| `datagen/embed_corpus.py` | GPU Job: embed preload → FAISS → publish index |
| `datagen/Dockerfile` | Nebius image for the GPU embed Job |
| `Dockerfile` | Nebius image for the Qwen/Ollama endpoint |

## Remaining stages

Partially implemented. **Qwen judge** (`judge_candidates.py`) is two-stage:
(1) batched title+abstract screen → `drop` or `read_full`; (2) download arXiv
PDF for survivors and judge **one paper per LLM call**. Final `judgment` has
`decision`, `relation`, `why`, `primary_level`, and `stage` (`screen` or
`full_text`). Kept papers then seed a small Semantic Scholar recommendation
neighborhood, which is judged the same way. Kept results are rendered as
suggestion cards (`*_cards.html` / `.md` / `.json`).

Still not implemented:

```text
optional second FAISS retrieval pass (rewrite query / more neighbors)
```

arXiv-only retrieval excludes journal-only and substantial biomedical literature;
another corpus can be added behind the same query interface later.

