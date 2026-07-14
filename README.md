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
  → optional Semantic Scholar metadata enrichment
  → ranked candidates with query provenance
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

The production index is arXiv computer science only (`set=cs`), title+abstract
only — not full PDFs.

### Semantic Scholar

Semantic Scholar is not a second search. After FAISS returns candidates, their
arXiv IDs are looked up via `POST /graph/v1/paper/batch` (`ARXIV:<id>`) to attach
citation counts, venue, publication date, fields of study, and open-access PDF
links when available.

Enrichment is batched, cached in SQLite, retried on 429/5xx, and optional
(`--no-s2` or missing `S2_API_KEY`). Missing or unavailable S2 records never
remove local arXiv results; title/abstract/URL from the index stay authoritative.

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
NEBIUS_INDEX_PREFIX=arxiv-index
S2_API_KEY=...
ARXIV_USER_AGENT=EmbedXivResearchAgent/0.1 your-email@example.com
```

`S2_API_KEY` is optional when search is run with `--no-s2`.
`NEBIUS_INDEX_PREFIX` must match the Object Storage path the Job publishes
(default `arxiv-index`, same as `/output/arxiv-index` when the bucket is mounted
at `/output`).

## Nebius deployments

Two separate Nebius deployments — same pattern as each other: build an image,
push it, create the resource in the Nebius UI.

### Qwen endpoint (`Dockerfile`)

Always-on Ollama container with `qwen3:32b`. `extract_claims.py` calls it over
the OpenAI-compatible API (`NEBIUS_ENDPOINT_URL` / `NEBIUS_ENDPOINT_TOKEN`).

### SPECTER2 corpus Job (`datagen/`)

One-shot GPU Job that builds the searchable arXiv index. Image entrypoint is
`python -m datagen.create_corpus`: harvest CS metadata → SPECTER2-embed
title+abstract → build FAISS → verify → write artifacts under `/output`
(mount your Object Storage bucket there in the Job UI).

```text
docker build -f datagen/Dockerfile -t <registry>/embedxiv-specter2:<tag> .
docker push <registry>/embedxiv-specter2:<tag>
```

Then in Nebius: create a Job from that image, GPU node, bucket mounted at
`/output`. The Job writes the corpus into Object Storage.

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
| `embedxiv_main.py` | Glue CLI: PDF/text → extract → search |
| `datagen/create_corpus.py` | Job only: harvest, embed, build FAISS, publish to Object Storage |
| `datagen/Dockerfile` | Nebius image for the corpus Job |
| `Dockerfile` | Nebius image for the Qwen/Ollama endpoint |

## Remaining stages

Not implemented yet. Planned agentic loop after FAISS candidates:

```text
FAISS candidates (title + abstract)
  → Qwen 32b judge: keep / drop + short why
  → optional second pass: rewrite query or fetch more neighbors / full text
  → Semantic Scholar enrich survivors only
  → rank and present
```

Qwen runs on the existing Nebius Ollama endpoint (`qwen3:32b`). It decides
autonomously, per candidate and per hierarchy level, whether to keep a paper as:

- Same-problem competitor or foundation.
- Claim support, extension, qualification, or contradiction.
- Implementation alternative.
- Irrelevant (drop).

If the first pass is thin, Qwen may request another retrieval round (reformulated
query text, more neighbors, or full-text for a few keepers). Semantic Scholar
runs only after judgment, for citation/venue metadata on survivors — not during
search. Final ranking and suggestion cards use the kept set.

arXiv-only retrieval excludes journal-only and substantial biomedical literature;
another corpus can be added behind the same query interface later.
