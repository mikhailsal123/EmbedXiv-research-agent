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
  → local FAISS search over arXiv title-and-abstract embeddings
  → optional Semantic Scholar metadata enrichment
  → ranked candidates with query provenance
```

### Extraction

`extract_claims.py` uses `qwen3:32b` through the Nebius OpenAI-compatible
endpoint. The Pydantic schema constrains counts, required fields, duplicates,
and extra fields.

### Vector queries

`retrieve.py` generates:

- Problem: problem statement + domain + keywords.
- Claim direct: problem context + claim.
- Claim functional: problem context + functional role.
- Detail direct: parent claim + concrete detail + role.
- Detail alternative: parent claim + functional role, omitting the current
  mechanism.

No words are truncated. Each full contextual query is encoded with
`allenai/specter2_adhoc_query`.

### arXiv index

`arxiv_index.py` encodes each arXiv title and abstract with the
`allenai/specter2` proximity adapter. Both adapters share
`allenai/specter2_base` and produce compatible 768-dimensional vectors.
The documented SPECTER2 setup uses unnormalized vectors and L2 distance.

The index builder provides:

- Official arXiv OAI-PMH sample/update harvesting.
- Local arXiv metadata-snapshot ingestion.
- SQLite metadata, content hashes, and resumable embedding jobs.
- Atomic NumPy embedding shards.
- Exact `IndexFlatL2` validation indexes.
- Configurable IVF and IVF-SQ8 indexes for the full corpus.

### Semantic Scholar

Semantic Scholar is not the primary retriever. After vector search, arXiv IDs
can be enriched through `POST /graph/v1/paper/batch` with citation, influence,
venue, publication, and open-access metadata.

Enrichment is batched, cached, retried conservatively, and optional. Missing or
unavailable Semantic Scholar records never remove local arXiv results.

## Install

Extraction only:

```bash
python3 -m pip install -r requirements.txt
```

Vector indexing and retrieval:

```bash
python3 -m pip install -r requirements-index.txt
```

Required `.env` values:

```text
NEBIUS_ENDPOINT_URL=...
NEBIUS_ENDPOINT_TOKEN=...
S2_API_KEY=...
ARXIV_USER_AGENT=EmbedXivResearchAgent/0.1 your-email@example.com
```

`S2_API_KEY` is optional when retrieval is run with `--no-s2`.

## Build a real validation index

Harvest approximately 1,000 real arXiv records and build an exact index:

```bash
python3 build_arxiv_index.py \
  --oai-sample 1000 \
  --from-date 2025-01-01 \
  --set cs \
  --index-dir data/arxiv-index \
  --index-type flat
```

Use `--device mps` on Apple silicon or `--device cuda` on a GPU machine.
Harvesting is resumable and observes a three-second OAI request interval.
If harvesting completed but embedding stopped, resume without downloading
another metadata page:

```bash
python3 build_arxiv_index.py \
  --existing-metadata \
  --index-dir data/arxiv-index \
  --device mps
```

## Build from the full metadata snapshot

```bash
python3 build_arxiv_index.py \
  --metadata-jsonl /path/to/arxiv-metadata-oai-snapshot.json \
  --index-dir data/arxiv-index \
  --device cuda \
  --index-type ivf
```

Benchmark a sample before starting the full build. At roughly 3.1 million
papers:

- Float32 vectors alone occupy about 8.9 GiB.
- `IVF4096,SQ8` can reduce index memory to roughly 2.2 GiB with a recall cost.
- Full CPU embedding is likely a multi-day operation.

The full embedding job is suited to a Nebius GPU. Persist the generated
SQLite, shard, manifest, and FAISS files in durable object storage after the
job. Supabase/pgvector can replace FAISS later, but storing and indexing
millions of 768-dimensional vectors requires a paid database tier and does not
improve embedding quality.

The SPECTER2 model cache itself uses several gigabytes. On macOS, query
encoding runs in a worker process because PyTorch and FAISS can otherwise
conflict through their OpenMP runtimes.

## Run retrieval

Use an existing extraction:

```bash
python3 retrieve.py grokking_paper_claims.json \
  --index-dir data/arxiv-index \
  --output grokking_vector_results.json
```

Run extraction and retrieval together:

```bash
python3 embedxiv_main.py papers/Grokking_paper.pdf \
  --index-dir data/arxiv-index \
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

## Remaining stages

Dense retrieval finds related papers, not argumentative stance. The next stage
must use Qwen to classify candidates as:

- Same-problem competitor or foundation.
- Claim support, extension, qualification, or contradiction.
- Implementation alternative.
- Irrelevant.

Full-text fetching and final suggestion-card ranking are not yet implemented.
arXiv-only retrieval also excludes journal-only and substantial biomedical
literature; another corpus can be added behind the same query interface later.
