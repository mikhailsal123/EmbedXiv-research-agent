# Pilot evaluation (3-hour workflow)

Generate evidence, score kept cards, and paste tables into your blog or README.

## Hour 1 — Generate evidence

Run the full pipeline plus two ablations on the same paper:

```bash
chmod +x eval/run_pilot.sh
./eval/run_pilot.sh papers/CBAM_paper.pdf
```

Or run variants manually:

```bash
mkdir -p eval/runs/manual
python3 embedxiv_main.py papers/CBAM_paper.pdf --no-open \
  --run-label full -o eval/runs/manual/full.json

python3 embedxiv_main.py papers/CBAM_paper.pdf --no-open \
  --run-label no_refinement --no-search-refinement \
  -o eval/runs/manual/no_refinement.json

python3 embedxiv_main.py papers/CBAM_paper.pdf --no-open \
  --run-label no_cap --no-per-node-cap \
  -o eval/runs/manual/no_cap.json
```

Print funnel + ablation tables:

```bash
python3 eval/summarize_run.py --compare \
  eval/runs/manual/full.json \
  eval/runs/manual/no_refinement.json \
  eval/runs/manual/no_cap.json
```

Each results JSON stores `run.elapsed_seconds` and config flags for cost reporting.

## Hour 2 — Human rubric

Export kept cards to a spreadsheet:

```bash
python3 eval/export_rubric.py eval/runs/manual/full.json \
  -o eval/runs/manual/rubric.csv
```

Score each row (1–5):

| Column | Criterion |
| --- | --- |
| `specific_relevance_1_5` | How directly does this help the displayed problem / claim / detail? |
| `actionability_1_5` | Would the author cite, compare, revise, or experiment? |
| `non_redundancy_1_5` | Adds something distinct from other kept papers on that node? |

Aggregate:

```bash
python3 eval/aggregate_scores.py eval/runs/manual/rubric.csv
```

**High-quality rate** = % of rated cards with relevance ≥ 4 and actionability ≥ 4.  
**Redundancy rate** = % of rated cards with non-redundancy ≤ 2/5.

A blank template lives at `eval/rubric_template.csv`.

## Hour 3 — Presentation

Use this framing sentence:

> We evaluate final suggestions by human ratings on specific relevance,
> actionability, and non-redundancy — not vector similarity alone.

Report honestly: “pilot eval, n=15–20 cards across 1–2 papers.”

Record Nebius cost from the console for the runs you just did (endpoint uptime while judging, Postgres already provisioned).

## Teardown (after demo)

When you are done with the challenge demo:

1. Stop or delete the Qwen Serverless Endpoint if you do not need it running.
2. Delete or scale down Managed Postgres if you no longer need the live index.
3. Remove Object Storage preload artifacts if Postgres is your only backend.
4. Rotate any API keys that were used on a shared machine.

This keeps spend predictable and signals operational discipline in the writeup.
