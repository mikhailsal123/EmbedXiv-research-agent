# Eval scripts

Small scripts for turning a pipeline run into numbers you can put in the README
or blog post.

## What to run

### 1. Funnel counts (from an existing results JSON)

```bash
python3 eval/summarize_run.py output/full_run_results.json
```

Gives retrieved → screened → kept → capped counts, plus whether the input paper
leaked into results.

### 2. Export cards for manual scoring

```bash
python3 eval/export_rubric.py output/full_run_results.json \
  -o eval/runs/cbam_rubric.csv
```

Open the CSV. For each kept suggestion, fill in three scores (1–5):

| Column | What you're asking |
| --- | --- |
| `specific_relevance_1_5` | Does this actually help the problem/claim/detail it's under? |
| `actionability_1_5` | Would you cite it, compare against it, or try something from it? |
| `non_redundancy_1_5` | Is it meaningfully different from the other cards in that section? |

### 3. Aggregate scores

```bash
python3 eval/aggregate_scores.py eval/runs/cbam_rubric.csv
```

Outputs:

- **High-quality rate** — both relevance and actionability ≥ 4
- **Redundancy rate** — non-redundancy ≤ 2
- Means for each column

Blank template: `eval/rubric_template.csv`

## Optional: full run + ablations

If you want to compare with/without refinement or per-node cap:

```bash
chmod +x eval/run_pilot.sh
./eval/run_pilot.sh path/to/paper.pdf
```

Or run variants yourself and compare:

```bash
python3 eval/summarize_run.py --compare \
  eval/runs/full.json \
  eval/runs/no_refinement.json \
  eval/runs/no_cap.json
```

Flags: `--no-search-refinement`, `--no-per-node-cap`, `--run-label NAME`.

## After the demo

Turn off stuff you're not using so Nebius doesn't keep billing:

1. Stop the Qwen endpoint
2. Scale down or delete Postgres if you don't need the index live
3. Delete Object Storage preload files if Postgres is enough
4. Rotate keys if you ran this on a shared machine
