# Evaluating EmbedXiv

This folder contains the evaluation workflow for EmbedXiv pipeline runs. It is
meant to answer three practical questions:

1. How many retrieved papers survive each pipeline stage?
2. Did the agentic query refinement and per-section cap change the final output?
3. Are the final suggestions useful to a human author?

The scripts operate on the JSON file produced by `embedxiv_main.py`, usually
`output/full_run_results.json`.

## Inputs and outputs

Input:

- A completed EmbedXiv run JSON, for example `output/full_run_results.json`.
- Optional comparison runs created with `--no-search-refinement` or
  `--no-per-node-cap`.

Outputs:

- Funnel tables showing retrieved candidates, abstract-screen decisions,
  full-text judge decisions, final kept suggestions, and source-paper leakage.
- A rubric CSV with one row per kept suggestion.
- Aggregate human scores for specific relevance, actionability, and
  non-redundancy.

## 1. Summarize a pipeline run

Run this after a normal EmbedXiv run:

```bash
python3 eval/summarize_run.py output/full_run_results.json
```

The summary reports:

- `Retrieved candidates`: every paper collected for the draft sections.
- `Abstract screen -> read_full`: papers that passed the first lightweight
  screening step.
- `Full-text judge -> keep`: papers accepted by the full judge before the
  per-section cap.
- `Per-node cap -> final kept`: the final suggestions shown to the author.
- `Source contamination`: whether the source paper was accidentally returned as
  a suggestion.
- `Kept from agentic refinement only`: final suggestions that came from the
  agentic query-refinement loop.

## 2. Export suggestions for human scoring

Convert the final kept suggestions into a CSV:

```bash
python3 eval/export_rubric.py output/full_run_results.json \
  -o eval/runs/cbam_rubric.csv
```

Each row is one paper suggestion attached to one extracted draft node. The CSV
includes the paper title, arXiv id, node path, node type, relation label, and the
model's explanation for why the paper matters.

Fill in these three score columns from 1 to 5:

| Column | What to score |
| --- | --- |
| `specific_relevance_1_5` | Does this paper directly help with the displayed problem, claim, or implementation detail? |
| `actionability_1_5` | Does it give the author something concrete to cite, compare against, test, integrate, or revise? |
| `non_redundancy_1_5` | Does it add something different from other suggestions attached to the same section? |

Suggested scoring anchors:

| Score | Meaning |
| ---: | --- |
| 1 | Broadly related, unclear use, or mostly duplicate. |
| 3 | Relevant background, but generic or partly overlapping. |
| 5 | Directly improves that part of the draft and gives a clear next action. |

The blank template is `eval/rubric_template.csv`. The filled CBAM pilot file is
`eval/runs/cbam_rubric.csv`.

## 3. Aggregate human scores

After scoring the CSV:

```bash
python3 eval/aggregate_scores.py eval/runs/cbam_rubric.csv
```

The script prints:

- Rated suggestion count.
- High-quality rate: suggestions with relevance >= 4 and actionability >= 4.
- Redundancy rate: suggestions with non-redundancy <= 2.
- Mean score for each rubric column.
- Mean overall score.

Rows with any blank score are skipped, so unfinished scoring will not crash the
aggregation.

## 4. Run a full pilot with ablations

Use this when you want one command that runs the main pipeline plus two
comparison variants:

```bash
chmod +x eval/run_pilot.sh
./eval/run_pilot.sh examples/sample_research_note.txt
```

You can pass a PDF or text draft instead:

```bash
./eval/run_pilot.sh papers/your_paper.pdf
```

The script writes a timestamped folder under `eval/runs/` containing:

- `full.json`: normal pipeline with agentic query refinement and per-node cap.
- `no_refinement.json`: disables agentic query refinement.
- `no_cap.json`: disables the per-node cap.
- `rubric.csv`: rubric rows exported from all three variants.

It also prints an ablation comparison table:

```bash
python3 eval/summarize_run.py --compare \
  eval/runs/<timestamp>/full.json \
  eval/runs/<timestamp>/no_refinement.json \
  eval/runs/<timestamp>/no_cap.json
```

Use this table to show whether the agentic refinement loop found extra useful
papers and whether the cap reduced repeated suggestions within the same draft
section.

## Manual scoring rules

Score the suggestion as displayed by EmbedXiv, not the paper in isolation. The
question is whether that retrieved paper helps the specific draft node it was
attached to.

For example:

- A paper proposing a better attention block for a draft's attention module can
  score high on relevance and actionability.
- A famous but already obvious baseline may score high on relevance and lower on
  non-redundancy.
- A paper from a distant domain can still score well if its mechanism clearly
  transfers to the draft's technical choice.

Keep notes short and concrete. Useful notes explain why a score was given, such
as "direct architecture replacement", "same claim tested empirically", or
"overlaps with other channel-attention suggestions".

## Reproducible eval checklist

For a reproducible evaluation run, record:

- The input draft path.
- The EmbedXiv command and flags.
- The results JSON path.
- Whether agentic refinement was enabled.
- Whether the per-node cap was enabled.
- The rubric CSV path.
- The date and model/backend used for the run.

The evaluation scripts do not call Nebius directly. They read saved pipeline
outputs, so another practitioner can rerun the same scoring workflow from the
checked-in JSON or from a newly generated run.
