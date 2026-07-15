#!/usr/bin/env bash
# Run full pipeline + two ablations, then export funnel tables and rubric CSV.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PAPER="${1:-examples/sample_research_note.txt}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="eval/runs/${STAMP}"
mkdir -p "$OUTDIR"

echo "Paper: $PAPER"
echo "Output: $OUTDIR"
echo

run_variant() {
  local label="$1"
  shift
  echo ">>> $label"
  python3 embedxiv_main.py "$PAPER" --no-open --no-cards \
    --run-label "$label" \
    -o "$OUTDIR/${label}.json" \
    "$@"
}

run_variant full
run_variant no_refinement --no-search-refinement
run_variant no_cap --no-per-node-cap

python3 eval/summarize_run.py --compare \
  "$OUTDIR/full.json" \
  "$OUTDIR/no_refinement.json" \
  "$OUTDIR/no_cap.json"

python3 eval/export_rubric.py "$OUTDIR/full.json" -o "$OUTDIR/rubric.csv"
python3 eval/export_rubric.py "$OUTDIR/no_refinement.json" --append -o "$OUTDIR/rubric.csv"
python3 eval/export_rubric.py "$OUTDIR/no_cap.json" --append -o "$OUTDIR/rubric.csv"

cat <<EOF

Next steps:
1. Open $OUTDIR/rubric.csv and score specific_relevance / actionability / non_redundancy (1–5).
2. python3 eval/aggregate_scores.py $OUTDIR/rubric.csv
3. Copy funnel + ablation tables into README or your blog post.

EOF
