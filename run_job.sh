#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_job.sh <tag> [provider] [model]
# Example: ./run_job.sh med-xr-pm openai gpt-4o-mini
TAG="${1:-company-role}"
PROVIDER="${2:-openai}"
MODEL="${3:-gpt-4o-mini}"

# Ensure venv
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
  else
    echo "No venv found. Create with: python3 -m venv .venv && source .venv/bin/activate && pip -r requirements.txt"
    exit 2
  fi
fi

OUTDIR="outputs/runs/${TAG}_$(date +%Y-%m-%d_%H-%M)"
echo "OUTDIR=$OUTDIR"

python job_copilot_starter.py \
  --provider "$PROVIDER" ${MODEL:+--model "$MODEL"} \
  --resume data/private/my_resume_bullets.txt \
  --jd data/private/my_jd.txt \
  --outdir "$OUTDIR"

# prettify and quick QC
python make_readable.py --files "$OUTDIR/suggestions.jsonl" "$OUTDIR/run_log.csv" || true

# optional helpers if present
if [[ -x tools/filter_supported.py ]]; then
  ./tools/filter_supported.py "$OUTDIR" || true
fi
if [[ -x tools/jd_coverage.py ]]; then
  ./tools/jd_coverage.py "$OUTDIR/suggestions.jsonl" || true
fi

echo "Done. Run folder: $OUTDIR"
