# Job-Application Copilot (truth-constrained)

A small, production-minded tool that tailors resume bullets to a job description **without fabricating facts**. It maps your bullets to JD requirements, rewrites for clarity, and shows provenance (which source bullet supports which suggestion).

- **Two modes:** local heuristic (no API) or LLM rewriter (OpenAI).
- **Groundedness:** suggestions must tie back to your source bullets; insufficient evidence → the tool says so.
- **Artifacts:** writes JSONL/CSV logs so you can curate and audit.
- **Safety:** minimal data handling; no server-side persistence by default.

## Quickstart

### Local (no API key)
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python job_copilot_starter.py \
  --provider local \
  --resume examples/resume_bullets.txt \
  --jd examples/jd.txt \
  --outdir "outputs/runs/example_$(date +%Y-%m-%d_%H-%M)"

python make_readable.py --open

export OPENAI_API_KEY="sk-..."       # or add it to ~/.zshrc
python job_copilot_starter.py \
  --provider openai --model gpt-4o-mini \
  --resume data/private/my_resume_bullets.txt \
  --jd data/private/my_jd.txt \
  --outdir "outputs/runs/company-role_$(date +%Y-%m-%d_%H-%M)"

python make_readable.py --open

q
q
