#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Job-Application Copilot (truth-constrained) — CLEAN STARTER (v2)

- Inputs:
    --resume : path to bullets file (one bullet per line)
    --jd     : path to JD text (whole posting OK; extractor keeps 'asks')
- Providers:
    local    : heuristic rewrite (no API)
    openai   : phrasing via OpenAI; still truth-guarded
- Outputs (per run; inside --outdir):
    suggestions.jsonl  : source_bullet, jd_req, suggested, similarity, support, latency_s
    run_log.csv        : run summary
- Notes:
    * No torch deps; uses scikit-learn TF-IDF for mapping.
"""

import argparse, os, time, json, csv, re, sys
from pathlib import Path
from typing import List, Tuple

# --- Similarity (TF-IDF) ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_openai_client():
    """Lazily import OpenAI and validate key only if provider=openai."""
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package not installed. Try: pip install openai") from e
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI()

# ----------------------------
# Helpers
# ----------------------------

def read_resume_bullets(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8")
    bullets = [ln.strip(" •\t").strip() for ln in text.splitlines()]
    return [b for b in bullets if b]

def extract_jd_reqs(jd_text: str) -> List[str]:
    """
    Extract JD 'asks':
    - Keep bullet lines (-, *, •)
    - Keep short requirement-like sentences
    - Skip section headers (e.g., 'Responsibilities')
    - Drop very long marketing sentences
    """
    lines = []
    for raw in jd_text.splitlines():
        s = raw.strip()
        if not s:
            continue

        # skip common headers
        if s.lower() in {
            "responsibilities", "requirements", "what you'll do", "what you will do",
            "about the role", "about the job"
        }:
            continue

        if s.startswith(("-", "*", "•")):
            s = s.lstrip("-*•").strip()
            if s:
                lines.append(s)
        else:
            # split paragraph-ish text; keep short 'asks' with useful verbs
            parts = re.split(r"[.;]\s+", s)
            for p in parts:
                p = p.strip()
                if 5 <= len(p) <= 240 and re.search(
                    r"\b(own|drive|ship|deliver|build|lead|manage|scope|schedule|budget|risk|mitigat|"
                    r"communicat|stakeholder|analy|metric|report|api|platform|reliab|uptime|ml|ai|privacy|gdpr)\w*\b",
                    p, re.I
                ):
                    lines.append(p)

    # drop overly long items
    lines = [x for x in lines if len(x) <= 180]

    # de-dup, preserve order
    seen = set(); out = []
    for x in lines:
        if x not in seen:
            seen.add(x); out.append(x)
    return out[:200]

def map_similarity(bullets: List[str], reqs: List[str]) -> List[Tuple[int,int,float]]:
    """Return list of (req_idx, bullet_idx, score) for best bullet per req."""
    if not bullets or not reqs:
        return []
    corpus = bullets + reqs
    vec = TfidfVectorizer(min_df=1, ngram_range=(1,2)).fit(corpus)
    B = vec.transform(bullets)
    R = vec.transform(reqs)
    sims = cosine_similarity(R, B)  # [len(reqs), len(bullets)]
    best = []
    for i in range(len(reqs)):
        j = int(sims[i].argmax())
        best.append((i, j, float(sims[i, j])))
    return best

def local_rewrite(source: str, req: str) -> str:
    """Heuristic rewrite, truth-only. If weak evidence, caller will mark insufficient."""
    return f"{source} — aligned to: {req}"

def openai_rewrite(client, model: str, source: str, req: str) -> Tuple[str, float]:
    prompt = f"""
Rewrite a single resume bullet so it directly addresses the JD requirement.

Rules:
- Use ONLY facts present in SOURCE. Do NOT add employers, titles, dates, new metrics, or tools not in source.
- If SOURCE does not support the JD requirement, answer exactly: INSUFFICIENT EVIDENCE — <why>.

JD REQUIREMENT:
{req}

SOURCE:
{source}

Write ONE polished bullet (single line). If insufficient, follow the rule above.
"""
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            temperature=0.3,
            max_tokens=120,
        )
        text = resp.choices[0].message.content.strip()
    except Exception as e:
        text = f"INSUFFICIENT EVIDENCE — API error: {e}"
    latency = time.time() - t0
    return text, latency

def is_insufficient(s: str) -> bool:
    return s.strip().upper().startswith(("INSUFFICIENT", "INS UFFICIENT"))

# ----------------------------
# Main run
# ----------------------------

def run_once(resume_path: Path, jd_path: Path, provider: str, model: str, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    sug_path = outdir / "suggestions.jsonl"
    log_path = outdir / "run_log.csv"

    bullets = read_resume_bullets(resume_path)
    jd_text = jd_path.read_text(encoding="utf-8")
    reqs = extract_jd_reqs(jd_text)

    best = map_similarity(bullets, reqs)

    client = None
    if provider == "openai":
        client = get_openai_client()
        if not model:
            raise RuntimeError("With --provider openai, pass --model (e.g., gpt-4o-mini).")

    # Threshold — conservative support, but not too strict for PM
    SIM_THRESH = 0.08  # was 0.18

    suggestions = []
    latencies = []

    with sug_path.open("w", encoding="utf-8") as f:
        for (req_idx, bul_idx, score) in best:
            req = reqs[req_idx]
            source = bullets[bul_idx]
            if score < SIM_THRESH:
                suggested = f"INSUFFICIENT EVIDENCE — source bullet does not support: {req}"
                latency = 0.0
            else:
                if provider == "openai":
                    suggested, latency = openai_rewrite(client, model, source, req)
                else:
                    suggested = local_rewrite(source, req)
                    latency = 0.0

            latencies.append(latency)
            rec = {
                "source_bullet": source,
                "jd_req": req,
                "suggested": suggested,
                "similarity": round(score, 4),
                "support": (score >= SIM_THRESH) and (not is_insufficient(suggested)),
                "latency_s": round(latency, 3),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            suggestions.append(rec)

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "ts","provider","model","resume","jd",
            "reqs","bullets","suggestions","mean_latency_s","insufficient_count"
        ])
        w.writerow([
            ts, provider, model or "", str(resume_path), str(jd_path),
            len(reqs), len(bullets), len(suggestions),
            round(sum(latencies)/len(latencies), 3) if latencies else 0.0,
            sum(1 for r in suggestions if is_insufficient(r["suggested"]))
        ])

    print(f"Wrote {sug_path}")
    print(f"Wrote {log_path}")
    return sug_path, log_path

# ----------------------------
# CLI
# ----------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", required=False, help="Path to resume bullets (one per line).")
    p.add_argument("--jd", required=False, help="Path to JD text.")
    p.add_argument("--provider", choices=["local","openai"], default="local")
    p.add_argument("--model", help="OpenAI model (when provider=openai), e.g. gpt-4o-mini.")
    p.add_argument("--eval", help="Path to golden_questions.csv (optional; heuristic eval).")
    p.add_argument("--outdir", default="outputs", help="Directory to write artifacts (default: outputs)")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.eval:
        print("Eval mode not included in this trimmed starter. (Keep using your previous eval script if needed.)")
        sys.exit(0)

    if not args.resume or not args.jd:
        print("ERROR: --resume and --jd are required (unless using --eval).", file=sys.stderr)
        sys.exit(2)

    resume_path = Path(args.resume)
    jd_path = Path(args.jd)
    if not resume_path.exists():
        print(f"ERROR: resume file not found: {resume_path}", file=sys.stderr); sys.exit(2)
    if not jd_path.exists():
        print(f"ERROR: JD file not found: {jd_path}", file=sys.stderr); sys.exit(2)

    print(f"Using resume: {resume_path}")
    print(f"Using JD: {jd_path}")
    print(f"Writing artifacts to: {outdir}")

    run_once(resume_path, jd_path, args.provider, args.model or "", outdir)

if __name__ == "__main__":
    main()
