#!/usr/bin/env python3
"""
job_copilot_starter.py — Tailor resume bullets to a Job Description (JD).

Usage:
  python job_copilot_starter.py --resume examples/resume_bullets.txt --jd examples/jd.txt --provider local
  python job_copilot_starter.py --resume my_resume.txt --jd jd.txt --provider openai --model gpt-4o-mini
"""
import argparse, os, time, json, csv, re
from pathlib import Path
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional: local embeddings (no API); falls back to TF-IDF if not installed.
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False

# Optional: OpenAI provider
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

console = Console()
load_dotenv()

def read_lines(path: str) -> List[str]:
    text = Path(path).read_text(encoding="utf-8").strip()
    # Split bullets either by newline or '•'
    parts = [x.strip(" •\t") for x in re.split(r"\n+|•", text) if x.strip()]
    return parts

def jd_requirements(jd_text: str) -> List[str]:
    # Very simple extractor: split into sentences; keep those with strong verbs/nouns.
    sents = re.split(r"(?<=[\.\?\!])\s+", jd_text.strip())
    keep = []
    for s in sents:
        if any(k in s.lower() for k in ["experience", "own", "deliver", "ship", "ai", "privacy", "safety", "experiments", "voice", "contact center", "llm", "rag", "uptime", "reliability", "ab test", "a/b", "risk"]):
            keep.append(s.strip())
    # Also split on commas if sentence is long
    reqs = []
    for s in keep:
        if len(s) > 160:
            reqs.extend([x.strip() for x in s.split(",") if len(x.strip()) > 0])
        else:
            reqs.append(s)
    # Dedup and clean
    uniq = []
    seen = set()
    for r in reqs:
        k = r.lower()
        if k not in seen:
            uniq.append(r)
            seen.add(k)
    return uniq

def embed(texts: List[str], method: str = "auto") -> np.ndarray:
    if method == "st" and _HAS_ST:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return model.encode(texts, normalize_embeddings=True)
    # fallback: TF-IDF cosine
    vec = TfidfVectorizer(max_features=4096, ngram_range=(1,2))
    X = vec.fit_transform(texts)
    return X.toarray() / (np.linalg.norm(X.toarray(), axis=1, keepdims=True) + 1e-9)

def map_bullets_to_reqs(bullets: List[str], reqs: List[str]) -> Dict[int, List[Tuple[int, float]]]:
    embs = embed(bullets + reqs, method="st" if _HAS_ST else "tfidf")
    B = embs[:len(bullets)]
    R = embs[len(bullets):]
    sims = cosine_similarity(B, R)
    mapping = {}
    for i in range(len(bullets)):
        # top 3 matches
        idxs = np.argsort(-sims[i])[:3]
        mapping[i] = [(int(j), float(sims[i][j])) for j in idxs]
    return mapping

ACTION_VERBS = ["Led","Owned","Delivered","Shipped","Drove","Implemented","Launched","Coordinated","Optimized","Reduced","Increased","Improved"]

def heuristic_rewrite(source: str, req: str) -> str:
    # Keep metrics if present; mirror nouns/phrases from req; keep result to ~1 line.
    # Extract %/numbers
    m = re.findall(r"(\d+\.?\d*%|\d+\.?\d*)", source)
    metric = m[0] if m else None
    # Mirror a few nouns from req
    nouns = re.findall(r"\b([A-Za-z]{4,})\b", req)
    nouns = [n for n in nouns if n.lower() not in {"with","that","into","your","will","have","plus","needed","experience"}]
    mirror = ", ".join(list(dict.fromkeys(nouns))[:3])
    verb = ACTION_VERBS[hash(source) % len(ACTION_VERBS)]
    # Build sentence
    parts = [verb, source[0].lower() + source[1:]]
    if mirror:
        parts.append(f"— aligned to {mirror}")
    if metric and "%" in metric and metric not in source:
        parts.append(f" (result: {metric})")
    out = " ".join(parts)
    # Normalize
    out = out.replace("  ", " ").strip()
    # Trim to ~180 chars
    return (out[:177] + "…") if len(out) > 180 else out

def openai_rewrite(source: str, req: str, model: str) -> str:
    if not _HAS_OPENAI:
        raise RuntimeError("openai package not available. Use --provider local or pip install openai.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment.")
    client = OpenAI(api_key=api_key)
    sys = (
        "You tailor resume bullets to a JD without inventing facts. "
        "Never add employers, dates, titles, or metrics not present in SOURCE. "
        "Be ATS-friendly, 1 line, strong verb, include metric if present."
    )
    prompt = f"JD REQUIREMENT:\n{req}\n\nSOURCE BULLET:\n{source}\n\nRewrite a single bullet. If insufficient evidence, say: 'INSUFFICIENT EVIDENCE — <reason>'."
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":sys},{"role":"user","content":prompt}],
        temperature=0.2,
        max_tokens=120,
    )
    latency = time.time() - t0
    text = resp.choices[0].message.content.strip()
    return text, latency

def suggest(bullets: List[str], reqs: List[str], provider: str, model: str) -> List[Dict[str, Any]]:
    mapping = map_bullets_to_reqs(bullets, reqs)
    results = []
    for bi, matches in mapping.items():
        source = bullets[bi]
        for (rj, score) in matches[:2]:
            req = reqs[rj]
            if provider == "openai":
                try:
                    text, latency = openai_rewrite(source, req, model=model)
                except Exception as e:
                    text = f"ERROR: {e}"
                    latency = None
            else:
                t0 = time.time()
                text = heuristic_rewrite(source, req)
                latency = time.time() - t0
            # Truthfulness gate: if the model says insufficient, mark low support
            support = "supported"
            if text.upper().startswith("INSUFFICIENT EVIDENCE"):
                support = "insufficient"
            results.append({
                "jd_req": req,
                "source_bullet": source,
                "suggested": text,
                "similarity": round(score, 3),
                "support": support,
                "latency_s": None if latency is None else round(latency, 3),
            })
    return results

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_log(path: str, rows: List[Dict[str, Any]]):
    cols = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp"]+cols)
        w.writeheader()
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        for r in rows:
            rr = {"timestamp": ts, **r}
            w.writerow(rr)

def run_eval(eval_csv: str, provider: str, model: str):
    import pandas as pd
    df = pd.read_csv(eval_csv)
    rows = []
    for _, row in df.iterrows():
        out = suggest([row["resume_bullet"]], [row["job_requirement"]], provider, model)
        sug = out[0]["suggested"]
        grounded = ("INSUFFICIENT" not in sug.upper()) and all(
            tok.lower() in (row["resume_bullet"].lower()+" "+row["job_requirement"].lower())
            or tok.isdigit() or tok.lower() in {"the","a","an","to","and","of","for","with","by","on"}
            for tok in re.findall(r"[A-Za-z0-9%]+", sug)
        )
        rows.append({
            "id": row["id"],
            "suggested": sug,
            "grounded_guess": int(grounded),
            "notes": ""
        })
    out_path = "outputs/eval_results.csv"
    Path("outputs").mkdir(exist_ok=True, parents=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    console.print(f"[bold green]Eval results written to {out_path}[/bold green]")
    # Pretty print
    table = Table(title="Eval — groundedness (heuristic)")
    table.add_column("ID"); table.add_column("Grounded?"); table.add_column("Suggestion", overflow="fold", max_width=80)
    for r in rows:
        table.add_row(r["id"], "✅" if r["grounded_guess"] else "⚠️", r["suggested"][:400])
    console.print(table)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, help="Path to resume bullets (.txt, one per line)")
    parser.add_argument("--jd", type=str, help="Path to job description (.txt)")
    parser.add_argument("--provider", type=str, choices=["local","openai"], default="local")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--eval", type=str, help="Path to golden_questions.csv to run evaluation")
    args = parser.parse_args()

    Path("outputs").mkdir(exist_ok=True, parents=True)

    if args.eval:
        run_eval(args.eval, args.provider, args.model)
        return

    if not args.resume or not args.jd:
        console.print("[red]Provide --resume and --jd or use --eval[/red]")
        return

    bullets = read_lines(args.resume)
    jd_text = Path(args.jd).read_text(encoding="utf-8")
    reqs = jd_requirements(jd_text)

    console.print(f"[bold]Loaded[/bold] {len(bullets)} bullets and {len(reqs)} JD requirements.")
    results = suggest(bullets, reqs, args.provider, args.model)

    write_jsonl("outputs/suggestions.jsonl", results)
    write_log("outputs/run_log.csv", results)

    # Show a small table
    table = Table(title="Top suggestions")
    table.add_column("Sim"); table.add_column("Support"); table.add_column("Source", overflow="fold", max_width=50)
    table.add_column("JD Req", overflow="fold", max_width=50); table.add_column("Suggested", overflow="fold", max_width=70)
    for r in results[:10]:
        table.add_row(str(r["similarity"]), r["support"], r["source_bullet"], r["jd_req"], r["suggested"])
    console.print(table)
    console.print("[green]Wrote outputs to outputs/suggestions.jsonl and outputs/run_log.csv[/green]")

if __name__ == "__main__":
    main()
