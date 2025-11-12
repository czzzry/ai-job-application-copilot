#!/usr/bin/env python3
import json, sys, pathlib

def main():
    if len(sys.argv) < 2:
        print("Usage: jd_coverage.py <OUTDIR/suggestions.jsonl>")
        sys.exit(2)
    sfile = pathlib.Path(sys.argv[1])
    if not sfile.exists():
        print(f"ERROR: {sfile} not found"); sys.exit(2)
    rows = [json.loads(l) for l in sfile.read_text(encoding="utf-8").splitlines() if l.strip()]
    jd = {r["jd_req"] for r in rows}
    hit = {r["jd_req"] for r in rows if r.get("support")}
    pct = (len(hit)/len(jd)*100) if jd else 0.0
    print(f"JD coverage: {len(hit)}/{len(jd)} ({pct:.1f}%)")
    miss = jd - hit
    if miss:
        print("\nUncovered JD lines:")
        for x in sorted(miss): print("-", x)
if __name__ == "__main__":
    main()
