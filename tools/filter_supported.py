#!/usr/bin/env python3
import json, csv, sys, pathlib

def main():
    if len(sys.argv) < 2:
        print("Usage: filter_supported.py <OUTDIR>")
        sys.exit(2)
    outdir = pathlib.Path(sys.argv[1])
    sfile = outdir / "suggestions.jsonl"
    if not sfile.exists():
        print(f"ERROR: {sfile} not found"); sys.exit(2)
    rows = []
    with sfile.open(encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            rec = json.loads(line)
            if rec.get("support"):
                rows.append(rec)
    p = outdir / "supported.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["similarity","source_bullet","jd_req","suggested","latency_s"])
        for r in rows:
            w.writerow([r["similarity"], r["source_bullet"], r["jd_req"], r["suggested"], r["latency_s"]])
    print(f"Wrote {p} with {len(rows)} supported lines")

if __name__ == "__main__":
    main()
