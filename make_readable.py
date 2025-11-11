#!/usr/bin/env python3
import argparse, json, csv, sys, shutil, subprocess
from pathlib import Path

DEFAULT_SUG = Path("outputs/suggestions.jsonl")
DEFAULT_LOG = Path("outputs/run_log.csv")

def jsonl_to_csv(src: Path, dst: Path):
    cols = ["similarity","support","source_bullet","jd_req","suggested","latency_s"]
    rows = []
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    return dst

def jsonl_to_pretty_json(src: Path, dst: Path):
    arr = []
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        arr.append(json.loads(line))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(arr, indent=2, ensure_ascii=False), encoding="utf-8")
    return dst

def csv_to_pretty_txt(src: Path, dst: Path):
    column = shutil.which("column")
    if column:
        out = subprocess.check_output([column, "-t", "-s,", str(src)])
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(out)
    else:
        rows = [r.split(",") for r in src.read_text(encoding="utf-8").splitlines()]
        widths = [max(len(c) for c in col) for col in zip(*rows)]
        lines = []
        for r in rows:
            parts = [c.ljust(w) for c, w in zip(r, widths)]
            lines.append("  ".join(parts))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("\n".join(lines), encoding="utf-8")
    return dst

def quicklook(paths):
    ql = shutil.which("qlmanage")
    if not ql:
        print("Quick Look (qlmanage) not found; skipping previews.", file=sys.stderr)
        return
    for p in paths:
        try:
            subprocess.Popen([ql, "-p", str(p)])
        except Exception as e:
            print(f"Quick Look failed for {p}: {e}", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser(description="Make outputs easy to read.")
    ap.add_argument("--open", action="store_true", help="Open Quick Look previews after generating files")
    ap.add_argument("--no-pretty", action="store_true", help="Skip pretty JSON/TXT outputs; only write CSV")
    ap.add_argument("--files", nargs="*", help="Override inputs: [suggestions.jsonl, run_log.csv]")
    args = ap.parse_args()

    if args.files:
        sug_src = Path(args.files[0]) if len(args.files) >= 1 else DEFAULT_SUG
        log_src = Path(args.files[1]) if len(args.files) >= 2 else DEFAULT_LOG
    else:
        sug_src = DEFAULT_SUG
        log_src = DEFAULT_LOG

    if not sug_src.exists():
        print(f"ERROR: {sug_src} not found. Run the generator first.", file=sys.stderr)
        sys.exit(1)
    if not log_src.exists():
        print(f"WARNING: {log_src} not found. Will skip log prettifying.", file=sys.stderr)

    sug_csv = Path("outputs/suggestions.csv")
    sug_pretty_json = Path("outputs/suggestions_pretty.json")
    log_pretty_txt = Path("outputs/run_log_pretty.txt")

    out_csv = jsonl_to_csv(sug_src, sug_csv)
    print(f"Wrote {out_csv}")

    open_paths = [out_csv]

    if not args.no_pretty:
        out_json = jsonl_to_pretty_json(sug_src, sug_pretty_json)
        print(f"Wrote {out_json}")
        open_paths.append(out_json)

        if log_src.exists():
            out_txt = csv_to_pretty_txt(log_src, log_pretty_txt)
            print(f"Wrote {out_txt}")
            open_paths.append(out_txt)

    if args.open:
        quicklook(open_paths)

if __name__ == "__main__":
    main()
