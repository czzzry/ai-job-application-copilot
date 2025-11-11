

## Make outputs easy to read (helper)
From the project folder:
```bash
python make_readable.py --open
```
This writes:
- `outputs/suggestions.csv` (spreadsheet-friendly)
- `outputs/suggestions_pretty.json` (valid JSON array)
- `outputs/run_log_pretty.txt` (aligned columns)

Use `--no-pretty` to skip the pretty JSON/TXT and only produce the CSV.
