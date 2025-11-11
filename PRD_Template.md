# PRD — Job‑Application Copilot (Resume → JD Tailoring)

## 1) Problem & Context
Hiring teams skim quickly; generic applications underperform. Tailoring verbs, metrics, and language to the JD increases recruiter response rates but is time‑consuming and error‑prone.

## 2) Users & JTBD (Job-to-be-done)
- Primary user: job seeker (mid/senior IC or lead. 
- JTBD: “When applying to a role, help me quickly produce *truthful*, *ATS‑friendly* bullets that mirror the JD’s language without fabricating.”

## 3) Scope (MVP)
- Inputs: resume bullets (text), Job Description (text).
- Output: 6–12 bullet suggestions tailored to the JD, each linked to a source resume line (provenance).

### Non‑goals
- No auto‑submission or cover‑letter generation (phase 2).
- No fact invention; if proof is missing, mark as insufficient.

## 4) Success Metrics (leading → lagging)
- TTR (time‑to‑ready bullets) ↓ 70% vs. manual.
- % bullets with explicit metric/impact ≥ 80%.
- Recruiter reply rate (self‑reported) ↑.
- User quality rating ≥ 4.3/5 across 30+ uses.

## 5) Core Experience
- Paste or upload resume bullets (one per line) + JD.
- See a ranked list of suggested bullets with:
  - **Provenance**: links to source bullet(s).
  - **JD coverage**: which requirement(s) it addresses.
  - **Truthfulness badge**: “supported / weak evidence / insufficient.”

## 6) Safety & Compliance
- No fabrication. Prefer omission over hallucination.
- PII: never persist inputs server‑side by default.
- Data retention: local‑only unless the user opts in to cloud sync.
- Disclosure: “AI‑assisted; you are responsible for accuracy.”
- Guardrails: block adding employers, dates, titles, or metrics not present in the source resume unless the user provides them explicitly.

## 7) Technical Approach (MVP)
- Parse JD → extract skills/requirements.
- Map resume bullets ↔ requirements via embeddings + heuristics.
- Rewrite with controlled templates (+ optional LLM for polish).
- Log latency, approx token usage, and cost when available.
- Evaluation: golden‑questions set; groundedness and coverage metrics.

## 8) Risks & Mitigations
- **Risk:** Output invents a metric → **Mitigation:** require numeric evidence from source; otherwise tag as “weak/insufficient.”
- **Risk:** Overfitting to keywords → **Mitigation:** semantic mapping + human preview.
- **Risk:** ATS parsing quirks → **Mitigation:** plain‑text friendly formatting (no bullets > 2 lines, avoid special chars).

## 9) Rollout
- Alpha: 1 users, rapid iteration on eval failures.
- Beta: add small UI, export to .txt/.docx.
- GA: monitoring, versioned prompts/templates, optional vendor swap.

## 10) Open Questions
- Should we allow users to add *additional* facts (e.g., new metrics) and tag them as “user‑provided”?
- Add multilingual support?
