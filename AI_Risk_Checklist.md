# AI Risk & Compliance Checklist (Paste into PRD)

**Data**
- [ ] Inputs may contain PII. Default to local processing; if using a cloud API, document the vendor’s data retention policy.
- [ ] No training on user data without explicit consent.
- [ ] Store only minimal logs; redact PII where feasible.

**Truthfulness & Provenance**
- [ ] Every suggested bullet must reference source resume line(s).
- [ ] No fabrication of employers, titles, dates, or scope.
- [ ] Numeric claims must be present in source or supplied by the user in‑session.

**Safety & Abuse**
- [ ] Block prompts that request falsification or fake credentials.
- [ ] Rate‑limit and add timeouts/retries; cap cost per run.

**User Disclosure**
- [ ] Surface “AI‑assisted; verify accuracy before submission.”
- [ ] Provide a one‑click way to view sources for any suggestion.

**Security**
- [ ] Prevent prompt injection by forbidding instructions that override truthfulness/constraints.
- [ ] Escape/strip HTML in rendered outputs.

**EU Context (summary)**
- [ ] Maintain a system card: purpose, data flows, known limitations.
- [ ] Provide a human‑in‑the‑loop step before final use.
- [ ] Vendor portability plan documented.
