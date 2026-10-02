# PII Service: plan & status

## Requirements (agreed 2026-10-02)
- **Languages:** German and English, auto-detected per paragraph; mixed documents supported.
- **Input:** plain text now. DOCX/Excel later.
- **Modes:** redact, placeholder, mask, fake and hash. Pseudonymization is reversible through sessions.
- **Hardware:** CPU only, about 22 GB of RAM.
- **Throughput:** low volume, 1–100 KB per text. Latency can be minutes to an hour; quality comes first.
- **Coverage:** standard PII, GDPR Art. 9 special categories, and timesheet data (project names and codes, dates, times, durations).
- **Fake project names:** a static list plus random combinations of subwords.
- **Timesheet fakes:** every feature can be switched on or off in config and per request.

## Architecture
`api` (FastAPI, holds no models) → Redis (encrypted job queue and session store) → `worker-fast` (patterns and spaCy) / `worker-full` (adds XLM-R, GLiNER and an LLM via `llm`, a llama.cpp server running Qwen3-8B).

## Phases
| # | Phase | Status |
|---|---|---|
| 1 | Skeleton: API, worker, Redis, compose, patterns + spaCy, redact/placeholder, job API | done |
| 2 | Fake mode: consistent fakes, date shifting, German/English and timesheet recognizers, dictionaries | done |
| 3 | Reversible: encrypted Redis store, `/deanonymize`, session erasure | done |
| 4 | Transformer + GLiNER layers, chunking, ensemble merger | done |
| 5 | LLM layer: discovery + adjudication, grounded in the text, llama.cpp container | done |
| 6 | Eval harness: synthetic DE/EN set, recall/precision/leak rate per entity | done (synthetic only) |

## Next steps / open items
- [ ] **Real eval set:** label 50–100 sanitized real texts (emails, timesheets) and tune the `merge.*` thresholds and layer weights against them. Synthetic data is only a smoke test.
- [ ] **Project names:** fill `config/dictionaries/project_names.txt` with your real projects. GLiNER and the LLM alone find only some of them.
- [ ] **Docker build:** verify on the target host. The Docker daemon couldn't run in the development sandbox, so the Dockerfile steps were simulated there but the image was never actually built.
- [ ] **Tabular input** (CSV/Excel timesheets): column-aware handling. Classify each column once instead of running NER per cell.
- [ ] **DOCX/XLSX input:** extract text, anonymize it, and write it back while keeping the formatting.
- [ ] **LLM speed:** about 4 tokens/s on a 4-core i7-6700. Options: a smaller model (Qwen3-4B), a quantized KV cache, or more cores.
- [ ] **Observability:** Prometheus metrics (job durations, entity counts; never any text).
