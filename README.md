# PII Service

A local, Docker-based service that detects personally identifiable information
in German and English text and then **redacts** it, **pseudonymizes** it (with
placeholders or realistic fakes), or **restores** it later. It runs fully
offline and keeps no plaintext at rest.

Detection runs as an ensemble of local layers on top of Microsoft Presidio:

| Layer | What | Strengths |
|---|---|---|
| `patterns` | Presidio recognizers (checksums: IBAN, credit card, German IDs …), custom regexes (employee IDs, cost centres, project codes, addresses, dates, clock times, durations), dictionaries (your project names, GDPR Art. 9 special-category terms with context) | precise, structured data |
| `spacy` | `de_core_news_lg` / `en_core_web_lg` NER, language detected per paragraph | fast names/places |
| `transformer` | `Davlan/xlm-roberta-large-ner-hrl` (multilingual) | high-quality PER/ORG/LOC |
| `gliner` | `urchade/gliner_multi_pii-v1`, zero-shot labels | health, religion, project names … |
| `llm` | Qwen3-8B (GGUF, llama.cpp) | context-dependent PII; settles uncertain spans |

**Profiles** select layers: `fast` (patterns+spacy, ~0.02 s/doc), `balanced` (+transformer, gliner,
~0.5 s/doc), `thorough` (+llm, about a minute per 4 KB chunk on a 4-core CPU). The default is `thorough`.

## Quick start

```bash
cp .env.example .env
# set PII_VAULT_KEY=$(openssl rand -base64 32) and PII_API_KEYS=<some-secret>
scripts/fetch_llm.sh            # once, ~5 GB into ./models/gguf
docker compose up -d --build    # first build downloads ~4 GB of models into the image
curl localhost:8000/ready
```

The services are `api` (published port), `worker-fast`, `worker-full`, `llm` and `redis`.
Only the api is attached to an outside network. RAM is roughly llm 9–10 GB,
worker-full 5–6 GB, worker-fast 2 GB, everything else under 1 GB. Thread counts are set in `.env`.

## API

Interactive docs: `http://localhost:8000/docs`. Send `X-API-Key` with every request when `PII_API_KEYS` is set.

```bash
# synchronous (waits up to ?wait= seconds, default 60, then answers 202 + job id)
curl -s -X POST localhost:8000/v1/anonymize -H 'content-type: application/json' -H "X-API-Key: $KEY" \
  -d '{"text": "Max Müller (max@firma.de) war am 03.10.2026 krank.", "mode": "fake", "profile": "fast"}'

# long texts: submit a job, poll it
curl -s -X POST localhost:8000/v1/jobs -H 'content-type: application/json' -H "X-API-Key: $KEY" \
  -d '{"kind": "anonymize", "request": {"text": "...", "mode": "placeholder"}}'
curl -s localhost:8000/v1/jobs/<job_id> -H "X-API-Key: $KEY"
```

| Endpoint | Purpose |
|---|---|
| `POST /v1/anonymize` | detect + replace (sync with timeout) |
| `POST /v1/analyze` | detect only. Original strings are returned only with `include_text: true` |
| `POST /v1/jobs`, `GET/DELETE /v1/jobs/{id}` | async jobs; results kept 24 h (`vault.job_ttl_seconds`) |
| `POST /v1/deanonymize` | restore originals in any text containing a session's replacements |
| `DELETE /v1/sessions/{id}` | erase a session's mapping |
| `GET /v1/entities`, `/health`, `/ready` | metadata, liveness, readiness (redis + one worker per queue) |

### Anonymize request

```jsonc
{
  "text": "...",
  "mode": "placeholder",          // redact | placeholder | mask | fake | hash
  "profile": "thorough",          // fast | balanced | thorough
  "operators": {"IBAN_CODE": "mask", "ORGANIZATION": "keep"},  // per entity, "*" = all
  "reversible": true,             // store the mapping -> returns session_id
  "session_id": "…",              // reuse: same originals get the same replacements across requests
  "timesheet": {"clock_times": {"fake": true}, "durations": {"detect": false}},
  "allow_list": ["ACME Cloud"], "deny_list": ["Codename X"],
  "entities": null, "ignore_entities": ["LOCATION"],
  "language": null,               // de | en; default: auto-detect per paragraph
  "include_originals": false      // include original strings in the entity report
}
```

### Modes

| Mode | Example | Reversible |
|---|---|---|
| `redact` | `[REDACTED]` | no |
| `placeholder` | `<PERSON_1>` (numbered consistently) | yes |
| `mask` | `**** **** **** **** **30 00` (keeps the last 4 characters of IBANs, cards and phone numbers only) | no |
| `fake` | `Jonas Schmidt`, a valid fake IBAN, a shifted date … | yes |
| `hash` | `<PERSON_3fa9c1d2e0>` (HMAC with the session seed) | yes |

How fake mode works:
- **Person names** are mapped token by token, so "Max Müller" → "Jonas Schmidt" and a later "Müller" → "Schmidt". The gender of a first name is kept when it is known.
- **Formats are preserved:** fake IBANs pass the checksum, fake credit cards pass Luhn, phone numbers and IDs keep their shape, and the country code is kept.
- **Collisions are avoided:** a fake never equals another original's fake and never occurs in the source text, which is what makes reversing unambiguous.
- **Special-category data** (health, religion, …) is never faked. A fake diagnosis would be misleading, so it gets a placeholder (`anonymize.no_fake_entities`).
- **No fake available:** when no faker exists for a value, or a date can't be parsed, the placeholder operator is used.

### Timesheet switches

Defaults are in `config.yaml → timesheet`, and each request can override them. `detect: false` leaves an entity untouched. `fake: false` makes the entity use a placeholder even in fake mode.

| Feature | Entity | Fake behaviour | Default |
|---|---|---|---|
| `project_names` | PROJECT_NAME | static list + random subword combinations (~9.6 million names); keeps a leading "Projekt"/"Project" | fake on |
| `project_codes` | PROJECT_CODE | letters/digits substituted, format kept | fake on |
| `dates` | DATE_TIME, DATE_OF_BIRTH | **one offset per session** (multiple of 7 days ⇒ weekdays kept), every format: `03.10.2026`, `3. Oktober`, `Oct 3rd`, `KW 40` … | fake on |
| `clock_times` | CLOCK_TIME | one offset per session (intervals preserved), `08:30`, `8 Uhr`, `9:30 pm` | fake off |
| `durations` | DURATION | jitter ±15 %, rounded to 15 min, `7,5 h`, `7:30 h`, `2 PT` | fake off |

## Configuration

Everything is in [`config/config.yaml`](config/config.yaml). The most important settings:
- profiles, and the queue that serves each one
- model names: changing them and rebuilding the image downloads the new models
- custom regexes (`layers.patterns.custom_patterns`); `group` selects the part to redact
- merge thresholds and layer weights
- timesheet switches

Dictionaries:
- `config/dictionaries/project_names.txt`: **put your real project names here**, one per line, aliases as extra lines. This is by far the most reliable way to catch them.
- `config/dictionaries/special_category.yaml`: GDPR Art. 9 term lists in DE and EN. Weak terms count only with nearby context words.
- `config/dictionaries/allow_list.txt`: terms that are never redacted.
- `app/data/fake/project_words.yaml`: the word lists for fake project names.

To change config without rebuilding, mount your own file: `-v ./my-config:/app/config:ro`.

## How the ensemble decides

All layers run, and then the merger (`app/detection/merger.py`) combines their results:
1. **Score:** each span's score is its own score times the layer's weight, plus 0.15 for every other layer that found an overlapping span of the same entity family.
2. **Thresholds:** a span found by a single layer needs a higher score (`single_source_min_score`) than one several layers agree on.
3. **Selection:** spans are picked greedily without overlap. A precise span that several layers back beats a sloppy long one.
4. **Absorption:** an ADDRESS that contains LOCATION spans absorbs them.
5. **Type on identical spans:** when spans have the same boundaries, a pattern or dictionary type wins (e.g. ver.di → TRADE_UNION, not ORGANIZATION).
6. **Recall bias:** when an overlapping span loses, its uncovered remainder is kept, so text a layer considered PII never leaks.
7. **LLM (thorough profile only):**
   - It sees confidently detected strings as "already found" and reports only what is missing. Output tokens are the slow part on CPU.
   - It gives yes/no verdicts on uncertain spans.
   - Its findings must occur verbatim in the text, so a hallucinated finding is discarded.

## Security

- **Encrypted at rest:** job payloads, results and session mappings are AES-256-GCM encrypted in Redis with `PII_VAULT_KEY`. Without that key, sessions can't be read. Keep the key, and rotate it only together with flushing Redis.
- **No plaintext in logs:** request texts are never logged, and the api runs with `--no-access-log`.
- **Offline:** containers are read-only, run as non-root with no capabilities, and the backend network is `internal` (no internet). Models are baked into the image and the HF libraries run in offline mode.
- **Keep reports text-free:** `analyze` and `anonymize` reports contain only positions and types unless `include_text` / `include_originals` is set.

## Development

```bash
uv sync --all-extras
uv pip install "de_core_news_lg @ https://github.com/explosion/spacy-models/releases/download/de_core_news_lg-3.8.0/de_core_news_lg-3.8.0-py3-none-any.whl" \
               "en_core_web_lg @ https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl"
HF_HOME=models/hf python scripts/download_models.py
redis-server --daemonize yes
pytest -q                                         # API tests need Redis; ML tests need the models

# run locally
export PII_INSECURE_DEV=1 HF_HOME=models/hf HF_HUB_OFFLINE=1
PII_QUEUE=pii:fast arq app.worker.WorkerSettings &
PII_QUEUE=pii:full PII_LLM_URL=http://127.0.0.1:8080 arq app.worker.WorkerSettings &
uvicorn app.api.main:app --port 8000
```

### Evaluation

```bash
python -m eval.generate --n 200                       # synthetic labeled DE/EN data
python -m eval.evaluate --profiles fast,balanced      # recall / precision / leak rate per entity
python -m eval.evaluate --profiles thorough --limit 20 --llm-url http://127.0.0.1:8080
```

The **leak rate** is the share of PII characters that no prediction covers. It is the number that matters for redaction.
Synthetic data is only a smoke test. Before tuning `merge.*` thresholds, build a small labeled set from your real (sanitized) texts.
