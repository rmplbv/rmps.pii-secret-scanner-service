# Benchmark: compare NER models and LLMs for the PII service

```bash
python -m bench run --data <files-or-dirs> --out bench/out/<name>     # run + report
python -m bench report --out bench/out/<name>                          # re-render the report
```

It produces `bench/out/<name>/report.md` and `report.html` (tables plus a per-document view with
highlights). Raw detections are cached per model in `bench/out/<name>/raw/`, so you can re-run and
only missing work is computed. Use `--force` to recompute.

## Your own texts

Put files in a folder: `.txt`, `.md`, `.eml`, `.csv`, `.tsv`, `.html`, `.htm`. HTML is analysed the
way the service does it: visible text, attributes such as `href`/`title`, and comments.

**Unlabeled files** need no preparation. Every model runs on them and `report.html` shows the
detections of the best ensemble per document, so you can review them by eye. Recall and precision
can only be computed for labeled files.

**Labeled files** give you real numbers. Wrap each piece of PII in `[[ENTITY|text]]`:

```text
Hallo [[PERSON|Anna Huber]], bitte die Rechnung an [[ADDRESS|Eisvogelweg 1, 91161 Hilpoltstein]]
schicken. Sie ist seit Montag [[HEALTH_DATA|krankgeschrieben]].
```

In HTML, the markup goes inside the text: `<a href="mailto:[[EMAIL_ADDRESS|anna@example.org]]">`.
The `[[…]]` markers are removed before the models see the text.

Entity names to use:
- **Names and places:** PERSON, ORGANIZATION, LOCATION, ADDRESS
- **Contact and identifiers:** EMAIL_ADDRESS, PHONE_NUMBER, IBAN_CODE, CREDIT_CARD, ID_NUMBER, EMPLOYEE_ID, USERNAME, URL, IP_ADDRESS
- **Dates:** DATE_OF_BIRTH, DATE_TIME
- **Timesheet data:** PROJECT_NAME, PROJECT_CODE, CLOCK_TIME, DURATION
- **Special categories:** HEALTH_DATA, RELIGION, POLITICAL_OPINION, TRADE_UNION, ETHNIC_ORIGIN, SEXUAL_ORIENTATION, GENETIC_BIOMETRIC

Types in the same family count as a match: LOCATION/ADDRESS, the ID types, PROJECT_NAME/PROJECT_CODE,
DATE_TIME/DATE_OF_BIRTH, and the special categories.

Label as the service should behave. In ordinary e-mails or letters, a meeting date, time or duration
is not PII, so leave it unlabeled. In timesheets, label it. To state which kind a file is, put
`#preset: timesheet` (or `general`) on the first line, or `<!-- preset: timesheet -->` in HTML.
The report then also shows whether `preset: auto` guessed right.

A JSONL file in the format of `eval/generate.py` works too
(`{"id", "text", "spans": [{"start","end","entity"}], "format", "preset", "kind"}`).

## Examples

```bash
# everything in candidates.yaml on your folder (NER on all files, LLMs on the first 16)
python -m bench run --data ~/pii-samples --out bench/out/real

# only some models; the current production models are always included
python -m bench run --data ~/pii-samples --out bench/out/real --only bardsai-eu-pii,gemma4-26b-a4b

# skip the LLMs (fast), or run them on all documents
python -m bench run --data ~/pii-samples --out bench/out/real --no-llm
python -m bench run --data ~/pii-samples --out bench/out/real --llm-limit 0

# LLMs through Docker instead of a local llama-server binary
python -m bench run --data ~/pii-samples --out bench/out/real --llm-runner docker
```

## LLM runners
- **`binary`** (default): starts `llama-server` from `--llama-server PATH` or `$LLAMA_SERVER`, one model at a time.
- **`docker`**: starts `ghcr.io/ggml-org/llama.cpp:server` with `./models/gguf` mounted.
- **`external`**: uses `--llm-url` (one model per run; start it yourself, e.g. with `docker compose up llm`).

GGUF files are downloaded to `models/gguf/` on first use. Pass `--offline` to forbid downloads.

## Notes
- **GLiNER2 models** need `transformers<5`, so they run in `.venv-gliner2` (see the `python:` key in `candidates.yaml`). Create it with:
  ```bash
  uv venv --python 3.11 .venv-gliner2
  uv pip install --python .venv-gliner2/bin/python --extra-index-url https://download.pytorch.org/whl/cpu \
     "torch>=2.4" "transformers>=4.38,<5" "gliner2[local]>=2.0" sentencepiece protobuf pydantic pyyaml \
     presidio-analyzer "spacy>=3.8,<3.9" faker lingua-language-detector httpx
  ```
- **RAM** is the peak resident memory of each model's own process. **Time** is per document on the machine that ran it.
- **Combinations:** all combinations of up to `--max-combo` NER models (default 2) are scored from the cache. The LLMs then run on top of the best combination, or on `--llm-base`.
