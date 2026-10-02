# PII benchmark: synth

Documents: 240 (240 labeled), data: ['eval/out/synthetic.jsonl']

*Leak rate* = share of gold PII characters not covered by any prediction (lower is better). *Recall* counts a gold span as found when a prediction of the same entity family overlaps it. *RAM* = peak resident memory of the processes involved (models loaded separately; in production shared libraries make the sum somewhat smaller). Time = seconds per document on this machine.

## Ensembles (patterns + spaCy + NER models)

| # | layers | leak rate | recall | precision | preset acc. | RAM GB | s/doc |
|---|---|---|---|---|---|---|---|
| 1 | patterns + spacy + bardsai-eu-pii + gliner-multi-v2.1 | 1.0% | 0.984 | 0.983 | 100.0% | 6.4 | 0.41 |
| 2 | patterns + spacy + davlan-xlmr-ner-hrl + gliner-multi-pii-v1 **(current)** | 1.2% | 0.979 | 0.988 | 100.0% | 6.8 | 0.63 |
| 3 | patterns + spacy + bardsai-eu-pii + gliner2.5-multi | 1.3% | 0.981 | 0.979 | 100.0% | 6.3 | 0.41 |
| 4 | patterns + spacy + bardsai-eu-pii + gliner2-multi | 1.3% | 0.977 | 0.980 | 100.0% | 6.2 | 0.45 |
| 5 | patterns + spacy + bardsai-eu-pii + gliner-multi-pii-v1 | 1.3% | 0.976 | 0.987 | 100.0% | 6.1 | 0.41 |
| 6 | patterns + spacy + openai-privacy-filter + bardsai-eu-pii | 1.5% | 0.945 | 0.974 | 100.0% | 6.8 | 0.72 |
| 7 | patterns + spacy + gliner-multi-pii-v1 + gliner-multi-v2.1 | 1.5% | 0.967 | 0.987 | 100.0% | 7.8 | 0.60 |
| 8 | patterns + spacy + bardsai-eu-pii | 1.5% | 0.976 | 0.990 | 100.0% | 3.3 | 0.12 |
| 9 | patterns + spacy + bardsai-eu-pii + gliner2-privacy-pii-multi | 1.5% | 0.974 | 0.968 | 100.0% | 6.4 | 0.47 |
| 10 | patterns + spacy + openai-privacy-filter + gliner2-privacy-pii-multi | 1.5% | 0.941 | 0.955 | 100.0% | 8.6 | 0.97 |
| 11 | patterns + spacy + openmed-pf-multilingual-v2 + bardsai-eu-pii | 1.5% | 0.981 | 0.984 | 100.0% | 5.6 | 0.68 |
| 12 | patterns + spacy + davlan-xlmr-ner-hrl + gliner2.5-multi | 1.5% | 0.982 | 0.981 | 100.0% | 7.0 | 0.63 |
| 13 | patterns + spacy + nym-pii-multilingual + gliner-multi-v2.1 | 1.5% | 0.992 | 0.979 | 100.0% | 6.7 | 0.48 |
| 14 | patterns + spacy + nym-pii-multilingual + gliner-multi-pii-v1 | 1.6% | 0.987 | 0.983 | 100.0% | 6.3 | 0.48 |
| 15 | patterns + spacy + davlan-xlmr-ner-hrl + gliner2-privacy-pii-multi | 1.6% | 0.976 | 0.970 | 100.0% | 7.1 | 0.70 |
| 16 | patterns + spacy + nym-pii-multilingual + gliner2-privacy-pii-multi | 1.6% | 0.986 | 0.967 | 100.0% | 6.7 | 0.54 |
| 17 | patterns + spacy + openai-privacy-filter + gliner-multi-pii-v1 | 1.6% | 0.940 | 0.974 | 100.0% | 8.2 | 0.91 |
| 18 | patterns + spacy + davlan-xlmr-ner-hrl + openai-privacy-filter | 1.6% | 0.943 | 0.974 | 100.0% | 7.6 | 0.94 |
| 19 | patterns + spacy + openmed-pf-multilingual-v2 + gliner2-privacy-pii-multi | 1.6% | 0.967 | 0.966 | 100.0% | 7.3 | 0.93 |
| 20 | patterns + spacy + openmed-pf-multilingual-v2 + gliner-multi-pii-v1 | 1.6% | 0.968 | 0.985 | 100.0% | 7.0 | 0.87 |
| 21 | patterns + spacy + nym-pii-multilingual + gliner2-multi | 1.6% | 0.989 | 0.974 | 100.0% | 6.5 | 0.52 |
| 22 | patterns + spacy + openai-privacy-filter + gliner2.5-multi | 1.6% | 0.941 | 0.965 | 100.0% | 8.5 | 0.91 |
| 23 | patterns + spacy + openmed-pf-nemotron-v2 + bardsai-eu-pii | 1.6% | 0.976 | 0.983 | 100.0% | 5.5 | 0.72 |
| 24 | patterns + spacy + openai-privacy-filter + gliner-multi-v2.1 | 1.6% | 0.945 | 0.973 | 100.0% | 8.6 | 0.91 |
| 25 | patterns + spacy + openai-privacy-filter + gliner2-multi | 1.6% | 0.943 | 0.966 | 100.0% | 8.4 | 0.95 |

## NER models

| model | kind | alone: leak / recall / precision | with patterns+spaCy: leak / recall / precision | load s | s/doc | peak RAM GB |
|---|---|---|---|---|---|---|
| nym-pii-multilingual | hf_token | 14.7% / 0.803 / 0.990 | 2.2% / 0.977 / 0.985 | 21.6 | 0.169 | 1.64 |
| gliner2-privacy-pii-multi | gliner2 | 19.7% / 0.726 / 0.973 | 2.4% / 0.962 / 0.971 | 23.1 | 0.345 | 3.09 |
| openai-privacy-filter | hf_token | 21.5% / 0.641 / 0.985 | 3.2% / 0.925 / 0.977 | 35.0 | 0.594 | 3.53 |
| bardsai-eu-pii | hf_token | 21.5% / 0.702 / 0.995 | 1.5% / 0.976 / 0.990 | 18.5 | 0.094 | 1.37 |
| gliner2-multi | gliner2 | 22.3% / 0.718 / 0.989 | 2.9% / 0.965 / 0.983 | 21.8 | 0.322 | 2.93 |
| gliner-multi-pii-v1 | gliner | 22.9% / 0.720 / 0.998 | 1.9% / 0.960 / 0.992 | 7.0 | 0.283 | 2.77 |
| gliner2.5-multi | gliner2 | 24.7% / 0.693 / 0.988 | 2.7% / 0.964 / 0.983 | 22.6 | 0.283 | 3.01 |
| openmed-pf-multilingual-v2 | hf_token | 26.2% / 0.733 / 0.991 | 3.2% / 0.962 / 0.987 | 37.3 | 0.554 | 2.29 |
| gliner-multi-v2.1 | gliner | 30.4% / 0.686 / 0.993 | 2.8% / 0.962 / 0.988 | 28.1 | 0.286 | 3.09 |
| openmed-pf-nemotron-v2 | hf_token | 35.7% / 0.623 / 0.989 | 4.1% / 0.952 / 0.987 | 36.0 | 0.599 | 2.21 |
| davlan-xlmr-ner-hrl | hf_token | 38.9% / 0.601 / 0.997 | 2.9% / 0.972 / 0.990 | 2.3 | 0.321 | 2.1 |
| ai4privacy-modernbert | hf_token | 55.9% / 0.482 / 0.954 | 8.1% / 0.927 / 0.968 | 12.0 | 0.151 | 1.06 |

## LLM layer (on top of `patterns + spacy + bardsai-eu-pii + gliner-multi-v2.1`)

Each LLM is compared with the same ensemble *without* LLM on the same documents (the first N of the dataset; dense large models get fewer).

| LLM | docs | leak rate | Δ leak | recall | Δ recall | precision | Δ precision | GGUF GB | peak RAM GB | prompt tok/s | gen tok/s | s/doc | warnings |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nuextract3-addonly | 16 | 1.2% | +0.00 pp | 0.980 | +0.000 | 0.993 | +0.000 | 2.8 | 5.55 | 33.5 | 7.35 | 17.3 | 0 |
| qwen3-4b-instruct-2507-addonly | 16 | 1.2% | +0.00 pp | 0.980 | +0.000 | 0.989 | -0.004 | 2.5 | 5.99 | 34.4 | 7.11 | 18.3 | 0 |
| qwen3-30b-a3b-instruct-2507-addonly | 16 | 1.2% | +0.00 pp | 0.980 | +0.000 | 0.968 | -0.025 | 17.7 | 32.14 | 27.7 | 8.47 | 26.9 | 0 |
| gemma4-26b-a4b-addonly | 16 | 1.3% | +0.10 pp | 0.980 | +0.000 | 0.993 | +0.000 | 14.4 | 31.12 | 24.5 | 7.21 | 32.2 | 0 |
| gemma4-e4b | 16 | 1.5% | +0.21 pp | 0.980 | +0.000 | 0.982 | -0.011 | 5.2 | 8.03 | 27.4 | 7.12 | 24.0 | 0 |
| qwen3-8b | 16 | 1.5% | +0.21 pp | 0.980 | +0.000 | 0.978 | -0.014 | 5.0 | 10.07 | 19.2 | 4.31 | 44.2 | 0 |
| qwen3-30b-a3b-instruct-2507 | 16 | 1.5% | +0.21 pp | 0.980 | +0.000 | 0.971 | -0.022 | 17.7 | 32.14 | 27.9 | 8.48 | 26.6 | 0 |
| gemma4-12b | 16 | 1.6% | +0.32 pp | 0.980 | +0.000 | 0.996 | +0.004 | 7.0 | 18.75 | 9.6 | 2.97 | 61.8 | 0 |
| gemma4-26b-a4b | 16 | 1.6% | +0.36 pp | 0.976 | -0.004 | 0.996 | +0.004 | 14.4 | 31.12 | 25.6 | 7.23 | 31.2 | 0 |
| qwen3.8-27b | 8 | 1.2% | +0.41 pp | 0.993 | +0.000 | 0.986 | -0.014 | 14.3 | 17.84 | 5.1 | 1.25 | 130.5 | 0 |
| qwen3.6-35b-a3b | 16 | 1.7% | +0.43 pp | 0.980 | +0.000 | 0.967 | -0.025 | 17.7 | 18.84 | 26.7 | 5.89 | 31.0 | 0 |
| qwen3.5-9b | 16 | 1.7% | +0.47 pp | 0.976 | -0.004 | 1.000 | +0.007 | 5.7 | 9.74 | 18.5 | 4.24 | 42.2 | 0 |
| nuextract3 | 16 | 1.7% | +0.47 pp | 0.976 | -0.004 | 0.996 | +0.004 | 2.8 | 5.55 | 33.6 | 7.29 | 17.2 | 0 |
| qwen3-4b-instruct-2507 | 16 | 1.7% | +0.47 pp | 0.976 | -0.004 | 0.993 | -0.000 | 2.5 | 5.99 | 34.6 | 7.16 | 18.2 | 0 |
| gpt-oss-20b | 16 | 1.7% | +0.47 pp | 0.976 | -0.004 | 0.993 | -0.000 | 12.1 | 21.42 | 21.6 | 8.19 | 39.8 | 0 |
| qwen3.5-4b | 16 | 1.7% | +0.47 pp | 0.976 | -0.004 | 0.957 | -0.035 | 2.7 | 6.66 | 35.5 | 7.27 | 43.9 | 0 |

## Recall per entity (top ensembles)

| entity | n | #1 | #2 | #3 | #4 | #5 | #6 |
|---|---|---|---|---|---|---|---|
| PERSON | 709 | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 | 0.99 |
| ORGANIZATION | 344 | 0.88 | 0.85 | 0.89 | 0.85 | 0.84 | 0.83 |
| LOCATION | 108 | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 |
| ADDRESS | 84 | 0.89 | 0.94 | 0.92 | 0.92 | 0.92 | 0.91 |
| EMAIL_ADDRESS | 114 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| PHONE_NUMBER | 148 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
| IBAN_CODE | 60 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| DATE_OF_BIRTH | 30 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| EMPLOYEE_ID | 30 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| PROJECT_NAME | 402 | 0.97 | 0.94 | 0.94 | 0.93 | 0.94 | 0.90 |
| PROJECT_CODE | 402 | 1.00 | 0.99 | 0.99 | 0.98 | 0.97 | 0.99 |
| DATE_TIME | 402 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| CLOCK_TIME | 804 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.86 |
| DURATION | 402 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| HEALTH_DATA | 83 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| RELIGION | 30 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| TRADE_UNION | 30 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| POLITICAL_OPINION | 17 | 1.00 | 0.94 | 0.88 | 1.00 | 1.00 | 0.94 |

## Leak rate per document kind (top ensembles)

| kind | #1 | #2 | #3 | #4 | #5 | #6 |
|---|---|---|---|---|---|---|
| booking_list | 0.1% | 0.4% | 0.6% | 0.2% | 0.5% | 0.6% |
| chat | 0.7% | 1.1% | 0.7% | 0.7% | 0.7% | 1.7% |
| email | 3.5% | 3.3% | 3.6% | 3.8% | 3.6% | 4.2% |
| hr_note | 3.1% | 3.3% | 4.9% | 5.2% | 5.1% | 3.6% |
| html | 1.3% | 0.8% | 0.8% | 0.7% | 1.4% | 1.7% |
| nature | 0.0% | 0.7% | 0.0% | 0.0% | 0.0% | 0.0% |
| signature | 0.5% | 2.1% | 0.9% | 0.9% | 0.6% | 1.8% |
| timesheet_table | 0.5% | 0.6% | 0.5% | 0.8% | 0.5% | 0.7% |

## Misses and false positives of #1 (patterns + spacy + bardsai-eu-pii + gliner-multi-v2.1)

**Missed (gold not covered by a same-family prediction)**

- ORGANIZATION (42): `Gordon Group`×3, `Killer`×2, `Miller, Matthews and Baker`, `Roberts-Henderson`, `Iqbal-Stevenson`, `Byrne, Smith and Rees`, `Norton-Brooks`, `Ditschlerin`
- PROJECT_NAME (11): `Linde`×4, `Rowan`×3, `Grünfink`×2, `Wacholder`, `Schneeeule`
- ADDRESS (9): `588 Burton Burg`, `Studio 41 Sarah Brooks`, `8 Naomi Isle`, `470 Geoffrey Club`, `9 Marion Club`, `5 Frances Tunnel`, `Joanna-Losekann-Ring 3/2`, `5 Brian Dam`
- PERSON (3): `Sakarya`, `Rahiye Akar`, `Steuer`
- PHONE_NUMBER (1): `0797627308`
- PROJECT_CODE (1): `PRJ-2024-907`

**False positives (prediction overlapping no gold span)**

- PROJECT_NAME (14): `Project`×8, `Projektnr`×3, `project`×3
- LOCATION (14): `am weiher`×2, `stor`, `Wachtelkönig-Kartierung`, `137 Reviere`, `33 Reviere`, `22.11.2025`, `analysis`, `Weißstorch-Kartierung`
- IBAN_CODE (12): `IBAN`×12
- DATE_OF_BIRTH (12): `2026-07-22`, `01 October 2027`, `01 June 2026`, `18 May 2026`, `August 28, 2027`, `15 March 2026`, `May 02, 2026`, `2027-02-05`
- PERSON (12): `Date;From;To;Project;Project no.;Client;Activity;Hours`×5, `Donations`×3, `alignment`, `Species Protection Officer`, `Be`, `Biber`
- ORGANIZATION (7): `the`×7
- EMAIL_ADDRESS (6): `mail`×6
