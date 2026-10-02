# Model candidates (for review), 2026-10-02

Constraints: CPU only (8 threads, AVX2); ~42 GB RAM for the whole stack, of which ~8 GB goes
to the non-LLM parts, so an LLM may use up to ~30 GB, though smaller is better. German and English,
commercial use allowed, and offline use. Facts come from the Hugging Face model cards and configs as
checked on 2026-10-02. Speeds are measured by the benchmark, not taken from the cards.
`bench/candidates.yaml` is the machine-readable version.

## NER / PII span models

| id | model | size | languages | labels | licence | why it's in the list |
|---|---|---|---|---|---|---|
| davlan-xlmr-ner-hrl **(current)** | Davlan/xlm-roberta-large-ner-hrl | 559M, 2.2 GB | 10 incl. DE | PER/ORG/LOC | AFL-3.0 ✓ | baseline |
| gliner-multi-pii-v1 **(current)** | urchade/gliner_multi_pii-v1 | ~0.3B, 1.2 GB | multilingual | zero-shot | Apache ✓ | baseline |
| gliner-multi-v2.1 | urchade/gliner_multi-v2.1 | 1.2 GB | multilingual | zero-shot | Apache ✓ | general multilingual GLiNER |
| gliner2.5-multi | fastino/gliner2.5-multi-v1 (2026-08) | 287M, 1.2 GB | multilingual | zero-shot, 4k context | Apache ✓ | newest GLiNER generation; separate venv (transformers<5) |
| gliner2-multi | fastino/gliner2-multi-v1 | 205M, 1.2 GB | incl. DE | zero-shot | Apache ✓ | separate venv |
| gliner2-privacy-pii-multi | fastino/gliner2-privacy-filter-PII-multi (2026-05) | 205M, 1.2 GB | trained on DE among 7 languages | 42 PII labels | Apache ✓ | best published span F1 on SPY; low precision; separate venv |
| openai-privacy-filter | openai/privacy-filter (2026-04) | 1.4B MoE / 50M active, 2.8 GB | "primarily English" | 8 | Apache ✓ | strong English reference, 128k context |
| openmed-pf-multilingual-v2 | OpenMed/privacy-filter-multilingual-v2 | same, 2.8 GB | 16 incl. DE | 54 | Apache ✓ ⚠️ | German fine-tune of the above. ⚠️ trained partly on AI4Privacy data, which asks commercial users to contact them |
| openmed-pf-nemotron-v2 | OpenMed/privacy-filter-nemotron-v2 | same, 2.8 GB | 20 EU incl. DE | 55 incl. religion/politics/ethnicity/sexuality | Apache ✓ ⚠️ | has Art. 9 labels; same data caveat |
| bardsai-eu-pii | bardsai/eu-pii-anonimization-multilang (2026-03) | 278M, 1.1 GB | 24 EU incl. DE | 35 incl. GDPR Art. 9 | Apache ✓ | EU/GDPR-oriented, small |
| nym-pii-multilingual | Wismut/nym-pii-multilingual v3 (2026-07) | 307M, 1.2 GB | ~23 incl. DE | 40 | MIT ✓ | top score on the REDACT benchmark, 8k context |
| ai4privacy-modernbert | ai4privacy/…-multilingual-categorical-anonymiser-openpii | 150M, 0.6 GB | tagged DE, English-centric base | 20 | MIT ✓ | smallest; reference point |

**Excluded:**
- piiranha-v1 (CC-BY-NC-ND, no commercial use)
- HuggingLil/pii-sensitive-ner-german (piiranha derivative, no licence)
- knowledgator gliner-pii-* and nvidia/gliner-PII (English-only encoders)
- gliner2.5-base/small (English only)
- GLiNER2.5-Decide (a classifier, not NER)
- flair/ner-german-large (no licence declared; strong German baseline if you clear the licence)
- OpenMed German clinical models (medical domain)
- nym-pii-multilingual-small (ONNX only; could be added with an onnxruntime adapter)

## LLMs (GGUF via llama.cpp)

| id | model | params (active) | Q4 file | licence | thinking | notes |
|---|---|---|---|---|---|---|
| qwen3-8b **(current)** | Qwen3-8B | 8B dense | 5.0 GB | Apache ✓ | off via template | baseline |
| qwen3-4b-instruct-2507 | Qwen3-4B-Instruct-2507 | 4B dense | 2.5 GB | Apache ✓ | none | **low memory** |
| qwen3.5-4b | Qwen3.5-4B (2026-02) | 4B dense | 2.7 GB | Apache ✓ | off via template | **low memory**, 201 languages |
| qwen3.5-9b | Qwen3.5-9B | 9B dense | 5.7 GB | Apache ✓ | off | same class as current |
| nuextract3 | numind/NuExtract3 (2026-04, Qwen3.5-4B base) | 4.5B | 2.8 GB | Apache ✓ | off | **low memory**, extraction-specialised |
| gemma4-e4b | Gemma 4 E4B (QAT) | 4.5B effective | 5.2 GB | Apache ✓ | off by default | **low memory**; best at robust JSON extraction in NuMind's benchmark |
| gemma4-12b | Gemma 4 12B (QAT) | 12B dense | 7.0 GB | Apache ✓ | off by default | |
| gemma4-26b-a4b | Gemma 4 26B-A4B (QAT) | 25B (3.8B) MoE | 14.4 GB | Apache ✓ | off by default | MoE: big-model quality at small-model speed |
| gpt-oss-20b | gpt-oss-20b | 21B (3.6B) MoE | 12.1 GB | Apache ✓ | can't be disabled (effort "low") | reference |
| qwen3-30b-a3b-instruct-2507 | Qwen3-30B-A3B-Instruct-2507 | 30B (3.3B) MoE | 17.7 GB | Apache ✓ | none | reference |
| qwen3.6-35b-a3b | Qwen3.6-35B-A3B (2026-04) | 35B (3B) MoE | 17.7 GB (IQ4_XS) | Apache ✓ | off | strongest Qwen that fits |
| qwen3.8-27b | Qwen3.8-27B (2026-08) | 27B dense | 14.3 GB (IQ4_XS) | Apache ✓ | off | newest; dense, so slow on CPU (benchmarked on 8 docs only) |

**Excluded:**
- Qwen3.8-Flash-Next (≥72 GB) and GLM-5.3-Flash (≥93 GB): too big.
- DeepSeek V4.1 Flash: not a llama.cpp format.
- NuExtract-2.0-4B: non-commercial Qwen Research licence.
- LFM2.5-2.6B: commercial use only under $10M revenue, and it always thinks.
- Ornith-1.5: coding fine-tunes of Qwen.
- Qwen3.6-27B: Qwen3.8-27B is newer.
- NuExtract-2.0-8B: superseded by NuExtract3.
