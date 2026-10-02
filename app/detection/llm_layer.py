"""Layer L4: a local LLM (llama.cpp server, OpenAI-compatible API).

Two jobs per chunk, in a single call:
  * discovery  - list PII strings the other layers may have missed
  * adjudication - yes/no verdicts on uncertain spans from the other layers

The LLM never produces spans directly: it returns verbatim strings which are
located in the text. Strings that do not occur in the text are discarded, so
hallucinations cannot become detections.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import httpx

from ..config import Settings
from ..models import Span
from ..text_utils import chunk_text

log = logging.getLogger(__name__)

ENTITY_TYPES = {
    "PERSON": "name of a natural person, also partial (surname only, first name only, nickname, initials with surname)",
    "ORGANIZATION": "a specific company, client, authority, school, club (not generic words like 'team' or 'customer')",
    "LOCATION": "city, village, region, specific site or building that can locate a person",
    "ADDRESS": "street address, postal address",
    "EMAIL_ADDRESS": "email address",
    "PHONE_NUMBER": "phone or fax number",
    "IBAN_CODE": "bank account number / IBAN",
    "ID_NUMBER": "personal identifier: tax id, social security, passport, ID card, insurance, customer or contract number",
    "EMPLOYEE_ID": "employee / personnel number",
    "USERNAME": "user name, login, account handle",
    "DATE_OF_BIRTH": "date of birth or age of a person",
    "PROJECT_NAME": "name of an internal project, codename, product under development or client engagement",
    "PROJECT_CODE": "project number, cost center, order number, WBS element",
    "HEALTH_DATA": "illness, diagnosis, symptom, disability, pregnancy, medication, therapy, sick leave of a person",
    "RELIGION": "religion, belief or church membership of a person",
    "POLITICAL_OPINION": "political opinion or party membership of a person",
    "TRADE_UNION": "trade union membership of a person",
    "ETHNIC_ORIGIN": "racial or ethnic origin of a person",
    "SEXUAL_ORIENTATION": "sexual orientation or sex life of a person",
    "GENETIC_BIOMETRIC": "genetic or biometric data",
    "OTHER_PII": "any other information that identifies a specific person",
}

SYSTEM_PROMPT = """You are a meticulous data-protection auditor (GDPR). You find personal data in German or English business text such as e-mails, notes and timesheets.

Rules:
- Copy every finding EXACTLY as it appears in the text (same spelling, case, spacing). Never paraphrase, translate or invent.
- Report the shortest string that contains the personal data (e.g. "Diabetes", not the whole sentence; "Müller", not "Herr Müller").
- Special-category data (health, religion, politics, trade union, ethnicity, sexual orientation, genetics/biometrics) counts ONLY when it relates to a person, but then always report it.
- Do not report: generic words, job titles, weekdays, months, plain dates/times/durations, public figures mentioned only as such, software products (Excel, Jira, Teams).
- If unsure whether something is personal data, report it (recall matters more than precision).

Entity types:
{types}
"""

USER_PROMPT = """TEXT:
<<<
{text}
>>>

{known_block}{candidates_block}Return JSON with:
- "findings": personal data in TEXT that is NOT already detected, as {{"text": <verbatim string>, "type": <entity type>}}. List each distinct string once. Empty list if nothing was missed.
- "pii": ids of the candidates that ARE personal data here
- "not_pii": ids of the candidates that are NOT personal data here"""

KNOWN_PROMPT = """ALREADY DETECTED (do not repeat these, look for what is missing):
{items}

"""

CANDIDATES_PROMPT = """CANDIDATES other detectors are unsure about (id: text [suggested type]). Put every id in "pii" or "not_pii":
{items}

"""


def _schema() -> dict:
    types = list(ENTITY_TYPES)
    ids = {"type": "array", "maxItems": 120, "items": {"type": "integer"}}
    return {
        "type": "object",
        "properties": {
            "findings": {"type": "array", "maxItems": 60, "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}, "type": {"type": "string", "enum": types}},
                "required": ["text", "type"],
            }},
            "pii": ids,
            "not_pii": ids,
        },
        "required": ["findings", "pii", "not_pii"],
    }


@dataclass
class Verdict:
    span: Span
    is_pii: bool
    entity: str | None


class LlmLayer:
    name = "llm"

    def __init__(self, settings: Settings, client: httpx.Client | None = None, **overrides):
        self.cfg = settings.layers.llm.model_copy(update=overrides)
        self.client = client or httpx.Client(base_url=self.cfg.base_url, timeout=self.cfg.timeout_seconds)
        types = "\n".join(f"- {k}: {v}" for k, v in ENTITY_TYPES.items())
        self.system = SYSTEM_PROMPT.format(types=types)

    def health(self) -> bool:
        try:
            return self.client.get("/health", timeout=5).status_code == 200
        except httpx.HTTPError:
            return False

    def detect(self, text: str, candidates: list[Span] | None = None, known: list[Span] | None = None,
               warnings: list[str] | None = None) -> tuple[list[Span], list[Verdict]]:
        """`candidates`: uncertain spans to adjudicate. `known`: confident spans
        the LLM need not repeat (generation is the slow part on CPU)."""
        candidates = candidates or []
        known = known or []
        warnings = warnings if warnings is not None else []
        found: list[Span] = []
        verdicts: list[Verdict] = []
        for chunk in chunk_text(text, self.cfg.chunk_chars, self.cfg.overlap_chars):
            if not chunk.text.strip():
                continue
            local = [c for c in candidates if c.start >= chunk.start and c.end <= chunk.end]
            local_known = sorted({k.text for k in known if k.start >= chunk.start and k.end <= chunk.end})
            try:
                data = self._ask(chunk.text, local, local_known, chunk.start)
            except RuntimeError as err:
                if self.cfg.on_error == "fail":
                    raise
                warnings.append(f"LLM layer skipped chars {chunk.start}-{chunk.end}: {err}")
                continue
            for f in data.get("findings", []):
                for s, e in _locate(chunk.text, str(f.get("text", ""))):
                    a, b = chunk.start + s, chunk.start + e
                    found.append(Span(a, b, f.get("type") or "OTHER_PII", self.cfg.score_found,
                                      {self.name}, text[a:b]))
            for key, is_pii in (("pii", True), ("not_pii", False)):
                for idx in data.get(key) or []:
                    if isinstance(idx, int) and 0 <= idx < len(local):
                        verdicts.append(Verdict(local[idx], is_pii, None))
            for v in data.get("verdicts") or []:  # older response shape
                idx = v.get("id") if isinstance(v, dict) else None
                if isinstance(idx, int) and 0 <= idx < len(local):
                    verdicts.append(Verdict(local[idx], bool(v.get("is_pii")), v.get("type")))
        return found, verdicts

    def _ask(self, text: str, candidates: list[Span], known: list[str], offset: int) -> dict:
        items = "\n".join(f'{i}: "{c.text}" [{c.entity}]' for i, c in enumerate(candidates))
        user = USER_PROMPT.format(
            text=text,
            known_block=KNOWN_PROMPT.format(items="; ".join(f'"{k}"' for k in known)) if known else "",
            candidates_block=CANDIDATES_PROMPT.format(items=items) if candidates else "")
        body = {
            "model": self.cfg.model,
            "messages": [{"role": "system", "content": self.system}, {"role": "user", "content": user}],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "pii", "schema": _schema(), "strict": True}},
        }
        if self.cfg.disable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        body.update(self.cfg.extra_body or {})
        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            if attempt:  # greedy decoding would repeat a degenerate answer verbatim
                body["temperature"] = max(self.cfg.temperature, 0.3 * attempt)
            try:
                r = self.client.post("/v1/chat/completions", json=body)
                r.raise_for_status()
                msg = r.json()["choices"][0]["message"]
                content = msg.get("content") or ""
                if "{" not in content and msg.get("reasoning_content"):
                    # template put the answer into the reasoning channel (thinking models)
                    content = msg["reasoning_content"]
                return _parse_json(content)
            except (httpx.HTTPError, KeyError, ValueError) as err:
                last_err = err
                log.warning("LLM call failed (offset %d, attempt %d): %s", offset, attempt + 1, err)
        raise RuntimeError(f"LLM layer failed: {last_err}")


def _parse_json(content: str) -> dict:
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except ValueError:
            pass
    # truncated output (max_tokens hit, runaway list): keep every complete entry
    findings = [{"text": t, "type": y} for t, y in re.findall(
        r'\{\s*"text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"type"\s*:\s*"(\w+)"\s*\}', content)]
    out: dict = {"findings": findings}
    for key in ("pii", "not_pii"):
        m = re.search(rf'"{key}"\s*:\s*\[([\d,\s]*)', content)
        out[key] = [int(x) for x in re.findall(r"\d+", m.group(1))] if m else []
    if not findings and not out["pii"] and not out["not_pii"]:
        raise ValueError("no usable JSON in LLM response")
    log.warning("LLM response was truncated; salvaged %d findings, %d verdicts",
                len(findings), len(out["pii"]) + len(out["not_pii"]))
    return out


def _locate(text: str, needle: str) -> list[tuple[int, int]]:
    """All occurrences of `needle` in `text`, exact first, then case- and
    whitespace-insensitive. Very short needles must match whole words."""
    needle = needle.strip()
    if len(needle) < 2:
        return []
    pattern = r"\s+".join(re.escape(p) for p in needle.split())
    if len(needle) <= 3 or needle[0].isalnum():
        pattern = r"(?<!\w)" + pattern
    if len(needle) <= 3 or needle[-1].isalnum():
        pattern = pattern + r"(?!\w)"
    hits = [m.span() for m in re.finditer(pattern, text)]
    if not hits:
        hits = [m.span() for m in re.finditer(pattern, text, flags=re.IGNORECASE)]
    return hits
