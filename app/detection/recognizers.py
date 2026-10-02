"""Custom Presidio recognizers: regex with capture groups, dictionaries with
wildcards/context, and timesheet patterns (dates, clock times, durations)."""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from presidio_analyzer import EntityRecognizer, RecognizerResult


class RegexGroupRecognizer(EntityRecognizer):
    """Regex recognizer whose reported span is a capture group, so keywords
    such as "Personalnummer:" are used for matching but not redacted."""

    def __init__(self, name: str, entity: str, pattern: str, group: int = 0,
                 score: float = 0.8, language: str = "en", flags: int = 0):
        self.regex = re.compile(pattern, flags)
        self.group = group
        self.score = score
        super().__init__(supported_entities=[entity], name=name, supported_language=language)

    def load(self) -> None:  # noqa: D401 - nothing to load
        pass

    def analyze(self, text, entities, nlp_artifacts=None):
        out = []
        for m in self.regex.finditer(text):
            s, e = m.span(self.group)
            if s < 0 or s == e:
                continue
            out.append(self._result(s, e, self.score))
        return out

    def _result(self, start: int, end: int, score: float) -> RecognizerResult:
        return RecognizerResult(
            entity_type=self.supported_entities[0], start=start, end=end, score=score,
            recognition_metadata={
                RecognizerResult.RECOGNIZER_NAME_KEY: self.name,
                RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: self.id,
            },
        )


def _term_regex(terms: list[str]) -> re.Pattern | None:
    """Case-insensitive whole-word alternation. A trailing '*' allows any word
    continuation (prefix match)."""
    parts = []
    for t in sorted({t.strip() for t in terms if t and t.strip()}, key=len, reverse=True):
        prefix = t.endswith("*")
        core = re.escape(t.rstrip("*")).replace(r"\ ", r"\s+")
        parts.append(core + (r"\w*" if prefix else ""))
    if not parts:
        return None
    return re.compile(r"(?<![\w])(?:" + "|".join(parts) + r")(?![\w])", re.IGNORECASE)


class DictionaryRecognizer(RegexGroupRecognizer):
    """Term list matcher. Weak terms only count when a context word is nearby
    (within `window` characters)."""

    def __init__(self, name: str, entity: str, terms: list[str], language: str,
                 score: float = 0.6, weak_terms: list[str] | None = None,
                 context: list[str] | None = None, weak_score: float = 0.3,
                 context_boost: float = 0.3, window: int = 60):
        self.term_re = _term_regex(terms)
        self.weak_re = _term_regex(weak_terms or [])
        self.context_re = _term_regex([c if c.endswith("*") else c + "*" for c in (context or [])])
        self.weak_score = weak_score
        self.context_boost = context_boost
        self.window = window
        super().__init__(name, entity, r"(?!x)x", 0, score, language)

    def analyze(self, text, entities, nlp_artifacts=None):
        out = []
        seen: list[tuple[int, int]] = []
        for regex, base in ((self.term_re, self.score), (self.weak_re, self.weak_score)):
            if regex is None:
                continue
            for m in regex.finditer(text):
                s, e = m.span()
                if any(s < pe and ps < e for ps, pe in seen):
                    continue
                score = base
                if self.context_re is not None:
                    lo, hi = max(0, s - self.window), min(len(text), e + self.window)
                    if self.context_re.search(text[lo:s]) or self.context_re.search(text[e:hi]):
                        score = min(1.0, score + self.context_boost)
                seen.append((s, e))
                out.append(self._result(s, e, score))
        return out


def read_list(path: Path | None) -> list[str]:
    if not path or not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def special_category_recognizers(path: Path | None, languages: list[str]) -> list[EntityRecognizer]:
    if not path or not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: list[EntityRecognizer] = []
    for cat in data.get("categories", []):
        for lang in languages:
            # terms of all languages are matched (mixed-language text), context per language first
            terms = [t for ts in (cat.get("terms") or {}).values() for t in ts]
            weak = [t for ts in (cat.get("weak_terms") or {}).values() for t in ts]
            ctx = [t for ts in (cat.get("context") or {}).values() for t in ts]
            out.append(DictionaryRecognizer(
                name=f"special_{cat['entity'].lower()}", entity=cat["entity"], terms=terms,
                weak_terms=weak, context=ctx, score=float(cat.get("score", 0.6)), language=lang,
            ))
    return out


# --- Timesheet / date-time patterns -------------------------------------------------

MONTHS_DE = ("Januar|Jänner|Februar|März|Maerz|April|Mai|Juni|Juli|August|September|"
             "Oktober|November|Dezember|Jan|Feb|Mär|Mrz|Apr|Jun|Jul|Aug|Sep|Sept|Okt|Nov|Dez")
MONTHS_EN = ("January|February|March|April|May|June|July|August|September|October|November|"
             "December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec")
MONTHS = f"(?:{MONTHS_DE}|{MONTHS_EN})"

DATE_PATTERNS = [
    # 03.10.2026, 3.10.26, 03/10/2026, 10/03/2026, 03-10-2026
    r"\b\d{1,2}[./-]\d{1,2}[./-](?:\d{4}|\d{2})\b",
    # 2026-10-03, 2026/10/03
    r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b",
    # 03.10. (no year) — only if followed by non-digit
    r"\b\d{1,2}\.\d{1,2}\.(?!\d)",
    # 3. Oktober 2026, 3 October 2026, 3rd of October
    rf"\b\d{{1,2}}(?:\.|st|nd|rd|th)?\s*(?:of\s+)?{MONTHS}\.?(?:\s+\d{{4}})?\b",
    # October 3, 2026 / Oct 3rd
    rf"\b{MONTHS}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b",
    # Oktober 2026
    rf"\b{MONTHS}\s+\d{{4}}\b",
    # KW 40, KW40/2026, CW 40
    r"\b(?:KW|CW|Kalenderwoche|calendar\s+week|week)\s?\d{1,2}(?:\s?[/-]\s?\d{2,4})?\b",
]

CLOCK_PATTERNS = [
    # 08:30, 8:30, 17:00h, 9:30 pm, 08.30 Uhr (dot only with "Uhr" to avoid dates)
    r"(?<![\d:.,])(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?(?:\s?(?:Uhr|am|pm|AM|PM)\b)?(?!\s?(?:h|Std|Stunden|hours|hrs|min)\b)(?![\d:])",
    r"\b(?:[01]?\d|2[0-3])\.[0-5]\d\s?Uhr\b",
    # 8 Uhr, 9am, 5 pm
    r"(?<![\d:.,])(?:[01]?\d|2[0-3])\s?(?:Uhr|am|pm|AM|PM)\b",
]

DURATION_PATTERNS = [
    # 7,5 h, 7.5h, 7:30 h, 2 Std., 1,5 Stunden, 45 min, 3 hours, 2 PT, 1.5 days, 7h30
    r"(?<![\d:.,])\d{1,4}(?:[.,]\d{1,2})?\s?(?:h|hrs?|hours?|Std\.?|Stunden?|Stdn\.?|min\.?|Min\.?|Minuten?|minutes?|PT|Personentage?|MT|Manntage?|person[- ]?days?|days?|Tage?)(?![\w])",
    r"(?<![\d:.,])\d{1,2}:[0-5]\d\s?(?:h|Std\.?|Stunden|hours)(?![\w])",
    r"\b\d{1,2}h\s?[0-5]\d(?:\s?min)?\b",
]


_CAP = r"[A-ZÄÖÜ][\wäöüßÄÖÜ.\-]*"
_STREET_SUFFIX = (r"(?:straße|strasse|str\.|weg|gasse|platz|allee|ring|damm|ufer|steig|pfad|"
                  r"chaussee|markt|hof|garten|berg|feld)")
ADDRESS_PATTERNS = [
    # Hauptstraße 5, Berliner Straße 12a, Am Weinberg 3 — optional ", 80331 München"
    rf"(?:{_CAP}\s){{0,2}}(?:{_CAP}{_STREET_SUFFIX}|(?:Straße|Strasse|Weg|Platz|Allee|Gasse|Ring|Damm)\b)"
    r"\s+\d{1,4}(?:\s?[a-zA-Z]\b)?(?:\s?[-/]\s?\d{1,4}[a-zA-Z]?)?"
    r"(?:,?[ \t]+(?:D-)?\d{5}[ \t]+[A-ZÄÖÜ][\wäöüß\-]+(?:\s(?:an\sder|am|im|bei|in)\s[A-ZÄÖÜ][\wäöüß]+)?)?",
    # 80331 München (postal code + city on its own)
    r"\b(?:D-)?\d{5}[ \t]+[A-ZÄÖÜ][a-zäöüß]+(?:[\- ][A-ZÄÖÜ][a-zäöüß]+)?(?=[\s,.;)]|$)",
    # 221B Baker Street, 10 Downing St
    r"\b\d{1,5}[A-Za-z]?\s+(?:[A-Z][a-z]+\s){1,3}(?:Street|St\.?|Road|Rd\.?|Avenue|Ave\.?|Lane|Ln\.?|"
    r"Drive|Dr\.?|Boulevard|Blvd\.?|Way|Court|Ct\.?|Place|Pl\.?|Square|Sq\.?|Terrace|Close|Crescent)\b",
]


def address_recognizers(language: str) -> list[EntityRecognizer]:
    return [RegexGroupRecognizer(f"address_{i}", "ADDRESS", p, 0, 0.7, language)
            for i, p in enumerate(ADDRESS_PATTERNS)]


def timesheet_recognizers(language: str) -> list[EntityRecognizer]:
    flags = re.IGNORECASE
    out: list[EntityRecognizer] = []
    for i, p in enumerate(DATE_PATTERNS):
        out.append(RegexGroupRecognizer(f"date_{i}", "DATE_TIME", p, 0, 0.8, language, flags))
    for i, p in enumerate(CLOCK_PATTERNS):
        out.append(RegexGroupRecognizer(f"clock_{i}", "CLOCK_TIME", p, 0, 0.75, language))
    for i, p in enumerate(DURATION_PATTERNS):
        out.append(RegexGroupRecognizer(f"duration_{i}", "DURATION", p, 0, 0.75, language))
    # date of birth with keyword: report the date only
    out.append(RegexGroupRecognizer(
        "date_of_birth", "DATE_OF_BIRTH",
        r"(?i)\b(?:geb\.?|geboren(?:\s+am)?|geburtsdatum|date\s+of\s+birth|d\.?o\.?b\.?|born(?:\s+on)?)\s*[:]?\s*"
        rf"(\d{{1,2}}[./-]\d{{1,2}}[./-]\d{{2,4}}|\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}\.?\s*{MONTHS}\.?\s+\d{{4}}|{MONTHS}\s+\d{{1,2}},?\s+\d{{4}})",
        1, 0.95, language,
    ))
    return out


# --- person names: first-name gazetteer + honorifics ----------------------------------

# First names that are ordinary words far more often than names.
_AMBIGUOUS_FIRST = {
    "will", "mark", "rose", "may", "june", "april", "august", "grant", "page", "bill", "art", "dean", "hope",
    "faith", "joy", "summer", "autumn", "sunny", "rich", "dawn", "chance", "case", "chase", "cash", "price",
    "major", "king", "earl", "duke", "baron", "prince", "royal", "jack", "pat", "sue", "ray", "guy", "hunter",
    "mai", "juni", "juli", "maria", "frank", "ernst", "fritz", "hans", "kurt", "otto", "heinz", "karl", "paul",
    "ella", "ida", "lisa", "max", "ben", "tim", "tom", "jan", "kai", "leo", "emil", "jona", "noah",
}  # (common German names are listed so they do not *alone* trigger; they still match with a surname)
_SINGLE_OK = _AMBIGUOUS_FIRST - {"will", "mark", "rose", "may", "june", "april", "august", "grant", "page",
                                 "bill", "art", "dean", "hope", "faith", "joy", "summer", "autumn", "sunny",
                                 "rich", "dawn", "chance", "case", "chase", "cash", "price", "major", "king",
                                 "earl", "duke", "baron", "prince", "royal", "hunter", "mai", "juni", "juli"}
_LOCALES = ("de_DE", "de_AT", "de_CH", "en_US", "en_GB", "tr_TR", "pl_PL", "it_IT", "fr_FR", "es_ES", "hr_HR")


def first_names() -> set[str]:
    import importlib

    names: set[str] = set()
    for loc in _LOCALES:
        try:
            prov = importlib.import_module(f"faker.providers.person.{loc}").Provider
        except ImportError:
            continue
        for attr in ("first_names", "first_names_male", "first_names_female"):
            vals = getattr(prov, attr, ())
            names.update(vals.keys() if isinstance(vals, dict) else vals)
    words = {w for w in names if re.fullmatch(r"[A-ZÄÖÜŞÇĞİ][a-zäöüßşçğıéèáàíóúñ]{2,}", w)}
    return {w for w in words if w.lower() not in _AMBIGUOUS_FIRST - _SINGLE_OK}


# Role nouns after "Herr"/"Frau" ("Frau Ministerin", "Herr Bürgermeister") are not names.
_ROLE = re.compile(
    r"(?:Minister|Bürgermeister|Oberbürgermeister|Landrat|Kolleg|Doktor|Professor|Präsident|Vorsitzend|"
    r"Direktor|Pfarrer|Kanzler|Abgeordnet|Richter|Anwalt|Anwält|Kund|Nachbar|Geschäftsführer|Leiter|"
    r"Referent|Sachbearbeiter|Vorstand|Kassier|Schriftführer|Beauftragt|Gemeinderat|Stadtrat)"
    r"(?:in|innen|e|en|er|s)?")


class PersonNameRecognizer(RegexGroupRecognizer):
    """"Anna Huber", "Mehmet Yılmaz", "Herr Fink", "Dr. Weber". Complements spaCy,
    which often mislabels names next to "&", in lists or at line starts."""

    SURNAME = r"(?:(?:von|van|de|der|den|zu|da|di|del)\s+){0,2}[A-ZÄÖÜŞÇ][a-zäöüßşçğıéèáàíóú]+(?:-[A-ZÄÖÜ][a-zäöüß]+)?"

    def __init__(self, language: str, names: set[str] | None = None, stop: set[str] | None = None):
        names = names if names is not None else first_names()
        self.stop = stop or set()
        alt = "|".join(sorted(map(re.escape, names), key=len, reverse=True))
        self.name_re = re.compile(rf"(?<![\w-])((?:{alt})(?:[ -](?:{alt}))?\s+{self.SURNAME})(?![\w])")
        title = r"(?:Herrn?|Frau|Hr\.|Fr\.|Dr\.(?:\s?(?:med|rer\.\s?nat|phil|jur)\.)?|Prof\.|Dipl\.-\w+\.?|Mr\.?|Mrs\.?|Ms\.?|Miss|Mx\.?)"
        self.title_re = re.compile(
            rf"(?<![\w]){title}(?:\s+{title})*\s+"
            rf"((?!(?:Dr|Prof|Dipl|Herr|Frau)\b)(?:[A-ZÄÖÜ][a-zäöüß]+\s+)?{self.SURNAME})(?![\w])")
        super().__init__("person_names", "PERSON", r"(?!x)x", 0, 0.6, language)

    def analyze(self, text, entities, nlp_artifacts=None):
        out = []
        for rx, score in ((self.title_re, 0.75), (self.name_re, 0.6)):
            for m in rx.finditer(text):
                last = m.group(1).split()[-1].split("-")[-1]
                if last.lower() in self.stop or _ROLE.fullmatch(m.group(1).split()[-1]):
                    continue  # "Anna Und", "Frau Ministerin"
                out.append(self._result(m.start(1), m.end(1), score))
        return out
