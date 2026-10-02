"""Realistic, format-preserving fake values per entity type."""

from __future__ import annotations

import hashlib
import random
import re
import string

from faker import Faker

from ..config import Settings, Timesheet
from .project_names import ProjectNameGenerator
from .session import SessionState, norm
from .timesheet import jitter_duration, shift_clock, shift_date

LOCALES = {"de": "de_DE", "en": "en_US"}
ID_LIKE = {"EMPLOYEE_ID", "ID_NUMBER", "PROJECT_CODE", "POSTAL_CODE", "CRYPTO", "MAC_ADDRESS",
           "US_SSN", "UK_NHS", "UK_NINO"}
NAME_PARTICLES = {"von", "van", "de", "der", "den", "zu", "vom", "zum", "di", "da", "del", "le", "la", "mc"}


class FakeFactory:
    def __init__(self, settings: Settings, state: SessionState, timesheet: Timesheet,
                 source_text: str = ""):
        self.settings = settings
        self.state = state
        self.ts = timesheet
        self.source_lower = source_text.lower()
        self._fakers: dict[str, Faker] = {}
        self.projects = ProjectNameGenerator(settings.path(settings.dictionaries.fake_project_words),
                                             timesheet.project_names.generated_ratio)

    # -- infrastructure ---------------------------------------------------------------
    def _rng(self, *parts: str) -> random.Random:
        h = hashlib.sha256("\x00".join((self.state.seed,) + parts).encode()).digest()
        return random.Random(int.from_bytes(h[:8], "big"))

    def _faker(self, language: str, rng: random.Random) -> Faker:
        loc = LOCALES.get(language, "de_DE")
        if loc not in self._fakers:
            self._fakers[loc] = Faker(loc)
        f = self._fakers[loc]
        f.seed_instance(rng.getrandbits(64))
        return f

    def _date_offset(self) -> int:
        if self.state.date_offset_days is None:
            lo, hi = self.ts.dates.shift_days
            rng = self._rng("date-offset")
            if self.ts.dates.preserve_weekday:
                weeks = [w for w in range(lo // 7, hi // 7 + 1) if w != 0] or [1]
                self.state.date_offset_days = rng.choice(weeks) * 7
            else:
                self.state.date_offset_days = rng.choice([d for d in range(lo, hi + 1) if d != 0] or [1])
        return self.state.date_offset_days

    def _time_offset(self) -> int:
        if self.state.time_offset_minutes is None:
            lo, hi = self.ts.clock_times.shift_minutes
            step = max(1, self.ts.clock_times.round_minutes)
            choices = [m for m in range(lo - lo % step, hi + 1, step) if m != 0] or [step]
            self.state.time_offset_minutes = self._rng("time-offset").choice(choices)
        return self.state.time_offset_minutes

    def _usable(self, candidate: str, original: str) -> bool:
        """A fake must not equal another original's fake (reversibility) nor
        occur in the source text (ambiguity)."""
        if not candidate or norm(candidate) == norm(original):
            return False
        if self.state.replacement_taken(candidate, original):
            return False
        return candidate.lower() not in self.source_lower

    # -- public -------------------------------------------------------------------------
    def fake(self, entity: str, original: str, language: str) -> str | None:
        """Return a fake for `original` or None when no faker applies."""
        if entity == "PERSON":
            return self._person(original, language)
        gen = self._generator(entity, original, language)
        if gen is None:
            return None
        # shift-based fakes are deterministic by design: no collision retries
        if entity in ("DATE_TIME", "DATE_OF_BIRTH", "CLOCK_TIME", "DURATION"):
            return gen(self._rng(entity, norm(original)))
        for attempt in range(25):
            candidate = gen(self._rng(entity, norm(original), str(attempt)))
            if candidate is None:
                return None
            if self._usable(candidate, original):
                return candidate
        return candidate + f" {attempt}" if candidate else None

    def _generator(self, entity: str, original: str, language: str):
        F = lambda rng: self._faker(language, rng)  # noqa: E731
        if entity in ("DATE_TIME", "DATE_OF_BIRTH"):
            return lambda rng: shift_date(original, self._date_offset(), language)
        if entity == "CLOCK_TIME":
            return lambda rng: shift_clock(original, self._time_offset())
        if entity == "DURATION":
            d = self.ts.durations
            return lambda rng: jitter_duration(original, rng, d.jitter_pct, d.round_minutes, language)
        if entity == "PROJECT_NAME":
            return lambda rng: self.projects.generate(rng, original)
        if entity == "EMAIL_ADDRESS":
            return lambda rng: _email(F(rng), rng)
        if entity == "PHONE_NUMBER":
            return lambda rng: substitute(original, rng, keep_prefix=_phone_prefix(original))
        if entity == "IBAN_CODE":
            return lambda rng: fake_iban(original, rng)
        if entity == "CREDIT_CARD":
            return lambda rng: fake_luhn(original, rng)
        if entity == "IP_ADDRESS":
            return lambda rng: F(rng).ipv6() if ":" in original else F(rng).ipv4_private()
        if entity == "URL":
            return lambda rng: F(rng).url()
        if entity == "LOCATION":
            return lambda rng: F(rng).city()
        if entity == "ADDRESS":
            return lambda rng: _address(F(rng), original)
        if entity == "ORGANIZATION":
            return lambda rng: F(rng).company()
        if entity == "USERNAME":
            return lambda rng: F(rng).user_name()
        if entity in ID_LIKE or entity.startswith(("DE_", "US_", "UK_")):
            return lambda rng: substitute(original, rng)
        return None

    # -- persons --------------------------------------------------------------------------
    def _person(self, original: str, language: str) -> str:
        """Token-wise mapping so partial mentions stay consistent: once
        "Max Müller" -> "Jonas Schmidt", a later "Müller" -> "Schmidt"."""
        words = original.split()
        out = []
        for i, word in enumerate(words):
            last = i == len(words) - 1
            parts = word.split("-")
            out.append("-".join(self._person_token(p, last or len(words) == 1, language) for p in parts))
        return " ".join(out)

    def _person_token(self, token: str, is_last: bool, language: str) -> str:
        if not token or token.lower() in NAME_PARTICLES:
            return token
        m = re.fullmatch(r"([A-ZÄÖÜ])\.?", token)
        if m:  # initial
            return self._rng("initial", token).choice(string.ascii_uppercase) + ("." if "." in token else "")
        key = f"PERSON\x00{token.casefold()}"
        if key in self.state.token_forward:
            return self.state.token_forward[key]
        for attempt in range(25):
            rng = self._rng("person", token.casefold(), str(attempt))
            f = self._faker(language, rng)
            if is_last:
                cand = f.last_name()
            else:
                gender = _gender(token)
                cand = f.first_name_female() if gender == "f" else f.first_name_male() if gender == "m" else f.first_name()
            words = [w for w in re.split(r"[\s-]+", cand) if re.fullmatch(r"[A-ZÄÖÜ][a-zäöüß]{2,}", w)]
            cand = words[-1] if words else cand
            if cand.lower() in _stopwords():
                continue  # fake "Lange"/"Gute" would make deanonymize rewrite ordinary words
            taken = {k.casefold() for k in self.state.token_reverse}
            if cand.casefold() != token.casefold() and cand.casefold() not in taken \
                    and cand.lower() not in self.source_lower:
                break
        if token.isupper() and len(token) > 1:
            cand = cand.upper()
        self.state.token_forward[key] = cand
        self.state.token_reverse[cand] = token  # original case: casefold("ß") == "ss" would never match
        return cand


_GENDER_CACHE: dict[str, str] = {}
_STOP: set[str] = set()


def _stopwords() -> set[str]:
    if not _STOP:
        from spacy.lang.de.stop_words import STOP_WORDS as DE
        from spacy.lang.en.stop_words import STOP_WORDS as EN
        _STOP.update(w.lower() for w in DE | EN)
    return _STOP


def _gender(first_name: str) -> str | None:
    if not _GENDER_CACHE:
        from faker.providers.person.de_DE import Provider as De
        from faker.providers.person.en_US import Provider as En
        for prov in (De, En):
            for n in prov.first_names_female:
                _GENDER_CACHE.setdefault(n.casefold(), "f")
            for n in prov.first_names_male:
                _GENDER_CACHE.setdefault(n.casefold(), "m")
    return _GENDER_CACHE.get(first_name.casefold())


# -- helpers ------------------------------------------------------------------------------
def substitute(value: str, rng: random.Random, keep_prefix: int = 0) -> str:
    """Replace every digit by a random digit and every letter by a random letter
    of the same case; separators and length are kept."""
    out = list(value[:keep_prefix])
    for ch in value[keep_prefix:]:
        if ch.isdigit():
            out.append(str(rng.randint(0, 9)))
        elif ch.isalpha() and ch.isascii():
            letter = rng.choice(string.ascii_uppercase)
            out.append(letter if ch.isupper() else letter.lower())
        else:
            out.append(ch)
    return "".join(out)


def _phone_prefix(value: str) -> int:
    """Keep the country code (+49 / 0049) or the leading trunk 0 plus the next digit."""
    m = re.match(r"\s*(\+\d{1,3}|00\d{1,3})", value)
    if m:
        return m.end()
    m = re.match(r"\s*\(?0\d", value)
    return m.end() if m else 0


def fake_iban(value: str, rng: random.Random) -> str:
    compact = re.sub(r"\s", "", value).upper()
    country, bban = compact[:2], compact[4:]
    new_bban = substitute(bban, rng)
    check = 98 - _iban_mod(new_bban + country + "00")
    new = f"{country}{check:02d}{new_bban}"
    # re-apply original grouping
    out, i = [], 0
    for ch in value:
        if ch.isspace():
            out.append(ch)
        else:
            out.append(new[i] if i < len(new) else "")
            i += 1
    return "".join(out)


def _iban_mod(s: str) -> int:
    digits = "".join(str(int(c, 36)) for c in s)
    return int(digits) % 97


def iban_valid(value: str) -> bool:
    c = re.sub(r"\s", "", value).upper()
    return len(c) > 4 and _iban_mod(c[4:] + c[:4]) == 1


def fake_luhn(value: str, rng: random.Random) -> str:
    digits = [c for c in value if c.isdigit()]
    new = [digits[0]] + [str(rng.randint(0, 9)) for _ in digits[1:-1]]
    total = 0
    for i, d in enumerate(reversed(new)):
        n = int(d)
        if i % 2 == 0:
            n *= 2
            n -= 9 if n > 9 else 0
        total += n
    new.append(str((10 - total % 10) % 10))
    it = iter(new)
    return "".join(next(it) if c.isdigit() else c for c in value)


def _email(f: Faker, rng: random.Random) -> str:
    user = f.user_name()
    return f"{user}@{rng.choice(['example.com', 'example.org', 'example.net'])}"


def _address(f: Faker, original: str) -> str:
    if re.fullmatch(r"\s*(?:D-)?\d{5}\s+\S.*", original):
        return f"{f.postcode()} {f.city()}"
    street = f.street_address()
    if re.search(r"\b\d{5}\s+\S", original):
        return f"{street}, {f.postcode()} {f.city()}"
    return street
