"""Format-preserving fakes for timesheet values: dates are shifted by a session
offset, clock times shifted by a session offset, durations jittered."""

from __future__ import annotations

import random
import re
from datetime import date, timedelta

MONTHS = {
    "de": ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September",
           "Oktober", "November", "Dezember"],
    "en": ["January", "February", "March", "April", "May", "June", "July", "August", "September",
           "October", "November", "December"],
}
MONTHS_ABBR = {
    "de": ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"],
    "en": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
}
_MONTH_LOOKUP: dict[str, tuple[int, str, bool]] = {}
for _lang in ("de", "en"):
    for _i, _m in enumerate(MONTHS[_lang]):
        _MONTH_LOOKUP.setdefault(_m.lower(), (_i + 1, _lang, False))
    for _i, _m in enumerate(MONTHS_ABBR[_lang]):
        _MONTH_LOOKUP.setdefault(_m.lower(), (_i + 1, _lang, True))
_MONTH_LOOKUP.update({"jänner": (1, "de", False), "maerz": (3, "de", False), "mrz": (3, "de", True),
                      "sept": (9, "en", True)})
_MONTH_RE = "|".join(sorted(map(re.escape, _MONTH_LOOKUP), key=len, reverse=True))


def _pad(value: int, like: str) -> str:
    return str(value).zfill(2) if len(like) == 2 else str(value)


def _year(y: int, like: str) -> str:
    return str(y)[-2:] if len(like) == 2 else str(y)


def _parse_year(s: str) -> int:
    y = int(s)
    return y + (2000 if y < 70 else 1900) if len(s) == 2 else y


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def shift_date(value: str, days: int, language: str = "de", reference_year: int | None = None) -> str | None:
    """Shift a date string by `days`, keeping its format. Returns None if the
    format is not understood (the caller then falls back to a placeholder)."""
    v = value.strip()
    ref_year = reference_year or date.today().year

    # ISO-like: 2026-10-03, 2026/10/03
    m = re.fullmatch(r"(\d{4})([-/.])(\d{1,2})\2(\d{1,2})", v)
    if m:
        y, sep, mo, d = m.groups()
        nd = _safe_date(int(y), int(mo), int(d))
        if not nd:
            return None
        nd += timedelta(days=days)
        return f"{nd.year}{sep}{_pad(nd.month, mo)}{sep}{_pad(nd.day, d)}"

    # numeric d.m.y, d/m/y, m/d/y (US order only for English slash dates when unambiguous)
    m = re.fullmatch(r"(\d{1,2})([./-])(\d{1,2})\2(\d{2}|\d{4})", v)
    if m:
        a, sep, b, y = m.groups()
        us = sep == "/" and language == "en"
        mo, d = (a, b) if us else (b, a)
        if int(mo) > 12 and int(d) <= 12:
            mo, d, us = d, mo, not us
        nd = _safe_date(_parse_year(y), int(mo), int(d))
        if not nd:
            return None
        nd += timedelta(days=days)
        first, second = (_pad(nd.month, mo), _pad(nd.day, d)) if us else (_pad(nd.day, d), _pad(nd.month, mo))
        return f"{first}{sep}{second}{sep}{_year(nd.year, y)}"

    # 03.10. (no year)
    m = re.fullmatch(r"(\d{1,2})\.(\d{1,2})\.", v)
    if m:
        d, mo = m.groups()
        nd = _safe_date(ref_year, int(mo), int(d))
        if not nd:
            return None
        nd += timedelta(days=days)
        return f"{_pad(nd.day, d)}.{_pad(nd.month, mo)}."

    # 3. Oktober 2026 / 3 October 2026 / 3rd of October / 3. Okt.
    m = re.fullmatch(rf"(\d{{1,2}})(\.|st|nd|rd|th)?(\s*)(of\s+)?({_MONTH_RE})(\.?)(\s+(\d{{4}}))?", v, re.I)
    if m:
        d, suf, sp, of, mon, dot, ysp, y = m.groups()
        mo, lang, abbr = _MONTH_LOOKUP[mon.lower()]
        nd = _safe_date(int(y) if y else ref_year, mo, int(d))
        if not nd:
            return None
        nd += timedelta(days=days)
        suf = _ordinal(nd.day) if suf and suf.lower() in ("st", "nd", "rd", "th") else (suf or "")
        out = f"{_pad(nd.day, d)}{suf}{sp}{of or ''}{_month_name(nd.month, lang, abbr, mon)}{dot}"
        return out + (f" {nd.year}" if y else "")

    # October 3, 2026 / Oct 3rd
    m = re.fullmatch(rf"({_MONTH_RE})(\.?)\s+(\d{{1,2}})(st|nd|rd|th)?(,?\s+(\d{{4}}))?", v, re.I)
    if m:
        mon, dot, d, suf, ypart, y = m.groups()
        mo, lang, abbr = _MONTH_LOOKUP[mon.lower()]
        nd = _safe_date(int(y) if y else ref_year, mo, int(d))
        if not nd:
            return None
        nd += timedelta(days=days)
        out = f"{_month_name(nd.month, lang, abbr, mon)}{dot} {_pad(nd.day, d)}{_ordinal(nd.day) if suf else ''}"
        if y:
            out += ("," if "," in ypart else "") + f" {nd.year}"
        return out

    # Oktober 2026
    m = re.fullmatch(rf"({_MONTH_RE})\s+(\d{{4}})", v, re.I)
    if m:
        mon, y = m.groups()
        mo, lang, abbr = _MONTH_LOOKUP[mon.lower()]
        nd = date(int(y), mo, 15) + timedelta(days=days)
        return f"{_month_name(nd.month, lang, abbr, mon)} {nd.year}"

    # KW 40, KW40/2026, CW 40, week 40
    m = re.fullmatch(r"(KW|CW|Kalenderwoche|calendar\s+week|week)(\s?)(\d{1,2})(?:(\s?[/-]\s?)(\d{2,4}))?", v, re.I)
    if m:
        kw, sp, wk, ysep, y = m.groups()
        year = _parse_year(y) if y else ref_year
        try:
            monday = date.fromisocalendar(year, int(wk), 1)
        except ValueError:
            return None
        nd = monday + timedelta(days=days)
        iy, iw, _ = nd.isocalendar()
        out = f"{kw}{sp}{_pad(iw, wk)}"
        if y:
            out += f"{ysep}{_year(iy, y)}"
        return out
    return None


def _month_name(month: int, lang: str, abbr: bool, like: str) -> str:
    name = (MONTHS_ABBR if abbr else MONTHS)[lang][month - 1]
    if like.isupper():
        return name.upper()
    if like.islower():
        return name.lower()
    return name


def _safe_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def shift_clock(value: str, minutes: int) -> str | None:
    v = value.strip()
    m = re.fullmatch(r"(\d{1,2})(?:([:.])(\d{2}))?(?::(\d{2}))?(\s?)(Uhr|h|am|pm|AM|PM)?", v)
    if not m:
        return None
    h, sep, mi, sec, sp, suffix = m.groups()
    hour, minute = int(h), int(mi or 0)
    ampm = suffix and suffix.lower() in ("am", "pm")
    if ampm:
        hour = hour % 12 + (12 if suffix.lower() == "pm" else 0)
    total = (hour * 60 + minute + minutes) % (24 * 60)
    nh, nm = divmod(total, 60)
    if ampm:
        suffix_new = ("pm" if nh >= 12 else "am")
        suffix_new = suffix_new.upper() if suffix.isupper() else suffix_new
        nh = nh % 12 or 12
    else:
        suffix_new = suffix or ""
    out = _pad(nh, h)
    if sep:
        out += f"{sep}{nm:02d}"
        if sec:
            out += f":{sec}"
    elif nm:  # "8 Uhr" shifted to a non-full hour
        out += f":{nm:02d}"
    return out + (sp or "") + suffix_new


_HOUR_UNITS = r"h|hrs?|hours?|Std\.?|Stunden?|Stdn\.?"
_MIN_UNITS = r"min\.?|Min\.?|Minuten?|minutes?"
_DAY_UNITS = r"PT|Personentage?|MT|Manntage?|person[- ]?days?|days?|Tage?"


def jitter_duration(value: str, rng: random.Random, pct: float, round_minutes: int,
                    language: str = "de") -> str | None:
    v = value.strip()
    # 7:30 h
    m = re.fullmatch(r"(\d{1,2}):([0-5]\d)(\s?)(.*)", v)
    if m:
        h, mi, sp, unit = m.groups()
        total = _jitter(int(h) * 60 + int(mi), rng, pct, round_minutes)
        return f"{_pad(total // 60, h)}:{total % 60:02d}{sp}{unit}"
    # 7h30
    m = re.fullmatch(r"(\d{1,2})h(\s?)([0-5]\d)(\s?min)?", v)
    if m:
        h, sp, mi, tail = m.groups()
        total = _jitter(int(h) * 60 + int(mi), rng, pct, round_minutes)
        return f"{total // 60}h{sp}{total % 60:02d}{tail or ''}"
    m = re.fullmatch(rf"(\d{{1,4}})(?:([.,])(\d{{1,2}}))?(\s?)({_HOUR_UNITS}|{_MIN_UNITS}|{_DAY_UNITS})", v)
    if not m:
        return None
    whole, dec, frac, sp, unit = m.groups()
    num = float(f"{whole}.{frac}" if frac else whole)
    if re.fullmatch(_MIN_UNITS, unit):
        new = _jitter(int(num), rng, pct, round_minutes)
        return f"{new}{sp}{unit}"
    step_minutes = round_minutes if re.fullmatch(_HOUR_UNITS, unit) else 4 * 60  # days: half-day steps (8h days)
    unit_minutes = 60 if re.fullmatch(_HOUR_UNITS, unit) else 8 * 60
    new_min = _jitter(round(num * unit_minutes), rng, pct, step_minutes)
    new = new_min / unit_minutes
    if new == int(new) and not frac:
        text = str(int(new))
    else:
        digits = max(len(frac or ""), len(f"{new:.2f}".rstrip("0").split(".")[1]) if new != int(new) else 1)
        text = f"{new:.{digits}f}"
        if dec == "," or (dec is None and language == "de"):
            text = text.replace(".", ",")
    return f"{text}{sp}{unit}"


def _jitter(minutes: int, rng: random.Random, pct: float, step: int) -> int:
    if minutes <= 0:
        return minutes
    factor = 1 + rng.uniform(-pct, pct) / 100
    step = max(1, step)
    new = max(step, round(minutes * factor / step) * step)
    return new
