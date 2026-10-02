"""Resolve the `preset` of a request ("auto" -> "general" or "timesheet")."""

from __future__ import annotations

import re

from .recognizers import CLOCK_PATTERNS, DATE_PATTERNS, DURATION_PATTERNS

_TIME_RX = re.compile("|".join(f"(?:{p})" for p in CLOCK_PATTERNS + DURATION_PATTERNS), re.IGNORECASE)
_DATE_RX = re.compile("|".join(f"(?:{p})" for p in DATE_PATTERNS), re.IGNORECASE)


def looks_like_timesheet(text: str) -> bool:
    """Timesheet if several short lines are dense in dates/times/durations
    (tables, CSV exports, booking lists). Prose mentioning a meeting time is not."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    booking = 0
    for ln in lines:
        times = len(_TIME_RX.findall(ln))
        dates = len(_DATE_RX.findall(ln))
        words = max(1, len(ln.split()))
        if times and (dates or times >= 2) and len(ln) <= 250 and (times + dates) / words >= 0.12:
            booking += 1
    return booking >= 3 or (booking >= 2 and booking / len(lines) >= 0.5)


def resolve_preset(preset: str | None, default: str, text: str) -> str:
    preset = preset or default
    if preset == "auto":
        return "timesheet" if looks_like_timesheet(text) else "general"
    return preset
