import random
import re

import pytest

from app.anonymize.timesheet import jitter_duration, shift_clock, shift_date


@pytest.mark.parametrize("value,lang,expected", [
    ("03.10.2026", "de", "31.10.2026"),
    ("3.10.26", "de", "31.10.26"),
    ("2026-10-03", "de", "2026-10-31"),
    ("03.10.", "de", "31.10."),
    ("3. Oktober 2026", "de", "31. Oktober 2026"),
    ("October 3, 2026", "en", "October 31, 2026"),
    ("Oct 3rd", "en", "Oct 31st"),
    ("10/03/2026", "en", "10/31/2026"),
    ("KW 40", "de", "KW 44"),
    ("Oktober 2026", "de", "November 2026"),
])
def test_shift_date_keeps_format(value, lang, expected):
    assert shift_date(value, 28, lang, reference_year=2026) == expected


def test_shift_date_crosses_month_and_pads():
    assert shift_date("03.10.2026", -112, "de") == "13.06.2026"


@pytest.mark.parametrize("value", ["31.02.2026", "Montag", "gestern"])
def test_unparseable_dates_return_none(value):
    assert shift_date(value, 7, "de") is None


@pytest.mark.parametrize("value,expected", [
    ("08:30", "09:15"), ("17:00 Uhr", "17:45 Uhr"), ("8 Uhr", "8:45 Uhr"),
    ("11:45 pm", "12:30 am"), ("08.30 Uhr", "09.15 Uhr"),
])
def test_shift_clock(value, expected):
    assert shift_clock(value, 45) == expected


def test_shift_clock_preserves_intervals():
    a, b = shift_clock("08:30", -60), shift_clock("17:00", -60)
    assert (a, b) == ("07:30", "16:00")


@pytest.mark.parametrize("value", ["7,5 h", "7.5h", "7:30 h", "2 Std.", "45 min", "1.5 days", "7h30"])
def test_jitter_duration_keeps_unit_and_separator(value):
    out = jitter_duration(value, random.Random(3), 15, 15)
    assert out is not None
    unit = lambda v: re.sub(r"[\d.,:\s]", "", v.replace("Std.", "Std"))  # noqa: E731
    assert unit(out) == unit(value)
    if "," in value:
        assert "." not in out


def test_jitter_duration_german_decimal_comma():
    outs = {jitter_duration("2 Std.", random.Random(i), 30, 15, "de") for i in range(20)}
    assert all("." not in o.replace("Std.", "") for o in outs)
