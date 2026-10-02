import pytest

from app.anonymize.engine import Anonymizer, deanonymize, mask
from app.anonymize.session import SessionState
from app.models import Span


def spans_for(text, items):
    out = []
    for sub, entity in items:
        i = text.index(sub)
        out.append(Span(i, i + len(sub), entity, 0.9, {"patterns"}, sub))
    return out


TEXT = "Max Müller (max@firma.de) war am 03.10.2026 von 08:30 bis 17:00 krank wegen Diabetes. Müller kommt Montag."
ITEMS = [("Max Müller", "PERSON"), ("max@firma.de", "EMAIL_ADDRESS"), ("03.10.2026", "DATE_TIME"),
         ("08:30", "CLOCK_TIME"), ("17:00", "CLOCK_TIME"), ("Diabetes", "HEALTH_DATA")]


def run(settings, mode, text=TEXT, items=ITEMS, state=None, **kw):
    state = state or SessionState()
    out, reps = Anonymizer(settings).anonymize(text, spans_for(text, items), state, mode, **kw)
    return out, reps, state


def test_placeholder_numbering_and_consistency(settings):
    text = "Anna Berg trifft Tom Kurz. Anna Berg geht."
    out, _, _ = run(settings, "placeholder", text, [("Anna Berg", "PERSON"), ("Tom Kurz", "PERSON")])
    # only the first occurrence was passed as span; same original -> same placeholder within a session
    assert out.startswith("<PERSON_1> trifft <PERSON_2>.")


@pytest.mark.parametrize("mode", ["placeholder", "fake", "hash"])
def test_reversible_modes_roundtrip(settings, mode):
    out, reps, state = run(settings, mode)
    assert "Müller (" not in out and "Diabetes" not in out
    back, _ = deanonymize(out, state)
    assert back == TEXT


def test_redact_and_mask(settings):
    out, _, _ = run(settings, "redact")
    assert out.count("[REDACTED]") == len(ITEMS)
    assert mask("DE89 3704 0044 0532 0130 00", "*", 4) == "**** **** **** **** **30 00"
    out, _, _ = run(settings, "mask")
    assert "Max" not in out and "******" in out


def test_fake_mode_special_category_never_faked(settings):
    out, reps, _ = run(settings, "fake")
    health = [r for r in reps if r.span.entity == "HEALTH_DATA"][0]
    assert health.operator == "placeholder" and health.replacement == "<HEALTH_DATA_1>"


def test_timesheet_fake_switches(settings):
    on = settings.timesheet.merged({"clock_times": {"fake": True}, "dates": {"fake": True}})
    out, reps, state = run(settings, "fake", timesheet=on)
    clock = [r.replacement for r in reps if r.span.entity == "CLOCK_TIME"]
    assert all(":" in c for c in clock) and clock != ["08:30", "17:00"]
    date = [r for r in reps if r.span.entity == "DATE_TIME"][0]
    assert date.operator == "fake" and state.date_offset_days % 7 == 0  # preserve_weekday
    off = settings.timesheet.merged({"clock_times": {"fake": False}, "dates": {"fake": False}})
    out, reps, _ = run(settings, "fake", timesheet=off)
    assert "<CLOCK_TIME_1>" in out and "<DATE_TIME_1>" in out


def test_session_reuse_is_consistent(settings):
    out1, _, state = run(settings, "fake")
    text2 = "Gestern schrieb Max Müller."
    out2, _, _ = run(settings, "fake", text2, [("Max Müller", "PERSON")], state=state)
    fake_name = out1.split(" (")[0]
    assert out2 == f"Gestern schrieb {fake_name}."


def test_deanonymize_partial_person_mentions(settings):
    out, reps, state = run(settings, "fake")
    fake_name = [r.replacement for r in reps if r.span.entity == "PERSON"][0]
    answer = f"Laut Bericht war {fake_name.split()[-1]} krank."
    back, n = deanonymize(answer, state)
    assert back == "Laut Bericht war Müller krank." and n == 1


def test_operator_overrides(settings):
    out, reps, _ = run(settings, "fake", operators={"EMAIL_ADDRESS": "redact", "PERSON": "keep"})
    assert out.startswith("Max Müller ([REDACTED])")


def test_session_state_serialisation():
    s = SessionState()
    s.remember("fake:PERSON", "Max", "Tom")
    s2 = SessionState.from_json(s.to_json())
    assert s2.lookup("fake:PERSON", "max") == "Tom" and s2.reverse == {"Tom": "Max"}
