from app.detection.merger import Merger
from app.models import Span


def S(text, sub, entity, score, source, occurrence=0):
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(sub, start + 1)
    return Span(start, start + len(sub), entity, score, {source}, sub)


def test_precise_span_beats_sloppy_overlap(settings):
    text = "John Smith in London called on October 3, 2026."
    spans = [
        S(text, "London called on October", "LOCATION", 0.85, "spacy"),
        S(text, "London", "LOCATION", 0.99, "transformer"),
        S(text, "London", "LOCATION", 0.8, "gliner"),
        S(text, "October 3, 2026", "DATE_TIME", 0.8, "patterns"),
    ]
    out = Merger(settings).merge(spans, text)
    assert [(s.text, s.entity) for s in out] == [("London", "LOCATION"), ("October 3, 2026", "DATE_TIME")]


def test_single_weak_source_is_dropped_but_agreement_keeps_it(settings):
    text = "Treffen mit Acme heute."
    weak = S(text, "Acme", "ORGANIZATION", 0.38, "spacy")
    assert Merger(settings).merge([weak], text) == []
    agree = [weak, S(text, "Acme", "ORGANIZATION", 0.5, "gliner")]
    assert [s.text for s in Merger(settings).merge(agree, text)] == ["Acme"]


def test_address_absorbs_contained_locations(settings):
    text = "wohne in der Hauptstraße 5, 80331 München."
    spans = [
        S(text, "Hauptstraße 5, 80331 München", "ADDRESS", 0.7, "patterns"),
        S(text, "Hauptstraße", "LOCATION", 1.0, "transformer"),
        S(text, "München", "LOCATION", 1.0, "transformer"),
        S(text, "München", "LOCATION", 0.85, "spacy"),
    ]
    out = Merger(settings).merge(spans, text)
    assert [(s.text, s.entity) for s in out] == [("Hauptstraße 5, 80331 München", "ADDRESS")]


def test_pattern_type_wins_on_identical_boundaries(settings):
    text = "Code PRJ-2024-017 heute"
    spans = [S(text, "PRJ-2024-017", "PROJECT_NAME", 0.99, "gliner"),
             S(text, "PRJ-2024-017", "PROJECT_CODE", 0.7, "patterns")]
    out = Merger(settings).merge(spans, text)
    assert [(s.entity, s.sources) for s in out] == [("PROJECT_CODE", {"gliner", "patterns"})]


def test_remainder_of_rejected_span_is_kept(settings):
    text = "Kontakt: Dr. Hans Peter Meier-Schulze, Tel 0301234567"
    spans = [S(text, "Hans Peter Meier-Schulze, Tel 0301234567", "PERSON", 0.9, "gliner"),
             S(text, "0301234567", "PHONE_NUMBER", 1.0, "patterns")]
    out = Merger(settings).merge(spans, text)
    texts = {s.text for s in out}
    assert "0301234567" in texts and any("Meier-Schulze" in t for t in texts)
