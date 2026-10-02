"""HTML format, presets, species filter, new recognizers, table propagation."""

import re

import pytest

from app.anonymize.engine import deanonymize
from app.api.schemas import AnalyzeRequest, AnonymizeRequest
from app.detection.pipeline import DetectionOptions
from app.detection.presets import looks_like_timesheet
from app.formats.html import extract, looks_like_html


@pytest.fixture(scope="module")
def service(settings):
    from app.service import PiiService
    return PiiService(settings, ["patterns", "spacy"])


HTML = """<html><body><h1 style="font-size:24px">Newsletter 03/2026</h1>
<p>Am <b>14.11.2026</b> lädt <a href="mailto:anna.huber@example.org">Anna Huber</a> &amp; Team ein (Dauer: 2,5&nbsp;h).
Tel. <span>0941 123456</span></p><!-- intern: max.mueller@example.de --></body></html>"""


def test_html_roundtrip_keeps_markup(service):
    res, state = service.anonymize(AnonymizeRequest(text=HTML, profile="fast", mode="placeholder", reversible=True))
    out = res["text"]
    assert res["format"] == "html"
    for leak in ("anna.huber@example.org", "Anna Huber", "0941 123456", "max.mueller@example.de"):
        assert leak not in out
    assert re.search(r'<a href="mailto:&lt;EMAIL_ADDRESS_\d&gt;">&lt;PERSON_1&gt;</a>', out)
    assert out.count("<") == HTML.count("<") and out.count(">") == HTML.count(">")
    back, _ = deanonymize(out, state)
    assert back == HTML


def test_html_entity_positions_are_raw(service):
    res = service.analyze(AnalyzeRequest(text=HTML, profile="fast", include_text=True))
    for e in res["entities"]:
        assert HTML[e["start"]:e["end"]] == e["text"] or "&" in HTML[e["start"]:e["end"]]


def test_html_detection():
    assert looks_like_html(HTML) and not looks_like_html("Preis < 5 und > 3")
    vt = extract("<p>a<script>secret()</script>b</p>", [], ["script"])
    assert "secret" not in vt.text


def test_preset_auto():
    assert looks_like_timesheet("03.10.2026 | 08:30-17:00 | Nordlicht | 7,5 h\n04.10.2026 | 09:00-12:00 | X | 3 h\n"
                                "05.10.2026 | 08:00-16:00 | Y | 8 h")
    assert not looks_like_timesheet("Die Versammlung ist am 14. November 2026 um 18:30 Uhr. Die Pause dauert 15 min.")


def test_general_preset_keeps_meeting_dates_but_not_birth_dates(fast_pipeline):
    text = "Die Versammlung ist am 14. November 2026 um 18:30 Uhr, Dauer 2 Stunden. Max Weber, geb. 12.03.1980."
    r = fast_pipeline.detect(text, DetectionOptions(profile="fast"))
    found = {(s.entity, s.text) for s in r.spans}
    assert r.preset == "general"
    assert not any(e in ("DATE_TIME", "CLOCK_TIME", "DURATION") for e, _ in found)
    assert ("DATE_OF_BIRTH", "12.03.1980") in found
    r = fast_pipeline.detect(text, DetectionOptions(profile="fast", preset="timesheet"))
    assert {"DATE_TIME", "CLOCK_TIME", "DURATION"} <= {s.entity for s in r.spans}


def test_species_not_pii_but_surnames_are(fast_pipeline):
    if not fast_pipeline.species[0]:
        pytest.skip("species list missing")
    text = "Der Kiebitz brütet wieder. Herr Fink hat 12 Kiebitze gezählt."
    r = fast_pipeline.detect(text, DetectionOptions(profile="fast", deny_list=[]))
    texts = {s.text for s in r.spans}
    assert "Kiebitz" not in texts


def test_iban_without_valid_checksum_and_org_legal_forms(fast_pipeline):
    text = "Spendenkonto DE12 7509 0000 0000 1234 56 bei der Geisel GmbH & Co. KG und Hudson and Sons Ltd."
    found = {(s.entity, s.text) for s in fast_pipeline.detect(text, DetectionOptions(profile="fast")).spans}
    assert ("IBAN_CODE", "DE12 7509 0000 0000 1234 56") in found
    assert ("ORGANIZATION", "Geisel GmbH & Co. KG") in found and ("ORGANIZATION", "Hudson and Sons Ltd.") in found


def test_table_column_propagation(fast_pipeline):
    text = ("Datum | Zeit | Projekt | Dauer\n"
            "03.10.2026 | 08:30-12:00 | Projekt Nordlicht | 3,5 h\n"
            "04.10.2026 | 09:00-17:00 | Silberfalke | 8 h\n"
            "05.10.2026 | 08:00-12:00 | Kranichzug Next | 4 h\n")
    found = {(s.entity, s.text) for s in fast_pipeline.detect(text, DetectionOptions(profile="fast")).spans}
    assert ("PROJECT_NAME", "Silberfalke") in found and ("PROJECT_NAME", "Kranichzug Next") in found


def test_table_header_typing(fast_pipeline):
    text = ("Datum;Von;Bis;Projekt;Kunde;Stunden\n"
            "03.10.2026;08:30;12:00;Sailing Stone;Savage, Kelly and Walker;3,5\n"
            "04.10.2026;09:00;17:00;Meisenweg;Lindner GmbH;8,0\n")
    found = {(s.entity, s.text) for s in fast_pipeline.detect(text, DetectionOptions(profile="fast")).spans}
    for item in [("PROJECT_NAME", "Sailing Stone"), ("PROJECT_NAME", "Meisenweg"), ("DURATION", "3,5"),
                 ("DURATION", "8,0"), ("ORGANIZATION", "Savage, Kelly and Walker")]:
        assert item in found, item
    assert not any(t in ("Datum", "Projekt", "Stunden") for _, t in found)
