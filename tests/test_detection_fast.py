"""Fast profile (patterns + spaCy) on realistic German/English text."""

from app.detection.pipeline import DetectionOptions

TEXT = (
    "Hallo Frau Schneider,\n\n"
    "ich bin Max Müller (Personalnummer: 48213), Hauptstraße 5, 80331 München. "
    "Tel. +49 89 1234567, max.mueller@example.de, IBAN DE89 3704 0044 0532 0130 00.\n"
    "Am 03.10.2026 von 08:30 bis 17:00 Uhr am Projekt PRJ-2024-017 gearbeitet, 7,5 h, Kostenstelle 4711-20. "
    "Seit Montag krankgeschrieben wegen Depressionen; Mitglied bei ver.di und katholisch.\n\n"
    "John Smith from London called on October 3, 2026 about the Sample Project Osprey.\n"
)


def found(result):
    return {(s.entity, s.text) for s in result.spans}


def test_fast_profile_core_entities(fast_pipeline):
    r = fast_pipeline.detect(TEXT, DetectionOptions(profile="fast"))
    f = found(r)
    for item in [
        ("EMPLOYEE_ID", "48213"), ("ADDRESS", "Hauptstraße 5, 80331 München"),
        ("PHONE_NUMBER", "+49 89 1234567"), ("EMAIL_ADDRESS", "max.mueller@example.de"),
        ("IBAN_CODE", "DE89 3704 0044 0532 0130 00"), ("DATE_TIME", "03.10.2026"),
        ("CLOCK_TIME", "08:30"), ("CLOCK_TIME", "17:00 Uhr"), ("PROJECT_CODE", "PRJ-2024-017"),
        ("DURATION", "7,5 h"), ("PROJECT_CODE", "4711-20"), ("HEALTH_DATA", "Depressionen"),
        ("HEALTH_DATA", "krankgeschrieben"), ("TRADE_UNION", "ver.di"), ("RELIGION", "katholisch"),
        ("DATE_TIME", "October 3, 2026"), ("PROJECT_NAME", "Sample Project Osprey"),
        ("PERSON", "Max Müller"), ("PERSON", "John Smith"),
    ]:
        assert item in f, item
    assert r.languages == ["de", "en"]
    assert not any(s.text in ("Montag", "He", "Frau") for s in r.spans)


def test_timesheet_detect_switch(fast_pipeline):
    r = fast_pipeline.detect(TEXT, DetectionOptions(profile="fast", timesheet={"clock_times": {"detect": False},
                                                                             "durations": {"detect": False}}))
    assert not any(s.entity in ("CLOCK_TIME", "DURATION") for s in r.spans)


def test_allow_and_deny_lists(fast_pipeline):
    r = fast_pipeline.detect(TEXT, DetectionOptions(profile="fast", allow_list=["Max Müller"],
                                                    deny_list=["Osprey", "Hallo"]))
    f = found(r)
    assert ("PERSON", "Max Müller") not in f and ("CUSTOM", "Hallo") in f


def test_entity_filter(fast_pipeline):
    r = fast_pipeline.detect(TEXT, DetectionOptions(profile="fast", entities=["IBAN_CODE"]))
    assert {s.entity for s in r.spans} == {"IBAN_CODE"}
