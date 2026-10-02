import json

import httpx

from app.detection.llm_layer import LlmLayer, _locate
from app.models import Span


def make_layer(settings, payload, calls):
    def handler(request: httpx.Request):
        calls.append(json.loads(request.content))
        content = "<think>\n</think>\n" + json.dumps(payload)
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    client = httpx.Client(base_url="http://llm", transport=httpx.MockTransport(handler))
    return LlmLayer(settings, client=client)


def test_findings_are_grounded_and_hallucinations_dropped(settings):
    text = "Treffen mit jsmith88 und Frau Kowalski wegen Projekt Kiebitz."
    payload = {"findings": [{"text": "jsmith88", "type": "USERNAME"},
                            {"text": "kowalski", "type": "PERSON"},          # case differs -> still located
                            {"text": "Erika Mustermann", "type": "PERSON"}],  # not in text -> dropped
               "verdicts": []}
    calls = []
    found, verdicts = make_layer(settings, payload, calls).detect(text)
    assert sorted(s.text for s in found) == ["Kowalski", "jsmith88"]
    assert calls[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_verdicts_and_known_list(settings):
    text = "Termin mit Acme am Montag."
    cand = Span(11, 15, "ORGANIZATION", 0.4, {"spacy"}, "Acme")
    known = Span(19, 25, "DATE_TIME", 0.9, {"patterns"}, "Montag")
    calls = []
    layer = make_layer(settings, {"findings": [], "verdicts": [{"id": 0, "is_pii": True, "type": "ORGANIZATION"}]}, calls)
    found, verdicts = layer.detect(text, [cand], [known])
    assert verdicts[0].is_pii and verdicts[0].span is cand
    prompt = calls[0]["messages"][1]["content"]
    assert '"Montag"' in prompt and '0. "Acme"' in prompt


def test_locate_short_needles_need_word_boundaries():
    assert _locate("Max und Maxi", "Max") == [(0, 3)]


def test_llm_errors_are_skipped_with_warning(settings):
    def handler(request):
        return httpx.Response(500)

    layer = LlmLayer(settings, client=httpx.Client(base_url="http://llm", transport=httpx.MockTransport(handler)))
    warnings = []
    found, verdicts = layer.detect("Ein Text über Frau Kowalski.", warnings=warnings)
    assert found == [] and len(warnings) == 1 and "skipped" in warnings[0]


def test_truncated_json_is_salvaged():
    from app.detection.llm_layer import _parse_json
    raw = '{"findings": [{"text": "Kowalski", "type": "PERSON"}, {"text": "jsm'
    assert _parse_json(raw) == {"findings": [{"text": "Kowalski", "type": "PERSON"}], "verdicts": []}


def test_llm_cannot_veto_pattern_matches(settings, monkeypatch):
    from app.detection.pipeline import DetectionOptions, DetectionPipeline

    p = DetectionPipeline(settings, layers=["patterns", "spacy"])
    p.settings = settings.model_copy(update={"profiles": {**settings.profiles, "t": ["patterns", "spacy", "llm"]}})

    class Veto:
        def detect(self, text, candidates, known, warnings):
            from app.detection.llm_layer import Verdict
            assert all("patterns" not in c.sources for c in candidates)
            return [], [Verdict(c, False, None) for c in candidates]

    p.loaded["llm"] = Veto()
    r = p.detect("Gebucht auf PRJ-2024-017 am Montag.", DetectionOptions(profile="t"))
    assert [s.text for s in r.spans] == ["PRJ-2024-017"]
