"""Transformer + GLiNER layers (need downloaded models; slow)."""

import pytest

from app.detection.pipeline import DetectionOptions
from tests.conftest import requires_ml

pytestmark = [pytest.mark.ml, requires_ml]


@pytest.fixture(scope="module")
def balanced(settings):
    from app.detection.pipeline import DetectionPipeline
    return DetectionPipeline(settings, layers=["patterns", "spacy", "transformer", "gliner"])


def test_balanced_profile_adds_recall(balanced):
    text = ("Ich arbeite bei der Siemens AG. Dr. Weber hat Diabetes diagnostiziert. "
            "John Smith from Acme Corp in London called.")
    r = balanced.detect(text, DetectionOptions(profile="balanced"))
    f = {(s.entity, s.text) for s in r.spans}
    for item in [("ORGANIZATION", "Siemens AG"), ("PERSON", "Weber"), ("HEALTH_DATA", "Diabetes"),
                 ("PERSON", "John Smith"), ("ORGANIZATION", "Acme Corp"), ("LOCATION", "London")]:
        assert item in f, item
    assert all(len(s.sources) >= 1 for s in r.spans)
