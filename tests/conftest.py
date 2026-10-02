import os
from pathlib import Path

import pytest

os.environ.setdefault("PII_INSECURE_DEV", "1")
HF_HOME = Path(os.environ.get("HF_HOME", Path(__file__).resolve().parent.parent / "models" / "hf"))
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from app.config import load_settings  # noqa: E402


@pytest.fixture(scope="session")
def settings():
    return load_settings()


@pytest.fixture(scope="session")
def fast_pipeline(settings):
    from app.detection.pipeline import DetectionPipeline
    return DetectionPipeline(settings, layers=["patterns", "spacy"])


def _models_available() -> bool:
    return (HF_HOME / "hub").exists() and any((HF_HOME / "hub").glob("models--*gliner*"))


requires_ml = pytest.mark.skipif(not _models_available(), reason="HF models not downloaded")
