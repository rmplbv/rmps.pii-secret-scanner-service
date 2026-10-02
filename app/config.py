"""Configuration loading. The YAML file is the single source of truth; env vars
only select the file and provide secrets."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(os.environ.get("PII_ROOT", Path(__file__).resolve().parent.parent))


class _Model(BaseModel):
    model_config = ConfigDict(extra="allow")


class CustomPattern(_Model):
    name: str
    entity: str
    pattern: str
    group: int = 0
    score: float = 0.8
    languages: list[str] | None = None


class PatternsLayer(_Model):
    phone_regions: list[str] = ["DE", "US", "GB"]
    custom_patterns: list[CustomPattern] = []


class SpacyLayer(_Model):
    models: dict[str, str] = {"de": "de_core_news_lg", "en": "en_core_web_lg"}
    low_confidence_entities: list[str] = []
    low_confidence_multiplier: float = 0.45


class TransformerLayer(_Model):
    model: str = "Davlan/xlm-roberta-large-ner-hrl"
    label_map: dict[str, str] = {"PER": "PERSON", "ORG": "ORGANIZATION", "LOC": "LOCATION"}
    chunk_chars: int = 1500
    overlap_chars: int = 200
    batch_size: int = 4
    min_score: float = 0.5


class GlinerLayer(_Model):
    model: str = "urchade/gliner_multi_pii-v1"
    threshold: float = 0.45
    chunk_words: int = 220
    overlap_words: int = 40
    labels: dict[str, str] = {}


class LlmLayer(_Model):
    base_url: str = "http://llm:8080"
    model: str = "local"
    timeout_seconds: float = 900
    chunk_chars: int = 4000
    overlap_chars: int = 300
    max_tokens: int = 2048
    temperature: float = 0.0
    adjudicate_range: tuple[float, float] = (0.25, 0.75)
    score_found: float = 0.75
    disable_thinking: bool = True
    max_retries: int = 2
    on_error: str = "skip"   # skip: keep other layers' results for the chunk and warn | fail: fail the job


class Layers(_Model):
    patterns: PatternsLayer = PatternsLayer()
    spacy: SpacyLayer = SpacyLayer()
    transformer: TransformerLayer = TransformerLayer()
    gliner: GlinerLayer = GlinerLayer()
    llm: LlmLayer = LlmLayer()


class Merge(_Model):
    min_score: float = 0.4
    single_source_min_score: float = 0.55
    agreement_boost: float = 0.15
    entity_priority: dict[str, int] = {}


class Dictionaries(_Model):
    project_names: str | None = None
    special_category: str | None = None
    allow_list: str | None = None
    fake_project_words: str = "app/data/fake/project_words.yaml"


class Anonymize(_Model):
    default_mode: str = "placeholder"
    redact_text: str = "[REDACTED]"
    placeholder_format: str = "<{entity}_{n}>"
    hash_format: str = "<{entity}_{digest}>"
    mask_char: str = "*"
    mask_keep_last: int = 4
    mask_keep_last_entities: list[str] = ["IBAN_CODE", "CREDIT_CARD", "PHONE_NUMBER"]
    no_fake_entities: list[str] = []


class TimesheetFeature(_Model):
    detect: bool = True
    fake: bool = True


class TimesheetProjectNames(TimesheetFeature):
    generated_ratio: float = 0.7


class TimesheetDates(TimesheetFeature):
    shift_days: tuple[int, int] = (-120, 120)
    preserve_weekday: bool = True


class TimesheetClock(TimesheetFeature):
    fake: bool = False
    shift_minutes: tuple[int, int] = (-90, 90)
    round_minutes: int = 15


class TimesheetDurations(TimesheetFeature):
    fake: bool = False
    jitter_pct: float = 15
    round_minutes: int = 15


class Timesheet(_Model):
    project_names: TimesheetProjectNames = TimesheetProjectNames()
    project_codes: TimesheetFeature = TimesheetFeature()
    dates: TimesheetDates = TimesheetDates()
    clock_times: TimesheetClock = TimesheetClock()
    durations: TimesheetDurations = TimesheetDurations()

    def merged(self, override: dict[str, Any] | None) -> "Timesheet":
        """Return a copy with a (partial) per-request override applied."""
        if not override:
            return self
        data = self.model_dump()
        for feature, values in override.items():
            if feature in data and isinstance(values, dict):
                data[feature].update(values)
        return Timesheet.model_validate(data)


class Vault(_Model):
    key_env: str = "PII_VAULT_KEY"
    session_ttl_seconds: int = 30 * 24 * 3600
    job_ttl_seconds: int = 24 * 3600


class Settings(_Model):
    languages: list[str] = ["de", "en"]
    default_language: str = "de"
    profiles: dict[str, list[str]] = {"fast": ["patterns", "spacy"]}
    default_profile: str = "thorough"
    queues: dict[str, str] = {}
    layers: Layers = Layers()
    merge: Merge = Merge()
    entity_aliases: dict[str, str] = {}
    ignored_entities: list[str] = []
    dictionaries: Dictionaries = Dictionaries()
    anonymize: Anonymize = Anonymize()
    timesheet: Timesheet = Timesheet()
    vault: Vault = Vault()
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)

    def path(self, rel: str | None) -> Path | None:
        if not rel:
            return None
        p = Path(rel)
        return p if p.is_absolute() else ROOT / p

    def queue_for(self, profile: str) -> str:
        return self.queues.get(profile, "pii:full")


def load_settings(path: str | Path | None = None) -> Settings:
    path = Path(path or os.environ.get("PII_CONFIG", ROOT / "config" / "config.yaml"))
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return Settings.model_validate({**data, "raw": data})


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
