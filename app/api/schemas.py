"""Request/response models of the HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

Mode = Literal["redact", "placeholder", "mask", "fake", "hash"]
Operator = Literal["redact", "placeholder", "mask", "fake", "hash", "keep"]
MAX_TEXT = 2_000_000


class TimesheetFeatureOverride(BaseModel):
    detect: bool | None = None
    fake: bool | None = None


class TimesheetOverride(BaseModel):
    """Per-request switches; omitted values fall back to config.yaml."""
    project_names: TimesheetFeatureOverride | None = None
    project_codes: TimesheetFeatureOverride | None = None
    dates: TimesheetFeatureOverride | None = None
    clock_times: TimesheetFeatureOverride | None = None
    durations: TimesheetFeatureOverride | None = None

    def as_dict(self) -> dict[str, dict[str, bool]]:
        out = {}
        for name, value in self:
            if value is not None:
                out[name] = {k: v for k, v in value.model_dump().items() if v is not None}
        return out


class DetectionParams(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT)
    profile: Literal["fast", "balanced", "thorough"] | None = Field(
        None, description="Detection depth; default from config (thorough)")
    language: Literal["de", "en"] | None = Field(None, description="Force a language; default: auto per paragraph")
    entities: list[str] | None = Field(None, description="Only report these entity types")
    ignore_entities: list[str] | None = None
    allow_list: list[str] = Field(default_factory=list, description="Terms never to redact")
    deny_list: list[str] = Field(default_factory=list, description="Terms always to redact (entity CUSTOM)")
    timesheet: TimesheetOverride | None = None
    preset: Literal["auto", "general", "timesheet"] | None = Field(
        None, description="auto: timesheet rules if the text looks like a timesheet; general: plain dates/times/"
                          "durations are not PII. Default from config")
    format: Literal["auto", "text", "html"] = Field(
        "auto", description="html: only visible text and selected attributes are analysed; markup is kept")


class AnalyzeRequest(DetectionParams):
    include_text: bool = Field(False, description="Include the detected original strings in the response")


class AnonymizeRequest(DetectionParams):
    mode: Mode = "placeholder"
    operators: dict[str, Operator] = Field(
        default_factory=dict, description='Per-entity operator overrides, e.g. {"IBAN_CODE": "mask"}; "*" = all')
    session_id: str | None = Field(
        None, pattern=r"^[A-Za-z0-9_-]{8,64}$",
        description="Reuse a session: consistent replacements across requests, reversible via /deanonymize")
    reversible: bool = Field(False, description="Store the mapping so /deanonymize can restore originals")
    include_originals: bool = Field(False, description="Include original strings in the entity report")

    @model_validator(mode="after")
    def _check(self):
        if self.reversible and self.mode in ("redact", "mask"):
            raise ValueError("reversible=true needs mode placeholder, fake or hash")
        return self


class DeanonymizeRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT)
    session_id: str = Field(..., pattern=r"^[A-Za-z0-9_-]{8,64}$")


class JobRequest(BaseModel):
    kind: Literal["anonymize", "analyze"] = "anonymize"
    request: dict[str, Any]


class JobStatus(BaseModel):
    job_id: str
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None
