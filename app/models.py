"""Core data types shared by detection and anonymization."""

from __future__ import annotations

from dataclasses import dataclass, field

# Entity families (used for defaults and docs; recognizers may emit others).
SPECIAL_CATEGORY = {
    "HEALTH_DATA",
    "RELIGION",
    "POLITICAL_OPINION",
    "TRADE_UNION",
    "ETHNIC_ORIGIN",
    "SEXUAL_ORIENTATION",
    "GENETIC_BIOMETRIC",
}
TIMESHEET = {"PROJECT_NAME", "PROJECT_CODE", "DATE_TIME", "CLOCK_TIME", "DURATION"}

# Which timesheet feature switch governs which entity.
TIMESHEET_FEATURE = {
    "PROJECT_NAME": "project_names",
    "PROJECT_CODE": "project_codes",
    "DATE_TIME": "dates",
    "CLOCK_TIME": "clock_times",
    "DURATION": "durations",
}


# Entities whose *fake* behaviour follows a timesheet switch even though they
# are always detected (a date of birth is PII in every kind of text).
FAKE_FEATURE = {**TIMESHEET_FEATURE, "DATE_OF_BIRTH": "dates"}


@dataclass
class Span:
    start: int
    end: int
    entity: str
    score: float
    sources: set[str] = field(default_factory=set)
    text: str = ""

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end

    def __len__(self) -> int:
        return self.end - self.start

    def to_dict(self, include_text: bool = False) -> dict:
        d = {
            "start": self.start,
            "end": self.end,
            "entity": self.entity,
            "score": round(self.score, 3),
            "sources": sorted(self.sources),
        }
        if include_text:
            d["text"] = self.text
        return d
