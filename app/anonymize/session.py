"""Per-session pseudonymization state: the mapping original <-> replacement
plus offsets, so a session yields consistent replacements across requests and
can be reversed."""

from __future__ import annotations

import json
import re
import secrets
import uuid
from dataclasses import asdict, dataclass, field


def norm(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


@dataclass
class SessionState:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    seed: str = field(default_factory=lambda: secrets.token_hex(16))
    forward: dict[str, str] = field(default_factory=dict)       # "ENTITY\x00original" -> replacement
    reverse: dict[str, str] = field(default_factory=dict)       # replacement -> original
    token_forward: dict[str, str] = field(default_factory=dict)  # person name tokens
    token_reverse: dict[str, str] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    date_offset_days: int | None = None
    time_offset_minutes: int | None = None

    @staticmethod
    def key(entity: str, original: str) -> str:
        return f"{entity}\x00{norm(original)}"

    def lookup(self, entity: str, original: str) -> str | None:
        return self.forward.get(self.key(entity, original))

    def remember(self, entity: str, original: str, replacement: str, reversible: bool = True) -> None:
        self.forward[self.key(entity, original)] = replacement
        if reversible and replacement not in self.reverse:
            self.reverse[replacement] = original

    def replacement_taken(self, replacement: str, original: str) -> bool:
        owner = self.reverse.get(replacement)
        return owner is not None and norm(owner) != norm(original)

    def next_number(self, entity: str) -> int:
        self.counters[entity] = self.counters.get(entity, 0) + 1
        return self.counters[entity]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str | bytes) -> "SessionState":
        return cls(**json.loads(data))
