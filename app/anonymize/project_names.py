"""Fake project names: drawn from a static list or freely combined from
subwords (prefix/core/suffix), which gives a body of several million names."""

from __future__ import annotations

import random
import re
from functools import lru_cache
from pathlib import Path

import yaml

TEMPLATES = [
    "{core}",
    "{prefix}{core_lower}",       # compound: Silberfalke
    "{prefix} {core}",            # Blue Heron
    "{core} {suffix}",            # Falke Next
    "{prefix}{core_lower} {suffix}",
    "{prefix} {core} {suffix}",
]
LEAD_WORDS = re.compile(r"^((?:Sample\s+|Beispiel)?(?:Projekt|Project|Programm|Program|Initiative|Vorhaben)\s+)", re.I)


class ProjectNameGenerator:
    def __init__(self, path: Path, generated_ratio: float = 0.7):
        data = _load(str(path))
        self.names: list[str] = data.get("names", [])
        self.prefixes: list[str] = data.get("prefixes", [])
        self.cores: list[str] = data.get("cores", [])
        self.suffixes: list[str] = [str(s) for s in data.get("suffixes", [])]
        self.ratio = generated_ratio

    def generate(self, rng: random.Random, original: str = "") -> str:
        lead = ""
        m = LEAD_WORDS.match(original)
        if m:  # keep "Projekt " / "Project " so the sentence still reads naturally
            lead = m.group(1)
        if self.cores and (not self.names or rng.random() < self.ratio):
            tpl = rng.choice(TEMPLATES if self.prefixes and self.suffixes else ["{core}"])
            core = rng.choice(self.cores)
            name = tpl.format(
                core=core, core_lower=core[:1].lower() + core[1:],
                prefix=rng.choice(self.prefixes) if self.prefixes else "",
                suffix=rng.choice(self.suffixes) if self.suffixes else "",
            ).strip()
        elif self.names:
            name = rng.choice(self.names)
        else:
            name = f"Projekt {rng.randint(100, 999)}"
        if lead and LEAD_WORDS.match(name):
            name = LEAD_WORDS.sub("", name)
        return lead + name

    @property
    def body_size(self) -> int:
        p, c, s = len(self.prefixes), len(self.cores), len(self.suffixes)
        return len(self.names) + c + 2 * p * c + c * s + 2 * p * c * s


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
