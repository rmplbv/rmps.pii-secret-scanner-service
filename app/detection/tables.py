"""Column consistency for delimiter-separated tables (timesheet exports, pasted
CSV): if a column holds entity E in some rows, the other cells of that column
are very likely E as well (project names, clients, employees)."""

from __future__ import annotations

import re

from ..models import Span

DELIMS = ("|", "\t", ";")

# Column header -> entity of the data cells below it (timesheet / booking exports).
HEADERS = [
    (re.compile(r"^(?:stunden|std\.?|h|hours?|hrs|dauer|duration|zeit\s*\(h\)|aufwand|menge|pt|personentage|tage|days)$", re.I), "DURATION"),
    (re.compile(r"^(?:projekt(?:name)?|project(?:\s*name)?|vorhaben|auftrag)$", re.I), "PROJECT_NAME"),
    (re.compile(r"^(?:projekt-?\s*nr\.?|projektnummer|project\s*(?:no\.?|number|id|code)|kostenstelle|kst\.?|psp(?:-element)?|wbs|cost\s*cent(?:er|re))$", re.I), "PROJECT_CODE"),
    (re.compile(r"^(?:kunde|auftraggeber|client|customer|firma|company|organisation|organization)$", re.I), "ORGANIZATION"),
    (re.compile(r"^(?:mitarbeiter(?:in)?|name|person|employee|staff|bearbeiter(?:in)?|ansprechpartner(?:in)?|contact)$", re.I), "PERSON"),
    (re.compile(r"^(?:datum|date|tag|day)$", re.I), "DATE_TIME"),
    (re.compile(r"^(?:von|bis|beginn|ende|start|end|from|to|uhrzeit|time)$", re.I), "CLOCK_TIME"),
    (re.compile(r"^(?:personalnummer|pers\.?-?\s*nr\.?|employee\s*(?:id|no\.?))$", re.I), "EMPLOYEE_ID"),
    (re.compile(r"^(?:e-?mail|mail)$", re.I), "EMAIL_ADDRESS"),
    (re.compile(r"^(?:telefon|tel\.?|phone|mobil|handy)$", re.I), "PHONE_NUMBER"),
    (re.compile(r"^(?:ort|stadt|city|location|standort)$", re.I), "LOCATION"),
]
PROPAGATE = {"PROJECT_NAME", "ORGANIZATION", "PERSON", "LOCATION", "PROJECT_CODE", "EMPLOYEE_ID"}
_NOT_A_NAME = re.compile(r"^[\d\s.,:/%+\-–hHmin]*$")  # numbers, times, durations


def _cells(line: str, start: int, delim: str) -> list[tuple[int, int]]:
    out, pos = [], 0
    for part in line.split(delim):
        a, b = pos, pos + len(part)
        lead = len(part) - len(part.lstrip())
        trail = len(part) - len(part.rstrip())
        out.append((start + a + lead, start + b - trail))
        pos = b + len(delim)
    return out


def tables(text: str) -> list[list[list[tuple[int, int]]]]:
    """Runs of >= 2 consecutive lines with the same delimiter and column count (>= 3)."""
    result, run, run_key = [], [], None
    pos = 0
    for line in text.split("\n"):
        key = None
        for d in DELIMS:
            n = line.count(d)
            if n >= 2:
                key = (d, n)
                break
        if key and key == run_key:
            run.append(_cells(line, pos, key[0]))
        else:
            if len(run) >= 2:
                result.append(run)
            run = [_cells(line, pos, key[0])] if key else []
            run_key = key
        pos += len(line) + 1
    if len(run) >= 2:
        result.append(run)
    return result


def header_entity(cell: str) -> str | None:
    cell = cell.strip().strip(":*").strip()
    for rx, entity in HEADERS:
        if rx.match(cell):
            return entity
    return None


def propagate_columns(text: str, spans: list[Span], ignored: set[str], min_share: float = 0.5) -> list[Span]:
    """1. Header typing: a column titled "Projekt"/"Stunden"/"Kunde" labels every data cell.
    2. Majority: otherwise, if most cells of a column hold entity E, the rest are E too."""
    out: list[Span] = []
    for table in tables(text):
        rows = table
        header = None
        # header row: no digits while the data rows have some
        if not re.search(r"\d", text[rows[0][0][0]:rows[0][-1][1]]) and \
                any(re.search(r"\d", text[r[0][0]:r[-1][1]]) for r in rows[1:]):
            header, rows = rows[0], rows[1:]
        for col in range(len(rows[0])):
            cells = [row[col] for row in rows if col < len(row) and row[col][1] > row[col][0]]
            if not cells:
                continue
            typed = header_entity(text[header[col][0]:header[col][1]]) if header and col < len(header) else None
            if typed:
                entity, numeric_ok = typed, True
            else:
                numeric_ok = False
                hits: dict[str, int] = {}
                for a, b in cells:
                    for s in spans:
                        # entity covers most of the cell ("Projekt Nordlicht" -> "Nordlicht")
                        overlap = min(b, s.end) - max(a, s.start)
                        if s.entity in PROPAGATE and overlap >= 0.5 * (b - a):
                            hits[s.entity] = hits.get(s.entity, 0) + 1
                            break
                if not hits:
                    continue
                entity, n = max(hits.items(), key=lambda kv: kv[1])
                if n < 2 or n / len(cells) < min_share:
                    continue
            if entity in ignored:
                continue
            for a, b in cells:
                cell = text[a:b]
                if not numeric_ok and _NOT_A_NAME.match(cell):
                    continue
                # cells already holding this entity are fine; a different type (spaCy calling a
                # project "LOCATION") is overruled: pattern-origin types win on identical spans
                if any(s.start < b and a < s.end and s.entity == entity for s in spans):
                    continue
                out.append(Span(a, b, entity, 0.8 if typed else 0.6, {"patterns"}, cell))
    return out
