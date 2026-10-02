"""Generate a synthetic, labeled German/English PII dataset (JSONL).

    python -m eval.generate --n 200 --out eval/out/synthetic.jsonl

Each line: {"id", "language", "text", "spans": [{"start", "end", "entity"}]}.
Templates cover e-mails, notes, HR/sick-leave messages and timesheets. Real
project names are NOT in the dictionary here, so PROJECT_NAME recall measures
the model layers; in production add your projects to project_names.txt.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from faker import Faker

from app.anonymize.project_names import ProjectNameGenerator
from app.config import load_settings

TEMPLATES = {
    "de": [
        "Hallo {FIRST},\n\nanbei die Unterlagen von {PERSON} ({EMAIL}). Bitte ruf sie unter {PHONE} zurück.\n\nViele Grüße\n{PERSON2}",
        "{PERSON} (Personalnummer: {EMPID}) ist seit {DATE} {SICK}, Diagnose: {HEALTH}.",
        "Bitte überweisen Sie den Betrag auf die IBAN {IBAN} (Kontoinhaber: {PERSON}).",
        "Neue Anschrift von {PERSON}: {ADDRESS}.",
        "{DATE} | {TIME}-{TIME2} | {PROJECT} | {PCODE} | {DURATION} | Abstimmung mit {ORG}",
        "Am {DATE} habe ich von {TIME} bis {TIME2} Uhr am Projekt {PROJECT} gearbeitet ({DURATION}), Kostenstelle {PCODE}.",
        "Termin mit {PERSON} von der {ORG} in {CITY} am {DATE} um {TIME} Uhr.",
        "{PERSON} ist Mitglied bei {UNION} und {RELIGION}.",
        "Laut {PERSON2} leidet {LAST} an {HEALTH} und nimmt deshalb {DURATION} Sonderurlaub.",
        "Zeiterfassung {WEEK}: {PERSON}, {PROJECT}, {DURATION}; Rückfragen an {EMAIL}.",
    ],
    "en": [
        "Hi {FIRST},\n\nplease find attached the documents from {PERSON} ({EMAIL}). Call back on {PHONE}.\n\nBest,\n{PERSON2}",
        "{PERSON} (employee ID {EMPID}) has been on {SICK} since {DATE}, diagnosis: {HEALTH}.",
        "Please transfer the amount to IBAN {IBAN} (account holder: {PERSON}).",
        "New address of {PERSON}: {ADDRESS}.",
        "{DATE} | {TIME}-{TIME2} | {PROJECT} | {PCODE} | {DURATION} | call with {ORG}",
        "On {DATE} I worked from {TIME} to {TIME2} on project {PROJECT} ({DURATION}), cost center {PCODE}.",
        "Meeting with {PERSON} from {ORG} in {CITY} on {DATE} at {TIME}.",
        "{PERSON} is a member of {UNION} and is {RELIGION}.",
        "According to {PERSON2}, {LAST} suffers from {HEALTH} and takes {DURATION} of leave.",
        "Timesheet {WEEK}: {PERSON}, {PROJECT}, {DURATION}; questions to {EMAIL}.",
    ],
}
HEALTH = {"de": ["Diabetes", "Depressionen", "Migräne", "Bandscheibenvorfall", "Burnout", "Asthma"],
          "en": ["diabetes", "depression", "migraine", "a slipped disc", "burnout", "asthma"]}
RELIGION = {"de": ["katholisch", "evangelisch", "muslimisch", "jüdisch"],
            "en": ["Catholic", "Protestant", "Muslim", "Jewish"]}
UNION = {"de": ["ver.di", "der IG Metall", "der GEW"], "en": ["Unite", "the GMB", "UNISON"]}
SLOT_ENTITY = {
    "PERSON": "PERSON", "PERSON2": "PERSON", "FIRST": "PERSON", "LAST": "PERSON", "EMAIL": "EMAIL_ADDRESS",
    "PHONE": "PHONE_NUMBER", "IBAN": "IBAN_CODE", "ADDRESS": "ADDRESS", "DATE": "DATE_TIME",
    "TIME": "CLOCK_TIME", "TIME2": "CLOCK_TIME", "DURATION": "DURATION", "PROJECT": "PROJECT_NAME",
    "PCODE": "PROJECT_CODE", "EMPID": "EMPLOYEE_ID", "ORG": "ORGANIZATION", "CITY": "LOCATION",
    "HEALTH": "HEALTH_DATA", "RELIGION": "RELIGION", "UNION": "TRADE_UNION", "WEEK": "DATE_TIME", "SICK": "HEALTH_DATA",
}


def values(lang: str, fk: Faker, rng: random.Random, projects: ProjectNameGenerator) -> dict[str, str]:
    first, last = fk.first_name(), fk.last_name()
    d = fk.date_between("-2y", "+1y")
    h1 = rng.randint(7, 10)
    union = rng.choice(UNION[lang])
    return {
        "PERSON": f"{first} {last}", "FIRST": first, "LAST": last,
        "PERSON2": fk.name().replace("Dr. ", "").replace("Prof. ", ""),
        "EMAIL": f"{first.lower()}.{last.lower()}@{fk.domain_name()}".replace(" ", ""),
        "PHONE": fk.phone_number(), "IBAN": fk.iban(),
        "ADDRESS": fk.address().replace("\n", ", "),
        "DATE": d.strftime("%d.%m.%Y" if lang == "de" else rng.choice(["%B %d, %Y", "%Y-%m-%d"])),
        "TIME": f"{h1:02d}:{rng.choice(['00', '15', '30', '45'])}",
        "TIME2": f"{h1 + rng.randint(4, 9):02d}:{rng.choice(['00', '30'])}",
        "DURATION": (f"{rng.choice([1, 2, 4, 6, 7, 8])},{rng.choice([0, 5])} h" if lang == "de"
                     else f"{rng.choice([1, 2, 4, 6, 7, 8])}.{rng.choice([0, 5])} hours"),
        "PROJECT": projects.generate(rng),
        "PCODE": f"PRJ-{rng.randint(2019, 2026)}-{rng.randint(1, 999):03d}",
        "EMPID": str(rng.randint(10000, 999999)), "ORG": fk.company(), "CITY": fk.city(),
        "HEALTH": rng.choice(HEALTH[lang]), "RELIGION": rng.choice(RELIGION[lang]),
        "UNION": union, "WEEK": f"{'KW' if lang == 'de' else 'week'} {rng.randint(1, 52)}",
        "SICK": "krankgeschrieben" if lang == "de" else "sick leave",
    }


def render(template: str, vals: dict[str, str]) -> tuple[str, list[dict]]:
    out, spans, pos = [], [], 0
    for m in re.finditer(r"\{(\w+)\}", template):
        out.append(template[pos:m.start()])
        start = sum(len(x) for x in out)
        val = vals[m.group(1)]
        entity = SLOT_ENTITY[m.group(1)]
        if m.group(1) == "UNION" and val.startswith(("der ", "the ")):
            prefix, val = val.split(" ", 1)
            out.append(prefix + " ")
            start += len(prefix) + 1
        out.append(val)
        if entity:
            spans.append({"start": start, "end": start + len(val), "entity": entity})
        pos = m.end()
    out.append(template[pos:])
    return "".join(out), spans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="eval/out/synthetic.jsonl")
    args = ap.parse_args()
    s = load_settings()
    projects = ProjectNameGenerator(s.path(s.dictionaries.fake_project_words), 0.7)
    rng = random.Random(args.seed)
    fakers = {"de": Faker("de_DE"), "en": Faker("en_GB")}
    for f in fakers.values():
        f.seed_instance(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for i in range(args.n):
            lang = "de" if i % 2 == 0 else "en"
            # 2-3 templates per document
            parts = [rng.choice(TEMPLATES[lang]) for _ in range(rng.randint(2, 3))]
            text, spans = "", []
            for tpl in parts:
                t, sp = render(tpl, values(lang, fakers[lang], rng, projects))
                off = len(text)
                spans += [{**x, "start": x["start"] + off, "end": x["end"] + off} for x in sp]
                text += t + "\n\n"
            fh.write(json.dumps({"id": i, "language": lang, "text": text, "spans": spans}, ensure_ascii=False) + "\n")
    print(f"wrote {args.n} documents to {args.out}")


if __name__ == "__main__":
    main()
