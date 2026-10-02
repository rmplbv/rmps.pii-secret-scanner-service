"""Generate a synthetic, labeled German/English PII dataset (JSONL).

    python -m eval.generate --n 240 --out eval/out/synthetic.jsonl

Each line: {"id", "kind", "language", "format", "preset", "text", "spans": [{"start","end","entity"}]}
Document kinds (round-robin): email, timesheet_table, booking_list, html, hr_note, chat, nature, signature.

Labeling follows the service's presets: in general texts (e-mails, letters, web pages) plain
dates, clock times and durations are NOT PII and are not labeled; dates of birth always are.
In timesheets they are. Species names (Kiebitz, Lapwing) are never PII. Real project names are
NOT in the project dictionary here, so PROJECT_NAME recall measures the model layers.
"""

from __future__ import annotations

import argparse
import html as htmllib
import json
import random
import re
from pathlib import Path

from faker import Faker

from app.anonymize.project_names import ProjectNameGenerator
from app.config import load_settings

TIMESHEET_ONLY = {"DATE_TIME", "CLOCK_TIME", "DURATION"}
KINDS = ["email", "timesheet_table", "booking_list", "html", "hr_note", "chat", "nature", "signature"]

HEALTH = {"de": ["Diabetes", "Depressionen", "Migräne", "einem Bandscheibenvorfall", "Burnout", "Asthma",
                 "einer Gürtelrose", "Long Covid", "einer Angststörung"],
          "en": ["diabetes", "depression", "migraine", "a slipped disc", "burnout", "asthma", "shingles",
                 "long covid", "an anxiety disorder"]}
SICK = {"de": ["krankgeschrieben", "arbeitsunfähig", "in der Reha"], "en": ["on sick leave", "signed off sick"]}
RELIGION = {"de": ["katholisch", "evangelisch", "muslimisch", "jüdisch", "konfessionslos"],
            "en": ["Catholic", "Protestant", "Muslim", "Jewish", "an atheist"]}
UNION = {"de": ["ver.di", "IG Metall", "GEW", "IG BCE"], "en": ["Unite", "UNISON", "GMB"]}
PARTY = {"de": ["SPD", "CSU", "Grünen", "FDP", "Linken", "Freien Wähler"], "en": ["Labour Party", "Green Party"]}
SPECIES = {"de": ["Kiebitz", "Rotmilan", "Uhu", "Eisvogel", "Feldlerche", "Großer Brachvogel", "Weißstorch",
                  "Wanderfalke", "Laubfrosch", "Gelbbauchunke", "Biber", "Wildkatze", "Schwarzstorch",
                  "Wiesenweihe", "Wachtelkönig", "Braunkehlchen"],
           "en": ["lapwing", "Red Kite", "Eagle Owl", "Kingfisher", "Skylark", "Curlew", "White Stork",
                  "Peregrine", "Tree Frog", "Beaver", "Wildcat", "Black Stork", "Corncrake", "Whinchat"]}
ACTIVITY = {"de": ["Kartierung", "Abstimmung", "Bericht", "Workshop", "Telefonat", "Gutachten", "Exkursion",
                   "Datenauswertung", "Antrag", "Pressearbeit"],
            "en": ["survey", "alignment", "report", "workshop", "call", "assessment", "field trip",
                   "data analysis", "proposal", "press work"]}


class Doc:
    """Text builder that records gold spans while appending."""

    def __init__(self, fmt: str = "text"):
        self.parts: list[str] = []
        self.spans: list[dict] = []
        self.fmt = fmt
        self.len = 0

    def add(self, s: str) -> "Doc":
        self.parts.append(s)
        self.len += len(s)
        return self

    def ent(self, value: str, entity: str | None) -> "Doc":
        out = htmllib.escape(value, quote=True) if self.fmt == "html" else value
        if entity:
            self.spans.append({"start": self.len, "end": self.len + len(out), "entity": entity})
        return self.add(out)

    def text(self) -> str:
        return "".join(self.parts)


def fill(doc: Doc, template: str, vals: dict[str, tuple[str, str | None]], drop: set[str] = frozenset()) -> None:
    """Append `template`, replacing {SLOT} by vals[SLOT] = (value, entity)."""
    pos = 0
    for m in re.finditer(r"\{(\w+)\}", template):
        doc.add(template[pos:m.start()])
        value, entity = vals[m.group(1)]
        doc.ent(value, None if entity in drop else entity)
        pos = m.end()
    doc.add(template[pos:])


class Gen:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        s = load_settings()
        self.projects = ProjectNameGenerator(s.path(s.dictionaries.fake_project_words), 0.7)
        self.fk = {"de": Faker("de_DE"), "en": Faker("en_GB"), "tr": Faker("tr_TR"), "pl": Faker("pl_PL")}
        for f in self.fk.values():
            f.seed_instance(seed)

    def person(self, lang: str) -> tuple[str, str]:
        f = self.fk[self.rng.choice([lang] * 6 + ["tr", "pl"])]
        return f.first_name(), f.last_name()

    def vals(self, lang: str) -> dict[str, tuple[str, str | None]]:
        r, f = self.rng, self.fk[lang]
        first, last = self.person(lang)
        first2, last2 = self.person(lang)
        d = f.date_between("-1y", "+1y")
        dob = f.date_of_birth(minimum_age=20, maximum_age=65)
        iban = f.iban()
        if r.random() < 0.3:  # typo -> invalid checksum
            iban = iban[:-1] + str((int(iban[-1]) + 1) % 10) if iban[-1].isdigit() else iban
        if r.random() < 0.5:
            iban = " ".join(iban[i:i + 4] for i in range(0, len(iban), 4))
        h = r.randint(7, 10)
        return {
            "FIRST": (first, "PERSON"), "LAST": (last, "PERSON"), "PERSON": (f"{first} {last}", "PERSON"),
            "PERSON2": (f"{first2} {last2}", "PERSON"), "LAST2": (last2, "PERSON"),
            "first_lc": (first.lower(), "PERSON"),
            "EMAIL": (f"{first.lower()}.{last.lower()}@{f.free_email_domain()}".replace(" ", ""), "EMAIL_ADDRESS"),
            "PHONE": (f.phone_number(), "PHONE_NUMBER"), "IBAN": (iban, "IBAN_CODE"),
            "STREET": (f.street_address(), "ADDRESS"),
            "ADDRESS": (f.address().replace("\n", ", "), "ADDRESS"),
            "CITY": (f.city(), "LOCATION"), "ORG": (f.company(), "ORGANIZATION"),
            "DATE": (d.strftime("%d.%m.%Y" if lang == "de" else r.choice(["%d %B %Y", "%B %d, %Y", "%Y-%m-%d"])),
                     "DATE_TIME"),
            "DOB": (dob.strftime("%d.%m.%Y" if lang == "de" else "%d/%m/%Y"), "DATE_OF_BIRTH"),
            "TIME": (f"{h:02d}:{r.choice(['00', '15', '30', '45'])}", "CLOCK_TIME"),
            "TIME2": (f"{h + r.randint(3, 8):02d}:{r.choice(['00', '30'])}", "CLOCK_TIME"),
            "DUR": ((f"{r.randint(1, 8)},{r.choice([0, 5])} h" if lang == "de" else f"{r.randint(1, 8)}.{r.choice([0, 5])} hours"),
                    "DURATION"),
            "PROJECT": (self.projects.generate(r), "PROJECT_NAME"),
            "PCODE": (f"PRJ-{r.randint(2019, 2026)}-{r.randint(1, 999):03d}", "PROJECT_CODE"),
            "EMPID": (str(r.randint(10000, 999999)), "EMPLOYEE_ID"),
            "HEALTH": (r.choice(HEALTH[lang]).split(" ", 1)[-1] if lang == "de" and r.random() < 0 else r.choice(HEALTH[lang]), "HEALTH_DATA"),
            "SICK": (r.choice(SICK[lang]), "HEALTH_DATA"),
            "RELIGION": (r.choice(RELIGION[lang]), "RELIGION"), "UNION": (r.choice(UNION[lang]), "TRADE_UNION"),
            "PARTY": (r.choice(PARTY[lang]), "POLITICAL_OPINION"),
            "SPECIES": (r.choice(SPECIES[lang]), None), "SPECIES2": (r.choice(SPECIES[lang]), None),
            "N": (str(r.randint(2, 140)), None),
        }

    # -- document kinds --------------------------------------------------------------------
    def email(self, lang: str) -> Doc:
        v, doc = self.vals(lang), Doc()
        de = [
            "Hallo {FIRST},\n\nwie besprochen schicke ich dir die Unterlagen von {PERSON2} ({EMAIL}). "
            "Das Treffen mit der {ORG} ist am {DATE} um {TIME} Uhr in {CITY}. Ruf mich gern unter {PHONE} an.\n\n",
            "{LAST2} meinte gestern, dass die {SPECIES}-Kartierung bei {CITY} gut lief: {N} Reviere. "
            "Bitte die Rechnung an {STREET} schicken.\n\n",
            "Kurze Info: {LAST} ist bis Freitag {SICK}. Vertretung übernimmt {PERSON2}.\n\n",
        ]
        en = [
            "Hi {FIRST},\n\nas discussed I'm forwarding the documents from {PERSON2} ({EMAIL}). "
            "The meeting with {ORG} is on {DATE} at {TIME} in {CITY}. Feel free to call me on {PHONE}.\n\n",
            "{LAST2} said yesterday that the {SPECIES} survey near {CITY} went well: {N} territories. "
            "Please send the invoice to {STREET}.\n\n",
            "Quick note: {LAST} is {SICK} until Friday. {PERSON2} is covering.\n\n",
        ]
        for t in self.rng.sample(de if lang == "de" else en, k=self.rng.randint(2, 3)):
            fill(doc, t, v, drop=TIMESHEET_ONLY)
        fill(doc, "Viele Grüße\n{PERSON}\n" if lang == "de" else "Best regards,\n{PERSON}\n", v)
        return doc

    def signature(self, lang: str) -> Doc:
        v, doc = self.vals(lang), Doc()
        fill(doc, ("Sehr geehrte Frau {LAST2},\n\nanbei das Protokoll vom {DATE}.\n\nMit freundlichen Grüßen\n\n"
                   if lang == "de" else "Dear Ms {LAST2},\n\nplease find attached the minutes of {DATE}.\n\nKind regards,\n\n"),
             v, drop=TIMESHEET_ONLY)
        title = self.rng.choice(["Dr. ", "", "", "Prof. "])
        fill(doc, title + "{PERSON}\n" + ("Referentin Artenschutz\n" if lang == "de" else "Species Protection Officer\n")
             + "{ORG} | {ADDRESS}\nTel. {PHONE} | {EMAIL}\n", v)
        return doc

    def hr_note(self, lang: str) -> Doc:
        v, doc = self.vals(lang), Doc()
        t = ("Personalakte {PERSON}, geb. {DOB}, Personalnummer: {EMPID}.\n"
             "Seit {DATE} {SICK}, Diagnose: {HEALTH}. Konfession: {RELIGION}. Mitglied bei {UNION}.\n"
             "Gehalt bitte auf IBAN {IBAN} überweisen. Neue Anschrift: {ADDRESS}.\n"
             if lang == "de" else
             "Personnel file {PERSON}, born {DOB}, employee ID {EMPID}.\n"
             "Since {DATE} {SICK}, diagnosis: {HEALTH}. Religion: {RELIGION}. Member of {UNION}.\n"
             "Please pay the salary to IBAN {IBAN}. New address: {ADDRESS}.\n")
        fill(doc, t, v, drop=TIMESHEET_ONLY)
        if self.rng.random() < 0.5:
            fill(doc, ("{LAST} ist außerdem Mitglied der {PARTY}.\n" if lang == "de"
                       else "{LAST} is also a member of the {PARTY}.\n"), v)
        return doc

    def chat(self, lang: str) -> Doc:
        v, doc = self.vals(lang), Doc()
        t = ("hab grad mit {first_lc} telefoniert, der {SPECIES} am weiher ist wieder da!! "
             "ruf {first_lc} mal an: {PHONE}\nps: {LAST2} kommt morgen auch mit\n"
             if lang == "de" else
             "just talked to {first_lc}, the {SPECIES} at the pond is back!! give {first_lc} a call: {PHONE}\n"
             "ps {LAST2} is coming tomorrow too\n")
        fill(doc, t, v, drop=TIMESHEET_ONLY)
        return doc

    def nature(self, lang: str) -> Doc:
        v, doc = self.vals(lang), Doc()
        t = ("Der {SPECIES} ist in Bayern stark gefährdet. Bei der Zählung am {DATE} wurden {N} Brutpaare erfasst, "
             "dazu {SPECIES2}. Die Streuobstwiese bei {CITY} ist ein wichtiges Biotop. Die Exkursion dauert {DUR}.\n"
             if lang == "de" else
             "The {SPECIES} is endangered in Bavaria. The count on {DATE} recorded {N} breeding pairs, plus {SPECIES2}. "
             "The orchard meadow near {CITY} is an important habitat. The field trip takes {DUR}.\n")
        fill(doc, t, v, drop=TIMESHEET_ONLY)
        return doc

    def booking_list(self, lang: str) -> Doc:
        doc = Doc()
        projects = [self.projects.generate(self.rng) for _ in range(self.rng.randint(2, 3))]
        for _ in range(self.rng.randint(4, 8)):
            v = self.vals(lang)
            v["PROJECT"] = (self.rng.choice(projects), "PROJECT_NAME")
            fill(doc, ("{DATE}: {TIME}-{TIME2} Projekt {PROJECT} ({PCODE}), {DUR}, Abstimmung mit {PERSON2}\n"
                       if lang == "de" else
                       "{DATE}: {TIME}-{TIME2} project {PROJECT} ({PCODE}), {DUR}, call with {PERSON2}\n"), v)
        return doc

    def timesheet_table(self, lang: str) -> Doc:
        doc = Doc()
        sep = self.rng.choice([" | ", ";", "\t"])
        projects = [(self.projects.generate(self.rng), f"PRJ-{self.rng.randint(2019, 2026)}-{self.rng.randint(1, 999):03d}")
                    for _ in range(self.rng.randint(2, 4))]
        clients = [self.fk[lang].company() for _ in range(2)]
        head = (["Datum", "Von", "Bis", "Projekt", "Projektnr.", "Kunde", "Tätigkeit", "Stunden"] if lang == "de"
                else ["Date", "From", "To", "Project", "Project no.", "Client", "Activity", "Hours"])
        doc.add(sep.join(head) + "\n")
        for _ in range(self.rng.randint(5, 12)):
            v = self.vals(lang)
            p, code = self.rng.choice(projects)
            act = self.rng.choice(ACTIVITY[lang])
            cells = [(v["DATE"][0], "DATE_TIME"), (v["TIME"][0], "CLOCK_TIME"), (v["TIME2"][0], "CLOCK_TIME"),
                     (p, "PROJECT_NAME"), (code, "PROJECT_CODE"), (self.rng.choice(clients), "ORGANIZATION")]
            for val, ent in cells:
                doc.ent(val, ent)
                doc.add(sep)
            if self.rng.random() < 0.3:
                doc.add(act + (" mit " if lang == "de" else " with "))
                doc.ent(v["PERSON2"][0], "PERSON")
            else:
                doc.add(act)
            doc.add(sep)
            doc.ent(v["DUR"][0].replace(" hours", "").replace(" h", ""), "DURATION")
            doc.add("\n")
        return doc

    def html(self, lang: str) -> Doc:
        v, doc = self.vals(lang), Doc("html")
        doc.add('<!DOCTYPE html><html><head><meta charset="utf-8"><title>Newsletter</title>'
                "<style>p{margin:0}</style></head><body>\n")
        doc.add(f'<h1 style="font-size:22px">{"Rundbrief" if lang == "de" else "Newsletter"} {self.rng.randint(1, 12):02d}/2026</h1>\n<p>')
        fill(doc, ("Am <b>{DATE}</b> lädt " if lang == "de" else "On <b>{DATE}</b> "), v, drop=TIMESHEET_ONLY)
        doc.add('<a href="mailto:')
        doc.ent(v["EMAIL"][0], "EMAIL_ADDRESS")
        doc.add('">')
        doc.ent(v["PERSON"][0], "PERSON")
        doc.add("</a>")
        fill(doc, (" zur Exkursion an den Weiher bei {CITY} ein. Wir hoffen auf den {SPECIES}!</p>\n"
                   if lang == "de" else " invites you to a field trip to the pond near {CITY}. Fingers crossed for the {SPECIES}!</p>\n"),
             v, drop=TIMESHEET_ONLY)
        doc.add("<ul>\n")
        for _ in range(self.rng.randint(1, 3)):
            vv = self.vals(lang)
            doc.add("<li>")
            fill(doc, "<strong>{PERSON}</strong> ({ORG}): {PHONE}</li>\n", vv)
        doc.add("</ul>\n<p>")
        fill(doc, ("Spenden bitte an IBAN {IBAN}." if lang == "de" else "Donations to IBAN {IBAN}."), v)
        doc.add("</p>\n<!-- ")
        fill(doc, "intern: {PERSON2}, {EMAIL}", {**v, "EMAIL": (self.vals(lang)["EMAIL"][0], "EMAIL_ADDRESS")})
        doc.add(" -->\n</body></html>\n")
        return doc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="eval/out/synthetic.jsonl")
    args = ap.parse_args()
    g = Gen(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for i in range(args.n):
            kind = KINDS[i % len(KINDS)]
            lang = "de" if (i // len(KINDS)) % 2 == 0 else "en"
            doc = getattr(g, kind)(lang)
            rec = {"id": i, "kind": kind, "language": lang, "format": doc.fmt,
                   "preset": "timesheet" if kind in ("timesheet_table", "booking_list") else "general",
                   "text": doc.text(), "spans": doc.spans}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {args.n} documents to {args.out}")


if __name__ == "__main__":
    main()
