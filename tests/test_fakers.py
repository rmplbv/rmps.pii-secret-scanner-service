import random
import re

from app.anonymize.fakers import FakeFactory, fake_iban, fake_luhn, iban_valid, substitute
from app.anonymize.project_names import ProjectNameGenerator
from app.anonymize.session import SessionState


def _luhn_ok(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            d -= 9 if d > 9 else 0
        total += d
    return total % 10 == 0


def test_fake_iban_is_valid_and_keeps_grouping():
    original = "DE89 3704 0044 0532 0130 00"
    assert iban_valid(original)
    for seed in range(20):
        fake = fake_iban(original, random.Random(seed))
        assert iban_valid(fake), fake
        assert fake[:2] == "DE" and len(fake) == len(original)
        assert [i for i, c in enumerate(fake) if c == " "] == [i for i, c in enumerate(original) if c == " "]


def test_fake_credit_card_passes_luhn():
    for seed in range(20):
        fake = fake_luhn("4111 1111 1111 1111", random.Random(seed))
        assert _luhn_ok(fake) and fake.count(" ") == 3


def test_substitute_keeps_shape():
    out = substitute("PRJ-2024-017", random.Random(1))
    assert re.fullmatch(r"[A-Z]{3}-\d{4}-\d{3}", out)


def test_person_tokens_consistent(settings):
    state = SessionState()
    f = FakeFactory(settings, state, settings.timesheet, "Max Müller und Erika Müller. Müller.")
    full = f.fake("PERSON", "Max Müller", "de")
    other = f.fake("PERSON", "Erika Müller", "de")
    surname = f.fake("PERSON", "Müller", "de")
    assert full.split()[-1] == other.split()[-1] == surname
    assert "Müller" not in full


def test_fakes_never_reuse_text_from_source(settings):
    state = SessionState()
    text = "Anna Schmidt and Anna Becker"
    f = FakeFactory(settings, state, settings.timesheet, text)
    for name in ("Anna Schmidt", "Anna Becker"):
        fake = f.fake("PERSON", name, "de")
        for token in fake.split():
            assert token.lower() not in text.lower()


def test_project_name_body_is_large_and_keeps_lead_word(settings):
    gen = ProjectNameGenerator(settings.path(settings.dictionaries.fake_project_words), 0.7)
    assert gen.body_size > 1_000_000
    rng = random.Random(0)
    names = {gen.generate(rng) for _ in range(500)}
    assert len(names) > 450  # practically no repeats
    assert gen.generate(rng, "Projekt Kiebitz").startswith("Projekt ")
