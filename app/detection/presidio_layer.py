"""Layers L0 (patterns, checksums, dictionaries) and L1 (spaCy NER), both run
through one Presidio AnalyzerEngine."""

from __future__ import annotations

import logging

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer import predefined_recognizers as pr
from presidio_analyzer.nlp_engine import NlpEngineProvider

from ..config import Settings
from ..models import Span
from ..text_utils import Chunk, detect_language, split_blocks
from .recognizers import (
    DictionaryRecognizer,
    address_recognizers,
    RegexGroupRecognizer,
    read_list,
    special_category_recognizers,
    timesheet_recognizers,
)

log = logging.getLogger(__name__)

SPACY_ENTITIES = ["PERSON", "LOCATION", "ORGANIZATION"]

# Language-independent predefined recognizers (instantiated once per language).
GENERIC = [
    pr.EmailRecognizer, pr.IbanRecognizer, pr.CreditCardRecognizer, pr.IpRecognizer,
    pr.UrlRecognizer, pr.CryptoRecognizer, pr.MacAddressRecognizer,
]
# Country-specific ones are registered for every language: a German ID number
# can appear in English text and vice versa.
COUNTRY = [
    pr.DeTaxIdRecognizer, pr.DeTaxNumberRecognizer, pr.DePassportRecognizer, pr.DeIdCardRecognizer,
    pr.DeSocialSecurityRecognizer, pr.DeHealthInsuranceRecognizer, pr.DeKfzRecognizer,
    pr.DeHandelsregisterRecognizer, pr.DePlzRecognizer, pr.DeVatIdRecognizer,
    pr.DeFuehrerscheinRecognizer, pr.UsSsnRecognizer, pr.NhsRecognizer, pr.UkNinoRecognizer,
    pr.UkPostcodeRecognizer,
]


class PresidioLayer:
    name = "patterns+spacy"

    def __init__(self, settings: Settings):
        self.settings = settings
        langs = settings.languages
        cfg = settings.layers.spacy
        nlp_conf = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": lang, "model_name": cfg.models[lang]} for lang in langs],
            "ner_model_configuration": {
                "model_to_presidio_entity_mapping": {
                    "PER": "PERSON", "PERSON": "PERSON", "LOC": "LOCATION", "GPE": "LOCATION",
                    "FAC": "LOCATION", "ORG": "ORGANIZATION",
                },
                "labels_to_ignore": ["MISC", "NORP", "DATE", "TIME", "CARDINAL", "EVENT",
                                     "LANGUAGE", "LAW", "MONEY", "ORDINAL", "PERCENT",
                                     "PRODUCT", "QUANTITY", "WORK_OF_ART"],
                "low_score_entity_names": cfg.low_confidence_entities,
                "low_confidence_score_multiplier": cfg.low_confidence_multiplier,
            },
        }
        nlp_engine = NlpEngineProvider(nlp_configuration=nlp_conf).create_engine()
        registry = RecognizerRegistry(supported_languages=langs)
        for lang in langs:
            for rec in self._recognizers(lang):
                registry.add_recognizer(rec)
        self.analyzer = AnalyzerEngine(registry=registry, nlp_engine=nlp_engine,
                                       supported_languages=langs)

    def _recognizers(self, lang: str):
        s = self.settings
        recs = [cls(supported_language=lang) for cls in GENERIC + COUNTRY]
        recs.append(pr.PhoneRecognizer(supported_language=lang,
                                       supported_regions=s.layers.patterns.phone_regions))
        recs.append(pr.SpacyRecognizer(supported_language=lang, supported_entities=SPACY_ENTITIES))
        for p in s.layers.patterns.custom_patterns:
            if p.languages is None or lang in p.languages:
                recs.append(RegexGroupRecognizer(p.name, p.entity, p.pattern, p.group, p.score, lang))
        recs.extend(timesheet_recognizers(lang))
        recs.extend(address_recognizers(lang))
        projects = read_list(s.path(s.dictionaries.project_names))
        if projects:
            recs.append(DictionaryRecognizer("project_dictionary", "PROJECT_NAME", projects, lang, 0.95))
        recs.extend(special_category_recognizers(s.path(s.dictionaries.special_category), [lang]))
        return recs

    def blocks(self, text: str) -> list[Chunk]:
        """Paragraph blocks with detected language; adjacent same-language
        paragraphs are merged."""
        s = self.settings
        out: list[Chunk] = []
        for b in split_blocks(text, 0):
            lang = detect_language(b.text, s.languages, s.default_language)
            if out and out[-1].language == lang and len(out[-1].text) + len(b.text) <= 20000:
                out[-1] = Chunk(out[-1].start, out[-1].text + b.text, lang)
            else:
                out.append(Chunk(b.start, b.text, lang))
        return out

    def detect(self, text: str, language: str | None = None) -> list[Span]:
        blocks = [Chunk(0, text, language)] if language else self.blocks(text)
        spans: list[Span] = []
        for b in blocks:
            for r in self.analyzer.analyze(text=b.text, language=b.language):
                rec = (r.recognition_metadata or {}).get("recognizer_name", "")
                source = "spacy" if rec == "SpacyRecognizer" else "patterns"
                s, e = b.start + r.start, b.start + r.end
                spans.append(Span(s, e, r.entity_type, float(r.score), {source}, text[s:e]))
        return spans

    def languages_used(self, text: str) -> list[str]:
        return sorted({b.language for b in self.blocks(text)})
