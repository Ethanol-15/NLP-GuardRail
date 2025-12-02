from transformers import pipeline
from .patterns import EMAIL_REGEX, PHONE_REGEX
import re

class PIINER:
    def __init__(self):
        # Load pretrained NER model
        self.ner = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple"
        )

    def detect_entities(self, text: str):
        """
        Extract PERSON, ORG, LOC, MISC using HuggingFace NER model.
        """
        return self.ner(text)

    def detect_regex(self, text: str):
        """
        Detect emails and phone numbers using regex.
        """
        emails = re.findall(EMAIL_REGEX, text)
        phones = re.findall(PHONE_REGEX, text)
        return {
            "emails": emails,
            "phones": phones
        }

    def redact(self, text: str):
        """
        Replace detected entities with redaction markers.
        """
        entities = self.detect_entities(text)
        redacted = text

        # Redact NER-based entities
        for ent in entities:
            substring = text[ent["start"]:ent["end"]]
            redacted = redacted.replace(substring, "[REDACTED]")

        # Redact email addresses
        redacted = re.sub(EMAIL_REGEX, "[REDACTED_EMAIL]", redacted)

        # Redact Philippine phone numbers
        redacted = re.sub(PHONE_REGEX, "[REDACTED_PHONE]", redacted)

        return redacted

    def analyze(self, text: str):
        """
        Unified output for the entire GuardRail pipeline.
        """
        ner_entities = self.detect_entities(text)
        regex_entities = self.detect_regex(text)

        return {
            "NER": ner_entities,
            "Regex": regex_entities,
            "HasPII": bool(
                ner_entities
                or regex_entities["emails"]
                or regex_entities["phones"]
            ),
            "RedactedText": self.redact(text)
        }
