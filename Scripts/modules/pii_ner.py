import os
import re
from gliner import GLiNER


class PIINER:
    """
    PII Detection and Redaction module using GLiNER Multi-v2.1.
    Detects PII entities (name, location, email, phone, etc.),
    counts them, and optionally redacts them.
    """

    EMAIL_REGEX = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    PH_PHONE_REGEX = r"(09\d{9})"

    # PII entity types supported by GLiNER Multi-v2.1
    PII_LABELS = {
        
    }

    def __init__(self):
        # Prevent symlink errors in Windows cache
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

        self.model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")

    def detect_entities(self, text: str):
        """
        Detect GLiNER PII entities.

        Returns:
            list: Each item contains {text, label, start, end, score}
        """
        return self.model.predict_entities(text, list(self.PII_LABELS))

    def count_pii(self, entities):
        """
        Count only entities classified as PII.
        """
        return sum(1 for ent in entities if ent["label"] in self.PII_LABELS)

    def redact_entities(
        self,
        text: str,
        entities,
        redact_person=True,
        redact_location=True,
        redact_other=True
    ):
        """
        Redact detected PII entities.
        Selective flags let you choose what to redact.
        """
        redacted = text
        offset = 0

        for ent in entities:
            label = ent["label"]
            start, end = ent["start"], ent["end"]

            if label == "person" and not redact_person:
                continue
            if label == "location" and not redact_location:
                continue
            if label not in {"person", "location"} and not redact_other:
                continue

            replacement = f"[REDACTED_{label.upper()}]"

            redacted = (
                redacted[: start + offset]
                + replacement
                + redacted[end + offset :]
            )

            offset += len(replacement) - (end - start)

        return redacted

    def redact_regex(self, text: str):

        text = re.sub(self.EMAIL_REGEX, "[REDACTED_EMAIL]", text)
        text = re.sub(self.PH_PHONE_REGEX, "[REDACTED_PH_PHONE]", text)
        return text

    def analyze(
        self,
        text: str,
        redact_person=True,
        redact_location=True,
        redact_other=True
    ):
        """
        Run full PII detection + counting + redaction.

        Returns:
            dict {
                "entities": [...],
                "pii_count": int,
                "redacted_text": str
            }
        """
        entities = self.detect_entities(text)
        pii_count = self.count_pii(entities)

        redacted = self.redact_entities(
            text,
            entities,
            redact_person=redact_person,
            redact_location=redact_location,
            redact_other=redact_other
        )

        redacted = self.redact_regex(redacted)

        return {
            "entities": entities,
            "pii_count": pii_count,
            "redacted_text": redacted
        }
