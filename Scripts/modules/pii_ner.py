from transformers import pipeline
import re


class PIINER:

    EMAIL_REGEX = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    PH_PHONE_REGEX = r"(09\d{9})"

    def __init__(self):
        self.ner = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple"
        )

    def detect_entities(self, text: str):
        return self.ner(text)

    def count_persons_and_locations(self, entities):
        count = 0
        for ent in entities:
            if ent["entity_group"] in ["PER", "LOC"]:
                count += 1
        return count
    
    def redact_entities(
        self,
        text: str,
        entities,
        redact_person: bool = True,
        redact_location: bool = True
    ):
        redacted_text = text

        # Sort entities from right to left to avoid shifting indexes
        entities = sorted(entities, key=lambda e: e["start"], reverse=True)

        for ent in entities:
            ent_type = ent["entity_group"]
            start, end = ent["start"], ent["end"]

            if ent_type == "PER" and redact_person:
                replacement = "[REDACTED_PERSON]"
            elif ent_type == "LOC" and redact_location:
                replacement = "[REDACTED_LOCATION]"
            else:
                continue

            # Expand to full words (avoid leftover characters)
            while start > 0 and redacted_text[start - 1].isalnum():
                start -= 1
            while end < len(redacted_text) and redacted_text[end:end+1].isalnum():
                end += 1

            redacted_text = redacted_text[:start] + replacement + redacted_text[end:]

        return redacted_text
   
    def redact_regex(self, text: str):
        text = re.sub(self.EMAIL_REGEX, "[REDACTED_EMAIL]", text)
        text = re.sub(self.PH_PHONE_REGEX, "[REDACTED_PH_PHONE]", text)
        return text
    
    def analyze(
        self,
        text: str,
        redact_person: bool = True,
        redact_location: bool = True
    ):
        entities = self.detect_entities(text)
        person_location_count = self.count_persons_and_locations(entities)

        redacted = self.redact_entities(
            text,
            entities,
            redact_person=redact_person,
            redact_location=redact_location
        )

        redacted = self.redact_regex(redacted)

        return {
            "entities": entities,
            "person_location_count": person_location_count,
            "redacted_text": redacted
        }
