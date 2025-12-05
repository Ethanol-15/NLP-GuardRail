import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(current_dir, ".."))

sys.path.append(project_root)

from modules.pii_ner import PIINER

pii = PIINER()

text = (
    "My name is Shan Cabantugan and I live in Manila. "
    "You can email me at shan@gmail.com or text me at 09161234567."
)

result = pii.analyze(text, redact_person=True, redact_location=True)

print("\n=== ENTITIES ===")
for ent in result["entities"]:
    print(ent)

print("\n=== PERSON + LOCATION COUNT ===")
print(result["person_location_count"])

print("\n=== REDACTED TEXT ===")
print(result["redacted_text"])
