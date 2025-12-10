import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from modules.pii_ner import PIINER

pii = PIINER()

text = (
    "My name is Shan Cabantugan. I live in Manila. "
    "My email is shan@gmail.com, phone is 09161234567, "
    "username is shannyboy123, ID number AB-123456."
)

result = pii.analyze(text)

print("\n=== ENTITIES ===")
for ent in result["entities"]:
    print(ent)

print("\n=== PII COUNT ===")
print(result["pii_count"])

print("\n=== REDACTED TEXT ===")
print(result["redacted_text"])
