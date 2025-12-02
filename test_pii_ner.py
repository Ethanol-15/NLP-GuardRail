from modules.pii_ner.pii_ner import PIINER

detector = PIINER()

sample = "My name is Shan Cabantugan and my email is shancabantugan@gmail.com and my number is 09160001234."

result = detector.analyze(sample)

print("\n=== RESULT ===")
print(result)
