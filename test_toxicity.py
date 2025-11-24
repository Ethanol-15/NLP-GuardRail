from modules.toxicity_clasifier import ToxicityClassifier

model = ToxicityClassifier()
score = model.predict_toxicity("You filthy piece of trash, I hope you kill yourself")
print("Toxicity Score:", score)