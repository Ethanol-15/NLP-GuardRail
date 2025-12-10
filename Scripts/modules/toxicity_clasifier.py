from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class ToxicityClassifier:
    def __init__(self):
        self.model_name = "unitary/toxic-bert"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def predict_toxicity(self, text: str) -> float:
        """Returns combined toxicity probability between 0 and 1."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

        # Combine all 6 toxicity category probabilities
        toxicity_score = float(probs.detach().max())
        return toxicity_score

def toxicity_run(text: str):
    """
    Runs the toxicity module
    """
    return ToxicityClassifier().predict_toxicity(text)