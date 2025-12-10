from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

class PromptInjection:
    """
    A classifier for detecting prompt injection attempts in user-provided text.

    This class loads the `ProtectAI/deberta-v3-base-prompt-injection` model
    and provides a simple API for running inference to determine whether input
    text contains prompt injection behaviors. The model outputs a label and
    confidence score indicating the likelihood of injection.

    Attributes
    ----------
    model_name : ProtectAI/deberta-v3-base-prompt-injection
        The HuggingFace model identifier used for tokenization and inference.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer used to preprocess input text into model-readable tensors.
    model : transformers.PreTrainedModel
        The underlying Transformer model fine-tuned for prompt injection detection.
    classifier : transformers.Pipeline
        A HuggingFace inference pipeline that wraps the model and tokenizer with
        preprocessing and postprocessing utilities for text classification.
    """

    def __init__(self):
        """
        Initialize the prompt injection classifier.

        This loads the tokenizer, model, and text-classification pipeline into
        memory.
        """
        self.model_name = "ProtectAI/deberta-v3-base-prompt-injection"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            truncation=True,
            max_length=512,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    def predict_injection(self, text: str):
        """
        Predict whether the given text contains prompt injection behavior.

        The output is a list containing a dictionary with:
        - `label`: either "INJECTION" or "NO_INJECTION"
        - `score`: a confidence value between 0 and 1
        Example - [{'label': 'INJECTION', 'score': 0.9923}]
        """
        return self.classifier(text)

