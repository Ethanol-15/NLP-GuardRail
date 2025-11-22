# NLP-GuardRail
## Instructions:
Note: You should get small models so that we can run this...
1) Input Validation Module (IVM) + Output Validation Module
- What should be inside this Module
  - Rule-Based System (A Regex+Dictionary to handle common problematic prompts, and to be able to set policies)
    - This should be a simple word search with a dictionary and regex that is configurable.
  - Model Classifier (Get a Pre-Trained Model that can handle classification of toxicity)
  - Model Classifier (Get a Pre-Trained model that can handle classification of Prompt Injections)
  - Named Entity Recognition to handle personal identifiable information (Named-Entity-Recognition Model)
- What should this module handle
   - prompt injection, PII(Personal Identfiable Information, and Policy Violations (e.g., forbidden topic))
- What should the module return?
  - Number Personal Identifiable Information Count, Prompt Injection(Yes/No), Toxicity Probability and Policy Violation Counts
2) Contexualization Engine (RAG)
  -  Optional, since this requires a large database (Wikipedia maybe?)
  -  Should grab the documents.
  - ChromeDB (Vector Database, Information Retrieval), you may use another vector database
  - You may use this model (sentence-transformers/all-MiniLM-L6-v2)
What should this return?
  - retrieved documents Link, and a Function that can return a snippet of the retrieved document to use as context.
3) Response Assessment Module
- Semantic Similarity Module to use with RAG to determine hallucination (SBERT) (Takes Context and Raw LLM Input)
What should this return?
  - A semantic similarity Score
