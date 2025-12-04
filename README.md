# NLP-GuardRail
## Instructions:
For each task that has been assigned to you, create a separate python file. (So I can combine it together in an orchestration layer)
Note: You should get small models so that we can run this locally, but not too small. 
      All functions here should take a single string as input
      When you finish your task, create a pull request please, do not directly push to main.
### Input Validation Module (IVM) + Output Validation Module
#### What should be inside this Module:
  - Rule-Based System (A Regex+Dictionary to handle common problematic prompts, and to be able to set policies)
    - This should be a simple word search with a dictionary and regex that is configurable.
  - Model Classifier (Get a Pre-Trained Model that can handle classification of toxicity)
  - Model Classifier (Get a Pre-Trained model that can handle classification of Prompt Injections)
  - Named Entity Recognition to handle personal identifiable information (Named-Entity-Recognition Model)
#### What should this module handle
   - prompt injection, PII(Personal Identfiable Information, and Policy Violations (e.g., forbidden topic))
#### What should the module return?
  - Number Personal Identifiable Information Count, Prompt Injection(Yes/No), Toxicity Probability and Policy Violation Counts
### Contexualization Engine (RAG)
#### What should this module handle
  -  Optional, since this requires a large database (Wikipedia maybe?)
  -  Should grab the documents.
  - ChromeDB (Vector Database, Information Retrieval), you may use another vector database
  - You may use this model (sentence-transformers/all-MiniLM-L6-v2)
#### What should this return?
  - Retrieved documents Link, and a Function that can return a snippet of the retrieved document to use as context.
### Response Assessment Module
#### What should this module handle
- Semantic Similarity Module to use with RAG to determine hallucination (SBERT) (Takes Context and Raw LLM Input)
#### What should this return?
  - A Semantic Similarity Score
## Setting up the 
Once you have
Set your virtual environment with the following code in the terminal.
```python
python -m venv venv
```
