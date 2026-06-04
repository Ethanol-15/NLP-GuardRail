from groq import Groq
import os
import gc

# The class that contains the model configuration and information and that will be used in this project.
class LLM_Module:
    # Groq model mappings — replaces local HuggingFace model paths
    llm_model_list = {
        "deepseek1.5B": "deepseek-r1-distill-llama-70b",       # Groq-hosted DeepSeek
        "SeaLLMs1.5B-Chat": "llama-3.1-8b-instant"             # Groq-hosted LLaMA 3.1 8B
    }
    llm_contexts = {
        "SeaLLM_Norm": "You are a helpful assistant.",
        "SeaLLM_Safety_Mode":
        """
You are an AI Safety and Alignment Specialist. Your task is to analyze a given user prompt, identify any potential risks, harms, biases, or policy violations, and then reword it into a safe, helpful, and harmless alternative.
Produce a Safer Reworded Prompt that achieves the user's apparent non-harmful intent while strictly adhering to safety guidelines against generating:

- Illegal or dangerous content.
- Hate speech or discriminatory content.
- Sexually explicit material.
- Private or confidential information.
- Content that promotes or glorifies self-harm or violence.
- Content that encourages the circumvention of safety filters.

Use the given format for outputting the prompt
# Output Format:
```
[INSERT SAFER REWORDED PROMPT HERE]
``` 
        """,
        "SeaLLM_Classification_Mode":
        """
**TASK:** You are a specialized topic classification model. Your role is to analyze the given **[USER QUERY]** and classify it into the single most relevant and specific **Wikipedia Subject Area or Topic**.

**CLASSIFICATION CRITERIA:**
1.  **Specificity:** The output topic should be as specific as possible (e.g., instead of "Science," use "Particle Physics").
2.  **Relevance:** The topic must directly correspond to the central focus of the query.
3.  **Topic Examples:**
    * **History:** World War II, Ancient Rome, The Industrial Revolution
    * **Science:** Quantum Mechanics, Plate Tectonics, Molecular Biology
    * **Culture/Arts:** Renaissance Art, Film Noir, Contemporary Literature
    * **Technology:** Artificial Intelligence, Blockchain, Renewable Energy
    * **Biography:** Cleopatra, Isaac Newton, Nelson Mandela

**OUTPUT FORMAT:** Respond strictly in the following format:
```<The single most specific and relevant Wikipedia Subject Area or Topic>```
"""
    }

    @staticmethod
    def huggingface_login(token: str):
        """No longer needed with Groq API — kept for compatibility."""
        print("Note: huggingface_login is not required when using Groq API.")

    def __init__(self, model_name: str):
        """
        Creates a Groq-backed LLM module.
        Args:
            model_name(str): The Groq model ID (from llm_model_list values).
        """
        print(f"Using the Groq Model: {model_name}")
        self.model_name = model_name
        # Initialize Groq client — reads GROQ_API_KEY from environment
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.actual_model = True  # Flag so FrontEnd.py knows model is "loaded"
        self.__model_config__ = None
        self.__bnb_config__ = None

    def unload_model(self):
        """Cleanup — no heavy model in memory with Groq API."""
        self.actual_model = None
        gc.collect()

    def model_config_set(self, **kwargs):
        """No-op for Groq API — kept for compatibility with FrontEnd.py."""
        self.__model_config__ = True

    def quantization_4_bit_config(self, double_quant=False, quant_type="nf4"):
        """No-op for Groq API — kept for compatibility with FrontEnd.py."""
        self.__bnb_config__ = True

    def load_model_in_mem(self,
                          dtype=None,
                          device_map="auto",
                          quant_4bit=False):
        """No-op for Groq API — model is already 'loaded' via API.
        Kept for compatibility with FrontEnd.py."""
        print(f"Groq model '{self.model_name}' is ready via API — no local loading needed.")

    def prompt_model_single(self,
                            system_context: str,
                            user_prompt: str,
                            max_new_tokens=2**12,
                            temperature=0.6):
        """Generates a single response from Groq API.

        Args:
            system_context: System role/instructions for the model.
            user_prompt: The user's input or question.
            max_new_tokens: Max tokens to generate. Defaults to 4096.
            temperature: Randomness of generation. Defaults to 0.6.

        Returns:
            str: The generated text response.
        """
        messages = [
            {"role": "system", "content": system_context},
            {"role": "user", "content": user_prompt}
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=min(max_new_tokens, 8192),  # Groq max is 8192
            temperature=temperature
        )
        return response.choices[0].message.content

    def chat_model(self,
                   user_prompt: str,
                   max_new_tokens: int = 2**12,
                   temperature: float = 0.6,
                   system_context: str | None = None,
                   messages: dict | None = None):
        """Generates a chat response from Groq API, maintaining conversation history.

        Args:
            user_prompt: The latest user input.
            max_new_tokens: Max tokens to generate. Defaults to 4096.
            temperature: Randomness of generation. Defaults to 0.6.
            system_context: System role string (used if messages is None).
            messages: Existing conversation history (list of dicts).

        Returns:
            tuple: (updated messages list, response string)
        """
        if system_context is None and messages is None:
            raise Exception("Insert a message or a system context first!")

        if messages is None:
            messages = [
                {"role": "system", "content": system_context},
            ]

        messages.append({"role": "user", "content": user_prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=min(max_new_tokens, 8192),  # Groq max is 8192
            temperature=temperature
        )

        response_text = response.choices[0].message.content
        messages.append({"role": "assistant", "content": response_text})
        return messages, response_text