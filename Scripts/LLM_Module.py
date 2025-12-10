from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from huggingface_hub import login
from torch import bfloat16
from torch.cuda import is_available,empty_cache
import gc
# The class that contains the model configuration and information and that will be used in this project.
class LLM_Module:
    default_torch_dtype = bfloat16
    llm_model_list = {
        "deepseek1.5B":"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "SeaLLMs1.5B-Chat":"SeaLLMs/SeaLLMs-v3-1.5B-Chat"}
    llm_contexts = {
        "SeaLLM_Norm":"You are a helpful assistant.",
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
    def huggingface_login(token:str):
        """
        Args:
            token(str): The token for accessing the model
        A huggingface_login gateway.
        """
        login(token)
    def __init__(self,model_name:str):
        """
        Creates an unloaded model and allows for managing of a model in memory.
        Args:
            model_name(str): The name of the model in huggingface or locally.
        """
        print(f"Using the Model: {model_name})")
        self.model_name,self.tokenizer = model_name, AutoTokenizer.from_pretrained(model_name)
        self.__model_config__ = None
        self.__bnb_config__ = None
        self.actual_model = None
    def unload_model(self):
        """
        Call this to unload the model from memory. Calls the garbage collector as well.
        """
        del self.__model_config__
        del self.__bnb_config__ 
        del self.actual_model
        del self.tokenizer
        gc.collect()
        if is_available():
            empty_cache()
    def model_config_set(self,**kwargs):
        self.__model_config__ = AutoConfig.from_pretrained(self.model_name,**kwargs)
    def quantization_4_bit_config(self,double_quant=False,quant_type="nf4"):
        """Sets up the 4-bit quantization configuration for the model.

        This function initializes a `BitsAndBytesConfig` object and stores it in
        `self.__bnb_config__`. This configuration is then used by the
        `load_model_in_mem` function to load the model in 4-bit mode.

        Args:
            double_quant: If True, enables **double quantization** (a secondary
                quantization applied to the quantization constants) to save more memory.
                Defaults to False.
            quant_type: The type of 4-bit quantization to use. Common options are
                `"nf4"` (Normal Float 4) or `"fp4"` (Float Point 4). Defaults to `"nf4"`.
        """
        self.__bnb_config__ = BitsAndBytesConfig(  
        load_in_4bit= True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype= LLM_Module.default_torch_dtype,
        bnb_4bit_use_double_quant= double_quant,
    )
    def load_model_in_mem(self,
                          dtype = default_torch_dtype,
                          device_map = "auto",
                          quant_4bit = False):
        """Loads the pre-trained language model into memory for inference.

        The model is loaded using configuration set in `self.__model_config__`. It
        supports loading the model with optional 4-bit quantization if configured
        previously.

        Args:
            dtype: The torch data type to use for the model weights (e.g., torch.float16).
                Defaults to the class's `default_torch_dtype`.
            device_map: A dictionary or string specifying how to distribute the model
                across devices (e.g., CPU, GPU). Defaults to `"auto"`.
            quant_4bit: If True and a BitsAndBytesConfig (`self.__bnb_config__`) is
                available, the model is loaded with 4-bit quantization. Defaults to False.

        Raises:
            Exception: If `self.__model_config__` is None, indicating that the model
                configuration was not set prior to calling this function.

        Note:
            This function only loads the model if `self.actual_model` is currently None.
        """
        if(self.actual_model == None):
            print(f"Loading {self.model_name} into memory.")
            if(self.__model_config__ == None):
                raise Exception("Model Config not initialized! call model_config_set first!")
            if(self.__bnb_config__ is not None and quant_4bit):
                print("Using 4-Bit_Quantized Model with the selected configuration!")
            self.actual_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config = self.__bnb_config__,
                    config = self.__model_config__,
                    dtype = dtype,
                    device_map = device_map)
        else:
            print(f"Model {self.model_name} is already loaded!")
    def prompt_model_single(self,
                      system_context:str,
                      user_prompt:str,
                      max_new_tokens=2**12,
                      temperature=0.6):
        """Generates a single, non-streaming response from the LLM.

        This function formats a request consisting of a system context and a single
        user prompt, sends it to the language model, and returns the final
        generated text.

        Args:
            system_context: The string defining the system's role, instructions,
                or background context for the model.
            user_prompt: The user's input, question, or task for the model.
            max_new_tokens: The maximum number of tokens to generate in the
                response. Defaults to 4096.
            temperature: The generation temperature (randomness) to use.
                Higher values make the output more random. Defaults to 0.6.

        Returns:
            str: The decoded text response generated by the language model.

        Raises:
            Exception: If `self.actual_model` is not loaded (i.e., is `None`).
        """
        if(self.actual_model is None):
            raise Exception("Model is not loaded!")
        messages =  [{"role": "system", "content": system_context},
                  {"role": "user", "content": user_prompt}]
        llm_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([llm_input], return_tensors="pt").to(self.actual_model.device)
        generated_ids = self.actual_model.generate(
                **model_inputs,
                temperature=temperature,
                max_new_tokens= max_new_tokens
            )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        return self.tokenizer.decode(output_ids,skip_special_tokens=True)
      
    def chat_model(self,user_prompt:str,
                   max_new_tokens:int=2**12,
                   temperature:float = 0.6,
                   system_context:str|None = None,
                   messages:dict|None = None):
        """Generates a streaming response from the LLM based on user input and configuration.

        This method manages the conversation history, applies the chat template,
        tokenizes the input, and generates a streamed response using the language
        model with specified generation parameters. The conversation history is
        updated with the new user prompt and the model's response.

        Args:
            user_prompt: The latest input or question provided by the user.
            max_new_tokens: The maximum number of tokens the model is allowed to
                generate in the response. Defaults to 4096.
            temperature: The randomness parameter for text generation. Higher values
                increase creativity but lower coherence. Defaults to 0.6.
            system_context: An optional string defining the system's role or
                behavior. If `messages` is None, this is used to initialize the
                conversation history.
                Defaults to None.
            messages: An optional list of dictionaries representing the existing
                conversation history. Each dictionary must contain 'role' and 'content' keys.
                If None, a new conversation is started using `system_context`.
                Defaults to None.

        Returns:
            A tuple containing:
                * messages (list): The updated list of conversation messages,
                including the user prompt and the model's complete response.
                * response (str): The final, concatenated text response generated by the LLM.

        Raises:
            Exception: If `self.actual_model` is not loaded (None).
            Exception: If both `system_context` and `messages` are None, as an
                initial context or history is required to start the conversation.

        """
        if(self.actual_model is None):
            raise Exception("Model is not loaded!")
        if(system_context is None and messages is None):
            raise Exception("Insert a message or a system context first!")
        if(messages is None):
            messages = [
                {"role": "system", "content":system_context},
            ]
        messages.append({"role": "user", "content": user_prompt})
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.actual_model.device)
        
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generated_ids = self.actual_model.generate(**model_inputs,
                                                   do_sample = True, 
                                                   max_new_tokens=max_new_tokens,
                                                   temperature=temperature, 
                                                   streamer=streamer)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        messages.append({"role": "assistant", "content": response})
        return messages, response
