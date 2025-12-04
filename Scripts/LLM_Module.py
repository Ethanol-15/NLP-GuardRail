from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from huggingface_hub import login
from torch import bfloat16
from torch.cuda import is_available,empty_cache
import gc
from os import sep
from torch import cat, ones_like
# The class that contains the model configuration and information and that will be used in this project.
class llm_module:
    default_torch_dtype = bfloat16
    llm_model_list = {
        "deepseek1.5B":"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "SeaLLMs1.5B-Chat":"SeaLLMs/SeaLLMs-v3-1.5B-Chat"}
    llm_contexts = {
        "LLM_"
        "LLM"
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
        Creates an unloaded model and allows
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
        del self.tokenizer.padding_side
        del self.__model_config__
        del self.__bnb_config__ 
        del self.actual_model
        del self.tokenizer
        gc.collect()
        if is_available():
            empty_cache()
    def model_config_set(self,**kwargs):
        self.__model_config__ = AutoConfig.from_pretrained(self.model_name,**kwargs)
    def quantization_config(self,double_quant=False,quant_type="nf4",load_in_4bit = False,load_in_8bit = False):
        if(load_in_4bit == True and load_in_8bit == True):
            raise Exception("Load_in_4bit or Load_in_8bit cannot be both true")
        self.__bnb_config__ = BitsAndBytesConfig(  
        load_in_4bit= load_in_4bit,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype= llm_module.default_torch_dtype,
        bnb_4bit_use_double_quant= double_quant,
    )
    def load_model_in_mem(self,dtype = default_torch_dtype,
                          device_map = "auto",
                          quant_4bit = False):
        """
            Loads the model into memory.
            Args:
                dtype
            """
        def load_preTrained():
            print(f"Loading {self.model_name} into memory.")
            if(self.__model_config__ == None):
                raise Exception("Model Config not initialized! call model_config_set first!")
            if(self.__bnb_config__ is not None and quant_4bit):
                print("Using 4-Bit_Quantized Model with the selected configuration!")
            return AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config = self.__bnb_config__,
                    config = self.__model_config__,
                    dtype = dtype,
                    device_map = device_map)
        if(self.actual_model == None):
            self.actual_model = load_preTrained()
        else:
            print(f"Model {self.model_name} is already loaded!")
    def prompt_model_single(self,
                      system_context:str,
                      user_prompt:str,
                      hidden_prompt:str =""
                     ,thinking_budget =2**10
                     ,max_new_tokens=2**12,
                      temperature=0.6,
                      thinking= True,
                      return_all = True):
        if(self.actual_model is None):
            raise Exception("Model is not loaded!")
        if(thinking_budget+26 >= max_new_tokens and thinking):
            raise Exception(f"Thinking Budget too high!{thinking_budget}")
        messages =  [{"role": "system", "content": system_context},
                  {"role": "user", "content": user_prompt+hidden_prompt}]
        llm_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([llm_text], return_tensors="pt").to(self.actual_model.device)
        input_length = model_inputs.input_ids.size(-1)
        if(thinking):
            generated_ids = self.actual_model.generate(
                **model_inputs,
                temperature=temperature,
                max_new_tokens= thinking_budget
            )

            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

            # 151643 is the end Token
            if 151643 not in output_ids:
                # 151649 is the thinking end token
                if 151649 not in output_ids:
                    print("Thinking budget has been reached!")
                    early_stopping_text = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
                    early_stopping_ids = self.tokenizer([early_stopping_text], return_tensors="pt", return_attention_mask=False).input_ids.to(self.actual_model.device)
                    input_ids = cat([generated_ids, early_stopping_ids], dim=-1)
                else:
                    input_ids = generated_ids
                attention_mask = ones_like(input_ids)

                generated_ids = self.actual_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        temperature=temperature,
                        max_new_tokens= input_length + max_new_tokens - input_ids.size(-1)  # could be negative if max_new_tokens is not large enough (early stopping text is 24 tokens)
                    )
                output_ids = generated_ids[0][input_length:].tolist()
        else:
            generated_ids = self.actual_model.generate(
                **model_inputs,
                temperature=temperature,
                max_new_tokens= max_new_tokens
            )

            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        if(return_all):
            return self.tokenizer.decode(output_ids,skip_special_tokens=True)
        # Attempt to parse thinking content
        try:
            index = len(output_ids) - output_ids[::-1].index(151649)
        except ValueError:
            index = 0
        
        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return thinking_content, content
    def chat_mode(self,system_context:str,messages:dict|None = None):
        if(messages is None):
            messages = [
                {"role": "system", "content":system_context},
            ]
        while True:
            prompt = input("User:")
            messages.append({"role": "user", "content": prompt})
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.actual_model.device)
            
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            generated_ids = self.actual_model.generate(model_inputs.input_ids, max_new_tokens=512, streamer=streamer)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            messages.append({"role": "assistant", "content": response})

