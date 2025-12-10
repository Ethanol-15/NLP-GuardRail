from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
import ast
import datasets

class MaliciousLLMPrompts:
    """
        All of the Malicious LLM prompts used for the project.
        
        Each dataset is a huggingface dataset with the following columns/features:
        - prompt: The prompt text
        - attack_type: The type of attack (if applicable)
        - malicious: Whether the prompt is malicious or not

        Dataset Sizes:
        - Fil: 1009 
        - EN: 987 
        - codesagar: 1758 
        - Combined (all datasets): 3754
        """

    def __init__(self):
        """
        Initializes all the datasets used in this class.
        """
        
        # prompts from the red teaming dataset
        ds_redTeaming = self.__load_redTeaming__()
        self.ds_filipino = ds_redTeaming["filipino"]
        self.ds_pure_english = ds_redTeaming["english"]
        
        # prompts from the german/english codesagar/malicious-llm-prompts dataset
        # important note: some of the data does not have a proper attack type
        self.ds_codesagar = self.__load_codesagar__() # combination of english and german prompts
        
        # combined dataset if you want to evaluate it on the full dataset
        self.combined_datasets = concatenate_datasets([self.ds_codesagar, self.ds_filipino, self.ds_pure_english]) # type: ignore
    

    ### ============== Internal Implementation =============== ###

    def __load_codesagar__(self):
        """
        Loads and processes the malicious LLM prompts dataset from codesagar/malicious-llm-prompts.

        Has the following columns/features:
        - prompt: The prompt text
        - attack_type: The type of attack (if applicable)
        - malicious: Whether the prompt is malicious or not
        """
        ds = load_dataset("codesagar/malicious-llm-prompts")
        full_ds = concatenate_datasets([ds["train"], ds["validation"], ds["test"]]) # type: ignore

        # Function fill None values with empty string to avoid null errors
        def fillNone(example):
            if example['attack_type'] is None:
                return {'attack_type': ""}
            return example

        # we'll then only use the responses
        # full_ds = full_ds.filter(lambda example: example['malicious'] == True)
        full_ds = full_ds.map(fillNone)
        reduced_dataset = full_ds.select_columns(['prompt', 'attack_type', 'malicious'])
        # dataset.filter(lambda example: example['malicious'] == False)
        return reduced_dataset
    
    def __load_redTeaming__(self):
        """
        Loads and processes the read teaming LLM prompts dataset from CohereLabs/aya_redteaming.
        See Ahmadian et al., 2024 for more details.

        Has the following columns/features:
        - prompt: The prompt text
        - attack_type: The type of attack (if applicable)
        - malicious: All of these prompts are malicious
        """
        # full dataset
        ds = load_dataset("CohereLabs/aya_redteaming", "default")

        # Function to join the list elements
        def flatten_attack_type(example):
            # Get the list
            categories = ast.literal_eval(example['attack_type'])
            
            prompt_format = {
                'prompt': example['prompt'],
                'attack_type': "",
                'malicious': True # add this column
            }

            # Safety check: ensure it's not None before joining
            # if categories is None:
            #     return {'attack_type': ""}
            
            if categories is not None:
                prompt_format['attack_type'] = " | ".join(categories)
            return prompt_format
            # # Join with a comma and space (or just a space " " if preferred)
            # return {'attack_type': " | ".join(categories)}
        
        # since it's a dataset Dictionary, we'll need to only get two of these
        prompts = {
            "filipino": ds["filipino"]\
                .rename_column("harm_category", "attack_type")\
                .map(flatten_attack_type)\
                .select_columns(["prompt", "attack_type", 'malicious']),

            "english": ds["english"]\
                .rename_column("harm_category", "attack_type")\
                .map(flatten_attack_type)\
                .select_columns(["prompt", "attack_type", 'malicious'])
        }
        return prompts
    

    ### ============== Getters =============== ###

    # just to make it cleaner

    def get_filipino_dataset(self):
        """
        Returns the filipino dataset from RedTeaming.
        """
        return self.ds_filipino
    
    def get_english_dataset(self):
        """
        Returns the english dataset from RedTeaming.
        """
        return self.ds_pure_english
    
    def get_basic_prompts_dataset(self):
        """
        Returns the basic harmful prompts dataset from the codesagar/malicious-llm-prompts dataset.
        """
        return self.ds_codesagar
    
    def get_combined_dataset(self):
        """
        Returns the combined dataset of all the datasets.
        """
        return self.combined_datasets

    ### ============== Additional Functions =============== ###
    @staticmethod
    def create_torch_dataloader(dataset : datasets.arrow_dataset.Dataset, batch_size=1, shuffle=True):
        """
        Creates a torch dataloader from the loaded dataset.
        Params:
            dataset (Dataset): The dataset to be converted to a dataloader.
            batch_size (int, optional): The batch size for the dataloader. Defaults to 1.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        """
        dataset.set_format(type='torch')
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) # type: ignore