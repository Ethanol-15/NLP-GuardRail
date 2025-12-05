from modules.toxicity_clasifier import toxicity_run
from LLM_Module import llm_module
# Import patterns here.
import logging
logging.basicConfig(filename='results.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
class LLMFilteredException(Exception):
    def __init__(self,reason:str,message="Filtered."):
        self.message = message
        self.reason = reason
        super().__init__(self.message)
class OrchestratorConfiguration:
    """Configures the orchestrator's modules sensitivities.

    This class holds the threshold scores and weights for various guardrail 
    checks, determining when a query or response should be flagged or blocked.
    """
    def __init__(
        self,
        banned_word_allcase:bool = True,
        threshold_overall_score:float = 0.7,
        # Thresholds, if not passed immediately filters the content. If negative, is disabled.
        threshold_banned_word_tolerance:int = 3,
        threshold_prompt_injection:float = 0.5,
        threshold_rag_context_distance:int = 0,
        threshold_semantic_similarity_threshold:float = 0.5,
        threshold_toxicity:float = 0.5,
        # Weights, affect the overall score if detected, setting to 0 disables it.
        weight_banned_word:float = 1,
        weight_prompt_injection:float = 1,
        weight_semantic_similarity_threshold:float = 1,
        weight_toxicity:float = 1,
        threshold_llm_rephrase:float = 0.5,
                 ):
        """Initializes the orchestrator's configuration parameters.

        Args:
            banned_word_allcase: If True, banned word checks are case-insensitive.
            threshold_overall_score: The final combined score above which the content is flagged.
            threshold_banned_word_tolerance: Maximum number of banned words allowed before filtering.
            threshold_prompt_injection: Score to flag potential Prompt Injection attempts (0.0 to 1.0).
            threshold_rag_context_distance: Max distance for RAG context relevance check. 0 disables the check.
            threshold_semantic_similarity_threshold: Score to flag semantically similar queries to known bad ones (0.0 to 1.0).
            threshold_toxicity: Score to flag toxic or harmful content (0.0 to 1.0).
            weight_banned_word: Weight applied to the banned word score for overall calculation.
            weight_prompt_injection: Weight applied to the prompt injection score.
            weight_semantic_similarity_threshold: Weight applied to the semantic similarity score.
            weight_toxicity: Weight applied to the toxicity score.
            threshold_llm_rephrase: Score threshold for filtering LLM rephrased output.
        """
        self.banned_word_allcase = banned_word_allcase
        self.banned_words = []
        self.threshold_overall_score = threshold_overall_score
        self.threshold_banned_word_tolerance = threshold_banned_word_tolerance
        self.threshold_llm_rephrase = threshold_llm_rephrase
        self.threshold_prompt_injection = threshold_prompt_injection
        self.threshold_RAG_context_distance = threshold_rag_context_distance
        self.threshold_semantic_similarity_threshold = threshold_semantic_similarity_threshold
        self.threshold_toxicity = threshold_toxicity
        self.regex = []
        self.weight_banned_word = weight_banned_word
        self.weight_prompt_injection = weight_prompt_injection
        self.weight_semantic_similarity_threshold = weight_semantic_similarity_threshold
        self.weight_toxicity = weight_toxicity
        self.sanity_check()
    def add_banned_words(self,banned_words:list):
        self.banned_words.extend(banned_words)
    def add_regex(self,regex:list):
        self.regex = regex.extend(regex)
    def default_english_regex(self):
        self.regex = []
        self.banned_words = []
    def sanity_check(self):
        error_message = ""
        if(self.threshold_toxicity < 0 or self.threshold_toxicity > 1):
            error_message.join(f"Toxicity threshold must be < 0 and > 1, found {self.toxicity_threshold}\n") 
        if(self.threshold_overall_score <0 or self.threshold_overall_score > 1):
            error_message.join(f"Overall score must be < 0 and > 1, found {self.threshold_overall_score}\n")
        if(self.threshold_banned_word_tolerance is not int):
            error_message.join(f"Banned word tolerance threshold must be an integer!, found {self.threshold_banned_word_tolerance}\n")
        if(self.threshold_prompt_injection is not int):
            error_message.join(f"Prompt injection tolerance threshold must be an integer!, found {self.threshold_banned_word_tolerance}\n")
        if(self.threshold_banned_word_tolerance):
            error_message.join(f"Banned word tolerance threshold")
        if(len(error_message>0)):
            raise Exception(f"Configuration failed, with the following errors:\n{error_message}")
    @staticmethod
    def default():
        """
        A default configuration with a default list of banned english words.
        """
        config = OrchestratorConfiguration()
        config.default_english_regex()
        return config
class Orchestrator:
    def __init__(self,config:OrchestratorConfiguration):
        self.config = config
    def validate_input(self,model:llm_module,input:str):
        try:
            toxicity_score = toxicity_run()
            input = self.decide_action(toxicity_score,self.config.threshold_toxicity)    
            pii_score
            overall_score = toxicity_score
            input = self.decide_action(overall_score)
            
            
        except LLMFilteredException as e:
            return e.message
        return input    
    def validate_output(self,llm_input,score:float):
        def determine_halluc(context:str,text:str):
            if(context is None):
                return
            pass
        
        return llm_input
    def decide_action(self,score,score_threshold,filter_reason:str,current_input:str):
        if(score_threshold<0):
            return current_input
        
        if(score_threshold>score):
            if(score_threshold*self.config.threshold_llm_rephrase>score):
                #TODO: Test this to get the actual reworded prompt, maybe change it?
                #NOTE: Since we are using the same model for safety llm (due to restrictions, we just pass)
                logging.debug("Safety_LLM_Triggered, rewording the prompt")
                
                cleaned_output = self.model.prompt_model_single(system_context=llm_module.llm_contexts["SeaLLM_Safety_Mode"],user_prompt=current_input)
                self.model.unload_model()
                return cleaned_output
            else:
                logging.debug(f"Filtered due to {filter_reason}") 
                raise LLMFilteredException(f"Filtered due to {filter_reason}")
        def attach_RAG_Context():
            #TODO: Pass the RAG Context here
            pass
