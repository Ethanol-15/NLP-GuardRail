from modules.toxicity_clasifier import toxicity_run
from LLM_Module import llm_module
import logging
logging.basicConfig(filename='results.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
class LLMFilteredException(Exception):
    def __init__(self,reason:str,message="Filtered."):
        self.message = message
        self.reason = reason
        super().__init__(self.message)
class OrchestratorConfiguration:
    """
    Configures the orchestrator's modules sensitivities
    """
    def __init__(self,
        banned_word_allcase = True,
        overall_score = 0.7,
        threshold_llm_rephrase = 0.5,
        #Thresholds, 
        threshold_banned_word_tolerance = 5,
        threshold_prompt_injection = 0.5,
        threshold_rag_context_distance = 0,
        threshold_semantic_similarity_threshold = 0.5,
        threshold_toxicity = 0.5
                 ):
        """
        Configures the orchestrator and setup the configurations
        
        """
        self.banned_word_allcase = banned_word_allcase
        self.banned_words = []
        self.overall_score = overall_score
        self.threshold_banned_word_tolerance = threshold_banned_word_tolerance
        self.threshold_llm_rephrase = threshold_llm_rephrase
        self.threshold_prompt_injection = threshold_prompt_injection
        self.threshold_RAG_context_distance = threshold_rag_context_distance
        self.threshold_semantic_similarity_threshold = threshold_semantic_similarity_threshold
        self.threshold_toxicity = threshold_toxicity
        self.regex = []
    def add_banned_words(self,**banned_word):
        self.banned_words.append(banned_word)
    def add_regex(self,**regex):
        self.regex = regex
    
    def sanity_check(self):
        error_message = ""
        if(self.threshold_toxicity < 0 or self.threshold_toxicity > 1):
            error_message.join(f"Toxicity threshold must be < 0 and > 1, found {self.toxicity_threshold}\n") 
        if(self.overall_score <0 or self.overall_score > 1):
            error_message.join(f"Overall score must be < 0 and > 1, found {self.overall_score}\n")
        if(self.threshold_banned_word_tolerance <0 and self.threshold_banned_word_tolerance is not int):
            error_message.join(f"Banned word tolerance threshold must be positive integer or 0!, found {self.threshold_banned_word_tolerance}\n")
        if(self.threshold_prompt_injection <0):
            error_message.join(f"Prompt injection tolerance threshold must be positive or 0!, found {self.threshold_banned_word_tolerance}\n")
        if(len(error_message>0)):
            raise Exception(f"Configuration failed, with the following errors:\n{error_message}")
    @staticmethod
    def default():
        """
        A default configuration with a default list of banned english words.
        """
        config = OrchestratorConfiguration()
        config.add_banned_words("")
        return config
class Orchestrator:
    def __init__(self,config:OrchestratorConfiguration,model:llm_module):
        self.config = config
    def validate_input(self,input:str):
        try:
            toxicity_score = toxicity_run()
            if(toxicity_score<=self.config.threshold_toxicity):
                pass
            overall_score = toxicity_score
            validated_input = self.decide_action(overall_score)
            if(validated_input == "Filtered"):
                print()
        except LLMFilteredException as e:
            return e.message
    def validate_output(self,score:float):
        def determine_halluc(context:str,text:str):
            if(context is None):
                return
            pass
        pass
    def decide_action(self,score,score_threshold,filter_reason:str,current_msg:str):
        if(score_threshold>score):
            if(score_threshold*self.config.threshold_llm_rephrase>score):
                
            else:
                filtered(filter_reason)
            
        def LLM_reword():
            logging.debug("Safety_LLM_Triggered, rewording the words")
            
        def LLM_allow():
            pass
        def attach_RAG_Context():
            #TODO: Pass the RAG Context here
            pass
        def filtered(reason_txt:str):
            logging.debug(f"Filtered due to {reason_txt}") 
            raise LLMFilteredException(reason_txt,)
    def finalize_output():
        if()

