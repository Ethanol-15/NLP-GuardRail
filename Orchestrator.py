from modules.toxicity_clasifier import ToxicityClassifier


class OrchestratorConfiguration:
    """
    Configures the orchestrator's modules sensitivities
    """
    def __init__(self,
        banned_word_tolerance = 5,
        banned_word_allcase = True,
        prompt_injection_threshold = 0.5,
        toxicity_threshold = 0.5,
        overall_score = 0.7,
        RAG_context_distance_threshold = 0,
        semantic_similarity_threshold = 0.5
                 ):
        """
        Configures the orchestrator
        """
        self.banned_word_tolerance = banned_word_tolerance
        self.banned_word_allcase = banned_word_allcase
        self.banned_words = []
        self.overall_score = overall_score
        self.prompt_injection_threshold = prompt_injection_threshold
        self.RAG_context_distance_threshold = RAG_context_distance_threshold
        self.semantic_similarity_threshold = semantic_similarity_threshold
        self.regex = []
        self.toxicity_threshold = toxicity_threshold
    def add_banned_words(self,**banned_word):
        self.banned_words.append(banned_word)
    def add_regex(self,**regex):
        self.regex = regex
    def sanity_check(self):
        error_message = ""
        if(self.toxicity_threshold < 0 or self.toxicity_threshold > 1):
            error_message.join(f"Toxicity threshold must be < 0 and > 1, found {self.toxicity_threshold}\n") 
        if(self.overall_score <0 or self.overall_score > 1):
            error_message.join(f"Overall score must be < 0 and > 1, found {self.overall_score}\n")
        if(self.banned_word_tolerance <0):
            error_message.join(f"Banned word tolerance threshold must be positive or 0!, found {self.banned_word_tolerance}\n")
        if(self.prompt_injection_threshold <0):
            error_message.join(f"Prompt injection tolerance threshold must be positive or 0!, found {self.banned_word_tolerance}\n")
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
    def __init__(self,config:OrchestratorConfiguration):
        self.config = config
    def validate_input(self,input:str):
        toxicity_score = 0 # TODO: Insert Toxicity Classifier
        overall_score = toxicity_score
        validated_input = self.decide_action(overall_score)
    def validate_output(self,score:float):
        def determine_halluc(context:str,text:str):
            if(context is None):
                return
            pass
        pass
    def decide_action(self,overall_score):
        pass
        def LLM_reword():
            pass
        def LLM_allow():
            pass
        def LLM_allow_with_RAG():
            pass
        def filtered(reason_txt:str):
            print(f"Filtered due to {reason_txt}")
            return "Filtered."
        


