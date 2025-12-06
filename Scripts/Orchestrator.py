from modules.toxicity_clasifier import toxicity_run
from modules.pii_ner import PIINER
from LLM_Module import llm_module
# Import patterns here.
import logging
logging.basicConfig(filename='results.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
class LLMFilteredException(Exception):
    def __init__(self,reason:str,message="Filtered."):
        self.message = message
        self.reason = reason
        super().__init__(self.message)
class LLMRewordedException(Exception):
    def __init__(self,reworded_input:str,message:str):
        self.input = reworded_input
        self.message = message
        super().__init__(self.message)
        
class OrchestratorConfiguration:
    """
    Configuration class for the LLM Orchestrator guardrail.

    It holds all the thresholds, weights, and flags necessary to define the
    behavior of the input and output validation checks. A negative threshold
    value disables the corresponding check. Weights are used to calculate
    the overall risk score.
    """
    def __init__(
        self,
        banned_word_allcase:bool = True,
        enable_rag:bool = True,
        weight_overall_score:float = 0.7,
        # Thresholds, if not passed immediately filters the content. If negative, is disabled.
        threshold_banned_word_tolerance:int = 3,
        threshold_prompt_injection:float = 0.5,
        threshold_pii:int = 3,
        threshold_rag_context_distance:int = 0,
        threshold_semantic_similarity_threshold:float = 0.5,
        threshold_toxicity:float = 0.5,
        # Weights, affect the overall score if detected, setting to 0 disables it.
        weight_banned_word:float = 0.2,
        weight_prompt_injection:float = 0.2,
        weight_pii:float = 0.2,
        weight_toxicity:float = 1,
        threshold_llm_rephrase:float = 0.5,
                 ):
        """
        Initializes the configuration parameters for the Orchestrator.

        Args:
            banned_word_allcase (bool, optional): If True, banned word checks
                                                  are case-insensitive. Defaults to True.
            enable_rag (bool, optional): If True, enables Retrieval-Augmented
                                         Generation (RAG) context attachment. Defaults to True.
            weight_overall_score (float, optional): The multiplier applied to the
                                                    max possible overall risk score to define
                                                    the overall filtering threshold. Defaults to 0.7.

            # Thresholds (Immediate Filtering)
            threshold_banned_word_tolerance (int, optional): Max number of banned words allowed.
                                                             Negative value disables check. Defaults to 3.
            threshold_prompt_injection (float, optional): Max probability (0-1) for prompt
                                                          injection allowed. Negative value disables. Defaults to 0.5.
            threshold_pii (int, optional): Max number of PII entities allowed. Negative value disables.
                                           Defaults to 3.
            threshold_rag_context_distance (int, optional): Maximum distance/score for RAG
                                                            context relevance. If exceeded, RAG is not used.
                                                            Defaults to 0.
            threshold_semantic_similarity_threshold (float, optional): Minimum semantic similarity
                                                                       (0-1) required for output validation (hallucination check).
                                                                       Defaults to 0.5.
            threshold_toxicity (float, optional): Max toxicity probability (0-1) allowed.
                                                  Negative value disables. Defaults to 0.5.

            # Weights (Overall Score Contribution)
            weight_banned_word (float, optional): Weight for banned word count in overall score.
                                                  Defaults to 0.2.
            weight_prompt_injection (float, optional): Weight for prompt injection probability in
                                                       overall score. Defaults to 0.2.
            weight_pii (float, optional): Weight for PII count in overall score. Defaults to 0.2.
            weight_toxicity (float, optional): Weight multiplier for toxicity probability in
                                               overall score. Defaults to 1.

            threshold_llm_rephrase (float, optional): Multiplier for the filter threshold. If a score
                                                      is between `threshold` and `threshold * (1 + threshold_llm_rephrase)`,
                                                      the input is rephrased instead of filtered. Defaults to 0.5.
        """
        self.banned_word_allcase = banned_word_allcase
        self.banned_words = []
        self.enable_rag = enable_rag
        self.threshold_banned_word_tolerance = threshold_banned_word_tolerance
        self.threshold_llm_rephrase = threshold_llm_rephrase
        self.threshold_pii = threshold_pii
        self.threshold_prompt_injection = threshold_prompt_injection
        self.threshold_RAG_context_distance = threshold_rag_context_distance
        self.threshold_semantic_similarity_threshold = threshold_semantic_similarity_threshold
        self.threshold_toxicity = threshold_toxicity
        self.regex = []
        self.weight_banned_word = weight_banned_word
        self.weight_pii = weight_pii
        self.weight_prompt_injection = weight_prompt_injection
        self.weight_toxicity = weight_toxicity
        self.weight_overall_score = weight_overall_score
        self.sanity_check()
    def add_banned_words(self,banned_words:list):
        """
        Adds a list of words to the internal banned words list.

        Args:
            banned_words (list): A list of strings to be added to the banned words list.
        """
        self.banned_words.extend(banned_words)
    def add_regex(self,regex:list):
        """
        Adds a list of regex patterns to the internal regex list for filtering.

        Args:
            regex (list): A list of compiled regex patterns or strings to be added.
        """
        self.regex = regex.extend(regex)
    def default_english_regex(self):
        """
        Sets the configuration to use a default set of English-language regex patterns
        and banned words 
        """
        self.regex = []
        self.banned_words = []
    def default_filipino_regex(self):
        """
        Sets the configuration to use a default set of Filipino-language regex patterns
        and banned words
        """
        self.regex = []
        self.banned_word = []
    def sanity_check(self):
        """
        Performs basic validation on the configuration values to ensure they are
        within acceptable bounds (e.g., probabilities between 0 and 1).

        Raises:
            Exception: If any configuration value is invalid.
        """
        error_message = ""
        if(self.threshold_toxicity < 0 or self.threshold_toxicity > 1):
            error_message.join(f"Toxicity threshold must be < 0 and > 1, found {self.threshold_toxicity}\n") 
        if(self.weight_overall_score <0 or self.weight_overall_score > 1):
            error_message.join(f"Overall score must be < 0 and > 1, found {self.weight_overall_score}\n")
        if(self.threshold_banned_word_tolerance is not int):
            error_message.join(f"Banned word tolerance threshold must be an integer!, found {self.threshold_banned_word_tolerance}\n")
        if(self.threshold_prompt_injection is not int):
            error_message.join(f"Prompt injection tolerance threshold must be an integer!, found {self.threshold_banned_word_tolerance}\n")
        if(self.threshold_banned_word_tolerance):
            error_message.join(f"Banned word tolerance threshold must be an ")
        if(len(error_message>0)):
            raise Exception(f"Configuration failed, with the following errors:\n{error_message}")
    @staticmethod
    def default():
        """
        A default configuration with a default list of banned english words.

        Returns:
            OrchestratorConfiguration: A new instance of OrchestratorConfiguration
                                       with default settings and English-based
                                       banned words/regex.
        """
        config = OrchestratorConfiguration()
        config.default_english_regex()
        return config
class Orchestrator:
    """
    Orchestrates the LLM guardrail process by validating inputs and outputs,
    and handling the prompting of the underlying language model.

    This class manages various safety checks (e.g., banned words, toxicity, PII,
    prompt injection) and decides on actions like filtering, rewording, or
    allowing the input/output based on configured thresholds.
    """
    def __init__(self,config:OrchestratorConfiguration):
        """
        Initializes the Orchestrator with a specific configuration.

        Args:
            config (OrchestratorConfiguration): An object holding all the
                                               thresholds, weights, and flags
                                               required for the guardrail logic.
        """
        self.config = config
    def validate_input(self,safety_model:llm_module,input:str):
        """
        Performs a series of safety checks on the user input before it is
        passed to the main LLM.

        Checks include: Banned Words, Toxicity Score, PII detection, and
        Prompt Injection probability, culminating in an overall weighted score.

        Args:
            safety_model (llm_module): The LLM module used for safety-related
                                       tasks, such as rewording an unsafe prompt.
            input (str): The raw user input string to be validated.

        Returns:
            str: The original input string if all checks pass, or the message
                 from an exception if the input is filtered or reworded.

        Raises:
            LLMFilteredException: If any single check or the overall score
                                  exceeds its filtering threshold.
            LLMRewordedException: If a prompt exceeds the rephrase threshold,
                                  but not the filtering threshold. The exception
                                  carries the rephrased input.
        """
        try:
            banned_word_score = 1 #INSERT BANNED WORD MODULE HERE, NUMBER OF BANNED WORKS
            self.decide_action(safety_model,banned_word_score,self.config.threshold_banned_word_tolerance,"Banned_Words",input)
            toxicity_probability = toxicity_run()
            self.decide_action(safety_model,toxicity_probability,self.config.threshold_toxicity,"Toxicity_Score",input)
            piiner_results_count = PIINER().analyze() #TODO: Insert results, NUMBER OF PII FOUND
            self.decide_action(safety_model,piiner_results_count,self.config.threshold_pii,"Personable_Interifiable_Information",input)
            prompt_injection_prob = 0 #TODO: PROMPT INJECT PROBABILITY
            self.decide_action(safety_model,prompt_injection_prob,self.config.threshold_prompt_injection,"Prompt_Injection",input)
            # Last check if the combined results do not pass the threshold
            overall_score = (
                            # Base Score
                            (banned_word_score * self.config.weight_banned_word + piiner_results_count * self.config.weight_pii)
                            # Multipliers
                            ((1+toxicity_probability) * self.config.weight_toxicity) *
                            ((1+prompt_injection_prob) * self.config.weight_prompt_injection)
                            )
            max_score = (
                        (banned_word_score + piiner_results_count) *
                        (1+prompt_injection_prob) * (1+toxicity_probability) 
                        )
            
            self.decide_action(overall_score,self.config.weight_overall_score * max_score,"Overall_Score,",input)
            return input    
        except LLMFilteredException as e:
            # NOTE: Can be improved by adding more checks to the LLM_Prompt, but we are merely passing it here.
            return e.message
        except LLMRewordedException as e:
            return e.input
    def prompt_model(self,llm_model:llm_module,input:str,max_tokens:int = 2**12,chat_log:dict | None = None):
        """
        Handles the interaction with the main LLM. It optionally attaches RAG
        (Retrieval-Augmented Generation) context to the input if RAG is enabled
        in the configuration and the context relevance is sufficient.

        Args:
            llm_model (llm_module): The main LLM module to be prompted.
            input (str): The validated user input.
            max_tokens (int, optional): The maximum number of new tokens to
                                        generate. Defaults to 4096.
            chat_log (dict | None, optional): The conversational history, if any.
                                              Defaults to None.

        Returns:
            tuple: A tuple containing the updated message log and the LLM's
                   generated response string.
        """
        def attach_RAG_Context(input):
            context = ""
            rag_distance = 1 # PLaceholder for now
            #TODO: Pass the RAG System here
            if(rag_distance> self.config.threshold_RAG_context_distance):
                return ""
            return context
        if(self.config.enable_rag):
            input = f"Use the following as context for the prompt: \"{attach_RAG_Context(input)}\"\n"+input
        messages, response =llm_model.chat_model(
            user_prompt=input,
            max_new_tokens=max_tokens,
            messages=chat_log,
            system_context=llm_module.llm_contexts["SeaLLM_Norm"]
        )
        return messages, response
    def validate_output(self,llm_output:str,context:str):
        """
        Performs safety checks on the LLM's output, primarily for hallucinations
        or incoherence, by checking semantic similarity against the context.

        Args:
            llm_output (str): The raw output string from the LLM.
            context (str): The context (e.g., RAG content or conversational
                           context) used to generate the output. Can be None.

        Returns:
            str: The LLM output string if it passes the validation check.

        Raises:
            LLMFilteredException: If the semantic similarity score is below the
                                  configured threshold (indicating hallucination).
        """
        if(context is None):
            return llm_output
        semantic_similarity_score = 1
        self.decide_action(semantic_similarity_score,self.config.threshold_semantic_similarity_threshold,"Hallucination",llm_output)
        return llm_output
    def decide_action(self,model:llm_module,score,score_threshold,filter_reason:str,current_input:str):
        """
        Core guardrail logic to determine the appropriate action (allow, reword,
        or filter) based on a specific safety score and its threshold.

        Args:
            model (llm_module): The LLM module used for safety operations (e.g.,
                                rewording).
            score (float | int): The computed safety score (e.g., toxicity
                                 probability, banned word count).
            score_threshold (float | int): The maximum acceptable score for the
                                           current check.
            filter_reason (str): A descriptive label for the type of check
                                 (e.g., "Toxicity_Score", "Hallucination").
            current_input (str): The input string being checked.

        Raises:
            LLMRewordedException: If the score is above the rephrase threshold
                                  but below the filter threshold.
            LLMFilteredException: If the score exceeds the primary filter
                                  threshold.
        """
        if(score_threshold<0):
            return 
        if(score>score_threshold):
            if(score_threshold*self.config.threshold_llm_rephrase>score and filter_reason is not "Hallucination"):
                #NOTE: Since we are using the same model for safety llm (due to restrictions, we just pass this to the same model)
                logging.debug("Safety_LLM_Prompt_Triggered, rewording the prompt into a safer prompt")
                cleaned_output = model.prompt_model_single(system_context=llm_module.llm_contexts["SeaLLM_Safety_Mode"],user_prompt=current_input)
                #TODO: Extract the cleaned output.
                raise LLMRewordedException(cleaned_output)
            else:
                logging.debug(f"Filtered due to {filter_reason}, {score} > {score_threshold}") 
                raise LLMFilteredException(f"Filtered due to {filter_reason}, {score} > {score_threshold}")
        logging.debug(f"Passed {filter_reason}")
        return
        