from Orchestrator import OrchestratorConfiguration,Orchestrator
from LLM_Module import LLM_Module
from eval_datasets.MaliciousLLMPrompts import MaliciousLLMPrompts

import logging
# 1. Define the numeric value and name
RESULTS_LEVEL_NUM = 60
RESULTS_LEVEL_NAME = "RESULTS"

# Check if the level is already defined to prevent redefinition errors
if not hasattr(logging, RESULTS_LEVEL_NAME):
    # 2. Add the level to the logging module's level names map
    logging.addLevelName(RESULTS_LEVEL_NUM, RESULTS_LEVEL_NAME)

    # 3. Define a convenience method on the Logger class
    def results(self, message, *args, **kws):
        # Check if the logger is enabled for this level
        if self.isEnabledFor(RESULTS_LEVEL_NUM):
            self._log(RESULTS_LEVEL_NUM, message, args, **kws)

    # Attach the new method to the logging.Logger class
    logging.Logger.results = results
logging.basicConfig(filename='results.log', level=RESULTS_LEVEL_NUM, format='Time: %(asctime)s - Level: %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

orchestrator = Orchestrator(OrchestratorConfiguration.default())
orchestrator.config.option_enable_rag = False
llm = LLM_Module(LLM_Module.llm_model_list["SeaLLMs1.5B-Chat"])
orchestrator.set_logging()

dataset = MaliciousLLMPrompts()
correct_correct = 0 #Positive 
wrong_correct = 0 #False Negative
wrong_wrong = 0 # False Positive
correct_wrong = 0 # Negative
for data in dataset.get_english_dataset():
    prompt = data['prompt']
    logger.results(f"Prompt:{prompt}")
    logger.results(f"Attack Type: {data["attack_type"]}")
    logger.results(f"Malicious: {data["malicious"]}")
    prompt = orchestrator.validate_input(llm,prompt)
    messages,context,response = orchestrator.prompt_model(llm,prompt)
    if(response.lower() !="filtered."):
        if(data["malicious"] == True):
            correct_wrong+=1
        else:
            wrong_correct+=1
    else:
        if(data["malicious"] == True):
            
        correct_correct+=1
logger.results(f"Correct: {correct_correct} | Incorrect: {wrong_correct}| Accuracy: {correct_correct/correct_correct+wrong_correct}%")