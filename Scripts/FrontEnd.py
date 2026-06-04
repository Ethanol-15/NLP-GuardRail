from contextlib import contextmanager
from LLM_Module import LLM_Module
import streamlit as st
from pathlib import Path
import gc
import torch
import traceback
# Make sure these imports are correct based on your file structure
from Orchestrator import Orchestrator, OrchestratorConfiguration, LLMFilteredException, LLMRewordedException 

# --- Constants and Session State Initialization ---

LLM_MODELS = {
    '1.5BSea_Chat': {'name': '1.5B_SeaLLM', 'path': LLM_Module.llm_model_list['SeaLLMs1.5B-Chat']},
    '1.5BDeepSeek_Chat': {'name': '1.5B_DeepSeek_Qwen_Distilled','path': LLM_Module.llm_model_list["deepseek1.5B"]}
}

# --- Cache and Context Management ---

@st.cache_resource
def load_model(model_name:str, enable_quantization:bool = True) -> LLM_Module:
    """Loads a pre-configured language model into memory and caches it."""
    model_config = LLM_MODELS[model_name]
    with st.spinner(f"Loading {model_config['name']}..."):
        # Path() conversion is important as the model name can be a string path
        model:LLM_Module = LLM_Module(model_name=model_config['path'])
        model.model_config_set()
        model.quantization_4_bit_config()
        # Ensure we use the Path object for consistency if needed, but model_config['path'] is fine
        model.load_model_in_mem(quant_4bit=enable_quantization)
        st.success(f"Successfully loaded: {model_config['name']}")
    return model

def handle_model_switch():
    """Handles memory cleanup when the active LLM is switched."""
    new_model_key = st.session_state.model_selector
    if 'active_model_key' in st.session_state and st.session_state.active_model_key != new_model_key:
        old_model_key = st.session_state.active_model_key
        # Clear the cached instance of the old model
        try:
            load_model.clear() 
        except:
             # Fallback if clear() on an individual key is not supported by current Streamlit version
             pass
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        st.session_state['active_model_key'] = new_model_key # Update the active key

@contextmanager
def loading_spinner():
    """Custom context manager for the loading box."""
    thinking_placeholder = st.empty()
    thinking_placeholder.info(" **Loading...** Processing request...")
    try:
        yield
    finally:
        thinking_placeholder.empty()

# --- Core Logic ---

def get_orchestrator_config() -> OrchestratorConfiguration:
    """Creates an OrchestratorConfiguration instance from Streamlit session state."""
    config = OrchestratorConfiguration(
        # General Options
        banned_word_allcase=st.session_state.banned_word_allcase_option,
        option_redact_pii_person=st.session_state.redact_pii_person,
        option_redact_pii_location=st.session_state.redact_pii_location,
        option_redact_pii_others=st.session_state.redact_pii_others,
        option_enable_rag=st.session_state.enable_rag,
        option_enable_llm_reword=st.session_state.enable_llm_reword,

        # Thresholds
        threshold_banned_word_tolerance=st.session_state.thresh_banned_word,
        threshold_prompt_injection=st.session_state.thresh_prompt_injection,
        threshold_pii=st.session_state.thresh_pii,
        threshold_rag_context_distance=st.session_state.thresh_rag_distance,
        threshold_semantic_similarity_threshold=st.session_state.thresh_semantic_similarity,
        threshold_toxicity=st.session_state.thresh_toxicity,
        
        # Weights
        weight_banned_word=st.session_state.weight_banned_word,
        weight_pii=st.session_state.weight_pii,
        weight_toxicity=st.session_state.weight_toxicity,
        weight_overall_score=st.session_state.weight_overall_score,
        threshold_llm_rephrase=st.session_state.thresh_llm_rephrase,
    )
    
    # Initialize default regex for safety checks
    
    config.default_english_regex()
    config.default_filipino_regex()
    return config

def submit_prompt():
    """Handles the submission, model running, and output display.""" 
    prompt = st.session_state.user_prompt_input
    model_key = st.session_state.model_selector
    
    if not prompt:
        st.error("Please enter a prompt before submitting.")
        return

    try:
        # 1. Load Model and Configuration
        config = get_orchestrator_config()
        orchestrator = Orchestrator(config=config)
        # Assuming the safety model is the same as the main model for simplicity
        model = load_model(model_key) 
        
        # Enable logging for results.log output
        orchestrator.set_logging(True) 
        
        with loading_spinner():
            # 2. Input Validation
            validated_input = orchestrator.validate_input(safety_model=model, input=prompt)
            st.session_state.model_output_context = None
            
            # 3. Prompting and Output Generation
            if validated_input.lower() == "filtered.":
                print("It got filtered!")
                st.session_state.model_output = f"**Input Filtered.** Reason: {validated_input}"
            elif validated_input.startswith("Filtered due to"):
                print("It got filtered as well")
                # Exception was caught and returned as a string from validate_input
                st.session_state.model_output = f"**Input Filtered.** Reason: {validated_input}"
            else:
                print("Generating")
                # Actual model generation
                messages_log = st.session_state.get('chat_history', None)
                messages, context, response = orchestrator.prompt_model(
                    llm_model=model,
                    input=validated_input,
                    max_tokens=st.session_state.overall_budget_input,
                    chat_log=messages_log
                )
                
                # 4. Output Validation (Hallucination check via semantic similarity)
                final_response = orchestrator.validate_output(model=model,llm_output=response, context=context)
                
                # Store chat history and context
                st.session_state.chat_history = messages
                st.session_state.model_output_context = context

                if final_response.lower() == "filtered.":
                    st.session_state.model_output = f"**Output Filtered.** Reason: Hallucination detected (Similarity too low)."
                else:
                    st.session_state.model_output = final_response
    except Exception as e:
        st.error(f"An unexpected error occurred: {traceback.format_exc()}")

def reset_fields():
    """Resets input, output, and chat history."""
    st.session_state.user_prompt_input = ""
    st.session_state.model_output = ""
    st.session_state.model_output_context = ""
    if 'chat_history' in st.session_state:
        del st.session_state.chat_history
    st.toast("Input, output, and history cleared!", icon="🧹")

# --- Streamlit UI Layout ---

st.set_page_config(
    page_title="LLM Guardrail Orchestrator",
    layout="wide"
)

# Initialize all critical session state keys here with default values
if 'active_model_key' not in st.session_state:
    st.session_state['active_model_key'] = list(LLM_MODELS.keys())[0]
if 'overall_budget_input' not in st.session_state:
    st.session_state['overall_budget_input'] = 4096
if 'model_output' not in st.session_state:
    st.session_state['model_output'] = ""
if 'user_prompt_input' not in st.session_state:
    st.session_state['user_prompt_input'] = ""
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = None
if 'model_output_context' not in st.session_state:
    st.session_state['model_output_context'] = ""

st.title("🛡️ LLM Guardrail Orchestrator Interface")
st.markdown("Configure the **Orchestrator** in the sidebar, select an **LLM**, and submit your prompt.")

# Top Row: Model Selection and Reset
col_model, col_reset = st.columns([3, 1])

with col_model:
    st.selectbox(
        "Select Model:",
        options=list(LLM_MODELS.keys()),
        key="model_selector",
        on_change=handle_model_switch,
        help="Swapping models will trigger a brief reload, but the model will be cached for later use."
    )

with col_reset:
    st.button(
        "Reset All Fields & History", 
        on_click=reset_fields,
        use_container_width=True
    )

# Input Area
st.text_area(
    "Natural Language Prompt Input", 
    key="user_prompt_input", 
    height=150, 
    placeholder="Example: Write a story about the destruction of a planet, but don't mention any toxic words or PII.",
)

# Submit Button
st.button(
    "**Submit Prompt to Orchestrator**", 
    on_click=submit_prompt, 
    type="primary",
    use_container_width=True
)

st.markdown("---")

# --- Sidebar: Orchestrator Configuration ---

with st.sidebar:
    st.header("⚙️ Generation & Budget")
    
    # Model Configuration
    st.number_input(
        "Max Output Tokens (Budget)",
        min_value=32,
        max_value=65536,
        step=512,
        key="overall_budget_input",
        help="Sets the maximum number of new tokens the model can generate.",
        value=4096 
    )
    
    st.header("🛡️ Orchestrator Configuration")
    
    # General Options
    st.subheader("Options")
    st.checkbox("Banned Word Case-Insensitive", key="banned_word_allcase_option", value=True)
    st.checkbox("Enable RAG Contextualization", key="enable_rag", value=True)
    st.checkbox("Enable LLM Rewording (Safety Mode)", key="enable_llm_reword", value=True)
    
    st.subheader("PII Redaction Options (On Detection)")
    st.checkbox("Redact Person/Name PII", key="redact_pii_person", value=False)
    st.checkbox("Redact Location PII", key="redact_pii_location", value=False)
    st.checkbox("Redact Other PII (Emails, IDs, etc.)", key="redact_pii_others", value=False)

    st.subheader("Filter Thresholds (Immediate Filter)")
    # Thresholds
    st.number_input("Toxicity Threshold (0-1)", key="thresh_toxicity", min_value=0.0, max_value=1.0, step=0.05, value=0.9)
    st.number_input("Banned Word Tolerance (Count)", key="thresh_banned_word", min_value=-1, max_value=10, step=1, value=3)
    st.number_input("PII Entity Count Threshold", key="thresh_pii", min_value=-1, max_value=10, step=1, value=2)
    st.number_input("Prompt Injection Threshold (0-1)", key="thresh_prompt_injection", min_value=0.0, max_value=1.0, step=0.05, value=0.5)
    st.number_input("RAG Context Distance (0-1, 0=Filter)", key="thresh_rag_distance", min_value=0.0, max_value=1.0, step=0.1, value=0.0, help="If context relevance score is above this distance (closer to 0 is better, but this should ideally be 0 to use context).")
    st.number_input("Semantic Similarity (Hallucination Check) Threshold (0-1)", key="thresh_semantic_similarity", min_value=0.0, max_value=1.0, step=0.05, value=0.5, help="Minimum similarity required between RAG context and output.")

    st.subheader("Reword/Filtering Range and Weights")
    # Rewording/Rephrase Threshold
    st.number_input("LLM Reword Multiplier (0-1)", key="thresh_llm_rephrase", min_value=0.0, max_value=1.0, step=0.1, value=0.5, help="Score must be > (Threshold) AND < (Threshold * (1 + Multiplier)) to trigger LLM Rewording.")
    
    # Weights for Overall Score
    st.number_input("Overall Score Filter Weight (0-1)", key="weight_overall_score", min_value=0.0, max_value=1.0, step=0.05, value=0.8, help="Max weight for overall score is calculated as MaxPossibleScore * this weight.")
    st.number_input("Banned Word Weight (0-1)", key="weight_banned_word", min_value=0.0, max_value=1.0, step=0.05, value=0.4)
    st.number_input("PII Count Weight (0-1)", key="weight_pii", min_value=0.0, max_value=1.0, step=0.05, value=0.4)
    st.number_input("Toxicity Probability Weight (0-1)", key="weight_toxicity", min_value=0.0, max_value=1.0, step=0.05, value=0.4)


# Output Area
st.header("Model Output")

st.info("RAG Context Used (If Enabled)", icon="📖")
st.code(st.session_state.model_output_context, language="text")

st.text_area(
    "Model Output", 
    key="model_output", # Renamed key to avoid conflict with model_output
    value=st.session_state.model_output,
    disabled=True,
    height=250, 
    placeholder="Model Output goes here.",
)
