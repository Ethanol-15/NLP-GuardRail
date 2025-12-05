from contextlib import contextmanager
from LLM_Module import llm_module
import streamlit as st
from pathlib import Path
import gc
import torch
from Orchestrator import Orchestrator,OrchestratorConfiguration

LLM_MODELS = {
    '1.5BSea_Chat': {'name': '1.5B_SeaLLM', 'path': llm_module.llm_model_list['SeaLLMs1.5B-Chat']},
    '1.5BDeepSeek_Chat': {'name': '1.5B_DeepSeek_Qwen_Distilled','path': llm_module.llm_model_list["deepseek1.5B"]}
}

@st.cache_resource
def load_model(model_name:str):
    """Loads a pre-configured language model into memory and caches it. 

    It retrieves the model's configuration, initializes the custom LLM module,
    sets the configuration, and loads the model's weights into memory.
   
    Args:
        model_name: A string key used to look up the model's configuration
        in the global dictionary `LLM_MODELS`.

    Returns:
        An instance of the custom `llm_module` class with the specified
        model loaded into its internal memory (`self.actual_model`).

    Raises:
        KeyError: If `model_name` is not found in `LLM_MODELS`.
    """
    # CURRENTLY DISABLED TO PREVENT ACCIDENTS, DO NOT USE THIS
    model_config = LLM_MODELS[model_name]
    with st.spinner(f"Loading {model_config['name']}..."):
        model:llm_module = llm_module(model_name=Path(model_config['path']))
        #model.model_config_set()
        #model.quantization_4_bit_config()
        #model.load_model_in_mem(quant_4bit=True)
        st.success(f"Successfully loaded: {model_config['name']}")
    return model


def handle_model_switch():
    """Handles memory cleanup when the active LLM is switched. 

    This function should be used as the `on_change` callback for the
    model selection widget (`st.session_state.model_selector`).

    If the selected model key differs from the `active_model_key`, it:
    *   **Clears the cache** for the old model using `load_model.clear()`.
    *   **Frees GPU memory** using `torch.cuda.empty_cache()` if available.
    *   **Runs the garbage collector** (`gc.collect()`) to reclaim system memory.
    """
    new_model_key = st.session_state.model_selector
    if 'active_model_key' in st.session_state and st.session_state.active_model_key != new_model_key:
        old_model_key = st.session_state.active_model_key
        load_model.clear(old_model_key)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

@contextmanager
def loading_spinner():
    """Custom context manager for the loading box."""
    thinking_placeholder = st.empty()
    thinking_placeholder.info(" **Loading...** Processing request...")
    try:
        yield
    finally:
        thinking_placeholder.empty()

def submit_prompt():
    """Handles the submission, model running, and output display.""" 
    # TODO: Create an OrchestratorConfig here.
    prompt = st.session_state.user_prompt_input
    model_key = st.session_state.model_selector
    st.session_state.model_output = "WIP Progress"
    # Fill the following
    # messages
    # Validate the input first from the orchestrator
    # Make sure to load the model and get the model loaded.
    # 
    #model = load_model(model_key)
    #messages, responses = model.chat_model(
    #        user_prompt=prompt,
    #        max_new_tokens=max_tokens,
    #        temperature = ,
    #        message = ,
    #        system_context= ,
    #    )    
    # Insert Chat Log somewhere
    #responses = 
    #return responses
   
    
def reset_fields():
    #TODO: Insert everything that is initialized and used below into this for the reset button
    st.session_state.user_prompt_input = ""
    st.toast("Input and output fields cleared!", icon="🧹")

st.set_page_config(
    page_title="LLM Interface",
    layout="wide"
)
    
if 'active_model_key' not in st.session_state:
    st.session_state['active_model_key'] = list(LLM_MODELS.keys())[0]
st.title("LLM Interface")
st.markdown("Select a model and enter your request below.")

# Initialize all state keys here,
if 'active_model_key' not in st.session_state:
    st.session_state['active_model_key'] = list(LLM_MODELS.keys())[0]
if 'overall_budget_input' not in st.session_state:
    st.session_state['overall_budget_input'] = 4096
if 'model_output' not in st.session_state:
    st.session_state['model_output'] = ""
col_model, col_reset = st.columns([3, 1])

with col_model:
    # 1. Model Selection Dropdown
    st.selectbox(
        "Select Model:",
        options=list(LLM_MODELS.keys()),
        key="model_selector",
        on_change=handle_model_switch,
        help="Swapping models will trigger a brief reload, but the model will be cached for later use."
    )

with col_reset:
    st.button(
        "Reset All Fields", 
        on_click=reset_fields,
        use_container_width=True
    )

st.text_area(
    "Natural Language Prompt Input", 
    key="user_prompt_input", 
    height=150, 
    placeholder="Hello, how may I help you today?",
)

# Submit Button
st.button(
    "**Submit**", 
    on_click=submit_prompt, 
    type="primary",
    use_container_width=True
)

st.markdown("---")

with st.sidebar:
    #TODO: Insert the rest of the orchestrator stuff config here here preferably with slider or number input
    # Used in submit prompt
    st.header("Orchestrator Control")
    
    overall_budget = st.slider(
        "Overall Budget (Tokens)",
        min_value=32,
        max_value=65536,
        step=2,
        key="overall_budget_input",
        help="Sets the maximum token limit or cost unit for the generation."
    )
    
    
st.header("Model Output")


st.text_area(
    "Model Output", 
    key="model_output", 
    disabled=True,
    height=150, 
    placeholder="Model Output goes here.",
)
