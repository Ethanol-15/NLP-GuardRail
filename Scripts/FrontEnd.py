from contextlib import contextmanager
from LLM_Module import llm_module
import streamlit as st
import pyperclip 
from pathlib import Path
import time 
import gc
import subprocess
import torch
import os
import tempfile

LLM_MODELS = {
    '1.5BSea_Chat': {'name': '1.5B_SeaLLM', 'path': llm_module.model_list['deepseek1.5B']},
    '1.5BDeepSeek_Chat': {'name': '1.5B_DeepSeek_Qwen_Distilled','path': llm_module.model_list["SeaLLMs1.5B-Chat"]}
}

@st.cache_resource
def load_model(model_name:str):
    model_config = LLM_MODELS[model_name]
    with st.spinner(f"Loading {model_config['name']}..."):
        model:llm_module = llm_module(model_name=Path(model_config['path']))
        model.model_config_set()
        model.load_model_in_mem()
        st.success(f"Successfully loaded: {model_config['name']}")
    return model

@contextmanager
def thinking_spinner():
    """Custom context manager for the 'thinking...' loading box."""
    # This creates the "thinking..." message
    thinking_placeholder = st.empty()
    thinking_placeholder.info(" **Loading...** Processing request...")
    try:
        yield
    finally:
        thinking_placeholder.empty()

def generate_code(model:llm_module, prompt, system_context,thinking_budget,max_tokens):
    if not prompt:
        return "", ""
 
    with thinking_spinner():
        thinking_process, code = model.prompt_model(
            user_prompt=prompt,
            system_context=system_context,
            thinking_budget=thinking_budget,
            max_new_tokens=max_tokens,
            return_all = False)
    
    return code, thinking_process
def submit_prompt():
    """Handles the submission, model running, and output display."""
    
    # 1. Get the current user input and selected model
    prompt = st.session_state.user_prompt_input
    model_key = st.session_state.model_selector
    current_thinking_budget = st.session_state.thinking_budget_input
    current_overall_budget = st.session_state.overall_budget_input
    if current_thinking_budget >= current_overall_budget:
        st.error("Your thinking budget is too high, should < Overall Budget")
        return
    if current_thinking_budget*2 > current_overall_budget:
        st.warning("Your thinking budget may be too low compared to overall budget, consider lowering it.")
    if not prompt:
        st.error("Please enter a prompt before submitting.")
        st.session_state['last_code'] = ""
        st.session_state['last_thinking'] = ""
        return

    # 2. Ensure the selected model is loaded/cached
    # Calling this first ensures the loading message appears if necessary
    loaded_model_key = load_model(model_key)
    system_context = def_system_context()
    # 3. Generate the code
    code, thinking = generate_code(loaded_model_key, prompt, system_context=system_context,thinking_budget=current_thinking_budget,max_tokens=current_overall_budget)
    
    # 4. Store the results in session state for display
    st.session_state['last_code'] = code
    st.session_state['last_thinking'] = thinking
    st.session_state['current_code_input'] = extract_code(code) # Auto-populate the execution box!
    st.session_state['execution_output'] = "" # Clear previous output
    # 5. Copy the Python code to the clipboard
    try:
        pyperclip.copy(code)
        st.toast("Python code successfully copied to clipboard!")
    except pyperclip.PyperclipException:
        st.warning("Could not copy code to clipboard. Check clipboard access.")
        
    # The display update happens automatically on the script re-run (which is triggered 
    # when st.button is pressed or st.rerun() is called, but we rely on the button press here).
def reset_fields():
    st.session_state.user_prompt_input = ""
    st.session_state['last_code'] = ""
    st.session_state['last_thinking'] = ""
    st.session_state['current_code_input'] = ""
    st.session_state['execution_output'] = ""
    st.toast("Input and output fields cleared!", icon="🧹")

st.set_page_config(
    page_title="LLM Interface",
    layout="wide"
)
def initiate_execution():
    """Button action to clear output and call execute_code."""
    code_to_run = st.session_state.current_code_input 
    raw_input_value = st.session_state.single_input_value.strip()
    st.session_state['execution_output'] = "Running..." 
    execute_code(raw_input_value,code_to_run)
def execute_code(stdin,code_string):
    """
    Executes the given Python code in a secure, separate subprocess 
    and captures stdout and stderr.
    """
    temp_file_path = ""
    with tempfile.NamedTemporaryFile("w", suffix=".py",encoding = "utf-8",delete=False) as f:
        f.write(str(BASE_IMPORTS+"\n"+SIMPLE_SANDBOX+"\n"+inject_input_type(st.session_state.single_input_type)+code_string))
        f.flush()
        temp_file_path = f.name
    command = ["python",  temp_file_path]
    try:
        result = subprocess.run(
            command,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout.strip()
    except subprocess.TimeoutExpired:
        output = "--- EXECUTION FAILED: TIMEOUT (10 seconds) ---\n"
    except Exception as e:
        output = f"--- EXECUTION FAILED: INTERNAL ERROR ---\n{e}"
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    # Store the output in the session state
    st.session_state['execution_output'] = output
    st.toast("Code execution complete!", icon="✅")

def handle_model_switch():
    new_model_key = st.session_state.model_selector
    if 'active_model_key' in st.session_state and st.session_state.active_model_key != new_model_key:
        old_model_key = st.session_state.active_model_key

        load_model.clear(old_model_key)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    st.session_state.active_model_key = new_model_key
    st.session_state['last_code'] = ""
    st.session_state['last_thinking'] = ""

if 'active_model_key' not in st.session_state:
    st.session_state['active_model_key'] = list(LLM_MODELS.keys())[0]
st.title("LLM Interface")
st.markdown("Select a model and enter your request below.")

# Initialize state keys
if 'single_input_value' not in st.session_state:
    st.session_state['single_input_value'] = "1" # Default test value
if 'active_model_key' not in st.session_state:
    st.session_state['active_model_key'] = list(LLM_MODELS.keys())[0]
if 'last_code' not in st.session_state:
    st.session_state['last_code'] = ""
if 'last_thinking' not in st.session_state:
    st.session_state['last_thinking'] = ""
if 'current_code_input' not in st.session_state:
    st.session_state['current_code_input'] = ""
if 'execution_output' not in st.session_state:
    st.session_state['execution_output'] = ""
if 'thinking_budget_input' not in st.session_state:
    st.session_state['thinking_budget_input'] = 2048
if 'overall_budget_input' not in st.session_state:
    st.session_state['overall_budget_input'] = 4096
if 'single_input_type' not in st.session_state:
    st.session_state['single_input_type'] = 'str'
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
    placeholder="Write a python script to download a CSV file from a URL and load it into a pandas DataFrame.",
)

# Submit Button
st.button(
    "**Submit**", 
    on_click=submit_prompt, 
    type="primary",
    use_container_width=True
)

st.markdown("---")

# Outputs (Code and Thinking Process)

st.header("Model Output")

col_code, col_thinking = st.columns(2)

with col_code:
    st.subheader("Code Section")
    st.code(
        st.session_state['last_code'],
        language='python',
    )
    if not st.session_state['last_code']:
        st.info("The generated Python code will appear here.")


with col_thinking:
    st.subheader("Thinking Process")
    st.text_area(
        "Internal Logic / Reasoning", 
        value=st.session_state['last_thinking'], 
        height=300, 
        disabled=True,
        placeholder="The model's internal thinking and steps will be displayed here."
    )
    if not st.session_state['last_thinking']:
        st.info("The model's step-by-step thinking process will be logged here.")
        
with st.sidebar:
    st.header("Thinking Budget Controls")
    
    thinking_budget = st.slider(
        "Thinking Budget (Token Count)",
        min_value=32,
        max_value=4096,
        step=1,
        key="thinking_budget_input",
        help="Simulates the maximum time (in seconds) the model can spend on internal thinking."
    )
    
    overall_budget = st.slider(
        "Overall Budget (Tokens)",
        min_value=32,
        max_value=65536,
        step=2,
        key="overall_budget_input",
        help="Simulates the maximum token limit or cost unit for the generation."
    )
    
st.header("Code Execution & Validation")
col_single_input_value, col_single_input_type, col_execution_button = st.columns([2.5, 1.5, 1])

with col_single_input_value:
    st.text_input(
        "Single Execution Input Value (e.g., test data for the function)",
        value="100",
        key="single_input_value",
        help="This value can be manually incorporated into the code for testing purposes."
    )

with col_execution_button:
    st.write(" ") 
    st.button(" **Run Code Subprocess**", on_click=initiate_execution, use_container_width=True)


col_execute_code, col_execute_output = st.columns(2)

with col_execute_code:
    st.subheader("Code to Execute (Editable)")
    st.text_area(
        "Edit and Run Code:", 
        key="current_code_input", 
        height=300, 
        placeholder="Generated code appears here. Edit it if needed before running."
    )
with col_single_input_type:
    st.selectbox(
        "Convert To Type:",
        options=['str', 'int', 'float', 'list (space-separated)'],
        key="single_input_type",
        help="Select the data type the input should be converted to before execution."
    )

with col_execute_output:
    st.subheader("Subprocess Output")
    st.text_area(
        "Subprocess Standard Output/Error:", 
        value=st.session_state['execution_output'], 
        height=300, 
        disabled=True
    )