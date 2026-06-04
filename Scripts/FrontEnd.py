from contextlib import contextmanager
from LLM_Module import LLM_Module
import streamlit as st
import gc
import traceback
from Orchestrator import Orchestrator, OrchestratorConfiguration, LLMFilteredException, LLMRewordedException 

# --- Constants and Session State Initialization ---

LLM_MODELS = {
    'LLaMA 3.3 70B (Best)': {
        'name': 'LLaMA 3.3 70B Versatile',
        'path': 'llama-3.3-70b-versatile'
    },
    'LLaMA 3.1 8B (Fastest)': {
        'name': 'LLaMA 3.1 8B Instant',
        'path': 'llama-3.1-8b-instant'
    },
    'Mixtral 8x7B (Long Context)': {
        'name': 'Mixtral 8x7B',
        'path': 'mixtral-8x7b-32768'
    },
    'Gemma 2 9B': {
        'name': 'Gemma 2 9B',
        'path': 'gemma2-9b-it'
    },
}

# --- Cache and Context Management ---

@st.cache_resource
def load_model(model_name: str, enable_quantization: bool = True) -> LLM_Module:
    """Loads a Groq-backed LLM module and caches it."""
    model_config = LLM_MODELS[model_name]
    with st.spinner(f"Connecting to {model_config['name']} via Groq API..."):
        model: LLM_Module = LLM_Module(model_name=model_config['path'])
        model.model_config_set()
        model.quantization_4_bit_config()
        model.load_model_in_mem(quant_4bit=enable_quantization)
        st.success(f"✅ Ready: {model_config['name']}")
    return model

def handle_model_switch():
    """Handles cleanup when the active LLM is switched."""
    new_model_key = st.session_state.model_selector
    if 'active_model_key' in st.session_state and st.session_state.active_model_key != new_model_key:
        try:
            load_model.clear()
        except:
            pass
        gc.collect()
        st.session_state['active_model_key'] = new_model_key

@contextmanager
def loading_spinner():
    """Custom context manager for the loading box."""
    thinking_placeholder = st.empty()
    thinking_placeholder.info("⏳ **Processing...** Running guardrail checks and generating response...")
    try:
        yield
    finally:
        thinking_placeholder.empty()

# --- Core Logic ---

def get_orchestrator_config() -> OrchestratorConfiguration:
    """Creates an OrchestratorConfiguration instance from Streamlit session state."""
    config = OrchestratorConfiguration(
        banned_word_allcase=st.session_state.banned_word_allcase_option,
        option_redact_pii_person=st.session_state.redact_pii_person,
        option_redact_pii_location=st.session_state.redact_pii_location,
        option_redact_pii_others=st.session_state.redact_pii_others,
        option_enable_rag=st.session_state.enable_rag,
        option_enable_llm_reword=st.session_state.enable_llm_reword,
        threshold_banned_word_tolerance=st.session_state.thresh_banned_word,
        threshold_prompt_injection=st.session_state.thresh_prompt_injection,
        threshold_pii=st.session_state.thresh_pii,
        threshold_rag_context_distance=st.session_state.thresh_rag_distance,
        threshold_semantic_similarity_threshold=st.session_state.thresh_semantic_similarity,
        threshold_toxicity=st.session_state.thresh_toxicity,
        weight_banned_word=st.session_state.weight_banned_word,
        weight_pii=st.session_state.weight_pii,
        weight_toxicity=st.session_state.weight_toxicity,
        weight_overall_score=st.session_state.weight_overall_score,
        threshold_llm_rephrase=st.session_state.thresh_llm_rephrase,
    )
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
        config = get_orchestrator_config()
        orchestrator = Orchestrator(config=config)
        model = load_model(model_key)
        orchestrator.set_logging(True)

        with loading_spinner():
            validated_input = orchestrator.validate_input(safety_model=model, input=prompt)
            st.session_state.model_output_context = None

            if validated_input.lower() == "filtered.":
                st.session_state.model_output = f"🚫 **Input Filtered.** Reason: Content flagged by guardrails."
            elif validated_input.startswith("Filtered due to"):
                st.session_state.model_output = f"🚫 **Input Filtered.** Reason: {validated_input}"
            else:
                messages_log = st.session_state.get('chat_history', None)
                messages, context, response = orchestrator.prompt_model(
                    llm_model=model,
                    input=validated_input,
                    max_tokens=st.session_state.overall_budget_input,
                    chat_log=messages_log
                )

                final_response = orchestrator.validate_output(model=model, llm_output=response, context=context)

                st.session_state.chat_history = messages
                st.session_state.model_output_context = context

                if final_response.lower() == "filtered.":
                    st.session_state.model_output = "⚠️ **Output Filtered.** Reason: Hallucination detected (similarity too low)."
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
    page_title="NLP LLM Guardrail Orchestrator",
    page_icon="🛡️",
    layout="wide"
)

# Initialize session state
defaults = {
    'active_model_key': list(LLM_MODELS.keys())[0],
    'overall_budget_input': 4096,
    'model_output': "",
    'user_prompt_input': "",
    'chat_history': None,
    'model_output_context': "",
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --- Header ---
st.title("🛡️ NLP LLM Guardrail Orchestrator")
st.markdown("Configure the **Orchestrator** in the sidebar, select a **Groq LLM**, and submit your prompt.")

# --- Groq API Key Input ---
groq_key = st.text_input(
    "🔑 Groq API Key",
    type="password",
    placeholder="Enter your Groq API key (get one free at console.groq.com)",
    help="Your key is only stored in session memory and never saved."
)
if groq_key:
    import os
    os.environ["GROQ_API_KEY"] = groq_key
    st.success("API key set!", icon="✅")
else:
    st.warning("Please enter your Groq API key above to use the app.", icon="⚠️")

st.markdown("---")

# --- Model Selection and Reset ---
col_model, col_reset = st.columns([3, 1])

with col_model:
    st.selectbox(
        "Select Groq Model:",
        options=list(LLM_MODELS.keys()),
        key="model_selector",
        on_change=handle_model_switch,
        help="All models are hosted on Groq's free API — no local loading needed!"
    )

with col_reset:
    st.button(
        "Reset All Fields & History",
        on_click=reset_fields,
        use_container_width=True
    )

# --- Input ---
st.text_area(
    "Natural Language Prompt Input",
    key="user_prompt_input",
    height=150,
    placeholder="Example: Write a story about the destruction of a planet.",
)

st.button(
    "**Submit Prompt to Orchestrator**",
    on_click=submit_prompt,
    type="primary",
    use_container_width=True,
    disabled=not groq_key
)

st.markdown("---")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Generation & Budget")
    st.number_input(
        "Max Output Tokens (Budget)",
        min_value=32,
        max_value=8192,
        step=512,
        key="overall_budget_input",
        help="Max tokens Groq will generate. Hard limit is 8192.",
        value=2048
    )

    st.header("🛡️ Orchestrator Configuration")

    st.subheader("Options")
    st.checkbox("Banned Word Case-Insensitive", key="banned_word_allcase_option", value=True)
    st.checkbox("Enable RAG Contextualization", key="enable_rag", value=True)
    st.checkbox("Enable LLM Rewording (Safety Mode)", key="enable_llm_reword", value=True)

    st.subheader("PII Redaction Options")
    st.checkbox("Redact Person/Name PII", key="redact_pii_person", value=False)
    st.checkbox("Redact Location PII", key="redact_pii_location", value=False)
    st.checkbox("Redact Other PII (Emails, IDs, etc.)", key="redact_pii_others", value=False)

    st.subheader("Filter Thresholds")
    st.number_input("Toxicity Threshold (0-1)", key="thresh_toxicity", min_value=0.0, max_value=1.0, step=0.05, value=0.9)
    st.number_input("Banned Word Tolerance (Count)", key="thresh_banned_word", min_value=-1, max_value=10, step=1, value=3)
    st.number_input("PII Entity Count Threshold", key="thresh_pii", min_value=-1, max_value=10, step=1, value=2)
    st.number_input("Prompt Injection Threshold (0-1)", key="thresh_prompt_injection", min_value=0.0, max_value=1.0, step=0.05, value=0.5)
    st.number_input("RAG Context Distance (0-1)", key="thresh_rag_distance", min_value=0.0, max_value=1.0, step=0.1, value=0.0)
    st.number_input("Semantic Similarity Threshold (0-1)", key="thresh_semantic_similarity", min_value=0.0, max_value=1.0, step=0.05, value=0.5)

    st.subheader("Reword/Filtering Weights")
    st.number_input("LLM Reword Multiplier (0-1)", key="thresh_llm_rephrase", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
    st.number_input("Overall Score Filter Weight (0-1)", key="weight_overall_score", min_value=0.0, max_value=1.0, step=0.05, value=0.8)
    st.number_input("Banned Word Weight (0-1)", key="weight_banned_word", min_value=0.0, max_value=1.0, step=0.05, value=0.4)
    st.number_input("PII Count Weight (0-1)", key="weight_pii", min_value=0.0, max_value=1.0, step=0.05, value=0.4)
    st.number_input("Toxicity Probability Weight (0-1)", key="weight_toxicity", min_value=0.0, max_value=1.0, step=0.05, value=0.4)

# --- Output ---
st.header("Model Output")

st.info("RAG Context Used (If Enabled)", icon="📖")
st.code(st.session_state.model_output_context or "", language="text")

st.text_area(
    "Model Output",
    value=st.session_state.model_output,
    disabled=True,
    height=250,
    placeholder="Model output will appear here after submission.",
)
