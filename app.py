# -------------------------------
# IMPORTS
# -------------------------------

# Streamlit → frontend UI framework
# Handles web app rendering, chat UI, sidebar, session state
import streamlit as st

# Groq SDK → backend client to call Groq LLM servers
from groq import Groq

# Google GenAI SDK → backend client to call Gemini LLM servers
from google import genai

# OS + dotenv → load secure API keys from environment (.env file)
import os
from dotenv import load_dotenv


# -------------------------------
# LOAD ENVIRONMENT VARIABLES
# -------------------------------

# This reads the .env file and loads keys into environment memory
# Without this, os.getenv() would return None
load_dotenv()

# These variables now live in backend memory only (not visible in UI)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# -------------------------------
# PAGE CONFIGURATION (UI LAYER)
# -------------------------------

# Configures browser tab title and layout
# Layout="wide" gives more horizontal space
st.set_page_config(page_title="Biomedical Chatbot", layout="wide")

# Main UI title
st.title("🧬 Biomedical Multi-LLM Chatbot")


# -------------------------------
# SIDEBAR CONFIGURATION (CONTROL PANEL)
# -------------------------------

# Sidebar acts as configuration control layer
st.sidebar.header("⚙️ Configuration")

# Provider selector → this controls backend routing
# It does NOT change UI — it changes which API is called
provider = st.sidebar.selectbox(
    "Select LLM Provider",
    ["Groq", "Gemini"]
)

# Memory toggle → controls whether past conversation is injected
# This directly affects backend prompt construction
memory_enabled = st.sidebar.toggle("Enable Memory", value=True)

# Clear Chat button → resets conversation state
# This modifies Streamlit's session memory
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []


# -------------------------------
# SESSION STATE INITIALIZATION (STATE MANAGEMENT LAYER)
# -------------------------------

# Streamlit re-runs the script on every interaction.
# session_state allows us to persist memory across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []


# -------------------------------
# SYSTEM PROMPT (MODEL BEHAVIOR CONTROL)
# -------------------------------

# This is the instruction layer that shapes model personality & scope.
# It constrains the model domain to biomedical.
BIOMED_SYSTEM_PROMPT = """
You are a biomedical research assistant.

You ONLY answer biomedical questions related to:
- Molecular biology
- Genetics
- Pharmacology
- Clinical research
- Pathophysiology
- Drug discovery

If the question is not biomedical, politely decline.

Provide a clear, structured biomedical explanation.
"""


# -------------------------------
# DOMAIN FILTER (SAFETY LAYER)
# -------------------------------

# Lightweight keyword-based guardrail before hitting the LLM.
# Prevents unnecessary API calls.
biomedical_keywords = [
    "gene", "protein", "dna", "rna",
    "mutation", "cancer", "drug",
    "disease", "therapy", "clinical",
    "enzyme", "pathway", "neuron"
]

def is_biomedical(question):
    # Returns True if any keyword appears
    return any(word in question.lower() for word in biomedical_keywords)


# -------------------------------
# DISPLAY CHAT HISTORY (FRONTEND RENDERING)
# -------------------------------

# Renders stored conversation on every rerun
# This gives illusion of persistent chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# -------------------------------
# MAIN CHAT INPUT LOOP (ENTRY POINT)
# -------------------------------

# This block executes ONLY when user submits a message
if prompt := st.chat_input("Ask a biomedical question..."):

    # Domain validation before calling expensive APIs
    if not is_biomedical(prompt):
        st.warning("⚠️ This chatbot only answers biomedical questions.")
        st.stop()

    # Display user message immediately (UI responsiveness)
    st.chat_message("user").markdown(prompt)

    # -------------------------------
    # MEMORY INJECTION (CONTEXT ENGINEERING)
    # -------------------------------

    if memory_enabled:
        # Store new user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Build conversation history string manually
        # This is because Gemini expects plain text, not role-based messages
        conversation_text = ""
        for m in st.session_state.messages:
            conversation_text += f"{m['role']}: {m['content']}\n"

        # Final prompt includes full history
        full_prompt = BIOMED_SYSTEM_PROMPT + "\n\nConversation:\n" + conversation_text
    else:
        # Stateless mode → only current question sent
        full_prompt = BIOMED_SYSTEM_PROMPT + "\n\nUser Question:\n" + prompt


    try:

        # -------------------------------
        # BACKEND ROUTING LAYER
        # -------------------------------

        # Provider selection determines which backend server receives request

        if provider == "Groq":

            # Validate key before making request
            if not GROQ_API_KEY:
                st.error("Groq API key missing in .env")
                st.stop()

            # Initialize Groq client
            client = Groq(api_key=GROQ_API_KEY)

            # Groq uses role-based chat message structure
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": BIOMED_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )

            # Extract model output
            answer = response.choices[0].message.content


        elif provider == "Gemini":

            if not GEMINI_API_KEY:
                st.error("Gemini API key missing in .env")
                st.stop()

            # Initialize Gemini client
            client = genai.Client(api_key=GEMINI_API_KEY)

            # Gemini expects plain text prompt
            response = client.models.generate_content(
                model="models/gemini-2.5-flash",
                contents=full_prompt,
                config={"temperature": 0.7}
            )

            answer = response.text


    except Exception as e:
        # Backend error handling layer
        st.error(f"Error: {str(e)}")
        st.stop()


    # -------------------------------
    # OUTPUT RENDERING
    # -------------------------------

    # Display assistant response in UI
    st.chat_message("assistant").markdown(answer)

    # Store assistant response for future memory injection
    if memory_enabled:
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
