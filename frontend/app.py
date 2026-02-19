import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(page_title="Biomedical Chatbot", layout="wide")
st.title("🧬 Biomedical Multi-LLM Chatbot")

st.sidebar.header("⚙️ Configuration")

provider = st.sidebar.selectbox("Select LLM Provider", ["Groq", "Gemini", "LocalTinyLlama"])
memory_enabled = st.sidebar.toggle("Enable Memory", value=True)

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a biomedical question..."):

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = requests.post(
        API_URL,
        json={
            "provider": provider,
            "prompt": prompt,
            "memory_enabled": memory_enabled,
            "history": st.session_state.messages
        }
    )

    if response.status_code == 200:
        answer = response.json()["answer"]
    else:
        answer = "Backend error occurred."

    st.chat_message("assistant").markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
