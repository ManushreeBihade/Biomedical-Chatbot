# Biomedical Multi-LLM Chatbot
A full-stack biomedical chatbot built with FastAPI (backend) and Streamlit (frontend) supporting:
🔀 Multi-LLM provider switching (Groq / Gemini)
🧠 Memory ON/OFF toggle
⚙️ Layered backend architecture
🔐 Environment-based API key management

# Features
Select between Groq (LLaMA 3.1) and Google Gemini
Toggle conversation memory dynamically
Biomedical domain filtering
REST API endpoint (/chat)
Swagger API docs for development

# Architecture
frontend/ → Streamlit UI  
backend/  → FastAPI server  
services/ → LLM routing & memory logic  
core/     → Config & environment handling  
models/   → Pydantic schemas  

# Run locally
Install Dependencies
Add API Keys (.env)
Start Backend
Start Frontend

# Tech Stack
FastAPI
Streamlit
Groq API
Google Gemini API
Pydantic
python-dotenv
