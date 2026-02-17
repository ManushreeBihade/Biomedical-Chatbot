# 🧬 Biomedical Multi-LLM Chatbot

A modular biomedical chatbot built using **FastAPI (backend)** and **Streamlit (frontend)** with support for multiple LLM providers.

Supports:

* ✅ Groq (Llama 3.1)
* ✅ Google Gemini
* ✅ Memory toggle (ON/OFF)
* ✅ Automatic context summarization
* ✅ Clean layered backend architecture

## 🏗 Architecture

User (Streamlit UI)
        ↓
FastAPI Backend
        ↓
Service Layer (LLM Logic)
        ↓
Groq / Gemini APIs

# Backend is structured into:

* `main.py` → App entry point
* `routes.py` → API endpoints
* `schemas.py` → Request/response validation
* `config.py` → Environment configuration
* `llm_service.py` → Core AI logic

## 🚀 Run Locally

### 1️⃣ Clone Repository

git clone https://github.com/ManushreeBihade/Biomedical-Chatbot.git
cd Biomedical-Chatbot

### 2️⃣ Create Virtual Environment

**Windows**

python -m venv venv
venv\Scripts\activate

**Mac/Linux**

python3 -m venv venv
source venv/bin/activate

### 3️⃣ Install Dependencies

pip install -r requirements.txt

### 4️⃣ Add API Keys

Create a `.env` file in the root directory:
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key

### 5️⃣ Run Backend (FastAPI)

uvicorn backend.main:app --reload

Backend runs at:
http://127.0.0.1:8000

API Docs:
http://127.0.0.1:8000/docs

### 6️⃣ Run Frontend (Streamlit)

Open a new terminal:
streamlit run frontend/app.py

The UI will open automatically in your browser.

## ⚙ Features

* 🔁 Multi-LLM provider toggle
* 🧠 Optional conversational memory
* 📉 Automatic summarization when chat context exceeds threshold
* 🧩 Layered backend design (production-ready structure)
* 🔐 Secure API key handling via environment variables

## 📌 Notes

* Run backend **before** frontend.
* Memory summarization triggers automatically when context exceeds configured limit.
* Designed for biomedical domain queries only.