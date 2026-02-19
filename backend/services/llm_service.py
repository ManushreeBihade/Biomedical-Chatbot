from groq import Groq
from google import genai
from backend.core.config import settings
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# -------------------------------
# Logging Configuration
# -------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# System Prompt
# -------------------------------

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
# Domain Filter
# -------------------------------

biomedical_keywords = [
    "gene", "protein", "dna", "rna",
    "mutation", "cancer", "drug",
    "disease", "therapy", "clinical",
    "enzyme", "pathway", "neuron"
]

def is_biomedical(question: str):
    return any(word in question.lower() for word in biomedical_keywords)

# -------------------------------
# Local TinyLlama Model
# -------------------------------

LOCAL_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading Local TinyLlama model...")

local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME)

local_model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="cpu"
)

print("Local TinyLlama ready.")

# -------------------------------
# LLM Call Functions
# -------------------------------

def call_groq(final_prompt: str):
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set")

    client = Groq(api_key=settings.GROQ_API_KEY)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0.7
    )

    return response.choices[0].message.content


def call_gemini(final_prompt: str):
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=final_prompt,
        config={"temperature": 0.7}
    )

    return response.text


def call_local_tinyllama(final_prompt: str):

    formatted_prompt = f"<|user|>\n{final_prompt}\n<|assistant|>"

    inputs = local_tokenizer(formatted_prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = local_model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True
        )

    decoded = local_tokenizer.decode(outputs[0], skip_special_tokens=True)

    response = decoded.split("<|assistant|>")[-1].strip()

    return response


# -------------------------------
# Summarization Logic
# -------------------------------

def summarize_history(conversation_text: str, provider: str):

    logger.info("Summarizing conversation history...")

    summary_prompt = f"""
Summarize the following biomedical conversation briefly and retain key scientific points:

{conversation_text}
"""

    if provider == "Groq":
        return call_groq(summary_prompt)

    elif provider == "Gemini":
        return call_gemini(summary_prompt)

    elif provider == "LocalTinyLlama":
        return call_local_tinyllama(summary_prompt)

    else:
        raise ValueError("Invalid provider")


# -------------------------------
# Main Response Generator
# -------------------------------

def generate_response(provider: str, prompt: str, memory_enabled: bool, history: list):

    if not is_biomedical(prompt):
        logger.info("Non-biomedical question detected.")
        return "⚠️ This chatbot only answers biomedical questions."

    # ---------------------------
    # Memory Enabled
    # ---------------------------

    if memory_enabled:

        conversation_text = ""

        for m in history:
            conversation_text += f"{m['role']}: {m['content']}\n"

        context_length = len(conversation_text)
        logger.info(f"Context length: {context_length}")

        if context_length > settings.CONTEXT_SIZE:
            logger.info("Context exceeded threshold. Triggering summarization.")

            summary = summarize_history(conversation_text, provider)

            logger.info("Summary generated successfully.")
            logger.info(f"Summary preview: {summary[:200]}")

            conversation_text = (
                f"Summary of previous conversation:\n{summary}\n"
            )

        final_prompt = (
            BIOMED_SYSTEM_PROMPT
            + "\n\nConversation:\n"
            + conversation_text
            + f"\nUser: {prompt}"
        )

    # ---------------------------
    # Memory Disabled
    # ---------------------------

    else:
        final_prompt = (
            BIOMED_SYSTEM_PROMPT
            + "\n\nUser Question:\n"
            + prompt
        )

    # ---------------------------
    # Route to Selected Provider
    # ---------------------------

    if provider == "Groq":
        return call_groq(final_prompt)

    elif provider == "Gemini":
        return call_gemini(final_prompt)

    elif provider == "LocalTinyLlama":
        return call_local_tinyllama(final_prompt)

    else:
        raise ValueError("Invalid provider")
