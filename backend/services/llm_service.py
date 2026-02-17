from groq import Groq
from google import genai
from backend.core.config import settings

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

biomedical_keywords = [
    "gene", "protein", "dna", "rna",
    "mutation", "cancer", "drug",
    "disease", "therapy", "clinical",
    "enzyme", "pathway", "neuron"
]

def is_biomedical(question: str):
    return any(word in question.lower() for word in biomedical_keywords)


def call_groq(final_prompt: str):
    client = Groq(api_key=settings.GROQ_API_KEY)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": final_prompt}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content


def call_gemini(final_prompt: str):
    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=final_prompt,
        config={"temperature": 0.7}
    )

    return response.text


def summarize_history(conversation_text: str, provider: str):
    summary_prompt = f"""
Summarize the following conversation briefly:

{conversation_text}
"""

    if provider == "Groq":
        return call_groq(summary_prompt)
    elif provider == "Gemini":
        return call_gemini(summary_prompt)


def generate_response(provider: str, prompt: str, memory_enabled: bool, history: list):

    if not is_biomedical(prompt):
        return "⚠️ This chatbot only answers biomedical questions."

    if memory_enabled:

        conversation_text = ""

        for m in history:
            conversation_text += f"{m['role']}: {m['content']}\n"

        # Check context size
        if len(conversation_text) > settings.CONTEXT_SIZE:
            summary = summarize_history(conversation_text, provider)
            conversation_text = f"Summary of previous conversation:\n{summary}\n"

        final_prompt = BIOMED_SYSTEM_PROMPT + "\n\nConversation:\n" + conversation_text

    else:
        final_prompt = BIOMED_SYSTEM_PROMPT + "\n\nUser Question:\n" + prompt

    if provider == "Groq":
        return call_groq(final_prompt)

    elif provider == "Gemini":
        return call_gemini(final_prompt)

    else:
        raise ValueError("Invalid provider")
