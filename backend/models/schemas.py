from pydantic import BaseModel
from typing import List, Dict

class ChatRequest(BaseModel):
    provider: str
    prompt: str
    memory_enabled: bool
    history: List[Dict[str, str]]

class ChatResponse(BaseModel):
    answer: str
