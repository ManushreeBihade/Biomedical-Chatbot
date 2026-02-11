from fastapi import FastAPI
from backend.api.routes import router  

app = FastAPI(title="Biomedical Multi-LLM API")

app.include_router(router)
