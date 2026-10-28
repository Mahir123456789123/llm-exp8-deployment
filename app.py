import logging, time
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from support_rag import ask_support

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("support_api")
app = FastAPI(title="Customer Support RAG API", version="1.0.0")

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question must not be blank.")
        return value.strip()

class AskResponse(BaseModel):
    answer: str
    intent: str
    sources: List[str]
    retries: int
    response_time_ms: float

@app.get("/")
def root():
    return {"message": "Customer Support RAG API is running.", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    start = time.perf_counter()
    try:
        result = ask_support(request.question)
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        logger.info("ask_success latency_ms=%.2f intent=%s", elapsed, result.get("intent", "unknown"))
        return {"answer": result.get("answer", ""), "intent": result.get("intent", "general_support"),
                "sources": result.get("sources", []), "retries": result.get("retry_count", 0),
                "response_time_ms": elapsed}
    except Exception:
        logger.exception("ask_failed")
        raise HTTPException(status_code=502, detail="The answer service is temporarily unavailable. Check the Render logs and inference configuration.")


