"""
FastAPI backend for UPI Intelligence Analyst.

Run with:
    uvicorn backend.main:app --reload

Then open: http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.router import intelligent_router

app = FastAPI(
    title="UPI Intelligence API",
    description="AI-powered UPI analytics using a multi-agent RAG system.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str

    model_config = {
        "json_schema_extra": {
            "examples": [{"query": "Which bank has the highest chargeback ratio?"}]
        }
    }


class QueryResponse(BaseModel):
    agent: str
    response: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"message": "UPI Intelligence API is running. Visit /docs for the API explorer."}


@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    """Route the query to the appropriate agent and return the response."""
    try:
        result = intelligent_router(request.query)
        return QueryResponse(agent=result["agent"], response=result["response"])
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}
