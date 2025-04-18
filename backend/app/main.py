from fastapi import FastAPI
from .api import router as api_router
from . import services # Import to trigger model loading on startup via cache

app = FastAPI(title="Classroom Captioning Backend API",
              description="Provides transcription, context retrieval, and summarization services.",
              version="0.1.0")

@app.on_event("startup")
async def startup_event():
    """Load models on startup to warm up cache."""
    print("Application startup: Pre-loading models...")
    services.get_whisper_model()
    services.get_llama_index_query_engine()
    services.get_gemini_model()
    print("Model loading initiated.")

app.include_router(api_router, prefix="/api")

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the Classroom Captioning Backend API. Visit /docs for details."}