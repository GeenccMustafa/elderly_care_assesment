# backend/app/main.py
import logging

from fastapi import FastAPI

from . import config, services
from .api import router as api_router

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) 

# Initialize FastAPI application
app = FastAPI(
    title="Elderly Care Assessment Assistant API",
    description=(
        "Provides assessment analysis based on user answers and "
        "uploaded personal documents."
    ),
    version="1.0.0",
    contact={  
        "name": "Mustafa Genc",
        "email": "mustafa.gencc94@gmail.com",
    },
)


@app.on_event("startup")
async def startup_event():
    """
    Handles application startup tasks.

    This function is executed when the FastAPI application starts.
    It pre-loads necessary models and initializes resources to ensure they are
    ready when the first request arrives, reducing initial request latency.
    """
    logger.info("Application startup: Initializing resources...")
    try:
        services.get_whisper_model()
        services.get_gemini_model()
        services.get_embedding_model()
        services.get_llama_index_retriever()
        logger.info("Resource initialization routines initiated successfully.")
    except Exception as e:
        logger.error(f"Error during startup initialization: {e}", exc_info=True)


app.include_router(api_router, prefix="/api", tags=["API Endpoints"])


@app.get("/", tags=["Root"])
async def read_root():
    """Provides a welcome message and basic API information."""
    return {
        "message": "Welcome to the Elderly Care Assessment Assistant API.",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "api_version": app.version,
        "contact": app.contact,
    }

@app.get("/health", tags=["Health Check"])
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}