from pydantic import BaseModel
from typing import List, Optional, Dict

class ProcessAudioResponse(BaseModel):
    """Response model for the /process_audio endpoint."""
    transcription: str
    summary: Optional[str] = None
    context: Optional[List[Dict]] = None # e.g., [{"file": "slide1.txt", "score": 0.85, "text_snippet": "..."}]
    error: Optional[str] = None

# No request model needed if using FastAPI's UploadFile