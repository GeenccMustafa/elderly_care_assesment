from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from . import schemas, services
import numpy as np
import soundfile as sf
import io # For reading UploadFile in memory

router = APIRouter()

@router.post("/process_audio", response_model=schemas.ProcessAudioResponse)
async def process_audio_endpoint(audio_file: UploadFile = File(...)):
    """
    Receives an audio file chunk, transcribes it, finds related context,
    generates a summary, and returns the results.
    """
    print(f"Received audio file: {audio_file.filename}, content type: {audio_file.content_type}")

    if not audio_file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")

    try:
        # Read audio file content into memory
        contents = await audio_file.read()
        # Use soundfile to read audio data and sample rate from bytes
        audio_data, sample_rate = sf.read(io.BytesIO(contents))

        # --- 1. Transcription ---
        transcription = services.transcribe_audio(audio_data, sample_rate)
        if "[Error" in transcription:
            return schemas.ProcessAudioResponse(transcription=transcription, error=transcription)

        # --- 2. Context Retrieval ---
        context_str, context_list = services.get_context(transcription)
        if "[Error" in context_str:
             # Don't fail the whole request, just note the context error maybe
             print(f"Warning: {context_str}")


        # --- 3. Summarization ---
        summary = services.summarize_text(transcription, context_str)
        if summary and "[Error" in summary:
            # Log summary error but don't fail the whole response
            print(f"Warning: {summary}")


        return schemas.ProcessAudioResponse(
            transcription=transcription,
            summary=summary,
            context=context_list
        )

    except Exception as e:
        print(f"Unhandled error processing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    finally:
        await audio_file.close()