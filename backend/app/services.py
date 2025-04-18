import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import google.generativeai as genai
import mlflow
import time
from functools import lru_cache # For caching models
import os

from . import config

# --- Model Loading (Cached) ---
@lru_cache(maxsize=1)
def get_whisper_model():
    print(f"Loading Whisper model: {config.WHISPER_MODEL_SIZE} ({config.WHISPER_DEVICE}, {config.WHISPER_COMPUTE_TYPE})")
    try:
        model = WhisperModel(config.WHISPER_MODEL_SIZE,
                             device=config.WHISPER_DEVICE,
                             compute_type=config.WHISPER_COMPUTE_TYPE)
        print("Whisper model loaded.")
        return model
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        return None

@lru_cache(maxsize=1)
def get_llama_index_query_engine():
    print("Setting up LlamaIndex...")
    try:
        if not os.path.exists(config.LLAMA_INDEX_PERSIST_DIR) or not os.listdir(config.LLAMA_INDEX_PERSIST_DIR):
            print(f"Warning: LlamaIndex persist directory '{config.LLAMA_INDEX_PERSIST_DIR}' not found or empty. Context retrieval will fail.")
            # Ideally, indexing should happen in a separate setup step/script
            return None

        Settings.embed_model = HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL, device=config.WHISPER_DEVICE) # Use same device?
        Settings.llm = None # Not using LlamaIndex LLM features

        storage_context = StorageContext.from_defaults(persist_dir=config.LLAMA_INDEX_PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        query_engine = index.as_query_engine(similarity_top_k=2)
        print("LlamaIndex query engine ready.")
        return query_engine
    except Exception as e:
        print(f"Error loading LlamaIndex: {e}")
        return None

@lru_cache(maxsize=1)
def get_gemini_model():
    print("Configuring Gemini model...")
    if not config.GOOGLE_API_KEY:
        print("Gemini API Key not found. Summarization disabled.")
        return None
    try:
        genai.configure(api_key=config.GOOGLE_API_KEY)
        model = genai.GenerativeModel(config.GEMINI_MODEL_NAME)
        print("Gemini model ready.")
        return model
    except Exception as e:
        print(f"Error configuring Gemini: {e}")
        return None

# --- Core Service Functions ---
def transcribe_audio(audio_data: np.ndarray, sample_rate: int) -> str:
    """Transcribes audio using the loaded Whisper model."""
    model = get_whisper_model()
    if not model:
        return "[Error: Whisper model not loaded]"

    # Ensure audio is float32
    if audio_data.dtype != np.float32:
       audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max # Normalize if int

    # Ensure mono
    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Whisper expects 16kHz, resampling might be needed if input differs
    # For simplicity, assume input is already 16kHz here
    # Add resampling logic if needed: import librosa; audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)

    print(f"Transcribing audio chunk (shape: {audio_data.shape}, sample_rate: {sample_rate})...")
    try:
        segments, info = model.transcribe(audio_data, beam_size=5, language="en", vad_filter=True)
        transcription = " ".join([seg.text for seg in segments]).strip()
        print(f"Transcription completed. Detected language: {info.language} (p={info.language_probability:.2f})")
        return transcription if transcription else "[No speech detected]"
    except Exception as e:
        print(f"Error during transcription: {e}")
        return f"[Error during transcription: {e}]"


def get_context(text: str) -> (str, list):
    """Queries LlamaIndex to get context related to the text."""
    query_engine = get_llama_index_query_engine()
    context_for_prompt = "No relevant context found in documents."
    retrieved_nodes_info = []

    if not query_engine or not text:
        return context_for_prompt, retrieved_nodes_info

    print(f"Querying LlamaIndex with text: '{text[:100]}...'")
    try:
        response = query_engine.query(text)
        if response and response.source_nodes:
            context_for_prompt = "\n\nRelevant Context from Documents:\n"
            for node_with_score in response.source_nodes:
                node = node_with_score.node
                score = node_with_score.score
                file_name = node.metadata.get('file_name', 'Unknown Document')
                node_text = node.get_content()
                info = f"- From '{file_name}' (Score: {score:.2f}): ...{node_text[:200]}...\n"
                context_for_prompt += info
                retrieved_nodes_info.append({"file": file_name, "score": float(score), "text_snippet": node_text[:100]})
            print(f"LlamaIndex found context in: {[n['file'] for n in retrieved_nodes_info]}")
        else:
            print("LlamaIndex: No relevant nodes found.")
    except Exception as e:
        print(f"Error querying LlamaIndex: {e}")
        context_for_prompt = "[Error retrieving context]"

    return context_for_prompt, retrieved_nodes_info


def summarize_text(text_to_summarize: str, context_str: str) -> Optional[str]:
    """Generates a summary using the Gemini API."""
    model = get_gemini_model()
    if not model or not text_to_summarize:
        return "[Summarization unavailable or no text provided]"

    print(f"Requesting summary for text (len: {len(text_to_summarize)})...")
    prompt = f"""Summarize the key points from the following classroom lecture transcript chunk.
Be concise (2-3 sentences). Incorporate the provided context if relevant.

Transcript Chunk:
---
{text_to_summarize}
---
{context_str if 'found' in context_str else ''}
---

Concise Summary:"""

    try:
        start_time = time.time()
        response = model.generate_content(prompt)
        api_latency = time.time() - start_time

        # Log API call to MLflow (optional, can be verbose)
        try:
            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
            with mlflow.start_run(nested=True):
                 mlflow.log_param("prompt_length", len(prompt))
                 mlflow.log_param("input_text_length", len(text_to_summarize))
                 mlflow.log_metric("gemini_api_latency", api_latency)
                 if response.parts:
                      summary = response.text
                      mlflow.log_text(summary, "summary_output.txt")
                      mlflow.log_metric("summary_generated", 1)
                 else:
                      mlflow.log_metric("summary_blocked_or_empty", 1)
        except Exception as mlflow_err:
            print(f"Warning: MLflow logging failed: {mlflow_err}")


        if response.parts:
             summary = response.text
             print(f"Gemini Summary received ({api_latency:.2f}s)")
             return summary
        else:
             print("Gemini Warning: Response might be blocked or empty.")
             return "[Summary generation failed or blocked]"

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return f"[Error generating summary: {e}]"