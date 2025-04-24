# backend/app/services.py

import html
import io
import logging
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

# --- Third-Party Library Imports ---
import google.generativeai as genai
import numpy as np
import soundfile as sf
import torch
from llama_index.core import (QueryBundle, Settings, StorageContext,
                              VectorStoreIndex, load_index_from_storage)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import pipeline as transformers_pipeline

# Attempt to import librosa, but don't fail if it's missing
try:
    import librosa
except ImportError:
    librosa = None
    logging.warning(
        "Librosa library not found. Audio reading capabilities will be limited "
        "to formats supported by soundfile."
    )

# --- Local Application Imports ---
from . import config, data_manager, schemas

# --- Configuration & Setup ---
logger = logging.getLogger(__name__)

# Module-level constants derived from config
PDF_STORAGE_DIR = config.PERSONAL_DOCS_STORAGE_DIR
LLAMA_INDEX_PERSIST_DIR = config.VECTOR_STORE_PERSONAL_DIR


# ============================
# Model Loading Functions
# ============================

@lru_cache(maxsize=1)
def get_whisper_model():
    """
    Loads and returns the Whisper ASR model pipeline using transformers.
    Caches the loaded model.
    """
    model_name = "openai/whisper-base.en"
    # Prefer GPU if available, fall back to CPU otherwise
    device_setting = "cuda" if torch.cuda.is_available() else "cpu"
    if device_setting == "cuda":
        device_index = 0 
        logger.info(f"Using CUDA device {device_index} for Whisper.")
    else:
        device_index = -1 # Use CPU
        logger.info("Using CPU for Whisper.")

    logger.info(f"Loading Whisper model ({model_name})...")
    try:
        # device_index: 0 for cuda:0, -1 for cpu
        model_pipeline = transformers_pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device_index
        )
        logger.info(f"Whisper model '{model_name}' loaded successfully on device {device_setting}.")
        return model_pipeline
    except Exception as e:
        logger.error(f"Failed to load Whisper model '{model_name}': {e}", exc_info=True)
        return None


@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Loads and returns the HuggingFace embedding model specified in config.
    Caches the loaded model.
    """
    model_name = config.EMBEDDING_MODEL
    device = config.EMBED_DEVICE
    logger.info(f"Loading embedding model ({model_name}) on device '{device}'...")
    try:
        embed_model = HuggingFaceEmbedding(model_name=model_name, device=device)
        logger.info(f"Embedding model '{model_name}' loaded successfully.")
        return embed_model
    except Exception as e:
        logger.error(f"Failed to load embedding model '{model_name}': {e}", exc_info=True)
        return None


@lru_cache(maxsize=1)
def get_llama_index_retriever(similarity_top_k: int = 3) -> Optional[VectorIndexRetriever]:
    """
    Loads the LlamaIndex vector store index from disk and returns a retriever.
    Requires the embedding model to be loaded first. Caches the retriever.

    Args:
        similarity_top_k: The number of top similar documents to retrieve.

    Returns:
        A VectorIndexRetriever instance or None if loading fails.
    """
    logger.info(f"Attempting to load LlamaIndex index from: {LLAMA_INDEX_PERSIST_DIR}")
    if not LLAMA_INDEX_PERSIST_DIR.exists() or not any(LLAMA_INDEX_PERSIST_DIR.iterdir()):
        logger.error(
            f"Cannot load index: Directory is missing or empty at "
            f"'{LLAMA_INDEX_PERSIST_DIR}'."
        )
        return None

    try:
        embed_model = get_embedding_model()
        if embed_model is None:
            logger.error(
                "Embedding model failed to load, cannot initialize index retriever."
            )
            return None

        logger.info("Loading index storage context...")
        storage_context = StorageContext.from_defaults(
            persist_dir=str(LLAMA_INDEX_PERSIST_DIR)
        )

        logger.info("Loading index from storage...")
        # Pass embed_model explicitly if required by the LlamaIndex version
        index = load_index_from_storage(
            storage_context,
            embed_model=embed_model
        )
        logger.info("Index loaded successfully.")

        retriever = VectorIndexRetriever(
            index=index, similarity_top_k=similarity_top_k
        )
        logger.info(
            f"LlamaIndex retriever created successfully (similarity_top_k={similarity_top_k})."
        )
        return retriever

    except Exception as e:
        logger.error(
            f"Failed to load LlamaIndex index or create retriever: {e}",
            exc_info=True
        )
        return None


@lru_cache(maxsize=1)
def get_gemini_model():
    """
    Configures and returns the Google Gemini generative model client.
    Requires GOOGLE_API_KEY from config. Caches the model client.
    """
    model_name = config.GEMINI_MODEL_NAME
    logger.info(f"Configuring Gemini model ({model_name})...")
    try:
        google_api_key = config.GOOGLE_API_KEY
        if not google_api_key:
            logger.error(
                "GOOGLE_API_KEY is not set in environment or config. "
                "Cannot configure Gemini model."
            )
            return None

        genai.configure(api_key=google_api_key)

        # Define safety settings to block harmful content
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        model = genai.GenerativeModel(
            model_name=model_name,
            safety_settings=safety_settings
        )
        logger.info(f"Gemini model '{model_name}' configured successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to configure Gemini model: {e}", exc_info=True)
        return None


# ============================
# Core Service Functions
# ============================

async def transcribe_audio(audio_content: bytes) -> str:
    """
    Transcribes audio content using the loaded Whisper model.
    Attempts to read audio using soundfile, falling back to librosa if needed and available.

    Args:
        audio_content: The audio data as bytes.

    Returns:
        The transcribed text, or an error message string.
    """
    if not audio_content:
        logger.warning("Transcribe request received with no audio data.")
        return "[Error: No audio data received]"

    whisper_model = get_whisper_model()
    if whisper_model is None:
        logger.error("Transcription failed: Whisper model is not available.")
        return "[Error: Whisper model not loaded]"

    logger.info(f"Attempting to transcribe audio ({len(audio_content)} bytes)...")
    audio_data = None
    sample_rate = None
    audio_read_method = "unknown"

    try:
        audio_data, sample_rate = sf.read(
            io.BytesIO(audio_content), dtype='float32', always_2d=False
        )
        audio_read_method = "soundfile"
        logger.debug("Audio read successfully using soundfile.")
    except sf.SoundFileError as sf_err:
        logger.warning(f"Soundfile failed to read audio ({sf_err}). Trying librosa...")
        if librosa:
            try:
                audio_data, sample_rate = librosa.load(
                    io.BytesIO(audio_content), sr=None, mono=True
                )
                audio_read_method = "librosa"
                logger.debug("Audio read successfully using librosa fallback.")
            except Exception as lb_err:
                logger.error(f"Librosa also failed to read audio: {lb_err}")
                return f"[Error: Could not read audio format (libs: soundfile, librosa)]"
        else:
            logger.error("Librosa fallback failed: Librosa library not installed.")
            return "[Error: Could not read audio format (soundfile failed, librosa missing)]"
    except Exception as read_err:
        logger.error(f"Unexpected error reading audio data: {read_err}", exc_info=True)
        return "[Error: Failed to process audio data]"

    if audio_data is None or sample_rate is None:
        logger.error("Audio data or sample rate is missing after read attempt.")
        return "[Error: Failed to extract audio data]"

    try:
        # Prepare input for Whisper pipeline
        audio_input = {"array": audio_data.astype(np.float32), "sampling_rate": sample_rate}
        logger.info(f"Performing transcription using {audio_read_method} data...")
        # Adjust parameters based on model needs or performance tuning
        result = whisper_model(audio_input, chunk_length_s=30, batch_size=8)
        transcription = result["text"].strip() if result and result.get("text") else ""

        if not transcription:
            logger.info("Transcription result is empty (no speech detected?).")
            return "[No speech detected]"

        logger.info(f"Transcription successful via {audio_read_method}.") # Log less text
        return transcription
    except Exception as e:
        logger.error(f"Error during Whisper model inference: {e}", exc_info=True)
        return "[Error: Transcription process failed]"


def _get_pdf_context(query: str, top_k: int = 3, person_id: Optional[str] = None) -> str:
    """
    Retrieves relevant context from indexed PDF documents using LlamaIndex.
    Filters results by person_id metadata if provided.
    Handles HTML unescaping for the final output context string.

    Args:
        query: The query string to search for.
        top_k: The final number of top relevant nodes to return.
        person_id: Optional person ID to filter metadata.

    Returns:
        A string containing the formatted context, or an error/status message.
        The returned string will have HTML entities decoded.
    """
    log_msg = f"Retrieving PDF context (top_k={top_k})"
    if person_id:
        log_msg += f" filtering for person_id='{person_id}'"
    log_msg += f" for query: '{query[:100]}...'"
    logger.info(log_msg)

    initial_retrieve_k = top_k * 2 if person_id else top_k
    retriever = get_llama_index_retriever(similarity_top_k=initial_retrieve_k)

    if retriever is None:
        logger.warning("PDF context retrieval failed: Retriever is not available.")
        return "No document context available (retriever not loaded)."

    try:
        nodes_with_scores = retriever.retrieve(query)

        if not nodes_with_scores:
            logger.info("No context nodes returned by retriever for the query.")
            # Static message, no unescaping needed
            return "No relevant information found in documents for this query."

        if person_id:
            filtered_nodes = []
            for n in nodes_with_scores:
                node_metadata = getattr(getattr(n, 'node', None), 'metadata', {})
                if node_metadata.get("person_id") == person_id:
                    filtered_nodes.append(n)

            logger.info(
                f"Retrieved {len(nodes_with_scores)} initial nodes, "
                f"filtered down to {len(filtered_nodes)} matching "
                f"person_id='{person_id}'."
            )
            relevant_nodes = filtered_nodes[:top_k]

            if not relevant_nodes:
                logger.info(f"No relevant nodes found *specifically* for '{person_id}' after filtering.")
                safe_person_id = html.unescape(person_id)
                message = (
                    f"No relevant information found specifically for '{safe_person_id}' "
                    "related to this query in the documents."
                )
                return html.unescape(message)
        else:
            relevant_nodes = nodes_with_scores[:top_k]
            logger.info(f"Retrieved {len(relevant_nodes)} nodes (no person filter applied).")

        context_parts = []
        for node_item in relevant_nodes:
            node = node_item.node
            filename = node.metadata.get("filename", "Unknown File")
            node_person_id = node.metadata.get("person_id", "Unknown Person")
            snippet = node.get_content(metadata_mode="none")
            context_parts.append(
                f"--- Context from: {filename} (Person: {node_person_id}) ---\n"
                f"{snippet}\n"
                f"--- End Context ---"
            )

        final_context = "\n\n".join(context_parts)


        unescaped_final_context = html.unescape(final_context)
        logger.info(f"Prepared final context string ({len(unescaped_final_context)} chars).")
        return unescaped_final_context

    except Exception as e:
        logger.error(f"Error during PDF context retrieval or processing: {e}", exc_info=True)
        return f"[Error retrieving document context: {e}]"


def _call_gemini_with_prompt(prompt: str, debug_label: str = "gemini_call") -> Tuple[Optional[str], Optional[str]]:
    """
    Calls the configured Gemini model with the provided prompt.

    Handles potential errors like API key issues, content blocking, and
    communication failures. Logs the raw response text before processing.
    Applies HTML unescaping and stripping to the result text.

    Args:
        prompt: The prompt string to send to the LLM.
        debug_label: A label for logging purposes to identify the call context.

    Returns:
        A tuple containing:
        - str | None: The processed (unescaped, stripped) response text, or None on error/block.
        - str | None: An error message if an error occurred or content was blocked, otherwise None.
    """
    llm = get_gemini_model()
    if not llm:
        logger.error(f"LLM call failed ({debug_label}): Gemini model not available.")
        return None, "LLM (Gemini) model not available."

    logger.debug(f"Sending prompt to Gemini ({debug_label}, {len(prompt)} chars)...")
    try:
        # Make the API call
        response = llm.generate_content(prompt)

        raw_text_for_log = "[Response object did not contain text content]"
        try:
            raw_text_from_api = getattr(response, 'text', None)
            if raw_text_from_api is not None:
                raw_text_for_log = raw_text_from_api
            logger.info(f"--- RAW GEMINI OUTPUT ({debug_label}) START ---")
            logger.info(raw_text_for_log)
            logger.info(f"--- RAW GEMINI OUTPUT ({debug_label}) END ---")
        except Exception as log_e:
            logger.warning(f"Could not access or log raw response text for {debug_label}: {log_e}")

        if response.prompt_feedback and response.prompt_feedback.block_reason:
            block_reason = response.prompt_feedback.block_reason
            logger.warning(f"LLM call blocked by API ({debug_label}): {block_reason}")
            return None, f"Content blocked by API: {block_reason}"

        if raw_text_from_api is not None:
            # Decode HTML entities and strip leading/trailing whitespace
            result_text = html.unescape(raw_text_from_api).strip()
            logger.debug(f"Successfully received and processed Gemini response ({debug_label}).")
            return result_text, None
        else:
            logger.warning(f"Gemini response for {debug_label} was successful but had no text content.")
            return "", None 

    except Exception as e:
        error_message = f"LLM communication error: {e}"
        if "API key not valid" in str(e):
            error_message = "Invalid Google API Key provided."
        elif "permission" in str(e).lower() and ("denied" in str(e).lower() or "403" in str(e)):
            error_message = "API Permission Denied (check API key permissions/billing)."

        logger.error(f"Error during LLM call ({debug_label}): {error_message}", exc_info=True)
        return None, error_message


async def analyze_assessment(
    person_id: str, answers: List[schemas.AssessmentAnswer]
) -> Tuple[
    schemas.FullAnalysisResponse, # Analysis object (with parsed notes)
    Dict[str, str],               # Prompts used
    Optional[str],               # Consolidated error message
    float,                       # PDF context latency (0.0 for this function)
    Dict[str, float],            # LLM call latencies
    Optional[str]                # Raw notification string from LLM
]:
    """
    Analyzes assessment answers using LLM prompts for different domains.

    Generates domain summaries and actionable notification notes based *only*
    on the provided assessment answers. Does NOT use external document context.

    Args:
        person_id: The ID of the person being assessed.
        answers: A list of assessment answers provided by the user.

    Returns:
        A tuple containing:
        - Base analysis response object (containing structured analysis and parsed notifications list).
        - Dictionary of prompts used for generation.
        - Optional consolidated error message if any step failed.
        - Placeholder for PDF context latency (always 0.0).
        - Dictionary of latencies for individual LLM calls.
        - The raw, unescaped, formatted notification string generated by the LLM.
    """
    start_time = time.time()
    logger.info(f"Starting assessment analysis for person '{person_id}' (using answers only).")

    prompts_used: Dict[str, str] = {}
    analysis_errors: List[str] = [] 
    llm_latencies: Dict[str, float] = {}
    raw_notification_output_text: Optional[str] = None 

    # 1. Combine Answers into a single string for prompts
    try:
        combined_answers = ""
        answer_map = {a.question_id: a.answer_text for a in answers}
        for i, q_text in enumerate(config.ASSESSMENT_QUESTIONS):
            q_id = i + 1
            answer = answer_map.get(q_id, "[No answer provided]")
            combined_answers += f"Q{q_id}: {q_text}\nA{q_id}: {str(answer)}\n\n"
        combined_answers = combined_answers.strip()
        prompts_used["input_combined_answers"] = combined_answers
        logger.info(f"Combined answers prepared for '{person_id}'.")
    except Exception as e:
        logger.error(f"Failed to combine answers for '{person_id}': {e}", exc_info=True)
        analysis_errors.append("Internal error: Failed to process answers.")
        error_analysis = schemas.DomainAnalysis(
            cognitive="[Error]", physical="[Error]", health="[Error]", personal_info="[Error]"
        )
        error_response = schemas.FullAnalysisResponse(
            analysis=error_analysis, notifications=[], error=", ".join(analysis_errors)
        )
        return error_response, prompts_used, ", ".join(analysis_errors), 0.0, llm_latencies, None

    domain_prompts_config = {
        "cognitive": config.PROMPT_TEMPLATE_COGNITIVE,
        "physical": config.PROMPT_TEMPLATE_PHYSICAL,
        "health": config.PROMPT_TEMPLATE_HEALTH,
        "personal_info": config.PROMPT_TEMPLATE_PERSONAL_INFO
    }
    analysis_results_text: Dict[str, str] = {}

    for domain_key, prompt_template in domain_prompts_config.items():
        logger.info(f"Generating analysis for domain: {domain_key}...")
        try:
            prompt = prompt_template.format(combined_answers=combined_answers)
            prompts_used[f"prompt_{domain_key}"] = prompt

            llm_call_start = time.time()
            result_text, error_msg = _call_gemini_with_prompt(
                prompt, debug_label=f"analysis_{domain_key}"
            )
            llm_latencies[f"{domain_key}_llm_latency_sec"] = time.time() - llm_call_start

            if error_msg:
                analysis_results_text[domain_key] = f"[Error: {error_msg}]"
                analysis_errors.append(f"Error in {domain_key} analysis: {error_msg}")
                logger.error(f"LLM error generating {domain_key} analysis: {error_msg}")
            else:
                analysis_results_text[domain_key] = result_text or f"[{domain_key} analysis not generated]"

        except KeyError as ke:
            error_msg = f"Prompt template misconfiguration for {domain_key}: {ke}"
            logger.error(error_msg, exc_info=False)
            analysis_results_text[domain_key] = f"[Error: Prompt Config Error]"
            analysis_errors.append(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during {domain_key} analysis: {e}"
            logger.error(error_msg, exc_info=True)
            analysis_results_text[domain_key] = f"[Error: Unexpected]"
            analysis_errors.append(error_msg)

    logger.info("Generating notification notes...")
    notification_notes_list: List[schemas.NotificationNote] = [] 

    try:
        notification_prompt = config.PROMPT_TEMPLATE_NOTIFICATIONS.format(
            combined_answers=combined_answers
        )
        prompts_used["prompt_notifications"] = notification_prompt

        llm_call_start = time.time()
        # Call LLM, get processed text and potential error
        processed_notification_text, notification_error = _call_gemini_with_prompt(
            notification_prompt, debug_label="notifications"
        )
        llm_latencies["notifications_llm_latency_sec"] = time.time() - llm_call_start

        # Store the processed text (unescaped, stripped) as the raw output for this purpose
        raw_notification_output_text = processed_notification_text

        if notification_error:
            error_msg = f"Error generating notification notes: {notification_error}"
            logger.warning(error_msg)
            analysis_errors.append(error_msg)
            # Store error state in raw text if needed for API response
            if raw_notification_output_text is None: # Avoid overwriting if partial text came back
                 raw_notification_output_text = f"[Error generating notes: {notification_error}]"

        elif processed_notification_text: # If LLM returned text successfully
             # Parse the processed text into a list of notes for structured storage/use
             notes = [
                 line.strip('-* ')
                 for line in processed_notification_text.splitlines()
                 if line.strip().startswith(('-', '*')) # Look for bullet points
             ]
             if notes:
                 notification_notes_list = [schemas.NotificationNote(note=n) for n in notes]
                 logger.info(f"Parsed {len(notification_notes_list)} notification notes from LLM output.")
             else:
                 logger.info("LLM generated notification text, but no standard bullet points ('- ' or '* ') found for parsing.")
        else: # LLM call was successful but returned empty string
            logger.info("LLM generated no text for notifications.")
            raw_notification_output_text = None # Explicitly set to None if result was empty

    except KeyError as ke:
        error_msg = f"Prompt template misconfiguration for notifications: {ke}"
        logger.error(error_msg, exc_info=False)
        analysis_errors.append(error_msg)
        raw_notification_output_text = "[Error: Prompt Config Error]"
    except Exception as e:
        error_msg = f"Unexpected error during notification generation: {e}"
        logger.error(error_msg, exc_info=True)
        analysis_errors.append(error_msg)
        raw_notification_output_text = "[Error: Unexpected]"

    # 4. Format Final Response Object
    analysis_output = schemas.DomainAnalysis(
        cognitive=analysis_results_text.get("cognitive", "[Analysis Unavailable]"),
        physical=analysis_results_text.get("physical", "[Analysis Unavailable]"),
        health=analysis_results_text.get("health", "[Analysis Unavailable]"),
        personal_info=analysis_results_text.get("personal_info", "[Analysis Unavailable]")
    )

    consolidated_error = ", ".join(analysis_errors) if analysis_errors else None

    # Create the base response object (API layer will add raw text)
    final_response_base = schemas.FullAnalysisResponse(
        analysis=analysis_output,
        notifications=notification_notes_list, # Use the parsed list here
        error=consolidated_error
    )

    end_time = time.time()
    logger.info(
        f"Assessment analysis service for '{person_id}' finished in "
        f"{end_time - start_time:.2f} seconds. Errors: {bool(consolidated_error)}"
    )

    return (
        final_response_base,
        prompts_used,
        consolidated_error,
        0.0,
        llm_latencies,
        raw_notification_output_text
    )


async def ask_unified_about_person(
    person_id: str, question: str
) -> Tuple[schemas.UnifiedAskResponse, Dict[str, Any]]:
    """
    Answers questions about a person using both indexed personal documents
    and their latest assessment analysis history.

    Args:
        person_id: The identifier for the person.
        question: The question being asked.

    Returns:
        A tuple containing:
        - The unified response object with answers from both sources.
        - A dictionary containing intermediate data (context, prompts, errors, latencies).
    """
    start_time = time.time()
    logger.info(f"Starting Unified Ask for '{person_id}': '{question[:100]}...'")

    intermediate_data: Dict[str, Any] = {"question": question}
    latencies: Dict[str, float] = {}
    doc_answer: Optional[str] = "[Not Available]"
    analysis_answer: Optional[str] = "[Not Available]"
    doc_error: Optional[str] = None
    analysis_error: Optional[str] = None
    general_errors: List[str] = [] # Collect general processing errors

    # 1. Get Document-Based Answer
    logger.info("Processing document-based answer...")
    try:
        pdf_context_start_time = time.time()
        pdf_context = _get_pdf_context(question, top_k=5, person_id=person_id)
        latencies["doc_context_retrieval_latency_sec"] = time.time() - pdf_context_start_time
        intermediate_data["retrieved_doc_context"] = pdf_context

        if pdf_context.startswith("[Error") or pdf_context.startswith("No document context"):
            doc_error = f"Context Retrieval Failed: {pdf_context}"
            doc_answer = pdf_context 
        elif "No relevant information found" in pdf_context:
            doc_answer = pdf_context
            logger.info("No relevant document context found specifically for the query/person.")
        else:
            # Proceed with LLM call only if valid context was found
            doc_prompt = config.PROMPT_TEMPLATE_ASK_DOCS.format(
                person_id=person_id, context=pdf_context, question=question
            )
            intermediate_data["prompt_document_based"] = doc_prompt

            llm_call_start = time.time()
            llm_doc_answer, llm_doc_error = _call_gemini_with_prompt(
                doc_prompt, debug_label="ask_docs"
            )
            latencies["doc_llm_latency_sec"] = time.time() - llm_call_start

            if llm_doc_error:
                doc_error = f"LLM Error (Docs): {llm_doc_error}"
                doc_answer = f"[LLM Error: {llm_doc_error}]"
            else:
                doc_answer = llm_doc_answer or "[LLM generated no answer from documents]"

    except Exception as e:
        error_msg = f"Unexpected error during document processing: {e}"
        logger.error(error_msg, exc_info=True)
        doc_error = error_msg
        doc_answer = "[Error processing documents]"
        general_errors.append("Error encountered during document processing.")

    intermediate_data["doc_error_details"] = doc_error

    # 2. Get Analysis-Based Answer
    logger.info("Processing analysis-based answer...")
    try:
        analysis_retrieval_start = time.time()
        latest_record = data_manager.get_latest_record_data(person_id)
        latencies["analysis_retrieval_latency_sec"] = time.time() - analysis_retrieval_start

        if latest_record is None:
            analysis_answer = f"No assessment record found for '{person_id}'."
            intermediate_data["analysis_summary_used"] = "N/A - Record not found"
            intermediate_data["assessment_answers_str"] = "N/A - Record not found"
            logger.info(f"No assessment history found for '{person_id}'.")
        else:
            record_analysis_error = latest_record.get("analysis_error")
            if record_analysis_error:
                # If the record itself contains an error, report that
                analysis_error = f"Latest assessment record indicates error: {record_analysis_error}"
                analysis_answer = f"[Error in saved record: {record_analysis_error}]"
                intermediate_data["analysis_summary_used"] = f"N/A - Error in record"
                intermediate_data["assessment_answers_str"] = f"N/A - Error in record"
                logger.warning(f"Using assessment record for '{person_id}' which contains an error.")
            else:
                # Extract data from the valid record
                analysis_dict = latest_record.get("analysis", {})
                notifications_list = latest_record.get("notifications", [])
                answers_dict = latest_record.get("answers", {})

                assessment_answers_str = "\n".join(
                    [f"Q{qid}: {ans}" for qid, ans in answers_dict.items()]
                ) if answers_dict else "No answers recorded."
                intermediate_data["assessment_answers_str"] = assessment_answers_str

                notes_str = "\n".join(
                    [f"- {n.get('note', 'N/A')}" for n in notifications_list]
                ) if notifications_list else "None"

                analysis_prompt = config.PROMPT_TEMPLATE_ASK_ANALYSIS.format(
                    person_id=person_id,
                    cognitive=analysis_dict.get('cognitive', "N/A"),
                    physical=analysis_dict.get('physical', "N/A"),
                    health=analysis_dict.get('health', "N/A"),
                    personal_info=analysis_dict.get('personal_info', "N/A"),
                    notifications=notes_str,
                    assessment_answers=assessment_answers_str,
                    question=question
                )
                intermediate_data["prompt_analysis_based"] = analysis_prompt
                intermediate_data["analysis_summary_used"] = ( # For logging
                     f"Cognitive: {analysis_dict.get('cognitive', 'N/A')}\n"
                     f"Physical: {analysis_dict.get('physical', 'N/A')}\n"
                     f"Health: {analysis_dict.get('health', 'N/A')}\n"
                     f"Personal Info: {analysis_dict.get('personal_info', 'N/A')}"
                 )

                # Call LLM
                llm_call_start = time.time()
                llm_analysis_answer, llm_analysis_error = _call_gemini_with_prompt(
                    analysis_prompt, debug_label="ask_analysis"
                )
                latencies["analysis_llm_latency_sec"] = time.time() - llm_call_start

                if llm_analysis_error:
                    analysis_error = f"LLM Error (Analysis): {llm_analysis_error}"
                    analysis_answer = f"[{analysis_error}]"
                else:
                    analysis_answer = llm_analysis_answer or "[LLM generated no answer from analysis]"

    except Exception as e:
        error_msg = f"Unexpected error during analysis processing: {e}"
        logger.error(error_msg, exc_info=True)
        analysis_error = error_msg
        analysis_answer = "[Error processing analysis]"
        general_errors.append("Error encountered during analysis processing.")

    intermediate_data["analysis_error_details"] = analysis_error

    # 3. Combine and Return
    consolidated_general_error = ", ".join(general_errors) if general_errors else None

    response = schemas.UnifiedAskResponse(
        document_answer=doc_answer,
        analysis_answer=analysis_answer,
        error=consolidated_general_error,
        document_error=doc_error,
        analysis_error=analysis_error
    )

    intermediate_data["final_response"] = response.dict()
    intermediate_data["latencies"] = latencies

    end_time = time.time()
    logger.info(
        f"Unified Ask for '{person_id}' finished in {end_time - start_time:.2f}s. "
        f"Doc Error: {bool(doc_error)}, Analysis Error: {bool(analysis_error)}"
    )

    return response, intermediate_data