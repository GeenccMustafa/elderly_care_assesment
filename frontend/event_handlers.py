# event_handlers.py

import datetime
import html
import json
import logging
import os
import socket
import time
import traceback
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import quote

import gradio as gr
import requests

# --- Constants ---
# Status & General Icons
ICON_SUCCESS = "✅"
ICON_ERROR = "❌"
ICON_WARNING = "⚠️"
ICON_INFO = "ℹ️"
ICON_PENDING = "⏳"
ICON_QUESTION = "❓"
ICON_REFRESH = "🔄"
ICON_LOADING = "⏳"
ICON_CHECK = "✔️"  # Note: Not used for manual prefixing

# Specific Action/Element Icons
ICON_PERSON = "👤"
ICON_AUDIO = "🎤"
ICON_TRANSCRIBE = "✍️"
ICON_SUBMIT = "➡️"
ICON_ANALYSIS = "📊"
ICON_NOTES = "📝"
ICON_UPLOAD = "☁️"
ICON_DOWNLOAD = "💾"
ICON_DOCUMENT = "📄"
ICON_ASK = "💬"
ICON_PLAY = "▶️"

MAX_QUESTIONS = 5  # Number of assessment questions expected in UI

# --- Backend API Configuration ---
# Read from environment variable or use default for local development
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8000").rstrip('/')
API_PREFIX = "/api"

# Construct full API endpoint URLs
START_ASSESSMENT_URL = f"{BACKEND_BASE_URL}{API_PREFIX}/start_assessment"
SUBMIT_ASSESSMENT_URL = f"{BACKEND_BASE_URL}{API_PREFIX}/submit_assessment"
GET_ANALYSIS_URL = f"{BACKEND_BASE_URL}{API_PREFIX}/analysis" # Needs /<person_id>
TRANSCRIBE_URL = f"{BACKEND_BASE_URL}{API_PREFIX}/transcribe_answer"
UPLOAD_PDF_URL = f"{BACKEND_BASE_URL}{API_PREFIX}/upload_personal_pdf"
LIST_PDFS_URL = f"{BACKEND_BASE_URL}{API_PREFIX}/list_personal_pdfs"
DOWNLOAD_PDF_BASE_URL = f"{BACKEND_BASE_URL}{API_PREFIX}/download_personal_pdf" # Needs /<filename>
ASK_QUESTION_URL = f"{BACKEND_BASE_URL}{API_PREFIX}/ask_unified"
LIST_PERSONS_URL = f"{BACKEND_BASE_URL}{API_PREFIX}/list_persons"

# Standard request timeout in seconds
REQUEST_TIMEOUT_SHORT = 15
REQUEST_TIMEOUT_MEDIUM = 60
REQUEST_TIMEOUT_LONG = 300 # For potentially long analysis/transcription

# --- Logging Setup ---
logger = logging.getLogger(__name__)
# Ensure logging is configured externally (e.g., in app.py)

# --- Backend Interaction Functions ---

def _handle_request_exception(
    e: Exception, context: str, person_id: Optional[str] = None
) -> str:
    """Handles common requests exceptions and returns a user-friendly error string."""
    person_context = f" for '{html.escape(person_id)}'" if person_id else ""
    if isinstance(e, requests.exceptions.Timeout):
        err_msg = f"{ICON_ERROR} Timeout during {context}{person_context}."
        logger.error(err_msg)
        gr.Error(f"Request timed out: {context}.")
        return f"[Timeout during {context}]"
    elif isinstance(e, requests.exceptions.HTTPError):
        status_code = e.response.status_code if hasattr(e, 'response') and e.response is not None else 'N/A'
        err_msg = f"{ICON_ERROR} Server Error ({status_code}) during {context}{person_context}."
        logger.error(f"{err_msg} Details: {e}")
        gr.Error(f"Server error ({status_code}) for {context}: {e}")
        return f"[Server Error {status_code}]"
    elif isinstance(e, requests.exceptions.RequestException):
        err_msg = f"{ICON_ERROR} Network Error during {context}{person_context}."
        logger.error(f"{err_msg} Details: {e}")
        gr.Error(f"Network error for {context}: {e}")
        return f"[Network Error]"
    else: # General Python exceptions during request handling
        err_msg = f"{ICON_ERROR} Frontend Error during {context}{person_context}."
        logger.error(f"{err_msg} Details: {e}", exc_info=True)
        gr.Error(f"Application error during {context}: {e}")
        return f"[Application Error]"


def fetch_assessment_questions() -> List[Dict]:
    """Fetches the list of assessment questions from the backend."""
    logger.info(f"{ICON_INFO} Fetching assessment questions from {START_ASSESSMENT_URL}")
    try:
        response = requests.get(START_ASSESSMENT_URL, timeout=REQUEST_TIMEOUT_MEDIUM)
        response.raise_for_status()
        questions_data = response.json()

        if isinstance(questions_data, list):
            # Validate basic structure and sort
            valid_questions = [q for q in questions_data if isinstance(q, dict) and 'question_id' in q]
            questions_list = sorted(valid_questions, key=lambda x: x.get('question_id', 0))
            if len(questions_list) != len(questions_data):
                logger.warning(f"{ICON_WARNING} Some invalid question data received from backend.")
            logger.info(f"{ICON_SUCCESS} Fetched {len(questions_list)} questions.")
            return questions_list
        else:
            logger.error(f"{ICON_ERROR} Unexpected format received for questions: {type(questions_data)}")
            gr.Error(f"Invalid format for questions list: {type(questions_data)}.")
            return []
    except json.JSONDecodeError as e:
        logger.error(f"{ICON_ERROR} Error decoding questions JSON: {e}")
        gr.Error(f"Failed to parse questions response: {e}")
        return []
    except Exception as e:
        _handle_request_exception(e, "fetching questions")
        return []


def fetch_person_list_choices() -> List[str]:
    """Fetches list of existing person IDs for dropdown choices."""
    person_choices = []
    logger.info(f"{ICON_INFO} Fetching person list from {LIST_PERSONS_URL}")
    try:
        response = requests.get(LIST_PERSONS_URL, timeout=REQUEST_TIMEOUT_SHORT)
        response.raise_for_status()
        data = response.json()
        persons = data.get("person_ids", [])
        if isinstance(persons, list):
            person_choices = sorted([str(p) for p in persons if p]) # Ensure strings and filter empty
            logger.info(f"{ICON_SUCCESS} Fetched {len(person_choices)} person choices.")
        else:
            logger.warning(f"{ICON_WARNING} Invalid format for person_ids in response: {type(persons)}")
    except json.JSONDecodeError as e:
        logger.warning(f"{ICON_WARNING} Error decoding person list JSON: {e}")
        gr.Warning(f"Could not parse person list response: {e}")
    except Exception as e:
        _handle_request_exception(e, "fetching person list")
        gr.Warning("Could not fetch person list.") # User-facing warning is sufficient here

    return person_choices


def fetch_person_list_for_load() -> gr.Dropdown:
    """Fetches person list and returns a Gradio update for the dropdown."""
    choices = fetch_person_list_choices()
    return gr.Dropdown(choices=choices, value=None)


def fetch_latest_analysis(person_id: Optional[str]) -> Tuple[str, str]:
    """
    Fetches and formats the latest saved analysis and notes for a given person_id.

    Prioritizes displaying the raw notification string from the backend if available.
    """
    # Default display texts
    analysis_display = f"## {ICON_ANALYSIS} Analysis Results\n\n*{ICON_INFO} Select a Person ID to load analysis.*"
    notifications_display = f"## {ICON_NOTES} Actionable Notes\n\n*{ICON_INFO} Select a Person ID to load notes.*"

    if not person_id or not person_id.strip():
        logger.debug("No person_id provided to fetch_latest_analysis.")
        return analysis_display, notifications_display # Return defaults

    person_id = person_id.strip()
    safe_person_id_display = html.escape(person_id) # Escape for display only
    url = f"{GET_ANALYSIS_URL}/{quote(person_id)}" # Use raw ID for URL
    logger.info(f"{ICON_PENDING} Fetching latest analysis for '{safe_person_id_display}' from {url}")

    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_MEDIUM)
        response.raise_for_status()
        data = response.json()
        logger.info(f"{ICON_SUCCESS} Received analysis data for '{safe_person_id_display}'.")

        analysis = data.get('analysis', {})
        notifications_list = data.get('notifications', []) # Parsed list from backend
        raw_notification_output = data.get("raw_notification_output") # Raw string from backend
        error = data.get('error') # Potential error message from backend

        # Format Analysis Display
        analysis_md = f"## {ICON_ANALYSIS} Latest Analysis ({safe_person_id_display})\n\n"
        analysis_md += f"**Cognitive:**\n> {html.escape(analysis.get('cognitive', 'N/A'))}\n\n"
        analysis_md += f"**Physical:**\n> {html.escape(analysis.get('physical', 'N/A'))}\n\n"
        analysis_md += f"**Health:**\n> {html.escape(analysis.get('health', 'N/A'))}\n\n"
        analysis_md += f"**Personal Info:**\n> {html.escape(analysis.get('personal_info', 'N/A'))}\n\n"
        analysis_display = analysis_md

        # Format Notification Display (Prioritize raw output)
        notifications_header = f"## {ICON_NOTES} Latest Notes ({safe_person_id_display})\n\n"
        if raw_notification_output is not None and isinstance(raw_notification_output, str):
             logger.info(f"Using raw_notification_output from API response for '{safe_person_id_display}'.")
             # Use the raw string directly - assume backend handles formatting. DO NOT escape this.
             notifications_display = notifications_header + raw_notification_output
        elif notifications_list and isinstance(notifications_list, list):
             logger.warning(f"raw_notification_output not found/invalid in API response for '{safe_person_id_display}', falling back to formatting notes list.")
             # Fallback: simple list, no specific formatting assumed
             notes_md = "\n".join([f"- {html.escape(note.get('note', 'N/A'))}" for note in notifications_list if isinstance(note, dict)])
             notifications_display = notifications_header + (notes_md if notes_md else f"*{ICON_INFO} No actionable items parsed from list.*")
        else:
             logger.info(f"No raw notification output or list found for '{safe_person_id_display}'.")
             notifications_display = notifications_header + f"*{ICON_INFO} No actionable items found in latest analysis.*"

        # Handle backend error messages
        if error:
            # Distinguish 'not found' from actual errors
            if "No assessment record found" in error or "No previous analysis found" in error:
                 analysis_display = f"## {ICON_ANALYSIS} Analysis ({safe_person_id_display})\n\n*{ICON_INFO} No prior analysis results found.*"
                 notifications_display = f"## {ICON_NOTES} Notes ({safe_person_id_display})\n\n*{ICON_INFO} No prior notes found.*"
                 logger.info(f"{ICON_INFO} No prior analysis found for '{safe_person_id_display}'.")
            else:
                 # Report other backend errors as warnings
                 gr.Warning(f"{ICON_WARNING} Issue retrieving analysis for '{safe_person_id_display}': {error}")

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
             analysis_display = f"## {ICON_ANALYSIS} Analysis ({safe_person_id_display})\n\n*{ICON_INFO} No analysis results found for this person.*"
             notifications_display = f"## {ICON_NOTES} Notes ({safe_person_id_display})\n\n*{ICON_INFO} No notes found for this person.*"
             logger.info(f"{ICON_INFO} No analysis found for '{safe_person_id_display}' (HTTP 404).")
        else:
             err_str = _handle_request_exception(e, "fetching analysis", person_id)
             analysis_display = f"## {ICON_ANALYSIS} Analysis ({safe_person_id_display})\n\n{err_str}"
             notifications_display = f"## {ICON_NOTES} Notes ({safe_person_id_display})\n\n{err_str}"
    except Exception as e:
        err_str = _handle_request_exception(e, "processing analysis", person_id)
        analysis_display = f"## {ICON_ANALYSIS} Analysis ({safe_person_id_display})\n\n{err_str}"
        notifications_display = f"## {ICON_NOTES} Notes ({safe_person_id_display})\n\n{err_str}"

    return analysis_display, notifications_display


def transcribe_answer_audio(
    audio_filepath: Optional[str], question_id: int, current_transcriptions: Dict
) -> Tuple[Dict, str, str]:
    """
    Sends audio file to backend for transcription, updates state and returns status.
    Cleans up temporary audio file afterwards.
    """
    if not isinstance(current_transcriptions, dict):
        current_transcriptions = {} # Initialize if state is invalid

    q_label = f"Q{question_id}"
    transcription_result_text = current_transcriptions.get(question_id, "") # Keep previous if error

    if audio_filepath is None:
        current_transcriptions[question_id] = "[Audio not provided]"
        status_msg = f"{ICON_WARNING} {q_label}: No audio file."
        return current_transcriptions, status_msg, "[Audio not provided]"

    status_msg = f"{ICON_PENDING} {q_label}: Transcribing..."
    logger.info(f"{ICON_PENDING} Transcribing {q_label} from temp file: {audio_filepath}...")

    try:
        with open(audio_filepath, 'rb') as f:
            files = {'audio_file': (os.path.basename(audio_filepath), f, 'audio/wav')} # Assume WAV, adjust if needed
            response = requests.post(TRANSCRIBE_URL, files=files, timeout=REQUEST_TIMEOUT_LONG)
        response.raise_for_status()
        data = response.json()

        transcription = data.get("transcription", "[Backend Transcription Error]")
        error = data.get("error") # Check for specific error from backend schema
        current_transcriptions[question_id] = transcription
        transcription_result_text = transcription # Update the display text
        logger.info(f"{ICON_INFO} Transcription result {q_label}: '{transcription[:50]}...'")

        # Update status based on result
        if error:
            status_msg = f"{ICON_ERROR} {q_label}: Error - {html.escape(error)}"
            gr.Warning(f"Transcription issue {q_label}: {error}")
        elif transcription == "[No speech detected]":
            status_msg = f"{ICON_WARNING} {q_label}: No speech detected."
        elif transcription.startswith("[Error"): # Catch other backend errors in transcription string
            status_msg = f"{ICON_ERROR} {q_label}: Backend Error"
            gr.Warning(f"Transcription issue {q_label}: {html.escape(transcription)}")
        else:
            status_msg = f"{ICON_SUCCESS} {q_label}: Transcribed."

    except Exception as e:
        err_str = _handle_request_exception(e, f"transcribing {q_label}")
        status_msg = f"{ICON_ERROR} {q_label}: {err_str}"
        current_transcriptions[question_id] = err_str # Store error in state
        transcription_result_text = err_str # Display error

    finally:
        # Attempt to clean up the temporary file Gradio created
        if audio_filepath and isinstance(audio_filepath, str):
            try:
                if os.path.exists(audio_filepath):
                    os.remove(audio_filepath)
                    logger.info(f"Removed temp audio file: {audio_filepath}")
                else:
                    logger.warning(f"Temp audio file not found for removal: {audio_filepath}")
            except OSError as rm_err:
                logger.warning(f"Could not remove temp audio file {audio_filepath}: {rm_err}")
        elif audio_filepath:
            logger.warning(f"Could not verify path for audio cleanup: {audio_filepath}")

    return current_transcriptions, status_msg, transcription_result_text


def submit_assessment(
    person_id: Optional[str],
    transcribed_answers_dict: Dict,
    questions_list: List[Dict]
):
    """
    Submits the collected transcriptions for analysis.
    Yields status updates and final analysis/notification display strings.
    Prioritizes raw notification output for display.
    """
    # Initial status yields
    status = f"{ICON_PENDING} Preparing submission..."
    analysis_display = f"## {ICON_ANALYSIS} Analysis Results\n\n{ICON_PENDING} Waiting for submission..."
    notifications_display = f"## {ICON_NOTES} Actionable Notes\n\n{ICON_PENDING} Waiting for submission..."
    yield status, analysis_display, notifications_display

    # --- Input Validation ---
    if not person_id or not person_id.strip():
        status = f"{ICON_ERROR} Person ID is required."
        gr.Error(status)
        analysis_display = f"## {ICON_ANALYSIS} Analysis Results\n\n*{ICON_ERROR} Submission failed: {html.escape(status)}*"
        notifications_display = f"## {ICON_NOTES} Actionable Notes\n\n*{ICON_ERROR} Submission failed: {html.escape(status)}*"
        yield status, analysis_display, notifications_display
        return

    person_id = person_id.strip()
    safe_person_id_display = html.escape(person_id) # For display

    if not questions_list or not isinstance(questions_list, list):
        status = f"{ICON_ERROR} Questions list not loaded or invalid."
        gr.Error(status)
        analysis_display = f"## {ICON_ANALYSIS} Analysis Results ({safe_person_id_display})\n\n*{ICON_ERROR} Submission failed: {html.escape(status)}*"
        notifications_display = f"## {ICON_NOTES} Actionable Notes ({safe_person_id_display})\n\n*{ICON_ERROR} Submission failed: {html.escape(status)}*"
        yield status, analysis_display, notifications_display
        return

    if not isinstance(transcribed_answers_dict, dict):
        status = f"{ICON_ERROR} Internal error: Transcription data invalid."
        gr.Error(status)
        analysis_display = f"## {ICON_ANALYSIS} Analysis Results ({safe_person_id_display})\n\n*{ICON_ERROR} Submission failed: {html.escape(status)}*"
        notifications_display = f"## {ICON_NOTES} Actionable Notes ({safe_person_id_display})\n\n*{ICON_ERROR} Submission failed: {html.escape(status)}*"
        yield status, analysis_display, notifications_display
        return

    # --- Prepare Payload ---
    payload_answers = []
    all_valid_answers = True
    missing_q_ids = []
    for q_data in questions_list:
        q_id = q_data.get('question_id')
        if q_id is None:
            logger.warning("Skipping question with missing ID in questions_list.")
            continue

        answer_text = transcribed_answers_dict.get(q_id)
        # Consider an answer missing/invalid if it's None, empty, or still an error placeholder
        is_invalid = (
            answer_text is None or
            not str(answer_text).strip() or
            str(answer_text).startswith("[") # Assumes errors start with "["
        )
        if is_invalid:
            final_answer_text = str(answer_text) if answer_text else "[Not Answered]"
            logger.warning(f"{ICON_WARNING} No valid transcription for Q{q_id} ('{safe_person_id_display}'). Using placeholder: '{final_answer_text}'.")
            all_valid_answers = False
            missing_q_ids.append(str(q_id))
        else:
            final_answer_text = str(answer_text)

        payload_answers.append({"question_id": q_id, "answer_text": final_answer_text})

    if not all_valid_answers:
        gr.Warning(f"{ICON_WARNING} Submitting with missing/invalid answers for Q(s): {', '.join(missing_q_ids)} for '{safe_person_id_display}'.")

    payload = {"person_id": person_id, "answers": payload_answers}
    status = f"{ICON_PENDING} Submitting analysis request for '{safe_person_id_display}'..."
    analysis_display = f"## {ICON_ANALYSIS} Analysis Results ({safe_person_id_display})\n\n{ICON_PENDING} Requesting analysis..."
    notifications_display = f"## {ICON_NOTES} Actionable Notes ({safe_person_id_display})\n\n{ICON_PENDING} Requesting analysis..."
    yield status, analysis_display, notifications_display

    # --- API Call ---
    try:
        response = requests.post(SUBMIT_ASSESSMENT_URL, json=payload, timeout=REQUEST_TIMEOUT_LONG)
        response.raise_for_status()
        data = response.json() # Expects FullAnalysisResponse schema
        logger.info(f"{ICON_SUCCESS} Received analysis response for '{safe_person_id_display}'.")

        analysis = data.get('analysis', {})
        notifications_list = data.get('notifications', []) # Parsed list
        raw_notification_output = data.get("raw_notification_output") # Raw string
        error = data.get('error') # Backend error message

        # Format Analysis Display
        analysis_md = f"## {ICON_ANALYSIS} Analysis Results ({safe_person_id_display})\n\n"
        analysis_md += f"**Cognitive:**\n> {html.escape(analysis.get('cognitive', 'N/A'))}\n\n"
        analysis_md += f"**Physical:**\n> {html.escape(analysis.get('physical', 'N/A'))}\n\n"
        analysis_md += f"**Health:**\n> {html.escape(analysis.get('health', 'N/A'))}\n\n"
        analysis_md += f"**Personal Info:**\n> {html.escape(analysis.get('personal_info', 'N/A'))}\n\n"
        analysis_display = analysis_md

        # Format Notification Display (Prioritize raw output)
        notifications_header = f"## {ICON_NOTES} Actionable Notes ({safe_person_id_display})\n\n"
        if raw_notification_output is not None and isinstance(raw_notification_output, str):
            logger.info(f"Using raw_notification_output for '{safe_person_id_display}' display.")
            # Use the raw string directly; DO NOT escape.
            notifications_display = notifications_header + raw_notification_output
        elif notifications_list and isinstance(notifications_list, list):
            logger.warning(f"raw_notification_output not in API response for '{safe_person_id_display}'. Falling back to list format.")
            notes_md = "\n".join([f"- {html.escape(note.get('note', 'N/A'))}" for note in notifications_list if isinstance(note, dict)])
            notifications_display = notifications_header + (notes_md if notes_md else f"*{ICON_INFO} No actionable items parsed.*")
        else:
            logger.info(f"No raw notification output or list found in response for '{safe_person_id_display}'.")
            notifications_display = notifications_header + f"*{ICON_INFO} No actionable items identified.*"

        # Final Status Update
        if error:
            status = f"{ICON_WARNING} Analysis completed with issues for '{safe_person_id_display}': {html.escape(error)}"
            gr.Warning(f"Analysis backend issue for '{safe_person_id_display}': {error}")
        else:
            status = f"{ICON_SUCCESS} Analysis Complete for '{safe_person_id_display}'."

    except Exception as e:
        err_str = _handle_request_exception(e, "submitting assessment", person_id)
        status = f"{ICON_ERROR} Submission failed for '{safe_person_id_display}'."
        analysis_display = f"## {ICON_ANALYSIS} Analysis Results ({safe_person_id_display})\n\n{err_str}"
        notifications_display = f"## {ICON_NOTES} Actionable Notes ({safe_person_id_display})\n\n{err_str}"

    yield status, analysis_display, notifications_display


def handle_pdf_upload(pdf_file_list: Union[List, object], progress=gr.Progress(track_tqdm=True)):
    """Handles uploading one or more PDF files to the backend."""
    if not pdf_file_list:
        return f"{ICON_INFO} No PDF files selected for upload.", update_pdf_list()

    # Ensure pdf_file_list is a list, Gradio might pass single object if only one selected
    if not isinstance(pdf_file_list, list):
        pdf_file_list = [pdf_file_list]

    upload_statuses = []
    success_count = 0
    fail_count = 0
    total_files = len(pdf_file_list)
    logger.info(f"{ICON_PENDING} Starting upload of {total_files} PDF file(s)...")

    for i, pdf_file_obj in enumerate(progress.tqdm(pdf_file_list, desc="Uploading PDFs", unit="file")):
        # Basic check for valid Gradio file object
        if pdf_file_obj is None or not hasattr(pdf_file_obj, 'name') or not pdf_file_obj.name:
             logger.warning(f"-- Skipping invalid/missing file object at index {i}")
             fail_count += 1 # Count as failure if object is invalid
             upload_statuses.append(f"{ICON_ERROR} Failed: Invalid file object provided by interface.")
             continue

        # Gradio File object '.name' attribute holds the temporary file path
        filepath = pdf_file_obj.name
        filename = os.path.basename(filepath) # Use original filename for upload 'name' field
        safe_display_filename = html.escape(filename)
        logger.info(f"-- Uploading PDF ({i+1}/{total_files}): '{safe_display_filename}' from temp path {filepath}")

        try:
            with open(filepath, 'rb') as f:
                files = {'pdf_file': (filename, f, 'application/pdf')}
                response = requests.post(UPLOAD_PDF_URL, files=files, timeout=REQUEST_TIMEOUT_MEDIUM)
            response.raise_for_status()
            data = response.json()
            # Use filename returned by backend if available, otherwise original
            returned_filename = data.get('filename', filename)
            upload_statuses.append(f"{ICON_SUCCESS} Success: {html.escape(returned_filename)}")
            success_count += 1
        except Exception as e:
            fail_count += 1
            err_context = f"uploading '{safe_display_filename}'"
            # Try to get detailed error from response if available
            error_detail = str(e)
            status_code = 'N/A'
            if isinstance(e, requests.exceptions.RequestException):
                if hasattr(e, 'response') and e.response is not None:
                    status_code = e.response.status_code
                    try: # Try to parse JSON detail from error response
                         error_detail = e.response.json().get('detail', str(e))
                    except (json.JSONDecodeError, AttributeError): pass
            # Use the helper, but customize the final message slightly
            err_str = _handle_request_exception(e, err_context).replace("[", "").replace("]", "") # Remove brackets for status line
            upload_statuses.append(f"{ICON_ERROR} Failed ({err_str} {status_code}): {safe_display_filename} - {html.escape(error_detail)}")

    # Final status summary
    summary_icon = ICON_SUCCESS if fail_count == 0 else (ICON_ERROR if success_count == 0 else ICON_WARNING)
    summary_status = (
        f"{summary_icon} Document Upload Complete: {success_count} succeeded, {fail_count} failed.\n\n"
        + "\n".join(upload_statuses)
    )
    # Add note about index rebuild only if uploads were successful
    rebuild_note = (
        f"\n\n**{ICON_WARNING} IMPORTANT:** If document content changed, run "
        "`python build_index.py` and restart the backend service "
        "to update the search index."
    ) if success_count > 0 else ""
    summary_status += rebuild_note

    # Return summary and trigger PDF list refresh
    return summary_status, update_pdf_list()


def update_pdf_list_choices() -> List[str]:
    """Fetches the list of available PDFs from the backend."""
    pdf_files = []
    logger.info(f"{ICON_INFO} Fetching PDF list from {LIST_PDFS_URL}")
    try:
        response = requests.get(LIST_PDFS_URL, timeout=REQUEST_TIMEOUT_SHORT)
        response.raise_for_status()
        data = response.json()
        files = data.get("pdf_files", [])
        if isinstance(files, list):
            pdf_files = sorted([str(f) for f in files if f]) # Ensure strings and filter empty
            logger.info(f"{ICON_SUCCESS} Fetched {len(pdf_files)} PDF file names.")
        else:
            logger.warning(f"{ICON_WARNING} Invalid format for pdf_files in response: {type(files)}")
    except json.JSONDecodeError as e:
        logger.warning(f"{ICON_WARNING} Error decoding PDF list JSON: {e}")
        gr.Warning(f"Could not parse PDF list response: {e}")
    except Exception as e:
        _handle_request_exception(e, "fetching PDF list")
        gr.Warning("Could not fetch PDF list.") # User warning is sufficient

    return pdf_files


def update_pdf_list() -> gr.Dropdown:
    """Fetches PDF list and returns Gradio update object for the dropdown."""
    choices = update_pdf_list_choices()
    return gr.Dropdown(choices=choices, value=None)


def generate_download_link(selected_filename: Optional[str]) -> str:
    """Creates a Markdown link to download the selected PDF file."""
    if not selected_filename or not isinstance(selected_filename, str):
        return f"*{ICON_INFO} Select a PDF file from the list to generate a download link.*"

    # Basic validation against common path traversal attempts
    if "/" in selected_filename or "\\" in selected_filename or ".." in selected_filename:
        logger.warning(f"Invalid characters detected in filename for download: {selected_filename}")
        return f"*{ICON_ERROR} Invalid filename format selected.*"

    safe_display_filename = html.escape(selected_filename)
    try:
        # URL-encode the filename for the path parameter
        encoded_filename = quote(selected_filename)
        download_url = f"{DOWNLOAD_PDF_BASE_URL}/{encoded_filename}"
        logger.info(f"{ICON_INFO} Generated download link for '{safe_display_filename}'. URL: {download_url}")
        # Create a Markdown link that uses the 'download' attribute
        return (
            f"Click to download: **<a href='{download_url}' target='_blank' "
            f"download='{safe_display_filename}'>{ICON_DOWNLOAD} {safe_display_filename}</a>**"
        )
    except Exception as e:
        logger.error(f"{ICON_ERROR} Error generating download link for '{safe_display_filename}': {e}", exc_info=True)
        return f"*{ICON_ERROR} Error creating download link for the selected file.*"


def ask_question_backend(person_id: Optional[str], question: Optional[str]):
    """
    Sends a question to the backend's unified Q&A endpoint.
    Yields updates for document-based answer, analysis-based answer, and status.
    Does NOT escape answers received from the backend (assumes backend handled it).
    Escapes locally generated error messages or user input for display.
    """
    # Initial pending states
    doc_answer_display = f"{ICON_PENDING} Asking document source..."
    analysis_answer_display = f"{ICON_PENDING} Asking analysis source..."
    qa_status = f"{ICON_PENDING} Sending question..."

    # --- Input Validation ---
    if not person_id or not person_id.strip():
        qa_status = f"{ICON_ERROR} Person ID is required."
        gr.Error(qa_status)
        # Escape locally generated error message
        error_msg_display = html.escape(f"{ICON_ERROR} Cannot ask: {qa_status}")
        yield error_msg_display, error_msg_display, qa_status
        return

    person_id = person_id.strip()
    safe_person_id_display = html.escape(person_id) # For display only

    if not question or not question.strip():
        qa_status = f"{ICON_ERROR} Question cannot be empty."
        gr.Error(qa_status)
         # Escape locally generated error message
        error_msg_display = html.escape(f"{ICON_ERROR} Cannot ask: {qa_status}")
        yield error_msg_display, error_msg_display, qa_status
        return

    question = question.strip()
    safe_question_display = html.escape(question[:50]) # For logging

    logger.info(f"{ICON_PENDING} Sending Unified Q&A about '{safe_person_id_display}': '{safe_question_display}...' to {ASK_QUESTION_URL}")
    # Yield initial pending status
    yield doc_answer_display, analysis_answer_display, qa_status

    # --- API Call ---
    try:
        # Use raw person_id and question for the API payload
        payload = {"person_id": person_id, "question": question}
        response = requests.post(ASK_QUESTION_URL, json=payload, timeout=REQUEST_TIMEOUT_LONG)
        response.raise_for_status()
        data = response.json() # Expects schemas.UnifiedAskResponse

        # Extract potentially raw answers and error flags from backend
        doc_answer_raw = data.get("document_answer", "[Backend response missing document answer]")
        analysis_answer_raw = data.get("analysis_answer", "[Backend response missing analysis answer]")
        doc_error = data.get("document_error")       # Specific error for doc part
        analysis_error = data.get("analysis_error") # Specific error for analysis part
        general_error = data.get("error")           # Overall processing error

        logger.info(f"Raw doc answer received for '{safe_person_id_display}': '{str(doc_answer_raw)[:100]}...'")
        logger.info(f"Raw analysis answer received for '{safe_person_id_display}': '{str(analysis_answer_raw)[:100]}...'")

        # Process Document Answer for Display
        if doc_error:
            doc_answer_display = f"{ICON_ERROR} Error: {html.escape(doc_error)}" # Escape backend error message
            gr.Warning(f"Unified Ask (Doc) issue for '{safe_person_id_display}': {doc_error}")
        elif isinstance(doc_answer_raw, str) and doc_answer_raw.startswith("[Error"):
             doc_answer_display = f"{ICON_ERROR} {html.escape(doc_answer_raw)}" # Escape backend error string in answer
             gr.Warning(f"Unified Ask Doc Answer Error ('{safe_person_id_display}'): {doc_answer_raw}")
        else:
            # Use the raw answer directly - DO NOT escape. Assumes backend provides safe HTML or plain text.
            doc_answer_display = doc_answer_raw if doc_answer_raw is not None else "[No document answer provided]"

        # Process Analysis Answer for Display
        if analysis_error:
            analysis_answer_display = f"{ICON_ERROR} Error: {html.escape(analysis_error)}" # Escape backend error message
            gr.Warning(f"Unified Ask (Analysis) issue for '{safe_person_id_display}': {analysis_error}")
        elif isinstance(analysis_answer_raw, str) and analysis_answer_raw.startswith("[Error"):
             analysis_answer_display = f"{ICON_ERROR} {html.escape(analysis_answer_raw)}" # Escape backend error string in answer
             gr.Warning(f"Unified Ask Analysis Answer Error ('{safe_person_id_display}'): {analysis_answer_raw}")
        else:
             # Use the raw answer directly - DO NOT escape.
            analysis_answer_display = analysis_answer_raw if analysis_answer_raw is not None else "[No analysis answer provided]"

        # Determine Overall Status
        status_parts = []
        if doc_error: status_parts.append(f"{ICON_WARNING} Doc Issue")
        if analysis_error: status_parts.append(f"{ICON_WARNING} Analysis Issue")
        if general_error:
            status_parts.append(f"{ICON_ERROR} General Issue")
            gr.Warning(f"Unified Ask (General) issue for '{safe_person_id_display}': {general_error}")
            # If a general error occurred, display it if no specific error was shown
            # Escape the general error message from backend
            general_error_display = f"{ICON_ERROR} Error: {html.escape(general_error)}"
            if not doc_error: doc_answer_display = general_error_display
            if not analysis_error: analysis_answer_display = general_error_display

        # Construct final status message
        if not status_parts:
             qa_status = f"{ICON_SUCCESS} Answers received for '{safe_person_id_display}'."
        else:
             qa_status = "; ".join(status_parts) + f" for '{safe_person_id_display}'."

    # --- Exception Handling ---
    except Exception as e:
        # Use helper to handle standard request errors and log appropriately
        err_str = _handle_request_exception(e, "asking unified question", person_id)
        # Use the safe string returned by the helper for display
        qa_status = f"{ICON_ERROR} Q&A Failed"
        doc_answer_display = err_str
        analysis_answer_display = err_str

    # Yield the final results for display
    yield doc_answer_display, analysis_answer_display, qa_status


# --- UI Update Helpers ---

def _reset_assessment_ui_elements() -> List:
    """Returns a list of gr.update calls to reset assessment question elements."""
    updates = []
    for _ in range(MAX_QUESTIONS):
        updates.extend([
            gr.update(value=None),    # Clear audio player
            gr.update(value=""),      # Clear transcription display Textbox
            gr.update(value=f"{ICON_INFO} Ready") # Reset question status Textbox
        ])
    return updates

def _reset_results_ui_elements() -> List:
    """Returns a list of gr.update calls to reset analysis/notes display."""
    return [
        gr.update(value=f"{ICON_INFO} Idle"), # Overall Status Textbox
        gr.update(value=f"## {ICON_ANALYSIS} Analysis Results\n\n*{ICON_INFO} Select or refresh Person ID.*"),
        gr.update(value=f"## {ICON_NOTES} Actionable Notes\n\n*{ICON_INFO} Select or refresh Person ID.*")
    ]

def _reset_qa_ui_elements() -> List:
    """Returns a list of gr.update calls to reset the Q&A tab elements."""
    return [
        gr.update(value=""), # Clear Q&A question_input
        gr.update(value=""), # Clear Q&A doc_answer_output
        gr.update(value=""), # Clear Q&A analysis_answer_output
        gr.update(value=f"{ICON_INFO} Idle") # Reset Q&A qa_status_output
    ]

def update_active_person_dd(selected_person_id: Optional[str]):
    """
    Handles selection change in the Person ID dropdown.
    Updates active person state, resets relevant UI sections (Assessment, Q&A, Results).
    Triggers fetching of latest analysis in a subsequent step (via .then()).
    """
    active_id = selected_person_id if selected_person_id else ""
    logger.info(f"{ICON_PERSON} Active Person ID changed to: '{html.escape(active_id)}'. Resetting UI sections.")

    updates = [
        active_id, # 1. Update Active Person State
        {},        # 2. Clear Transcriptions State
    ]
    updates.extend(_reset_results_ui_elements())    # Reset analysis/notes/status
    updates.extend(_reset_assessment_ui_elements()) # Reset question audio/text/status
    updates.extend(_reset_qa_ui_elements())         # Reset Q&A inputs/outputs

    # Also reset the question text itself to placeholder/loading
    for _ in range(MAX_QUESTIONS):
        updates.append(gr.update(value=f"*{ICON_LOADING}*")) # Reset question markdown text

    return updates


def handle_person_refresh():
    """
    Handles click on the Person list refresh button.
    Fetches updated person list, updates dropdown, resets active person,
    and resets all relevant UI sections.
    Triggers fetching assessment questions in a subsequent step (via .then()).
    """
    logger.info(f"{ICON_REFRESH} Handling person list refresh and full UI reset.")
    person_choices = fetch_person_list_choices()

    updates = [
        gr.update(choices=person_choices, value=None), # 1. Update Dropdown choices & selection
        "",        # 2. Clear Active Person State
        {},        # 3. Clear Transcriptions State
    ]
    updates.extend(_reset_results_ui_elements())    # Reset analysis/notes/status
    updates.extend(_reset_assessment_ui_elements()) # Reset question audio/text/status
    updates.extend(_reset_qa_ui_elements())         # Reset Q&A inputs/outputs

     # Also reset the question text itself to placeholder/loading
    for _ in range(MAX_QUESTIONS):
        updates.append(gr.update(value=f"*{ICON_LOADING}*")) # Reset question markdown text

    return updates


def switch_view(view_name: str):
    """
    Updates visibility of content columns and button styles based on the selected view tab.
    """
    logger.info(f"{ICON_INFO} Switching main view to: {view_name}")
    is_assessment = (view_name == "assessment")
    is_docs = (view_name == "docs")
    is_ask = (view_name == "ask")

    # Update button variants: primary for active, secondary for others
    assessment_btn_variant = "primary" if is_assessment else "secondary"
    docs_btn_variant = "primary" if is_docs else "secondary"
    ask_btn_variant = "primary" if is_ask else "secondary"

    return (
        view_name,                          # Update view_state
        gr.update(visible=is_assessment),   # assessment_column visibility
        gr.update(visible=is_docs),         # docs_column visibility
        gr.update(visible=is_ask),          # ask_column visibility
        gr.update(variant=assessment_btn_variant), # assessment_btn style
        gr.update(variant=docs_btn_variant),       # docs_btn style
        gr.update(variant=ask_btn_variant)         # ask_btn style
    )