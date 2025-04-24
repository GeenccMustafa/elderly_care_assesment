# backend/app/api.py
import datetime
import json
import logging
import os
import time
import traceback
from pathlib import Path as PyPath
from typing import Any, Dict, List, Optional

import mlflow
import pandas as pd
from fastapi import (APIRouter, Depends, File, Form, HTTPException, Path as FastApiPath,
                     UploadFile)
from fastapi.responses import FileResponse, JSONResponse
# --- MLflow Integration ---
from mlflow.data.pandas_dataset import PandasDataset

# Local application imports
from . import config, data_manager, schemas, services

# --- MLflow Configuration ---
MLFLOW_TRACKING_URI = config.MLFLOW_TRACKING_URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = config.MLFLOW_EXPERIMENT_NAME
experiment_id = None
try:
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    print(
        f"MLflow configured: Tracking URI='{MLFLOW_TRACKING_URI}', "
        f"Experiment='{EXPERIMENT_NAME}' (ID: {experiment_id})"
    )
except Exception as e:
    print(f"Error configuring MLflow experiment: {e}.")
# --- End MLflow Configuration ---

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

router = APIRouter()
PDF_STORAGE_DIR = config.PERSONAL_DOCS_STORAGE_DIR
PDF_STORAGE_DIR.mkdir(parents=True, exist_ok=True)  # Ensure directory exists


# === Helper Function for MLflow Logging ===
def _log_dict_as_json(
    data: Dict[str, Any], artifact_name: str, run_id: Optional[str] = None
):
    """Logs a dictionary as a JSON artifact in the current or specified MLflow run."""
    active_run = mlflow.active_run()
    if not active_run and not run_id:
        logger.warning(
            f"Cannot log artifact '{artifact_name}': No active MLflow run."
        )
        return
    current_run_id = run_id or active_run.info.run_id

    temp_dir = PyPath("./mlflow_temp_artifacts")
    temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / f"{artifact_name}.json"

    try:
        # Use helper to ensure data is JSON serializable before dumping
        cleaned_data = data_manager._convert_sets_to_lists(data)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=4)
        mlflow.log_artifact(
            str(file_path), artifact_path="run_data", run_id=current_run_id
        )
    except Exception as e:
        logger.warning(
            f"MLflow failed to log dictionary artifact '{artifact_name}.json' "
            f"for run {current_run_id}: {e}"
        )
    finally:
        # Clean up the temporary file and directory
        if file_path.exists():
            try:
                os.remove(file_path)
            except OSError as rm_err:
                logger.error(f"Error removing temp file {file_path}: {rm_err}")
        try:
            # Remove temp dir only if it's empty
            if temp_dir.exists() and not any(temp_dir.iterdir()):
                temp_dir.rmdir()
        except OSError as rmdir_err:
            # Ignore error if dir is not empty or other issue occurs
            logger.debug(f"Could not remove temp dir {temp_dir}: {rmdir_err}")


# ============================
# Transcription Endpoint
# ============================
class TranscriptionResponse(schemas.BaseModel):
    transcription: str
    error: Optional[str] = None

@router.post("/transcribe_answer", response_model=TranscriptionResponse)
async def transcribe_answer_endpoint(audio_file: UploadFile = File(...)):
    """Transcribes the provided audio file."""
    logger.info(
        f"Received audio: {audio_file.filename}, "
        f"type: {audio_file.content_type}"
    )
    start_time = time.time()

    if not audio_file.content_type or not audio_file.content_type.startswith(
        "audio/"
    ):
        logger.warning(f"Invalid audio type: {audio_file.content_type}")
        raise HTTPException(
            status_code=400, detail="Invalid audio file type."
        )

    contents = await audio_file.read()
    await audio_file.close()

    if not contents:
        logger.warning("Empty audio file received.")
        # Use JSONResponse for specific error structure matching response model
        return JSONResponse(
            status_code=400,
            content={"transcription": "", "error": "Empty audio file."}
        )

    try:
        transcription = await services.transcribe_audio(contents)
        latency = time.time() - start_time
        logger.info(
            f"Transcription done ({latency:.2f}s): '{transcription[:100]}...'"
        )

        # Check if the result indicates an error from the service
        error_msg = None
        if transcription.startswith(
            "[Error"
        ) or transcription.startswith(
            "[Transcription Error"
        ):
            error_msg = transcription

        return TranscriptionResponse(transcription=transcription, error=error_msg)
    except Exception as e:
        logger.error(f"Transcription endpoint error: {e}", exc_info=True)
        # Use JSONResponse for specific error structure matching response model
        return JSONResponse(
            status_code=500,
            content={
                "transcription": "",
                "error": f"Internal server error: {e}"
            }
        )


# ============================
# Assessment Endpoints
# ============================

@router.get("/start_assessment", response_model=List[schemas.AssessmentQuestion])
async def start_assessment_endpoint():
    """Provides the standard list of assessment questions."""
    logger.info("Request received for assessment questions.")
    return [
        schemas.AssessmentQuestion(question_id=i + 1, text=q_text)
        for i, q_text in enumerate(config.ASSESSMENT_QUESTIONS)
    ]


@router.post("/submit_assessment", response_model=schemas.FullAnalysisResponse)
async def submit_assessment_endpoint(request: schemas.AnalysisRequest):
    """
    Receives assessment answers, performs analysis, logs to MLflow,
    saves results, and returns the analysis including raw notifications.
    """
    request_start_time = time.time()
    person_id = request.person_id

    if not person_id or not person_id.strip():
        raise HTTPException(status_code=400, detail="person_id is required.")
    person_id = person_id.strip()
    logger.info(
        f"Received submission for '{person_id}' ({len(request.answers)} answers)."
    )

    run_name = (
        f"AssessmentAnalysis_{person_id}_"
        f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    answers_dict = {a.question_id: a.answer_text for a in request.answers}

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run '{run_id}' for Assessment '{person_id}'.")

        # --- MLflow Initial Logging ---
        mlflow.set_tags({
            "endpoint": "/submit_assessment",
            "person_id": person_id,
            "status": "PROCESSING",
            "llm_model": config.GEMINI_MODEL_NAME
        })
        mlflow.log_params({"num_answers_submitted": len(request.answers)})
        _log_dict_as_json(
            request.dict(), f"input_request_{person_id}", run_id=run_id
        )
        _log_dict_as_json(
            answers_dict, f"input_submitted_answers_{person_id}", run_id=run_id
        )

        final_status = "UNKNOWN"
        analysis_response_obj = None
        save_error = None
        raw_notification_text_from_service: Optional[str] = None

        try:
            # --- Call Analysis Service ---
            (
                analysis_result_base_obj,
                prompts_used,
                _,  # pdf_context is not used here
                _,  # pdf_context_latency is not used here
                llm_latencies,
                raw_notification_text_from_service
            ) = await services.analyze_assessment(person_id, request.answers)

            # --- Construct Final Response Object ---
            analysis_response_obj = schemas.FullAnalysisResponse(
                analysis=analysis_result_base_obj.analysis,
                notifications=analysis_result_base_obj.notifications,
                error=analysis_result_base_obj.error,
                raw_notification_output=raw_notification_text_from_service
            )

            # --- MLflow Results Logging ---
            mlflow.log_text(
                "N/A (Not Used)",
                f"intermediate_pdf_context_{person_id}.txt"
            )
            _log_dict_as_json(
                prompts_used,
                f"intermediate_prompts_used_{person_id}",
                run_id=run_id
            )
            mlflow.log_metric("pdf_context_retrieval_latency_sec", 0.0)
            mlflow.log_metrics(llm_latencies)
            mlflow.log_metric(
                "num_notification_notes_parsed",
                len(analysis_response_obj.notifications)
            )
            mlflow.log_metrics({
                "cognitive_analysis_length": len(
                    analysis_response_obj.analysis.cognitive or ""
                ),
                "physical_analysis_length": len(
                    analysis_response_obj.analysis.physical or ""
                ),
                "health_analysis_length": len(
                    analysis_response_obj.analysis.health or ""
                ),
                "personal_info_length": len(
                    analysis_response_obj.analysis.personal_info or ""
                ),
                "raw_notification_output_length": len(
                    analysis_response_obj.raw_notification_output or ""
                )
            })

            # Log response object and individual text components
            _log_dict_as_json(
                analysis_response_obj.dict(),
                f"output_analysis_response_{person_id}",
                run_id=run_id
            )
            mlflow.log_text(
                analysis_response_obj.analysis.cognitive or "N/A",
                f"output_analysis_cognitive_{person_id}.txt"
            )
            mlflow.log_text(
                analysis_response_obj.analysis.physical or "N/A",
                f"output_analysis_physical_{person_id}.txt"
            )
            mlflow.log_text(
                analysis_response_obj.analysis.health or "N/A",
                f"output_analysis_health_{person_id}.txt"
            )
            mlflow.log_text(
                analysis_response_obj.analysis.personal_info or "N/A",
                f"output_analysis_personal_info_{person_id}.txt"
            )
            mlflow.log_text(
                analysis_response_obj.raw_notification_output or "None",
                f"output_raw_notification_string_{person_id}.txt"
            )

            # --- MLflow Dataset Logging ---
            try:
                notes_str_parsed = "\n".join(
                    [n.note for n in analysis_response_obj.notifications]
                )
                dataset_df = pd.DataFrame([{
                    "person_id": person_id,
                    "num_answers": len(request.answers),
                    "input_answers": json.dumps(answers_dict),
                    "output_cognitive": (
                        analysis_response_obj.analysis.cognitive
                    ),
                    "output_physical": (
                        analysis_response_obj.analysis.physical
                    ),
                    "output_health": analysis_response_obj.analysis.health,
                    "output_personal": (
                        analysis_response_obj.analysis.personal_info
                    ),
                    "output_notifications_parsed": notes_str_parsed,
                    "output_notifications_raw": (
                        analysis_response_obj.raw_notification_output
                    ),
                    "has_error": bool(analysis_response_obj.error)
                }])
                mlflow_dataset = mlflow.data.from_pandas(
                    dataset_df,
                    source=f"api:/submit_assessment/{person_id}",
                    name=f"assessment_io_{person_id}"
                )
                mlflow.log_input(mlflow_dataset, context="assessment_io")
            except Exception as ds_log_err:
                logger.warning(
                    f"MLflow failed to log dataset for assessment "
                    f"'{person_id}': {ds_log_err}"
                )

            # --- Saving Logic ---
            if not analysis_response_obj.error:
                complete_record_dict = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "answers": answers_dict,
                    "analysis": analysis_response_obj.analysis.dict(),
                    "notifications": [
                        n.dict() for n in analysis_response_obj.notifications
                    ],
                    "analysis_error": analysis_response_obj.error,
                    "raw_notification_output": (
                        analysis_response_obj.raw_notification_output
                    )
                }
                try:
                    logger.info(
                        f"Attempting to save complete assessment record for "
                        f"'{person_id}' to history..."
                    )
                    data_manager.save_assessment_record(
                        person_id, complete_record_dict
                    )
                    logger.info(
                        "Successfully saved assessment record to history "
                        f"for '{person_id}'."
                    )
                    final_status = "SUCCESS"
                except Exception as e:
                    save_error = e
                    final_status = "COMPLETED_WITH_SAVE_ERROR"
                    logger.error(
                        f"Failed to save assessment record for '{person_id}': "
                        f"{save_error}", exc_info=True
                    )
                    mlflow.set_tag(
                        "error_type", f"SaveHistoryError,{type(save_error).__name__}"
                    )
                    mlflow.log_param("save_error_detail", str(save_error))
                    # Append save error to response object's error field
                    save_fail_msg = f"SAVE FAILED: {save_error}"
                    analysis_response_obj.error = (
                        f"{analysis_response_obj.error} | {save_fail_msg}"
                        if analysis_response_obj.error else save_fail_msg
                    )
            else:
                final_status = "COMPLETED_WITH_ANALYSIS_ISSUE"
                mlflow.set_tag("error_type", "AnalysisGenerationError")
                if analysis_response_obj.error:
                    mlflow.log_param(
                        "analysis_error_detail", analysis_response_obj.error
                    )
                logger.warning(
                    f"Assessment run '{run_id}' for '{person_id}' completed "
                    f"with analysis issue: {analysis_response_obj.error}"
                )

            mlflow.set_tag("status", final_status)
            return analysis_response_obj

        except Exception as e:
            final_status = "FAILED"
            mlflow.set_tags({
                "status": final_status,
                "error_type": type(e).__name__
            })
            error_trace = traceback.format_exc()
            mlflow.log_text(error_trace, f"error_traceback_{person_id}.txt")
            mlflow.log_param("error_detail", str(e))
            logger.error(
                f"Unhandled error during Assessment run '{run_id}' for "
                f"'{person_id}': {e}", exc_info=True
            )
            # Return default error response
            error_analysis = schemas.DomainAnalysis(
                cognitive="[Server Error]",
                physical="[Server Error]",
                health="[Server Error]",
                personal_info="[Server Error]"
            )
            return schemas.FullAnalysisResponse(
                analysis=error_analysis,
                notifications=[],
                error=f"Internal server error during analysis: {e}",
                raw_notification_output=None
            )
        finally:
            total_latency = time.time() - request_start_time
            mlflow.log_metric("total_latency_sec", total_latency)
            logger.info(
                f"Assessment run '{run_id}' for '{person_id}' finished. "
                f"Status: {final_status}. Total time: {total_latency:.2f}s"
            )
            # Log response even on completion errors for debugging
            if analysis_response_obj and (
                final_status.startswith("FAILED")
                or final_status.startswith("COMPLETED_WITH")
            ):
                try:
                    _log_dict_as_json(
                        analysis_response_obj.dict(),
                        f"output_analysis_response_on_"
                        f"{final_status.lower()}_{person_id}",
                        run_id=run_id
                    )
                except Exception as log_err:
                    logger.warning(f"Failed to log final error response: {log_err}")


@router.get("/analysis/{person_id}", response_model=schemas.FullAnalysisResponse)
async def get_latest_analysis_endpoint(
    person_id: str = FastApiPath(..., description="ID of the person")
):
    """
    Retrieves the analysis/notifications part of the most recent assessment record.
    Includes the raw notification string if available in the saved record.
    """
    logger.info(f"Request received for latest analysis of '{person_id}'")
    if not person_id or not person_id.strip():
        raise HTTPException(status_code=400, detail="person_id is required.")
    person_id = person_id.strip()

    try:
        latest_record_data = data_manager.get_latest_record_data(person_id)
        if latest_record_data is None:
            logger.info(f"No history record found for '{person_id}'.")
            raise HTTPException(
                status_code=404,
                detail="No assessment record found for this person."
            )

        # Reconstruct the response model from the saved record
        analysis_dict = latest_record_data.get("analysis", {})
        notifications_list = latest_record_data.get("notifications", [])
        analysis_error = latest_record_data.get("analysis_error")
        raw_notification_output_saved = latest_record_data.get(
            "raw_notification_output"
        )

        analysis_obj = schemas.DomainAnalysis(**analysis_dict)
        notifications_obj = [
            schemas.NotificationNote(**note) for note in notifications_list
        ]

        # Create response object including the raw text if found
        response_model = schemas.FullAnalysisResponse(
            analysis=analysis_obj,
            notifications=notifications_obj,
            error=analysis_error,
            raw_notification_output=raw_notification_output_saved
        )
        logger.info(
            "Returning latest analysis portion (incl. raw notes if available) "
            f"for '{person_id}'."
        )
        return response_model
    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly
        raise http_exc
    except Exception as e:
        logger.error(
            f"Error retrieving or processing latest record for '{person_id}': {e}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve/process latest analysis: {e}"
        )


# ============================
# Q&A Endpoint (Unified)
# ============================
@router.post("/ask_unified", response_model=schemas.UnifiedAskResponse)
async def ask_unified_endpoint(request: schemas.AskAboutPersonRequest):
    """Handles unified Q&A requests using both documents and analysis history."""
    request_start_time = time.time()
    person_id = request.person_id
    question = request.question

    if not person_id or not person_id.strip():
        raise HTTPException(status_code=400, detail="person_id required.")
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question required.")

    person_id = person_id.strip()
    question = question.strip()
    logger.info(
        f"API Unified Ask request for '{person_id}': '{question[:100]}...'"
    )

    run_name = (
        f"UnifiedAsk_{person_id}_"
        f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    )

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run '{run_id}' for Unified Ask '{person_id}'.")

        # --- MLflow Initial Logging ---
        mlflow.set_tags({
            "endpoint": "/ask_unified",
            "person_id": person_id,
            "status": "PROCESSING",
            "llm_model": config.GEMINI_MODEL_NAME,
            "embedding_model": config.EMBEDDING_MODEL,
            "retriever_type": "VectorIndexRetriever" # Assuming this is constant
        })
        mlflow.log_params({
            "question_length": len(question),
            "retriever_top_k_config": 5 # Example value, adjust if needed
        })
        _log_dict_as_json(
            request.dict(), f"input_request_{person_id}", run_id=run_id
        )
        mlflow.log_text(question, f"input_question_{person_id}.txt")

        final_status = "UNKNOWN"
        response: Optional[schemas.UnifiedAskResponse] = None
        intermediate_data: Dict[str, Any] = {}

        try:
            # --- Call Service ---
            response, intermediate_data = await services.ask_unified_about_person(
                person_id, question
            )

            # --- MLflow Results Logging ---
            # Log intermediate data
            mlflow.log_text(
                intermediate_data.get("retrieved_doc_context", "N/A"),
                f"intermediate_doc_context_{person_id}.txt"
            )
            mlflow.log_text(
                intermediate_data.get("analysis_summary_used", "N/A"),
                f"intermediate_analysis_summary_{person_id}.txt"
            )
            mlflow.log_text(
                intermediate_data.get("assessment_answers_str", "N/A"),
                f"intermediate_answers_used_{person_id}.txt"
            )
            mlflow.log_text(
                intermediate_data.get("prompt_document_based", "N/A"),
                f"intermediate_prompt_docs_{person_id}.txt"
            )
            mlflow.log_text(
                intermediate_data.get("prompt_analysis_based", "N/A"),
                f"intermediate_prompt_analysis_{person_id}.txt"
            )

            # Log latencies
            latencies = intermediate_data.get("latencies", {})
            if latencies:
                mlflow.log_metrics(latencies)

            # Log output metrics and artifacts
            mlflow.log_metrics({
                "document_answer_length": len(response.document_answer or ""),
                "analysis_answer_length": len(response.analysis_answer or "")
            })
            _log_dict_as_json(
                response.dict(),
                f"output_unified_response_{person_id}",
                run_id=run_id
            )
            mlflow.log_text(
                response.document_answer or "N/A",
                f"output_document_answer_{person_id}.txt"
            )
            mlflow.log_text(
                response.analysis_answer or "N/A",
                f"output_analysis_answer_{person_id}.txt"
            )

            # --- MLflow Dataset Logging ---
            try:
                dataset_df = pd.DataFrame([{
                    "person_id": person_id,
                    "question": question,
                    "document_answer": response.document_answer,
                    "analysis_answer": response.analysis_answer,
                    "has_general_error": bool(response.error),
                    "has_doc_error": bool(response.document_error),
                    "has_analysis_error": bool(response.analysis_error)
                }])
                mlflow_dataset = mlflow.data.from_pandas(
                    dataset_df,
                    source=f"api:/ask_unified/{person_id}",
                    name=f"unified_ask_io_{person_id}"
                )
                mlflow.log_input(mlflow_dataset, context="unified_ask_io")
            except Exception as ds_log_err:
                logger.warning(
                    f"MLflow failed to log dataset for unified ask "
                    f"'{person_id}': {ds_log_err}"
                )

            # --- Determine Final Status ---
            if (response.error or response.document_error
                    or response.analysis_error):
                final_status = "COMPLETED_WITH_ISSUE"
                error_types = []
                if response.error:
                    error_types.append("GeneralProcessingError")
                    mlflow.log_param("general_error_detail", response.error)
                if response.document_error:
                    error_types.append("DocumentProcessingError")
                    mlflow.log_param(
                        "document_error_detail", response.document_error
                    )
                if response.analysis_error:
                    error_types.append("AnalysisProcessingError")
                    mlflow.log_param(
                        "analysis_error_detail", response.analysis_error
                    )
                mlflow.set_tag("error_type", ",".join(error_types))
                logger.warning(
                    f"Unified Ask run '{run_id}' for '{person_id}' completed "
                    f"with issue(s). DocErr: {response.document_error}, "
                    f"AnalysisErr: {response.analysis_error}, "
                    f"GenErr: {response.error}"
                )
            else:
                final_status = "SUCCESS"

            mlflow.set_tag("status", final_status)
            return response

        except Exception as e:
            final_status = "FAILED"
            mlflow.set_tags({
                "status": final_status,
                "error_type": type(e).__name__
            })
            error_trace = traceback.format_exc()
            mlflow.log_text(error_trace, f"error_traceback_{person_id}.txt")
            mlflow.log_param("error_detail", str(e))
            logger.error(
                f"Unhandled error during Unified Ask run '{run_id}' for "
                f"'{person_id}': {e}", exc_info=True
            )
            # Return default error response
            return schemas.UnifiedAskResponse(
                document_answer="[Server Error]",
                analysis_answer="[Server Error]",
                error=f"Internal server error: {e}",
                document_error=f"Internal server error: {e}",
                analysis_error=f"Internal server error: {e}"
            )
        finally:
            total_latency = time.time() - request_start_time
            mlflow.log_metric("total_latency_sec", total_latency)
            logger.info(
                f"Unified Ask run '{run_id}' for '{person_id}' finished. "
                f"Status: {final_status}. Total time: {total_latency:.2f}s"
            )
            # Log response even on failure/issue for debugging
            if response and (
                final_status == "FAILED"
                or final_status.startswith("COMPLETED_WITH")
            ):
                try:
                    _log_dict_as_json(
                        response.dict(),
                        f"output_unified_response_on_"
                        f"{final_status.lower()}_{person_id}",
                        run_id=run_id
                    )
                except Exception as log_err:
                    logger.warning(f"Failed to log final error response: {log_err}")


# ============================
# Person Management Endpoint
# ============================
class PersonListResponse(schemas.BaseModel):
    person_ids: List[str]

@router.get("/list_persons", response_model=PersonListResponse)
async def list_persons_endpoint():
    """Lists all unique person IDs known to the system."""
    logger.info("Request received to list persons")
    try:
        person_ids = data_manager.list_persons()
        return {"person_ids": person_ids}
    except Exception as e:
        logger.error(f"Error listing persons: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Could not list persons: {e}"
        )


# ============================
# PDF Management Endpoints
# ============================

@router.post("/upload_personal_pdf", response_model=schemas.PDFUploadResponse, status_code=201)
async def upload_personal_pdf_endpoint(pdf_file: UploadFile = File(...)):
    """Uploads a personal PDF document, sanitizing the filename."""
    if not pdf_file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    if pdf_file.content_type != "application/pdf":
        logger.warning(f"Invalid PDF type: {pdf_file.content_type}")
        raise HTTPException(
            status_code=400, detail="Invalid file type (PDF only)."
        )

    # Sanitize filename
    base_filename = os.path.basename(pdf_file.filename)
    safe_filename = "".join(
        c for c in base_filename if c.isalnum() or c in ('.', '_', '-')
    ).strip()
    if not safe_filename:
        raise HTTPException(
            status_code=400, detail="Invalid filename after sanitization."
        )

    file_path = PDF_STORAGE_DIR / safe_filename
    logger.info(f"Attempting to save personal PDF to: {file_path}")
    if file_path.exists():
        logger.warning(f"PDF '{safe_filename}' exists. Overwriting.")

    try:
        content = await pdf_file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        logger.info(f"Successfully saved personal PDF: {safe_filename}")
        return {
            "filename": safe_filename,
            "detail": "PDF uploaded. Rebuild index might be needed."
        }
    except Exception as e:
        logger.error(
            f"Error saving personal PDF '{safe_filename}': {e}", exc_info=True
        )
        # Attempt to remove partially saved file on error
        if file_path.exists():
            try:
                os.remove(file_path)
                logger.info(f"Removed partially saved file: {file_path}")
            except OSError as rm_err:
                logger.error(
                    f"Could not remove partially saved file {file_path}: {rm_err}"
                )
        raise HTTPException(status_code=500, detail=f"Could not save PDF: {e}")
    finally:
        # Ensure the uploaded file handle is closed
        await pdf_file.close()


@router.get("/list_personal_pdfs", response_model=schemas.PDFListResponse)
async def list_personal_pdfs_endpoint():
    """Lists all PDF files currently stored in the personal documents directory."""
    logger.info(f"Listing personal PDFs in: {PDF_STORAGE_DIR}")
    try:
        if not PDF_STORAGE_DIR.is_dir():
            logger.warning(f"PDF storage directory not found: {PDF_STORAGE_DIR}")
            return {"pdf_files": []}

        pdf_files = sorted([
            f.name for f in PDF_STORAGE_DIR.iterdir()
            if f.is_file() and f.suffix.lower() == '.pdf'
        ])
        logger.info(f"Found {len(pdf_files)} personal PDF files.")
        return {"pdf_files": pdf_files}
    except Exception as e:
        logger.error(f"Error listing PDF files: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Could not list PDF files: {e}"
        )


@router.get("/download_personal_pdf/{filename}")
async def download_personal_pdf_endpoint(
    filename: str = FastApiPath(..., description="PDF filename to download")
):
    """Downloads a specific personal PDF file by filename."""
    logger.info(f"Download requested for PDF: {filename}")

    # Basic security checks
    if ".." in filename or filename.startswith("/"):
        logger.warning(f"Potential path traversal attempt: {filename}")
        raise HTTPException(status_code=400, detail="Invalid filename.")

    # Sanitize and validate filename format
    base_filename = os.path.basename(filename)
    safe_filename = "".join(
        c for c in base_filename if c.isalnum() or c in ('.', '_', '-')
    ).strip()
    if not safe_filename or safe_filename != filename:
        logger.warning(
            f"Invalid characters or mismatch after sanitization: "
            f"{filename} -> {safe_filename}"
        )
        raise HTTPException(status_code=400, detail="Invalid filename format.")

    try:
        file_path = (PDF_STORAGE_DIR / safe_filename).resolve()

        # Verify the file exists and is within the intended storage directory
        if not file_path.is_file() or not str(file_path).startswith(
            str(PDF_STORAGE_DIR.resolve())
        ):
            logger.error(
                f"PDF not found or path mismatch: {safe_filename} "
                f"(Resolved: {file_path})"
            )
            raise HTTPException(
                status_code=404, detail=f"PDF not found: {safe_filename}"
            )

        logger.info(f"Streaming PDF: {file_path}")
        return FileResponse(
            path=str(file_path),
            media_type='application/pdf',
            filename=safe_filename
        )
    except HTTPException as http_exc:
        # Re-raise known HTTP exceptions
        raise http_exc
    except Exception as e:
        logger.error(f"Error during PDF download for '{filename}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Could not process download request: {e}"
        )