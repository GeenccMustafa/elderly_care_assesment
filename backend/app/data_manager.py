# backend/app/data_manager.py
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Set up logger
logger = logging.getLogger(__name__)

try:
    from . import config, schemas
    PERSON_DATA_DIR = config.PERSON_DATA_DIR
except ImportError:
    logger.warning(
        "Could not perform relative import of config/schemas. "
        "Attempting fallback using project structure."
    )
    try:
        # Calculate project root assuming data_manager.py is in backend/app/
        project_root = Path(__file__).resolve().parent.parent
        PERSON_DATA_DIR = project_root / "data" / "person_data"
        PERSON_DATA_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Fallback: Ensured person data directory exists at {PERSON_DATA_DIR}")

        # Add project root to path to find 'app' module
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
            logger.debug(f"Fallback: Added {project_root} to sys.path")

        from app import schemas  
        logger.info("Fallback: Successfully imported schemas.")
    except Exception as e:
        schemas = None # Define schemas as None if import fails
        logger.error(
            f"Fallback import/setup failed: {e}. Schemas will not be available.",
             exc_info=True
        )
        if 'PERSON_DATA_DIR' not in locals():
             PERSON_DATA_DIR = Path("./data/person_data")
             logger.warning(f"Using last resort PERSON_DATA_DIR: {PERSON_DATA_DIR}")


# --- Helper Functions ---
def _convert_sets_to_lists(obj: Any) -> Any:
    """Recursively converts sets within a nested structure to lists."""
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: _convert_sets_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_sets_to_lists(elem) for elem in obj]
    else:
        return obj


def _get_person_filepath(person_id: str) -> Path:
    """
    Constructs the filepath for a person's JSON history data file.
    Sanitizes the person_id to create a safe filename.
    """
    if not isinstance(person_id, str) or not person_id.strip():
        raise ValueError("Invalid or empty person_id provided.")

    safe_filename_base = "".join(
        c for c in person_id if c.isalnum() or c in ('-', '_')
    ).strip()

    if not safe_filename_base:
        raise ValueError(
            f"Person_id '{person_id}' resulted in empty safe filename."
        )

    filename = f"{safe_filename_base}.json"
    return PERSON_DATA_DIR / filename


# --- Data Management Functions ---

def save_assessment_record(person_id: str, assessment_record: Dict[str, Any]):
    """
    Appends a new complete assessment record to the person's history file.
    The history is stored as a JSON list of records.
    """
    if not person_id or not assessment_record:
        logger.warning(
            "Attempted to save assessment record with missing person_id "
            "or empty data."
        )
        return

    try:
        filepath = _get_person_filepath(person_id)
    except ValueError as e:
        logger.error(f"Cannot save record due to invalid person_id: {e}")
        raise ValueError(f"Invalid person_id for saving: {person_id}") from e

    history: List[Dict[str, Any]] = []

    # Load existing history if file exists
    if filepath.exists():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            if isinstance(loaded_data, list):
                history = loaded_data
            else:
                logger.warning(
                    f"Overwriting non-list data in history file: {filepath}. "
                    "Expected a list of records."
                )
                history = []
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding JSON from history file {filepath}, "
                f"starting new history list. Error: {e}"
            )
            history = []
        except IOError as e:
            logger.error(f"Error reading history file {filepath}: {e}")
            raise IOError(f"Could not read history file: {filepath}") from e

    try:
        cleaned_record = _convert_sets_to_lists(assessment_record)
        history.append(cleaned_record)
    except Exception as clean_err:
        logger.error(
            f"Error cleaning assessment record for '{person_id}': {clean_err}",
            exc_info=True
        )
        raise RuntimeError(
            f"Failed to process record before saving for '{person_id}'"
        ) from clean_err

    try:
        PERSON_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4)
        logger.info(
            f"Saved assessment record for person '{person_id}' to {filepath}"
        )
    except IOError as e:
        logger.error(f"Error writing updated history to {filepath}: {e}", exc_info=True)
        raise IOError(f"Could not write history file: {filepath}") from e
    except TypeError as e:
        logger.error(
            f"TypeError during JSON dump for {filepath}. Data might not be fully "
            f"serializable: {e}", exc_info=True
        )
        raise TypeError(f"Data for '{person_id}' not JSON serializable") from e
    except Exception as e:
        logger.error(
            f"Unexpected error saving history {filepath}: {e}", exc_info=True
        )
        raise


def list_persons() -> List[str]:
    """Lists the IDs of persons with existing history data files (.json)."""
    if not PERSON_DATA_DIR.is_dir():
        logger.warning(f"Person data directory not found: {PERSON_DATA_DIR}")
        return []

    persons = []
    try:
        for filepath in PERSON_DATA_DIR.glob("*.json"):
            if filepath.is_file():
                persons.append(filepath.stem)
    except OSError as e:
        logger.error(
            f"Error accessing person data directory {PERSON_DATA_DIR}: {e}",
            exc_info=True
        )
        return []
    except Exception as e:
        logger.error(
            f"Unexpected error scanning person data directory {PERSON_DATA_DIR}: {e}",
            exc_info=True
        )
        return [] 

    return sorted(persons)


def get_latest_record_data(person_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves the dictionary of the most recent assessment record
    from the person's history file.
    """
    try:
        filepath = _get_person_filepath(person_id)
    except ValueError as e:
        logger.warning(f"Cannot get data due to invalid person_id: {e}")
        return None

    if not filepath.exists() or not filepath.is_file():
        logger.info(
            f"History file not found for '{person_id}' at {filepath}"
        )
        return None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            history = json.load(f)

        if isinstance(history, list) and history:
            latest_record = history[-1]
            if isinstance(latest_record, dict):
                logger.info(
                    f"Successfully retrieved latest record data for '{person_id}'."
                )
                return latest_record
            else:
                logger.warning(
                    f"Last item in history for '{person_id}' at {filepath} "
                    f"is not a dictionary (type: {type(latest_record)})."
                )
                return None
        else:
            logger.warning(
                f"History file for '{person_id}' at {filepath} is empty or "
                f"not a list (type: {type(history)})."
            )
            return None
    except json.JSONDecodeError as e:
        logger.error(
            f"Error decoding JSON from history file {filepath} for "
            f"'{person_id}': {e}"
        )
        return None
    except IOError as e:
        logger.error(
            f"Error reading history data file {filepath} for '{person_id}': {e}"
        )
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error loading history file {filepath} for "
            f"'{person_id}': {e}", exc_info=True
        )
        return None
