# build_index.py (place in project root)
import os
import logging
from pathlib import Path as PyPath # Use PyPath consistently
import re # For parsing filenames
import sys # To check python version or exit

# --- Use PyMuPDF for PDF Parsing ---
try:
    import fitz # PyMuPDF
except ImportError:
    print("Error: PyMuPDF (fitz) not installed.")
    print("Please install it: pip install PyMuPDF")
    sys.exit(1)

# --- LlamaIndex Imports ---
try:
    from llama_index.core import VectorStoreIndex, Settings, StorageContext
    from llama_index.core.schema import Document # Use Document for nodes
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError as e:
     print(f"Error: LlamaIndex components not installed ({e}).")
     print("Please install requirements: pip install -r requirements.txt (or specific packages)")
     sys.exit(1)

# --- Import Config from Backend App (ONLY for non-path settings) ---
try:
    # Assuming build_index.py is in the root, and config is in backend/app
    # We need this to get the correct embedding model name, device etc.
    from backend.app import config
except ImportError as e:
    print(f"Error importing config from backend.app: {e}")
    print("Please ensure build_index.py is run from the project root directory")
    print("and the backend structure (backend/app/config.py) is correct.")
    print("Also ensure backend dependencies are installed if needed for config loading.")
    sys.exit(1) # Exit if config cannot be imported

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Starting LlamaIndex build process for Personal Documents...")

# --- Define HOST Paths (relative to this script in the project root) ---
# These paths are on the machine running this script (your local machine)
# and correspond to the LEFT side of the volume mounts in docker-compose.yml
PROJECT_ROOT = PyPath(__file__).parent.resolve() # Gets the project root directory
DOCS_DIR = PROJECT_ROOT / "data" / "personal_documents"
PERSIST_DIR = PROJECT_ROOT / "vector_store_personal"

# --- Configuration from backend/app/config.py (Non-Path Settings) ---
# Fetch model configuration from the central config file
EMBED_MODEL_NAME = config.EMBEDDING_MODEL
EMBED_DEVICE = config.EMBED_DEVICE
# LLM_MODEL = config.GEMINI_MODEL_NAME # Not needed for indexing, but could be used later

# --- Log the HOST Paths being used by this script ---
logger.info(f"Project Root (Host): {PROJECT_ROOT}")
logger.info(f"Document Source Directory (Host Path): {DOCS_DIR}")
logger.info(f"Index Persist Directory (Host Path): {PERSIST_DIR}")
logger.info(f"Embedding Model: {EMBED_MODEL_NAME} on device '{EMBED_DEVICE}'")

# --- Safety Check: Source Directory (Using Host Path) ---
if not DOCS_DIR.exists():
    # Critical error if source docs don't exist on the host
    logger.error(f"Error: Document source directory '{DOCS_DIR}' does not exist.")
    logger.error("Please create the directory and add PDF documents before running this script.")
    sys.exit(1)
elif not any(DOCS_DIR.glob("*.pdf")): # More specific check for PDFs
    logger.warning(f"Warning: Document source directory '{DOCS_DIR}' contains no PDF files.")
    logger.warning("No documents found to index. Will create empty index structure if it doesn't exist.")
    # Allow proceeding to potentially create empty index dir structure if needed later

# --- Safety Check: Existing Index (Using Host Path) ---
if PERSIST_DIR.exists() and any(PERSIST_DIR.iterdir()):
    logger.warning(f"Index directory '{PERSIST_DIR}' already exists and is not empty.")
    # Provide clear instructions
    logger.error(f"To force a rebuild, please MANUALLY DELETE the directory: '{PERSIST_DIR}'")
    logger.info("Exiting without building index.")
    sys.exit(1) # Exit with error code to signify no build occurred due to existing index

# --- Configure LlamaIndex Settings ---
logger.info("Configuring LlamaIndex Settings...")
try:
    # Use HuggingFace embeddings (ensure transformer models are installed)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME, device=EMBED_DEVICE)
    Settings.llm = None # LLM not needed for just indexing text chunks
    # Optional: Configure node parser here if needed
    # from llama_index.core.node_parser import SimpleNodeParser
    # Settings.node_parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=20)
    logger.info("Settings configured successfully.")
except Exception as e:
    logger.error(f"Error configuring LlamaIndex settings: {e}", exc_info=True)
    logger.error("Ensure the embedding model is valid and dependencies (like PyTorch/TensorFlow, transformers) are installed.")
    sys.exit(1)

# --- PDF Parsing and Node Creation (Using Host Path DOCS_DIR) ---
documents = [] # List to hold LlamaIndex Document objects
logger.info(f"Processing PDF files from host directory: {DOCS_DIR}...")

# Use PyPath for iteration
for pdf_path in DOCS_DIR.glob("*.pdf"):
    logger.info(f"-- Processing file: {pdf_path.name}")

    # --- Attempt to extract person_id from filename ---
    # Assumes format like "PersonID_AnythingElse.pdf" or "PersonID-AnythingElse.pdf"
    # Regex allows letters, numbers, hyphen, underscore, space, dot in the ID part
    match = re.match(r"^([a-zA-Z0-9\-_ .]+?)[-_]", pdf_path.name) # Added '?' for non-greedy match
    if match:
        person_id = match.group(1).strip()
        logger.info(f"   Extracted Person ID: '{person_id}'")
    else:
        # If no match, use the entire filename stem (without .pdf) as the ID
        person_id = pdf_path.stem
        logger.warning(f"   Could not extract ID using pattern from '{pdf_path.name}'. Using full stem '{person_id}' as Person ID.")

    doc_fitz = None # Use PyMuPDF object
    try:
        doc_fitz = fitz.open(pdf_path) # Open PDF from host path
        num_pages = len(doc_fitz)
        logger.info(f"   Pages: {num_pages}")
        full_text = ""
        for page_num, page in enumerate(doc_fitz):
            # Extract text, attempt sorting to maintain reading order
            page_text = page.get_text("text", sort=True).strip()
            if page_text:
                full_text += page_text + "\n\n" # Add separator between pages

        if full_text.strip():
            # Create LlamaIndex Document with extracted text and metadata
            metadata = {
                "filename": pdf_path.name,
                "source_path": str(pdf_path), # Store the host path used
                "person_id": person_id,
                "total_pages": num_pages
                }
            doc = Document(text=full_text.strip(), metadata=metadata)
            documents.append(doc)
            logger.info(f"   Prepared LlamaIndex Document for '{pdf_path.name}' (Person ID: '{person_id}')")
        else:
             logger.warning(f"   No text extracted from {pdf_path.name}, skipping.")

    except fitz.fitz.FileNotFoundError:
         logger.error(f"   File not found error for {pdf_path.name}. Skipping.")
    except Exception as e:
        logger.error(f"   Error processing {pdf_path.name}: {e}", exc_info=False) # Set exc_info=False for cleaner logs unless debugging specific file
        # Continue to next file if one fails
    finally:
        if doc_fitz:
            try:
                doc_fitz.close()
            except Exception:
                pass # Ignore errors during close

# --- Post-Processing Checks ---
if not documents:
    logger.warning("No text documents were successfully extracted from any PDF files.")
    logger.warning("Index will be created but will be empty.")
    # Allow script to proceed to create the directory structure if needed

logger.info(f"Successfully prepared {len(documents)} documents for indexing.")

# --- Build and Persist Index (Using Host Path PERSIST_DIR) ---
if documents: # Only build if there are documents
    logger.info(f"Building VectorStoreIndex...")
    try:
        # Create index from the gathered LlamaIndex Document objects
        # show_progress=True is useful for large numbers of documents
        index = VectorStoreIndex(documents, show_progress=True)

        logger.info(f"Persisting index to host directory: {PERSIST_DIR}")
        # Ensure the persist directory exists on the HOST right before saving
        os.makedirs(PERSIST_DIR, exist_ok=True)
        index.storage_context.persist(persist_dir=str(PERSIST_DIR))

        logger.info(f"Index built successfully and saved to {PERSIST_DIR}")

    except Exception as e:
        logger.error(f"Error building or persisting the index: {e}", exc_info=True)
        sys.exit(1) # Exit if index building fails
else:
    # If no documents, still create the persist directory structure if it doesn't exist
    # This makes the volume mount work correctly even if the index is initially empty.
    if not PERSIST_DIR.exists():
        logger.info(f"No documents to index, creating empty persist directory: {PERSIST_DIR}")
        os.makedirs(PERSIST_DIR, exist_ok=True)
    else:
        # This case shouldn't happen because of the earlier check, but included for completeness
        logger.info(f"No documents to index. Existing directory found at {PERSIST_DIR}. No action taken.")


logger.info("Build process finished.")