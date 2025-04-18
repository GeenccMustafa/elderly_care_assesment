# Real-Time* Classroom Captioning System (Backend + Frontend)

This project demonstrates a system for captioning audio chunks in near real-time, providing context retrieval and summarization. It features a separate FastAPI backend and Gradio frontend, containerized with Docker.

*Note: "Real-time" here refers to processing discrete audio chunks rapidly, not continuous character-by-character transcription.*

## Features

*   **Backend API (FastAPI):**
    *   `/api/process_audio` endpoint accepts audio chunks.
    *   Uses `faster-whisper` for transcription.
    *   Uses `LlamaIndex` for context retrieval from pre-indexed slide text.
    *   Uses `Gemini API` for summarization.
    *   API documentation via Swagger UI at `/docs`.
*   **Frontend UI (Gradio):**
    *   Allows recording audio chunks via microphone.
    *   Sends audio to the backend API for processing.
    *   Displays transcription, summary, and context.
*   **MLOps Integration:**
    *   **Docker & Docker Compose:** For containerization and local orchestration.
    *   **DVC:** For managing the `slide_text_data` directory (setup required).
    *   **MLflow:** For logging API call performance (configure `MLFLOW_TRACKING_URI` in `backend/app/config.py`, run `mlflow ui` locally).

## Setup

1.  **Prerequisites:**
    *   Git & Git LFS (`git lfs install`)
    *   Docker & Docker Compose
    *   Python (for running DVC/MLflow locally if needed)
    *   DVC (`pip install dvc`)

2.  **Clone Repository:**
    ```bash
    git clone <your-repo-url>
    cd classroom-captioning-system
    ```

3.  **Environment Variables:**
    *   Create a `.env` file in the project root.
    *   Add your Google API key:
        ```dotenv
        GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
        ```

4.  **Prepare Data & Index:**
    *   **Add Slide Text:** Place your lecture slide content as `.txt` files inside `data/slide_text_data/`.
    *   **(Option 1: Use DVC - Recommended)**
        *   Initialize DVC remote (e.g., local):
            ```bash
            dvc init -q
            dvc remote add -d localremote ../dvc_storage # Or configure S3/GCS etc.
            ```
        *   Track the data directory:
            ```bash
            dvc add data/slide_text_data
            git add data/slide_text_data.dvc .gitignore
            git commit -m "Track slide text data"
            dvc push -r localremote # Push data to your DVC remote
            ```
        *   *If cloning later:* Run `dvc pull -r localremote` to retrieve data.
    *   **(Option 2: Manual)** Just ensure the `data/slide_text_data/` directory exists with text files before building the index.
    *   **Build LlamaIndex:** Run an indexing script *before* starting the containers, or adapt the backend service to create it if missing (can be slow on first startup). *For now, assume you run this locally first:*
        ```python
        # Create a simple script, e.g., `build_index.py` in the root:
        from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.core.node_parser import SentenceSplitter
        import os

        LLAMA_INDEX_DIR = "data/slide_text_data"
        LLAMA_INDEX_PERSIST_DIR = "./vector_store_local"
        EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
        DEVICE = "cpu"

        print("Building LlamaIndex...")
        if not os.path.exists(LLAMA_INDEX_DIR) or not os.listdir(LLAMA_INDEX_DIR):
             print(f"Error: Directory '{LLAMA_INDEX_DIR}' is empty or missing.")
             exit()

        Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL, device=DEVICE)
        Settings.llm = None

        documents = SimpleDirectoryReader(LLAMA_INDEX_DIR).load_data()
        parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        nodes = parser.get_nodes_from_documents(documents, show_progress=True)
        index = VectorStoreIndex(nodes, show_progress=True)
        index.storage_context.persist(persist_dir=LLAMA_INDEX_PERSIST_DIR)
        print(f"Index built and saved to {LLAMA_INDEX_PERSIST_DIR}")

        # --- Run this script: ---
        # python build_index.py
        ```

5.  **Build and Run Containers:**
    ```bash
    docker-compose up --build
    ```
    *   Use `-d` to run in detached mode.

## Usage

1.  **Frontend (Gradio UI):** Open your browser to `http://localhost:7860`.
    *   Click the microphone icon to record an audio chunk (e.g., 5-10 seconds).
    *   Click "Process Audio Chunk".
    *   View the transcription, summary, and context.
2.  **Backend (FastAPI Swagger UI):** Open your browser to `http://localhost:8000/docs`.
    *   Explore the `/api/process_audio` endpoint. You can test it here by uploading an audio file directly.
3.  **MLflow UI:**
    *   Open a separate terminal in the project root.
    *   Run `mlflow ui`.
    *   Open your browser to `http://localhost:5000` (or the URL provided).
    *   View the "Classroom Captioning API Calls" experiment to see logs from backend summarization calls (if enabled and working).

## Stopping the System

*   Press `Ctrl+C` in the terminal where `docker-compose up` is running.
*   Or, if running detached, use `docker-compose down`.