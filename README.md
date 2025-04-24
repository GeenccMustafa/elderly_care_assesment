# Elderly Care Assessment Assistant

## Overview

Elderly Care Assessment Assistant is a comprehensive system designed to assist caregivers in evaluating the well-being of elderly individuals. It leverages advanced Natural Language Processing (NLP) techniques, including Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG), to analyze assessment responses and provide intelligent Q&A capabilities using personal documents.

The system includes a web-based interface built with Gradio for ease of use, supporting audio input, document management, and analysis review. A FastAPI backend handles transcription, LLM interactions, document indexing, and experiment tracking with MLflow.

## Key Features

- Structured Assessments: Guided assessments using a predefined set of health and wellness questions.
- Audio Input & Transcription: Record verbal answers and transcribe them using OpenAI's Whisper model.
- LLM-Powered Analysis: Generate cognitive, physical, health, and personal domain analyses with actionable insights using Google's Gemini models.
- Document Management: Upload, manage, and download personal PDF documents like medical history and care plans.
- Retrieval-Augmented Q&A: Ask questions based on document content and assessment results using LlamaIndex and Gemini.
- MLflow Tracking: Track all assessments and Q&A interactions as experiments for auditability and performance monitoring.
- Dockerized Deployment: Easily deploy using Docker Compose for a consistent environment across development and production.

## Architecture Overview

The system follows a modular architecture with separate components:

- **Frontend**: Built with Gradio, communicates with the backend via REST API.
- **Backend**: Built with FastAPI, handles core logic, model calls, data management, and MLflow logging.
- **Vector Store**: Uses LlamaIndex to index and persist document embeddings.
- **MLflow**: Tracks experiments locally within the project.
- **Data Storage**: Local file-based storage for PDFs and JSON assessment history.

## Technology Stack

- **Frontend**: Gradio, Requests
- **Backend**: FastAPI, Uvicorn, python-multipart
- **NLP/ML**:
    - **LLM**: Google Gemini (via google-generativeai)
    - **Transcription**: OpenAI Whisper
    - **Embeddings**: BAAI/bge-small-en-v1.5 (via Hugging Face)
    - **Indexing & RAG**: LlamaIndex
- **Data Handling**: PyMuPDF (fitz), JSON
- **Experiment Tracking**: MLflow
- **Containerization**: Docker, Docker Compose
- **Language**: Python 3.10+

## Installation and Setup

### Prerequisites

- Docker + Docker Compose
- Python 3.10+ (if running indexing locally)
- Google API Key for Gemini models

### Steps

1. Clone the Repository

```
git clone <your-repository-url>
cd ELDERLY_CARE_ASSESSMENT
```

2. Set Up Environment Variables. Create a .env file in the root:

```
GOOGLE_API_KEY=your_actual_key_here
```

3. Build the Vector Index

- Place PDFs in data/personal_documents/.
- Run the indexing script:

```
python build_index.py
```
*Note: Delete vector_store_personal/ if re-indexing is needed.*

4. Start the Services

```
docker-compose up --build -d
```
5. Access the Application

- Frontend UI: http://localhost:7860
- Backend Docs (Swagger): http://localhost:8000/docs
- MLflow UI: http://localhost:5000

6. Stop the Services

```
docker-compose down
```

## Usage Guide

1. **Select/Enter Person ID**: Use the dropdown at the top. This ID associates assessments and filters document Q&A.
2. **Navigate Views**: Use the buttons ("Assessment & Analysis", "Document Management", "Ask About Person").
3. **Assessment**: Record audio answers, click "Transcribe" for each, then "Submit Transcribed Answers & Analyze". Review the generated results.
4. **Document Management**: Upload/download PDFs. Remember to stop services, run python build_index.py, and restart services after adding/changing documents if you want them included in Q&A.
5. **Ask About Person**: Select the Person ID, type your question, and click "Ask Question" to get answers derived from documents and analysis.
6. **MLflow**: Monitor the MLflow UI (http://localhost:5000) for detailed run information.

## Troubleshooting Notes (macOS/Docker)

- **Microphone Access**: If the browser doesn't prompt for mic access or shows errors when using the app via Docker:
    - Ensure you are accessing the UI via http://localhost:7860 or http://127.0.0.1:7860.
    - Check and allow microphone permissions for your browser in macOS System Settings -> Privacy & Security -> Microphone.
    - Check and allow microphone permissions for localhost:7860 in your browser's site settings (click the lock icon in the address bar).
    - If issues persist, consider running the frontend locally for testing mic features: cd frontend && export BACKEND_BASE_URL="http://localhost:8000" && pip install -r requirements.txt && python app.py (while the backend runs in Docker).

- **File Access Errors (Operation not permitted, Errno 35**): If docker-compose up fails with errors related to volume mounts or file reading (especially during LlamaIndex load):
    - Ensure Docker Desktop has Full Disk Access in macOS System Settings -> Privacy & Security.
    - Ensure the project directory is accessible via Docker Desktop's Settings -> Resources -> File Sharing (usually covered by the /Users default).
    - Try restarting Docker Desktop.
    - Try switching the file sharing implementation (e.g., VirtioFS vs gRPC FUSE) in Docker Desktop Settings -> General.
    - As a last resort, try Troubleshoot -> Reset disk permissions or Reset to factory defaults in Docker Desktop.

## MLflow Logging Details

**Assessment Submissions (/submit_assessment):**

- Tags: endpoint, person_id, status, llm_model.
- Parameters: num_answers_submitted, error details.
- Metrics: LLM latencies, text lengths, total_latency_sec.
- Artifacts: Input request/answers, prompts used, output analysis response (JSON/text), raw notification string, error tracebacks.
- Dataset: assessment_io (input/output).

**Q&A Sessions (/ask_unified):**

- Tags: endpoint, person_id, status, llm_model, embedding_model, retriever_type.
- Parameters: question_length, retriever_top_k_config, error details.
- Metrics: Retrieval/LLM latencies, answer lengths, total_latency_sec.
- Artifacts: Input request/question, intermediate contexts (docs, analysis, answers), prompts, output response (JSON/text), error tracebacks.
- Dataset: unified_ask_io (input/output).

## Future Improvements

**Advanced RAG**: Smarter retrieval strategies (re-ranking, HyDE, query rewriting)
**Quality Evaluation**: Integrate frameworks like RAGAs or DeepEval
**Database Backend**: Replace JSON storage with PostgreSQL or MongoDB
**User Auth**: Implement authentication/authorization
**Robust Error Handling**: Better UI feedback for failures
**Model Fine-Tuning**: Adapt models to domain-specific data if needed
**CI/CD Pipeline**: Add continuous integration and testing
**MLflow Containerization**: Include MLflow server in Docker setup

## Acknowledgments

- OpenAI Whisper (via Hugging Face transformers) for transcription
- Google Gemini (via google-generativeai) for LLM-powered insights
- Hugging Face (sentence-transformers) for embedding models
- LlamaIndex for document indexing and RAG framework
- MLflow for experiment tracking
- FastAPI & Gradio for web framework and UI

## Contact

For issues, suggestions, or contributions, please open an issue or pull request in the repository.