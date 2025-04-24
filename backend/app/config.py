# backend/app/config.py

import os
from pathlib import Path

# --- API Keys ---
# GOOGLE_API_KEY is expected to be set as an environment variable.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Assessment Configuration ---
ASSESSMENT_QUESTIONS = [
    "1. How have you been feeling overall lately, both physically and mentally?",
    "2. Can you describe your typical daily routine? What activities do you usually do?",
    "3. Have you noticed any changes in your memory or thinking recently? "
    "For example, remembering appointments or finding words?",
    "4. Are you taking any medications? If so, which ones and what are the "
    "dosages/times?",
    "5. Do you live alone or with others? How do you usually get around "
    "(e.g., driving, walking, public transport)? Do you need any help with "
    "daily tasks like cooking or cleaning?"
]

# --- Model/Service Configurations ---
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# Keep 'cpu' for broader compatibility unless GPU is configured
EMBED_DEVICE = 'cpu'
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"

# --- Path Configurations (Absolute paths INSIDE the container) ---
# Define paths based on the '/code' working directory set in the Dockerfile
# and the volume mounts defined in docker-compose.yml.
# These are used by the backend service running within the container.
CONTAINER_CODE_DIR = Path("/code")

PERSONAL_DOCS_STORAGE_DIR = CONTAINER_CODE_DIR / "data" / "personal_documents"
# Absolute path inside container
VECTOR_STORE_PERSONAL_DIR = CONTAINER_CODE_DIR / "vector_store_personal"
# Absolute path inside container
MLRUNS_DIR = CONTAINER_CODE_DIR / 'mlruns'
# Absolute path inside container
PERSON_DATA_DIR = CONTAINER_CODE_DIR / "data" / "person_data"

# --- Prompt Templates ---
PROMPT_TEMPLATE_COGNITIVE = """
Analyze the **Cognitive Abilities** of the elderly person based *only* on the following assessment answers.

**Assessment Answers:**
---
{combined_answers}
---

**Instructions for Cognitive Analysis:**
*   Focus strictly on aspects related to thinking, memory, attention, language, orientation, and decision-making mentioned in the answers.
*   Note any reported memory lapses, confusion, difficulty concentrating, problems finding words, or changes in thinking ability mentioned.
*   Mention expressed awareness or concern about cognitive function.
*   Evaluate clarity/coherence cautiously.
*   Do NOT include physical symptoms unless directly linked to a cognitive complaint in the answers.
*   If insufficient information is available in the answers, explicitly state that.
*   **Summarize the key cognitive findings briefly (1-3 sentences or concise bullet points). Be direct.**

**Cognitive Abilities Summary:**
"""

PROMPT_TEMPLATE_PHYSICAL = """
Analyze the **Physical Abilities** of the elderly person based *only* on the following assessment answers.

**Assessment Answers:**
---
{combined_answers}
---

**Instructions for Physical Analysis:**
*   Focus strictly on mobility, balance, strength, pain, energy levels, falls, assistive devices, and ability to perform physical ADLs mentioned in the answers.
*   Note mentioned sensory impairments (vision, hearing) if impacting physical function.
*   Do NOT include cognitive issues or general diagnoses unless explaining a physical limitation mentioned in the answers.
*   If insufficient information is available in the answers, explicitly state that.
*   **Briefly summarize the key physical abilities and limitations (1-3 sentences or concise bullet points). Focus on the main points.**

**Physical Abilities Summary:**
"""

PROMPT_TEMPLATE_HEALTH = """
Analyze the **Overall Health Status** of the elderly person based *only* on the following assessment answers.

**Assessment Answers:**
---
{combined_answers}
---

**Instructions for Health Analysis:**
*   Focus on self-reported general health, specific conditions/diagnoses mentioned, significant symptoms, medication details mentioned, doctor/hospital visits mentioned, and comments on diet/sleep if provided in the answers.
*   Synthesize the overall feeling with specific issues mentioned.
*   Do NOT analyze cognitive or mobility/pain here (covered elsewhere).
*   If insufficient information is available in the answers, explicitly state that.
*   **Provide a concise summary of the overall health status and key points mentioned (1-3 sentences or concise bullet points). Be brief.**

**Health Status Summary:**
"""

PROMPT_TEMPLATE_PERSONAL_INFO = """
Extract **Key Personal Information** about the elderly person based *only* on explicit statements in the following assessment answers.

**Assessment Answers:**
---
{combined_answers}
---

**Instructions for Personal Information Extraction:**
*   Extract *only* the information *directly stated* in the answers. Do **not** infer.
*   For each item below, report the mentioned information or state "Not Mentioned". Be concise.

**Personal Information Extracted:**
*   **Name:** [Extract name if mentioned, otherwise state Not Mentioned]
*   **Age:** [Extract age if mentioned, otherwise state Not Mentioned]
*   **Gender:** [Extract gender if mentioned, otherwise state Not Mentioned]
*   **Address:** [Extract address if mentioned, otherwise state Not Mentioned]
*   **Living Situation:** [e.g., Lives alone, Lives with spouse/family. State if mentioned, otherwise Not Mentioned]
*   **Support System:** [Mention of specific people involved? State if mentioned, otherwise Not Mentioned]
*   **Transportation:** [How they get around? State if mentioned, otherwise Not Mentioned]
*   **Assistance Needs (Tasks):** [Specific tasks mentioned? State if mentioned, otherwise Not Mentioned]
"""

PROMPT_TEMPLATE_NOTIFICATIONS = """
Review the following assessment answers *only*. Extract specific, actionable items needing follow-up or attention, and categorize them following the exact format below. Focus *only* on information explicitly stated in the answers.

**Assessment Answers:**
---
{combined_answers}
---

**Instructions & Output Format:**
*   Identify actionable items *directly from the answers*.
*   Categorize each item under the most relevant heading, following the exact structure shown below.
*   Use concise bullet points starting *exactly* with `- ` (dash and space) under the relevant heading for each actionable item. Do not add other prefixes like checkmarks.
*   If no actionable items fit a category based *only on the answers*, write "None noted in answers." under that heading (do not use a bullet point for this).
*   If absolutely no actionable items are found in any category, the overall output can simply be "No specific actionable items identified in the answers."
*   **Crucially, maintain the blank lines exactly as shown in the template below to ensure visual separation between categories.**

**Categorized Actionable Notes:**


**Medication:**
*(List specific mentions of medication names, dosages, schedules, reported side effects, or adherence issues needing attention found ONLY in the answers)*
- [Example: Review medication X schedule]
- [Example: Note reported dizziness after taking Y]


**Appointments/Scheduling:**
*(List specific mentions of upcoming/past appointments, tests, or scheduling needs requiring follow-up found ONLY in the answers)*
- [Example: Follow up on cardiology appointment]


**Cognitive Concerns:**
*(List specific mentions of memory lapses, confusion, orientation issues, or thinking changes requiring observation or follow-up found ONLY in the answers)*
- [Example: Investigate reported forgetfulness]


**Physical Needs/Limitations:**
*(List specific mentions of mobility difficulties, significant pain, fall risks, or requests for physical assistance needing action found ONLY in the answers)*
- [Example: Assess need for walker modification]


**Health Concerns:**
*(List specific mentions of non-physical symptoms, diet/sleep issues, or general health complaints requiring attention found ONLY in the answers)*
- [Example: Discuss reported poor sleep]


**Social/Support Needs:**
*(List specific mentions of needing help with tasks like cooking/cleaning/transport, expressed loneliness, or requests for contact/support found ONLY in the answers)*
- [Example: Clarify who helps with shopping]


**Other Important Notes:**
*(List any other specific actionable items mentioned ONLY in the answers that don't fit the categories above)*
- [Example: Note interest in joining activity group]

"""

PROMPT_TEMPLATE_ASK_DOCS = """You are an assistant answering questions about an elderly person based *only* on the provided excerpts from their personal documents. Focus specifically on information relevant to the person ID '{person_id}'. If the answer isn't in the excerpts provided below, clearly state that the information is not available in the documents. Do not infer or use outside knowledge. Be concise in your answer.

Relevant Information from Documents for '{person_id}':
--- START DOCUMENT EXCERPTS ---
{context}
--- END DOCUMENT EXCERPTS ---

Question about '{person_id}': {question}

Answer based only on the documents provided (be brief):"""

PROMPT_TEMPLATE_ASK_ANALYSIS = """You are an assistant answering questions about an elderly person based *only* on the provided assessment summary and their direct answers from a specific assessment. Do not use external knowledge. Provide a concise answer.

**Assessment Summary:**
Cognitive: {cognitive}
Physical: {physical}
Health: {health}
Personal Info: {personal_info}
Actionable Notes: {notifications}

**Direct Answers Given by {person_id} during the Assessment:**
--- START ANSWERS ---
{assessment_answers}
--- END ANSWERS ---

Based *only* on the summary and the answers provided above, answer the following question about {person_id}:
Question: {question}

Answer (be brief):"""

# --- MLflow Experiment ---
# Use the absolute container path for MLflow URI
MLFLOW_TRACKING_URI = f"file:{MLRUNS_DIR}"
MLFLOW_EXPERIMENT_NAME = "Elderly Care Assessment"

# --- Basic Checks & Logging (Run when module is loaded) ---
# These print statements run when the application imports this config.
if not GOOGLE_API_KEY:
    print(
        "Warning: GOOGLE_API_KEY environment variable not set! "
        "Analysis features requiring the Gemini API will fail."
    )

print("--- Configured Container Paths (from config.py) ---")
print(f"Container Base Dir              : {CONTAINER_CODE_DIR}")
print(f"Personal Docs Storage Dir       : {PERSONAL_DOCS_STORAGE_DIR}")
print(f"Personal Vector Store Dir       : {VECTOR_STORE_PERSONAL_DIR}")
print(f"MLflow Runs Dir (for tracking URI): {MLRUNS_DIR}")
print(f"Person Data Dir (History)       : {PERSON_DATA_DIR}")
print(f"MLflow Tracking URI             : {MLFLOW_TRACKING_URI}")
print("--- End Config Paths ---")