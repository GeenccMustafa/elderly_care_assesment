# ui_components.py

import html
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

# Import local modules: CSS, event handlers, and constants
from css_styles import css
from event_handlers import (ICON_ANALYSIS, ICON_ASK, ICON_AUDIO, ICON_CHECK,
                            ICON_DOCUMENT, ICON_DOWNLOAD, ICON_ERROR,
                            ICON_INFO, ICON_LOADING, ICON_NOTES, ICON_PERSON,
                            ICON_QUESTION, ICON_REFRESH, ICON_SUBMIT,
                            ICON_TRANSCRIBE, ICON_UPLOAD, ICON_WARNING,
                            MAX_QUESTIONS, ask_question_backend,
                            fetch_assessment_questions,
                            fetch_latest_analysis,
                            fetch_person_list_for_load,
                            generate_download_link, handle_pdf_upload,
                            handle_person_refresh, submit_assessment,
                            switch_view, transcribe_answer_audio,
                            update_active_person_dd, update_pdf_list)

# --- Logger Setup ---
logger = logging.getLogger(__name__)
# Ensure logging is configured externally (e.g., in app.py)

# --- Module-Level Variables ---
# Stores tuples of Gradio components for each question row. Populated by build_ui.
# Format: (group, question_md, audio_input, transcribe_btn, transcript_disp, status_textbox)
# While global state isn't ideal, it simplifies component referencing across functions here.
_question_elements: List[Tuple[gr.Group, gr.Markdown, gr.Audio, gr.Button, gr.Textbox, gr.Textbox]] = []


# --- UI Population Logic ---

def _generate_ui_updates(
    questions_data: Optional[List[Dict]] = None
) -> List[gr.update]:
    """
    Generates a list of Gradio update objects to show/hide/populate question rows.

    Args:
        questions_data: The list of question dictionaries fetched from the backend,
                        or None/empty if loading failed.

    Returns:
        A list of gr.update objects targeting the question loading messages,
        error messages, container visibility, and individual question components.
    """
    global _question_elements
    all_updates = []
    num_questions_fetched = len(questions_data) if questions_data else 0
    num_ui_elements = len(_question_elements)

    if not questions_data:
        logger.error("Failed to load questions from backend or data is empty.")
        all_updates.extend([
            gr.update(visible=False),  # Hide Loading message
            # Show Error message
            gr.update(value=f"**{ICON_ERROR} Error:** Failed to load questions from backend.", visible=True),
            gr.update(visible=False)   # Hide Questions container
        ])
        # Hide all question rows defined in the UI
        for i in range(num_ui_elements):
            all_updates.extend([gr.update(visible=False)] * len(_question_elements[i])) # group + 5 components
    else:
        logger.info(f"Populating UI based on {num_questions_fetched} fetched questions.")
        all_updates.extend([
            gr.update(visible=False),  # Hide Loading message
            gr.update(visible=False),  # Hide Error message
            gr.update(visible=True)    # Show Questions container
        ])
        num_to_show = min(num_questions_fetched, num_ui_elements)

        for i in range(num_ui_elements):
            if i < num_to_show:
                q_data = questions_data[i]
                q_id = q_data.get('question_id', f"ErrorID_{i+1}")
                q_text = q_data.get('text', '(Error loading question text)')

                # Format question text. DO NOT escape q_text here - assume backend sends safe text.
                formatted_question = f"**{ICON_QUESTION} Q{q_id}:** {q_text}"

                # Updates for the components in the i-th question row
                all_updates.extend([
                    gr.update(visible=True), # Show question group
                    gr.update(value=formatted_question), # Set question markdown text
                    gr.update(value=None),   # Clear audio input
                    gr.update(),             # Keep transcribe button as is
                    gr.update(value=""),     # Clear transcription display
                    gr.update(value=f"{ICON_INFO} Ready") # Reset status textbox
                ])
            else:
                # Hide question rows beyond the number of fetched questions
                all_updates.extend([gr.update(visible=False)] * len(_question_elements[i]))

    return all_updates


def populate_assessment_ui_initial(questions_data: List[Dict]) -> List[gr.update]:
    """
    Generates UI updates for initial load, including populating question elements
    and initializing the transcription state.

    Args:
        questions_data: List of question data from the backend.

    Returns:
        List of gr.update objects for UI elements and the initial transcription state.
    """
    ui_updates = _generate_ui_updates(questions_data)
    # Initialize transcription state with question IDs as keys and empty strings
    initial_state_dict = {
        q.get('question_id'): ""
        for q in questions_data
        if isinstance(q, dict) and q.get('question_id') is not None
    } if questions_data else {}
    # Append the state update to the UI updates
    ui_updates.append(gr.update(value=initial_state_dict))
    return ui_updates


def populate_assessment_ui_refresh(questions_data: List[Dict]) -> List[gr.update]:
    """
    Generates UI updates needed after refreshing the person list,
    primarily resetting the question text display.

    Args:
        questions_data: List of question data from the backend (likely from state).

    Returns:
        List of gr.update objects for UI elements.
    """
    return _generate_ui_updates(questions_data)


# --- UI Construction Function ---

def build_ui() -> gr.Blocks:
    """
    Builds the Gradio Blocks UI, including layout, components, states,
    and event listeners.

    Returns:
        The constructed gr.Blocks demo object.
    """
    global _question_elements
    _question_elements.clear() # Clear previous elements if rebuilding
    logger.info("Building Gradio UI layout and components...")

    with gr.Blocks(css=css, title="Elderly Care Assessment Assistant") as demo:
        gr.Markdown("# 🩺 Elderly Care Assessment Assistant")
        gr.Markdown("Enter/Select Person ID, record answers, upload documents, submit for analysis, and ask questions.")

        # --- States ---
        # Stores current transcriptions {q_id: text}
        transcribed_answers_state = gr.State({})
        # Stores the list of question dicts fetched on load
        fetched_questions_state = gr.State([])
        # Stores the currently selected/active Person ID
        active_person_id_state = gr.State("")
        # Stores the name of the currently visible view ('assessment', 'docs', 'ask')
        active_view_state = gr.State("assessment") # Default view

        # --- Top Row: Person Selection ---
        with gr.Row(elem_id="person_id_row"):
             person_id_input = gr.Dropdown(
                 label=f"{ICON_PERSON} Select/Enter Person ID",
                 interactive=True,
                 allow_custom_value=True,
                 elem_id="person_id_dd"
             )
             person_id_refresh_btn = gr.Button(
                 value=f"{ICON_REFRESH} Refresh Persons & Reset UI",
                 scale=0, # Take minimum width needed
                 min_width=150,
                 variant="secondary",
                 elem_id="person_refresh_btn"
             )
        gr.Markdown("---") # Separator

        # --- Navigation Row ---
        with gr.Row(elem_classes="view-nav-row"):
            assessment_btn = gr.Button(
                f"📝 Assessment & Analysis", scale=1, elem_classes="view-nav-button", variant="primary"
            )
            docs_btn = gr.Button(
                f"📄 Document Management", scale=1, elem_classes="view-nav-button", variant="secondary"
            )
            ask_btn = gr.Button(
                f"💬 Ask About Person", scale=1, elem_classes="view-nav-button", variant="secondary"
            )

        # --- Content Columns (Controlled by Navigation) ---

        # 1. Assessment & Analysis Column
        with gr.Column(visible=True, elem_id="assessment_view") as assessment_column:
            _build_assessment_column() # Delegate building this section

        # 2. Document Management Column
        with gr.Column(visible=False, elem_id="docs_view") as docs_column:
            pdf_upload_input, pdf_upload_button, pdf_upload_status, \
            pdf_download_select, pdf_refresh_button, pdf_download_link_display = \
                _build_docs_column() # Delegate building

        # 3. Ask About Person Column
        with gr.Column(visible=False, elem_id="ask_view") as ask_column:
            question_input, ask_button, doc_answer_output, \
            analysis_answer_output, qa_status_output = \
                _build_ask_column() # Delegate building

        # --- Footer ---
        _build_footer()

        # --- Define Listener Output Lists ---
        # Combine lists of components targeted by different actions for clarity
        change_outputs = _get_person_change_outputs(
            active_person_id_state, transcribed_answers_state, status_output,
            analysis_output_display, notifications_output_display,
            question_input, doc_answer_output, analysis_answer_output, qa_status_output
        )
        refresh_outputs = _get_person_refresh_outputs(
            person_id_input, active_person_id_state, transcribed_answers_state, status_output,
            analysis_output_display, notifications_output_display,
            question_input, doc_answer_output, analysis_answer_output, qa_status_output
        )
        populate_ui_outputs_list = _get_populate_ui_outputs(
            loading_msg, error_msg, questions_container
        )
        populate_outputs_on_load = populate_ui_outputs_list + [transcribed_answers_state]

        # --- Attach Event Listeners ---
        logger.info("Attaching Gradio event listeners...")
        _attach_listeners(
            demo, active_view_state, assessment_btn, docs_btn, ask_btn,
            assessment_column, docs_column, ask_column,
            person_id_input, person_id_refresh_btn, active_person_id_state,
            transcribed_answers_state, fetched_questions_state,
            submit_button, status_output, analysis_output_display, notifications_output_display,
            pdf_upload_input, pdf_upload_button, pdf_upload_status,
            pdf_download_select, pdf_refresh_button, pdf_download_link_display,
            question_input, ask_button, doc_answer_output, analysis_answer_output, qa_status_output,
            change_outputs, refresh_outputs, populate_outputs_on_load, populate_ui_outputs_list,
            loading_msg, error_msg, questions_container # Pass these for load sequence
        )
        logger.info("Event listeners attached.")

    logger.info("Gradio UI build complete.")
    return demo

# --- Helper Functions for Building UI Sections ---

def _build_assessment_column():
    """Builds the components within the Assessment & Analysis column."""
    global _question_elements, status_output, analysis_output_display, notifications_output_display
    global loading_msg, error_msg, questions_container, submit_button # Make globally accessible if needed by listeners

    gr.Markdown("## Assessment & Analysis") # Use H2 within the column
    gr.Markdown(f"### Step 1: {ICON_AUDIO} Record and Transcribe Answers")
    gr.Markdown("_Select/Enter Person ID first. Record audio for each question, then click Transcribe._")

    # Placeholders for dynamic question loading
    loading_msg = gr.Markdown(f"*{ICON_LOADING} Loading questions...*", visible=True)
    error_msg = gr.Markdown("", visible=False, elem_classes="error-message") # Add class for styling
    questions_container = gr.Column(visible=False, elem_id="questions_container")

    with questions_container:
        temp_question_elements = []
        for i in range(MAX_QUESTIONS):
            q_id = i + 1
            with gr.Group(visible=False) as question_group: # Start hidden
                question_md = gr.Markdown(f"*{ICON_LOADING}*", elem_classes="question-text-md")
                with gr.Row(equal_height=False): # Allow variable height
                    audio_input = gr.Audio(
                        show_label=False, sources=["microphone"], type="filepath",
                        interactive=True, show_download_button=False, container=False # Less padding
                    )
                    with gr.Column(scale=0, min_width=150):
                         # Pass q_id to button element ID for potential specific styling/targeting
                         transcribe_btn = gr.Button(value=f"{ICON_TRANSCRIBE} Transcribe", elem_id=f"transcribe_btn_{q_id}")
                         q_status = gr.Textbox(
                             label="Status", value=f"{ICON_INFO} Ready", interactive=False,
                             max_lines=1, container=False, # Less padding
                             elem_id=f"q_status_{q_id}"
                         )
                transcript_disp = gr.Textbox(
                    label="Transcription", interactive=False, lines=2, show_copy_button=True,
                    placeholder="Transcription appears here...", elem_id=f"transcript_disp_{q_id}"
                )
            temp_question_elements.append(
                (question_group, question_md, audio_input, transcribe_btn, transcript_disp, q_status)
            )
        _question_elements[:] = temp_question_elements # Update the global list

    gr.Markdown("---")
    gr.Markdown(f"### Step 2: {ICON_SUBMIT} Submit for Analysis")
    gr.Markdown(f"_*Ensure '{ICON_PERSON} Person ID' is selected/entered and answers are transcribed ({ICON_TRANSCRIBE})*_")
    submit_button = gr.Button(f"{ICON_SUBMIT} Submit Answers & Analyze", variant="primary", elem_id="submit_analysis_btn")
    status_output = gr.Textbox(label=f"{ICON_ANALYSIS} Overall Status", value=f"{ICON_INFO} Idle", interactive=False, max_lines=1, elem_id="overall_status")

    gr.Markdown("---")
    gr.Markdown(f"### Step 3: {ICON_CHECK} Review Analysis Results")
    with gr.Row():
        with gr.Column(scale=2):
            analysis_output_display = gr.Markdown(
                f"## {ICON_ANALYSIS} Analysis Results\n\n*{ICON_INFO} Select a Person ID or submit an assessment.*",
                elem_id="analysis_display"
            )
        with gr.Column(scale=1):
            notifications_output_display = gr.Markdown(
                f"## {ICON_NOTES} Actionable Notes\n\n*{ICON_INFO} Select a Person ID or submit an assessment.*",
                elem_id="notifications_display"
            )
    gr.Markdown("---")
    # Return components needed for listener attachment if not made global
    # return loading_msg, error_msg, questions_container, submit_button, status_output, analysis_output_display, notifications_output_display


def _build_docs_column():
    """Builds the components within the Document Management column."""
    gr.Markdown("## Document Management")
    gr.Markdown(
        f"Upload/download documents. **Naming Hint:** Use `{ICON_PERSON}PersonID_Description.pdf` or "
        f"`{ICON_PERSON}PersonID-Description.pdf` for potential auto-association during indexing (if implemented)."
    )
    gr.Markdown(
        f"**{ICON_WARNING} Manual Step Required:** If document content changes, run `python build_index.py` "
        "and restart the backend service to update the search index used by the 'Ask' feature."
    )
    gr.Markdown("---")
    gr.Markdown(f"### {ICON_UPLOAD} Upload Documents")
    with gr.Row():
        pdf_upload_input = gr.File(
            label="Select PDF(s) to Upload", file_types=['.pdf'], file_count="multiple", scale=2,
            elem_id="pdf_upload_input"
        )
        pdf_upload_button = gr.Button(
            value=f"{ICON_UPLOAD} Upload Selected PDF(s)", scale=1, variant="primary",
            elem_id="pdf_upload_btn"
        )
    pdf_upload_status = gr.Textbox(
        label=f"{ICON_INFO} Upload Status", interactive=False, lines=5, show_copy_button=True,
        max_lines=10, placeholder="Upload progress and results appear here...",
        elem_id="pdf_upload_status"
    )
    gr.Markdown("---")
    gr.Markdown(f"### {ICON_DOWNLOAD} Download Documents")
    with gr.Row():
        pdf_download_select = gr.Dropdown(
            label=f"{ICON_DOCUMENT} Select Document to Download", scale=2, interactive=True,
            elem_id="pdf_download_select"
        )
        pdf_refresh_button = gr.Button(
            value=f"{ICON_REFRESH} Refresh List", scale=1, min_width=100, variant="secondary",
            elem_id="pdf_refresh_btn"
        )
    pdf_download_link_display = gr.Markdown(
        value=f"*{ICON_INFO} Select PDF to generate download link.*",
        elem_id="pdf_download_link"
    )
    gr.Markdown("---")
    return (
        pdf_upload_input, pdf_upload_button, pdf_upload_status,
        pdf_download_select, pdf_refresh_button, pdf_download_link_display
    )


def _build_ask_column():
    """Builds the components within the Ask About Person column."""
    gr.Markdown("## Ask About Person")
    gr.Markdown(
        f"Ask questions about the **{ICON_PERSON} currently selected Person ID**. Answers are generated from "
        f"both uploaded documents ({ICON_DOCUMENT}) and the latest assessment analysis ({ICON_ANALYSIS})."
    )
    gr.Markdown("---")
    gr.Markdown(f"### {ICON_QUESTION} Ask a Question")
    with gr.Row():
         question_input = gr.Textbox(
             label=f"Your Question about the selected person",
             placeholder="e.g., What recent health concerns were noted? Any mobility issues mentioned?",
             lines=3, scale=3, elem_id="qa_question_input"
         )
         ask_button = gr.Button(
             value=f"{ICON_ASK} Ask Question", variant="primary", scale=1, elem_id="qa_ask_btn"
         )
    gr.Markdown("---")
    gr.Markdown(f"### {ICON_CHECK} Answers")
    with gr.Row():
         with gr.Column(scale=1):
            doc_answer_output = gr.Textbox(
                label=f"{ICON_DOCUMENT} Answer from Documents", lines=8, interactive=False,
                show_copy_button=True, placeholder="Answer based on uploaded PDFs appears here...",
                elem_id="qa_doc_answer"
            )
         with gr.Column(scale=1):
            analysis_answer_output = gr.Textbox(
                label=f"{ICON_ANALYSIS} Answer from Analysis", lines=8, interactive=False,
                show_copy_button=True, placeholder="Answer based on latest assessment analysis appears here...",
                elem_id="qa_analysis_answer"
            )
    with gr.Row():
         qa_status_output = gr.Textbox(
             label=f"{ICON_INFO} Q&A Status", value=f"{ICON_INFO} Idle", interactive=False,
             max_lines=1, elem_id="qa_status"
         )
    gr.Markdown("---")
    return (
        question_input, ask_button, doc_answer_output,
        analysis_answer_output, qa_status_output
    )


def _build_footer():
    """Builds the footer markdown components."""
    backend_url_for_footer = os.getenv("BACKEND_BASE_URL", "http://localhost:8000").rstrip('/')
    mlflow_ui_url_for_footer = os.getenv("MLFLOW_UI_URL", "http://localhost:5000").rstrip('/')

    gr.Markdown("---")
    # Use html.escape just in case URLs somehow contain characters needing it for display
    gr.Markdown(f"**Backend API Status:** Connecting to `{html.escape(backend_url_for_footer)}`")
    gr.Markdown(
        f"🔗 API Docs: [{html.escape(backend_url_for_footer)}/docs]({html.escape(backend_url_for_footer)}/docs) | "
        f"{ICON_ANALYSIS} MLflow UI: [{html.escape(mlflow_ui_url_for_footer)}]({html.escape(mlflow_ui_url_for_footer)})"
    )


# --- Helper Functions for Defining Listener Outputs ---

def _get_person_change_outputs(
    active_person_id_state, transcribed_answers_state, status_output,
    analysis_output_display, notifications_output_display,
    question_input, doc_answer_output, analysis_answer_output, qa_status_output
):
    """Returns the list of outputs for the person_id_input.change listener."""
    global _question_elements
    outputs = [
        active_person_id_state,
        transcribed_answers_state,
        status_output,
        analysis_output_display,
        notifications_output_display
    ]
    # Reset transcription display and status for each question
    for i in range(len(_question_elements)):
        outputs.extend([_question_elements[i][4], _question_elements[i][5]]) # transcript_disp, q_status
    # Reset Q&A tab elements
    outputs.extend([
        question_input,
        doc_answer_output,
        analysis_answer_output,
        qa_status_output
    ])
    # Reset Question Markdown Text
    for i in range(len(_question_elements)):
        outputs.append(_question_elements[i][1]) # question_md

    return outputs


def _get_person_refresh_outputs(
    person_id_input, active_person_id_state, transcribed_answers_state, status_output,
    analysis_output_display, notifications_output_display,
    question_input, doc_answer_output, analysis_answer_output, qa_status_output
):
    """Returns the list of outputs for the person_id_refresh_btn.click listener."""
    global _question_elements
    outputs = [
        person_id_input, # Update dropdown choices/value
        active_person_id_state,
        transcribed_answers_state,
        status_output,
        analysis_output_display,
        notifications_output_display
    ]
    # Reset transcription display and status for each question
    for i in range(len(_question_elements)):
        outputs.extend([_question_elements[i][4], _question_elements[i][5]]) # transcript_disp, q_status
    # Reset audio player for each question
    for i in range(len(_question_elements)):
        outputs.append(_question_elements[i][2]) # audio_input
    # Reset Question Markdown Text (will be repopulated by .then())
    for i in range(len(_question_elements)):
        outputs.append(_question_elements[i][1]) # question_md
    # Reset Q&A tab elements
    outputs.extend([
        question_input,
        doc_answer_output,
        analysis_answer_output,
        qa_status_output
    ])
    return outputs


def _get_populate_ui_outputs(loading_msg, error_msg, questions_container):
    """Returns the list of outputs for the UI population functions."""
    global _question_elements
    # Start with the general visibility controls
    outputs = [loading_msg, error_msg, questions_container]
    # Add all components from each question row
    for row_elements in _question_elements:
        outputs.extend(list(row_elements)) # Add all elements from the tuple
    return outputs


# --- Listener Attachment Function ---

def _attach_listeners(
    demo: gr.Blocks, active_view_state, assessment_btn, docs_btn, ask_btn,
    assessment_column, docs_column, ask_column,
    person_id_input, person_id_refresh_btn, active_person_id_state,
    transcribed_answers_state, fetched_questions_state,
    submit_button, status_output, analysis_output_display, notifications_output_display,
    pdf_upload_input, pdf_upload_button, pdf_upload_status,
    pdf_download_select, pdf_refresh_button, pdf_download_link_display,
    question_input, ask_button, doc_answer_output, analysis_answer_output, qa_status_output,
    change_outputs, refresh_outputs, populate_outputs_on_load, populate_ui_outputs_list,
    loading_msg, error_msg, questions_container # Needed for load sequence
):
    """Attaches all event listeners to the appropriate components."""
    global _question_elements

    # --- View Switching Listeners ---
    view_outputs = [
        active_view_state, assessment_column, docs_column, ask_column,
        assessment_btn, docs_btn, ask_btn
    ]
    assessment_btn.click(
        fn=switch_view, inputs=[gr.State("assessment")], outputs=view_outputs, show_progress="hidden"
    )
    docs_btn.click(
        fn=switch_view, inputs=[gr.State("docs")], outputs=view_outputs, show_progress="hidden"
    )
    ask_btn.click(
        fn=switch_view, inputs=[gr.State("ask")], outputs=view_outputs, show_progress="hidden"
    )

    # --- Person ID Selection/Refresh Listeners ---
    person_id_input.change(
        fn=update_active_person_dd,
        inputs=[person_id_input],
        outputs=change_outputs,
        show_progress="hidden" # Usually fast enough
    ).then(
        fn=fetch_latest_analysis, # Fetch analysis after state is updated
        inputs=[active_person_id_state],
        outputs=[analysis_output_display, notifications_output_display],
        show_progress="minimal"
    ).then(
        fn=populate_assessment_ui_refresh, # Also repopulate questions based on state
        inputs=[fetched_questions_state],
        outputs=populate_ui_outputs_list,
        show_progress="hidden"
    )

    person_id_refresh_btn.click(
        fn=handle_person_refresh,
        inputs=None,
        outputs=refresh_outputs,
        show_progress="hidden" # Usually fast enough
    ).then(
        fn=fetch_assessment_questions, # Fetch new questions after reset
        inputs=None,
        outputs=[fetched_questions_state],
        show_progress="hidden"
    ).then(
        fn=populate_assessment_ui_initial, # Repopulate UI and state
        inputs=[fetched_questions_state],
        outputs=populate_outputs_on_load,
        show_progress="hidden"
    ).then(
        # Ensure view is reset to assessment after a full refresh
        lambda: switch_view("assessment"),
        inputs=None,
        outputs=view_outputs,
        show_progress="hidden"
    )

    # --- Assessment Listeners ---
    for i, elements in enumerate(_question_elements):
        q_id = i + 1 # Use 1-based index for state key and API name
        group, q_md, audio_comp, btn_comp, disp_comp, status_comp = elements
        btn_comp.click(
            fn=transcribe_answer_audio,
            inputs=[audio_comp, gr.State(value=q_id), transcribed_answers_state],
            outputs=[transcribed_answers_state, status_comp, disp_comp],
            show_progress="minimal",
            api_name=f"transcribe_q{q_id}" # Unique API name if needed
        )

    submit_button.click(
        fn=submit_assessment,
        inputs=[active_person_id_state, transcribed_answers_state, fetched_questions_state],
        outputs=[status_output, analysis_output_display, notifications_output_display],
        show_progress="minimal", # Use minimal as it yields status updates
        api_name="submit_assessment_analysis"
    )

    # --- Document Management Listeners ---
    pdf_upload_button.click(
        fn=handle_pdf_upload,
        inputs=[pdf_upload_input],
        outputs=[pdf_upload_status, pdf_download_select], # Update status and refresh dropdown
        show_progress="full" # Show progress bar for uploads
    ).then(
        # Clear the file input component after upload attempt
        lambda: gr.File(value=None), outputs=[pdf_upload_input]
    )

    pdf_refresh_button.click(
        fn=update_pdf_list, # Fetches list and updates dropdown
        inputs=[],
        outputs=[pdf_download_select],
        show_progress="hidden"
    )

    pdf_download_select.change(
        fn=generate_download_link,
        inputs=[pdf_download_select],
        outputs=[pdf_download_link_display],
        show_progress="hidden" # Link generation is fast
    )

    # --- Q&A Listeners ---
    ask_button.click(
        fn=ask_question_backend,
        inputs=[active_person_id_state, question_input],
        outputs=[doc_answer_output, analysis_answer_output, qa_status_output],
        show_progress="minimal" # Use minimal as it yields status updates
    )
    # Allow submitting Q&A question via Enter key in the textbox
    question_input.submit(
        fn=ask_question_backend,
        inputs=[active_person_id_state, question_input],
        outputs=[doc_answer_output, analysis_answer_output, qa_status_output],
        show_progress="minimal"
    )

    # --- Initial Load Actions ---
    demo.load(
        fn=fetch_assessment_questions, # 1. Fetch questions
        inputs=[],
        outputs=[fetched_questions_state],
        show_progress="hidden"
    ).then(
        fn=populate_assessment_ui_initial, # 2. Populate UI based on questions & init state
        inputs=[fetched_questions_state],
        outputs=populate_outputs_on_load,
        show_progress="hidden"
    ).then(
        fn=update_pdf_list, # 3. Populate PDF download list
        inputs=[],
        outputs=[pdf_download_select],
        show_progress="hidden"
    ).then(
        fn=fetch_person_list_for_load, # 4. Populate Person ID dropdown
        inputs=None,
        outputs=[person_id_input],
        show_progress="hidden"
    )