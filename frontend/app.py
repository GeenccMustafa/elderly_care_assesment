import gradio as gr
import requests
import numpy as np
import os

# Use environment variable or hardcode for local dev
# Assumes backend is running and accessible at this URL
# If using docker-compose, this will be 'http://backend:8000/api/process_audio'
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000/api/process_audio")

print(f"Connecting to Backend API at: {BACKEND_API_URL}")

def process_audio_chunk(audio_filepath):
    """
    Sends the audio chunk (from Gradio filepath) to the backend API
    and returns the results for Gradio components.
    """
    if audio_filepath is None:
        return "[No audio input]", "", [], "[Please provide audio]"

    print(f"Processing audio file: {audio_filepath}")

    try:
        # Prepare the file for the request
        with open(audio_filepath, 'rb') as f:
            files = {'audio_file': (os.path.basename(audio_filepath), f, 'audio/wav')} # Adjust mime type if needed
            response = requests.post(BACKEND_API_URL, files=files, timeout=60) # Increase timeout

        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        data = response.json()
        print("Received response from backend:", data)

        # Extract data for Gradio components
        transcription = data.get("transcription", "[No transcription received]")
        summary = data.get("summary", "")
        context = data.get("context", []) # List of dicts
        error = data.get("error")

        if error:
            return transcription, summary, context, f"[Backend Error: {error}]"
        else:
            return transcription, summary, context, f"Processed successfully." # Status message

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to backend: {e}")
        return "[Error connecting to backend]", "", [], f"[Error: {e}]"
    except Exception as e:
        print(f"Error processing response: {e}")
        return "[Error processing response]", "", [], f"[Error: {e}]"

# --- Gradio Interface Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ Real-Time Classroom Captioning Demo")
    gr.Markdown("Record audio chunks using your microphone. The audio will be sent to the backend API for processing.")

    with gr.Row():
        # Input: Microphone audio chunk
        # type="filepath" saves audio to a temp file Gradio can access
        mic_input = gr.Audio(sources=["microphone"], type="filepath", label="Record Audio Chunk (e.g., 5-10 seconds)")

    with gr.Row():
        # Button to trigger processing
        submit_button = gr.Button("Process Audio Chunk")

    with gr.Row():
        # Outputs
        with gr.Column(scale=2):
            transcription_output = gr.Textbox(label="📜 Transcription", lines=10, interactive=False)
        with gr.Column(scale=1):
            summary_output = gr.Textbox(label="💡 Summary", lines=5, interactive=False)
            context_output = gr.JSON(label="🔗 Related Context") # Display context as JSON
            status_output = gr.Textbox(label="📊 Status", interactive=False)


    # Connect button click to the processing function
    submit_button.click(
        fn=process_audio_chunk,
        inputs=[mic_input],
        outputs=[transcription_output, summary_output, context_output, status_output]
    )

    gr.Markdown("---")
    gr.Markdown("Backend API Docs available at [http://localhost:8000/docs](http://localhost:8000/docs) (when running via docker-compose).")
    gr.Markdown("MLflow UI available at [http://localhost:5000](http://localhost:5000) (run `mlflow ui` locally in project root).")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860) # Run Gradio server