# frontend/Dockerfile

# --- Base Stage ---
# Use an official Python slim runtime for a smaller base image.
FROM python:3.10-slim as base

# Set common Python environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files to disc (improves caching layer behaviour)
# PYTHONUNBUFFERED: Prevents Python from buffering stdout/stderr (useful for logging in containers)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create a non-root user and group
ARG USER_UID=1001
ARG USER_GID=1001
RUN groupadd --gid ${USER_GID} appgroup && \
    useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /sbin/nologin appuser

# Set the working directory
WORKDIR /app

# --- Builder Stage ---
# Used to install dependencies without including build tools in the final image.
FROM base as builder

# Install build dependencies if any are needed for pip packages (e.g., gcc)
# RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
# --no-cache-dir reduces image size by not storing the pip cache
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --requirement requirements.txt

# --- Final Stage ---
# Start from the clean base image again for the final runtime image.
FROM base as final

# Copy installed Python packages from the builder stage's virtual environment
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
# Copy any executables installed by pip (if any)
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the application code into the working directory
# Assumes all necessary Python files are in the root of the build context (frontend dir)
# If subdirectories or other files are needed, change to: COPY . /app/
COPY *.py /app/

# Ensure the app directory is owned by the non-root user
RUN chown -R appuser:appgroup /app

# Switch to the non-root user
USER appuser

# Expose the port Gradio will run on (defined by GRADIO_SERVER_PORT or default 7860)
# This is informational for the user/orchestrator.
EXPOSE 7860

# Add standard labels (optional but good practice)
LABEL org.opencontainers.image.title="Elderly Care Frontend" \
      org.opencontainers.image.description="Gradio frontend for the Elderly Care Assessment Assistant" \
      org.opencontainers.image.source="https://github.com/GeenccMustafa/elderly_care_assesment" \
      org.opencontainers.image.version="1.0.0"

# Add a basic healthcheck (optional)
# Checks if the Gradio server is listening on the default port. Adjust if needed.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:7860/ || exit 1
# Note: If Gradio is slow to start or requires auth, this might need adjustment or a different check.

# Define the command to run the application.
# Expects BACKEND_BASE_URL to be passed as an environment variable at runtime.
# Example: docker run -e BACKEND_BASE_URL=http://backend-service:8000 ... image-name
CMD ["python", "app.py"]