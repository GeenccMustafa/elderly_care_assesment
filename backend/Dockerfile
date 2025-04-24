# backend/Dockerfile

# --- Base Stage ---
# Use an official Python runtime as a parent image
# Using slim version reduces image size
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1 # Prevents python from writing pyc files
ENV PYTHONUNBUFFERED 1     # Prevents python from buffering stdout/stderr
# Set the initial working directory in the container
WORKDIR /code # Set base WORKDIR early

# Install system dependencies
# - libsndfile1 is required by soundfile
# - ffmpeg is often required by librosa for various audio formats
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- Builder Stage (Optional but good practice) ---
# Install build dependencies if any, separate from runtime
FROM base as builder
# WORKDIR is inherited from base stage (/code)

# Install pip dependencies
# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
# Upgrade pip and install packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Final Stage ---
# Use the slim base image again for the final, smaller image
FROM base as final
# WORKDIR is inherited from base stage (/code)

# Copy installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the application code from the host into the container
# Copies the 'app' directory into '/code/app' inside the container
COPY ./app /code/app

# **** ADD THIS LINE ****
# Explicitly add the base code directory to the Python path
ENV PYTHONPATH "${PYTHONPATH}:/code"
# ***********************

# The final working directory should be /code (one level above the 'app' package)
# WORKDIR /code # This is inherited from the 'base' stage and remains /code

# Expose the port the app runs on
EXPOSE 8000

# Tell uvicorn the full module path to the app object, now findable from /code
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]