# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies if needed (e.g., for audio processing)
# RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
# Consider torch-cpu if not using GPU: pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend application code into the container at /app
COPY ./app /app/app

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable (optional, can be passed via docker-compose)
# ENV GOOGLE_API_KEY=YOUR_API_KEY

# Run uvicorn server when the container launches
# Use --host 0.0.0.0 to accept connections from outside the container
# Use --reload for development (mount code volume in docker-compose)
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]