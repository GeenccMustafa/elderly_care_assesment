FROM python:3.10-slim

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY ./app.py /app/app.py

EXPOSE 7860

# Set environment variable for backend URL (can be overridden in docker-compose)
ENV BACKEND_API_URL="http://backend:8000/api/process_audio"

CMD ["python", "app.py"]