FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create models directory if it doesn't exist
RUN mkdir -p models

# Set production env and default port
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_ENV=production \
    PORT=8000

# Expose port
EXPOSE ${PORT}

# Run with gunicorn using eventlet worker for WebSocket support
# -k eventlet: Use eventlet async worker (required for WebSockets)
# -w 1: Single worker (eventlet handles concurrency internally)
# --timeout 300: 5 minute timeout for long-running training tasks
# --log-level info: Better logging for debugging
CMD ["sh", "-c", "gunicorn -k eventlet -w 1 --timeout 300 --log-level info -b 0.0.0.0:${PORT:-8000} src.api_server:app"]
