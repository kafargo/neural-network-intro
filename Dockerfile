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

# Run with Gunicorn using eventlet worker for production WebSocket support
# -w 1: Single worker (required for Socket.IO with in-memory state)
# -k eventlet: Async worker class for WebSocket support
# --timeout 120: Longer timeout for training operations
# --log-level info: Production logging
CMD ["sh", "-c", "gunicorn -w 1 -k eventlet -b 0.0.0.0:${PORT:-8000} --timeout 120 --log-level info src.api_server:app"]
