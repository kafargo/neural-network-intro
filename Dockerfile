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

# Run with gunicorn (use shell form so ${PORT} is expanded)
CMD ["sh", "-c", "gunicorn -w 1 -b 0.0.0.0:${PORT:-8000} src.api_server:app"]
