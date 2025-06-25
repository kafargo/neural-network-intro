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

# Expose port - Railway will set PORT env var, but default to 8000
EXPOSE ${PORT:-8000}

# Run the application
CMD ["python", "src/api_server.py"]
