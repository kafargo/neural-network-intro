#!/bin/bash

# Script to shut down all instances of the neural network app
echo "Shutting down all instances of the neural network application..."

# Stop any Docker Compose services
if [ -f "docker-compose.yml" ]; then
    echo "Stopping Docker Compose services..."
    docker-compose down
fi

# Find and kill any directly running Python processes for api_server.py
echo "Checking for directly running Flask servers by process name..."
FLASK_PIDS=$(pgrep -f "python.*api_server.py")
if [ -n "$FLASK_PIDS" ]; then
    echo "Found Flask server processes. Stopping them..."
    echo $FLASK_PIDS | xargs kill -9
    echo "Flask server processes stopped."
else
    echo "No Flask server processes found by name."
fi

# Also check for processes running on port 8000
echo "Checking for processes running on port 8000..."
PORT_PIDS=$(lsof -t -i :8000 2>/dev/null)
if [ -n "$PORT_PIDS" ]; then
    echo "Found processes running on port 8000. Stopping them..."
    echo $PORT_PIDS | xargs kill -9
    echo "Port 8000 processes stopped."
else
    echo "No processes found running on port 8000."
fi

# Check the status after shutdown
echo "Checking status after shutdown..."
running_processes=$(ps aux | grep "[p]ython.*api_server.py")
port_check=$(lsof -i :8000 2>/dev/null)

if [ -z "$running_processes" ] && [ -z "$port_check" ]; then
    echo "✅ All neural network app processes have been shut down and port 8000 is free."
elif [ -n "$running_processes" ]; then
    echo "❌ Some api_server.py processes might still be running:"
    echo "$running_processes"
elif [ -n "$port_check" ]; then
    echo "❌ Port 8000 is still in use:"
    echo "$port_check"
fi

# Show Docker status
echo "Current Docker container status:"
docker ps

echo "Shutdown process completed."
