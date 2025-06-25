#!/bin/bash

# Navigate to project directory
cd "$(dirname "$0")"

# Check if virtual environment exists, create if it doesn't
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt
echo ""

# Run the API server from the project root
echo "Starting API server..."
echo "Access Information:"
echo "- Local URL: http://localhost:8000/"
echo "- Other devices on same network: http://<your-machine-ip>:8000/"
echo "  (Run ./access_info.sh to see all available IP addresses)"
echo
cd "$(dirname "$0")"
PYTHONPATH=$PYTHONPATH:. python src/api_server.py
