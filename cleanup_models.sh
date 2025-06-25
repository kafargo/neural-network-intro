#!/bin/bash

# This script cleans up saved models
echo "$(date): Running model cleanup..."

# Path to models directory
MODELS_DIR="/app/models"

# Check if models directory exists
if [ -d "$MODELS_DIR" ]; then
    # Count models before cleanup
    MODEL_COUNT=$(ls -1 "$MODELS_DIR" | wc -l)
    
    # Remove all files in the models directory
    rm -f "$MODELS_DIR"/*
    
    echo "$(date): Cleanup complete. Removed $MODEL_COUNT model files."
else
    echo "$(date): Models directory not found at $MODELS_DIR"
fi
