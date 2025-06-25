#!/bin/bash

# API testing script
API_URL="http://localhost:8000/api"
NETWORK_ID=""
JOB_ID=""

# Color codes for prettier output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Neural Network API Testing Script${NC}"
echo "=================================="
echo ""

# Check API status
echo -e "${BLUE}Checking API status...${NC}"
curl -s "$API_URL/status" | jq .
echo ""

# Create a new network
echo -e "${BLUE}Creating a new neural network...${NC}"
RESPONSE=$(curl -s -X POST "$API_URL/networks" \
    -H "Content-Type: application/json" \
    -d '{"layer_sizes": [784, 30, 10]}')

echo "$RESPONSE" | jq .

NETWORK_ID=$(echo "$RESPONSE" | jq -r '.network_id')
echo -e "${GREEN}Network created with ID: $NETWORK_ID${NC}"
echo ""

# Train the network
echo -e "${BLUE}Starting network training...${NC}"
RESPONSE=$(curl -s -X POST "$API_URL/networks/$NETWORK_ID/train" \
    -H "Content-Type: application/json" \
    -d '{"epochs": 1, "mini_batch_size": 10, "learning_rate": 3.0}')

echo "$RESPONSE" | jq .

JOB_ID=$(echo "$RESPONSE" | jq -r '.job_id')
echo -e "${GREEN}Training job started with ID: $JOB_ID${NC}"
echo ""

# Wait a bit for training to progress
echo -e "${BLUE}Waiting for training to progress...${NC}"
sleep 5

# Check training status
echo -e "${BLUE}Checking training status...${NC}"
curl -s "$API_URL/training/$JOB_ID" | jq .
echo ""

# Wait for training to complete (in a real script, would use a loop)
echo -e "${BLUE}Waiting for training to complete...${NC}"
echo -e "${RED}This may take some time depending on the number of epochs.${NC}"
echo -e "To view training progress, check: ${GREEN}$API_URL/training/$JOB_ID${NC}"
echo ""
echo -e "${BLUE}Press Enter when training has completed to continue...${NC}"
read -r

# Run a prediction
echo -e "${BLUE}Running prediction on a test example...${NC}"
curl -s -X POST "$API_URL/networks/$NETWORK_ID/predict" \
    -H "Content-Type: application/json" \
    -d '{"example_index": 42}' | jq .
echo ""

# Find misclassified examples
echo -e "${BLUE}Finding misclassified examples...${NC}"
curl -s "$API_URL/networks/$NETWORK_ID/misclassified?max_count=5&max_check=100" | jq .
echo ""

# Get network visualization
echo -e "${BLUE}Getting network visualization...${NC}"
echo "Visualization is a large base64 string, not showing it here."
echo -e "Visit ${GREEN}http://localhost:5000/${NC} to view visualizations in the web UI."
echo ""

# Get network stats
echo -e "${BLUE}Getting network statistics...${NC}"
curl -s "$API_URL/networks/$NETWORK_ID/stats" | jq .
echo ""

# Get a successful example
echo -e "${BLUE}Getting a successful example...${NC}"
echo "Response includes a large image and weights array, showing only partial output:"
curl -s "$API_URL/networks/$NETWORK_ID/successful_example" | jq '{network_id, example_index, predicted_digit, actual_digit}'
echo ""

# Get an unsuccessful example
echo -e "${BLUE}Getting an unsuccessful example...${NC}"
echo "Response includes a large image and weights array, showing only partial output:"
curl -s "$API_URL/networks/$NETWORK_ID/unsuccessful_example" | jq '{network_id, example_index, predicted_digit, actual_digit}'
echo ""

# List available networks
echo -e "${BLUE}Listing available networks...${NC}"
curl -s "$API_URL/networks" | jq .
echo ""

echo -e "${GREEN}API testing complete!${NC}"
echo "You can now visit http://localhost:5000/ to use the web interface."
