#!/bin/bash

# This script tests all neural network endpoints to verify they work correctly
# It will catch and report any errors

# Color codes for prettier output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

API_URL="http://localhost:8000/api"

echo -e "${BLUE}==== Neural Network API Full Test ====${NC}"
echo "Testing all endpoints including previously problematic ones"
echo

# Function to test an endpoint and report results
test_endpoint() {
    local endpoint=$1
    local method=$2
    local data=$3
    local description=$4
    
    echo -e "${YELLOW}Testing: ${description}${NC}"
    
    if [ "$method" == "POST" ]; then
        response=$(curl -s -X POST -H "Content-Type: application/json" -d "$data" "$endpoint")
    else
        response=$(curl -s "$endpoint")
    fi
    
    if [ -z "$response" ]; then
        echo -e "${RED}✘ Failed: No response${NC}"
        return 1
    fi
    
    error=$(echo "$response" | grep -o '"error":"[^"]*' | cut -d '"' -f 4)
    
    if [ -n "$error" ]; then
        echo -e "${RED}✘ Failed: $error${NC}"
        return 1
    else
        echo -e "${GREEN}✓ Success${NC}"
        echo "$response" | python3 -m json.tool | head -n 10
        echo
        return 0
    fi
}

# Step 1: Check API status
test_endpoint "$API_URL/status" "GET" "" "API Status Check"

# Step 2: Create a network
echo -e "${BLUE}=== Creating a neural network ===${NC}"
create_response=$(curl -s -X POST -H "Content-Type: application/json" -d '{"layer_sizes":[784, 30, 10]}' "$API_URL/networks")
NETWORK_ID=$(echo "$create_response" | python3 -c "import sys, json; print(json.load(sys.stdin)['network_id'])")

if [ -z "$NETWORK_ID" ]; then
    echo -e "${RED}Failed to create network. Exiting tests.${NC}"
    exit 1
fi

echo "Created network with ID: $NETWORK_ID"
echo

# Step 3: Train the network (just 1 epoch for testing)
echo -e "${BLUE}=== Training the network (1 epoch) ===${NC}"
train_response=$(curl -s -X POST -H "Content-Type: application/json" -d '{"epochs":1, "mini_batch_size":10, "learning_rate":3.0}' "$API_URL/networks/$NETWORK_ID/train")
JOB_ID=$(echo "$train_response" | python3 -c "import sys, json; print(json.load(sys.stdin)['job_id'])")

if [ -z "$JOB_ID" ]; then
    echo -e "${RED}Failed to start training job. Exiting tests.${NC}"
    exit 1
fi

echo "Training job started with ID: $JOB_ID"
echo -e "Waiting for training to complete..."

# Poll for training completion
while true; do
    status_response=$(curl -s "$API_URL/training/$JOB_ID")
    status=$(echo "$status_response" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))")
    
    echo -e "Training status: ${BLUE}$status${NC}"
    
    if [ "$status" = "completed" ] || [ "$status" = "failed" ]; then
        break
    fi
    
    sleep 3
done

if [ "$status" = "failed" ]; then
    error=$(echo "$status_response" | python3 -c "import sys, json; print(json.load(sys.stdin).get('error', 'Unknown error'))")
    echo -e "${RED}Training failed: $error${NC}"
    exit 1
fi

echo -e "${GREEN}Training completed successfully${NC}"
echo

# Step 4: Test previously problematic endpoints
echo -e "${BLUE}=== Testing visualization endpoints ===${NC}"

# 4.1 Test network visualization
test_endpoint "$API_URL/networks/$NETWORK_ID/visualize" "GET" "" "Network Visualization"

# 4.2 Test network statistics 
test_endpoint "$API_URL/networks/$NETWORK_ID/stats" "GET" "" "Network Statistics"

# 4.3 Test successful example endpoint
test_endpoint "$API_URL/networks/$NETWORK_ID/successful_example" "GET" "" "Successful Example"

# 4.4 Test unsuccessful example endpoint
test_endpoint "$API_URL/networks/$NETWORK_ID/unsuccessful_example" "GET" "" "Unsuccessful Example"

# 4.5 Test misclassified endpoint
test_endpoint "$API_URL/networks/$NETWORK_ID/misclassified?max_count=3" "GET" "" "Misclassified Examples"

# Step 5: Test basic prediction
echo -e "${BLUE}=== Testing prediction ===${NC}"
test_endpoint "$API_URL/networks/$NETWORK_ID/predict" "POST" '{"example_index": 42}' "Single Prediction"

# Step 6: Test batch prediction
test_endpoint "$API_URL/networks/$NETWORK_ID/predict_batch" "POST" '{"start_index": 100, "count": 2}' "Batch Prediction"

echo -e "${BLUE}==== All tests complete ====${NC}"
echo "If you see no error messages and all endpoints return success, the API is working correctly!"
echo
