#!/bin/bash

# Test the new endpoints that return successful and unsuccessful examples

API_URL="http://localhost:8000/api"
NETWORK_ID=""

# Get command line argument (network ID)
if [ -n "$1" ]; then
  NETWORK_ID="$1"
else
  # List available networks and get the first one
  NETWORKS_RESPONSE=$(curl -s "$API_URL/networks")
  NETWORK_ID=$(echo "$NETWORKS_RESPONSE" | jq -r '.networks[0].network_id')
  
  if [ "$NETWORK_ID" == "null" ]; then
    echo "Error: No networks available"
    echo "Please create and train a network first using test_api.sh"
    exit 1
  fi
  
  echo "Using network ID: $NETWORK_ID"
fi

# Test successful example endpoint
echo "Testing successful example endpoint..."
curl -s "$API_URL/networks/$NETWORK_ID/successful_example" | jq '{network_id, example_index, predicted_digit, actual_digit}'
echo ""

# Test unsuccessful example endpoint
echo "Testing unsuccessful example endpoint..."
curl -s "$API_URL/networks/$NETWORK_ID/unsuccessful_example" | jq '{network_id, example_index, predicted_digit, actual_digit}'
echo ""

echo "Done! To see the full output including images and weights, use the web interface at http://localhost:5000/"
