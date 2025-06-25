#!/bin/bash

# This script tests if the neural network server is accessible
# from various addresses and provides troubleshooting tips

echo "Neural Network Server Accessibility Test"
echo "========================================"
echo

# Define the port
PORT=8000

# Get the current time for checking log freshness
TIMESTAMP=$(date +%s)

# Color codes for prettier output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Step 1: Check if server process is running
echo -e "${BLUE}Step 1: Checking if server is running...${NC}"
SERVER_PROCESS=$(ps aux | grep "python.*api_server.py" | grep -v grep)

if [ -z "$SERVER_PROCESS" ]; then
    echo -e "${RED}✘ Server process not found!${NC}"
    echo "  Fix: Start the server using ./fix_and_run.sh"
    echo 
else
    echo -e "${GREEN}✓ Server process is running${NC}"
    echo "  Process: $SERVER_PROCESS"
    echo
fi

# Step 2: Check localhost access
echo -e "${BLUE}Step 2: Testing localhost access...${NC}"
LOCALHOST_RESPONSE=$(curl -s --connect-timeout 5 http://localhost:$PORT/api/status)

if [ -z "$LOCALHOST_RESPONSE" ]; then
    echo -e "${RED}✘ Cannot connect to localhost!${NC}"
    echo "  Fix: Make sure the server is running and listening on port $PORT"
else
    echo -e "${GREEN}✓ Successfully connected to localhost${NC}"
    echo "  Response: $LOCALHOST_RESPONSE"
fi
echo

# Step 3: Check 127.0.0.1 access
echo -e "${BLUE}Step 3: Testing 127.0.0.1 access...${NC}"
LOOPBACK_RESPONSE=$(curl -s --connect-timeout 5 http://127.0.0.1:$PORT/api/status)

if [ -z "$LOOPBACK_RESPONSE" ]; then
    echo -e "${RED}✘ Cannot connect to 127.0.0.1!${NC}"
    echo "  Fix: Make sure the server is running and listening on port $PORT"
else
    echo -e "${GREEN}✓ Successfully connected to 127.0.0.1${NC}"
    echo "  Response: $LOOPBACK_RESPONSE"
fi
echo

# Step 4: Check all available IPs
echo -e "${BLUE}Step 4: Testing all available IP addresses...${NC}"

# Get all non-loopback IPv4 addresses
IPS=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}')

if [ -z "$IPS" ]; then
    echo -e "${YELLOW}! No non-loopback IP addresses found${NC}"
else
    for IP in $IPS; do
        echo -e "Testing IP: ${BLUE}$IP${NC}"
        IP_RESPONSE=$(curl -s --connect-timeout 5 http://$IP:$PORT/api/status)
        
        if [ -z "$IP_RESPONSE" ]; then
            echo -e "  ${RED}✘ Cannot connect to $IP!${NC}"
            echo "    Fix: Check firewall settings and that server is listening on 0.0.0.0"
        else
            echo -e "  ${GREEN}✓ Successfully connected to $IP${NC}"
            echo "    Response: $IP_RESPONSE"
            echo -e "    External device URL: ${GREEN}http://$IP:$PORT/${NC}"
        fi
    done
fi
echo

# Step 5: Check for port conflicts
echo -e "${BLUE}Step 5: Checking for port conflicts...${NC}"
PORT_USAGE=$(lsof -i:$PORT | grep LISTEN)

if [ -z "$PORT_USAGE" ]; then
    echo -e "${YELLOW}! No process appears to be actively listening on port $PORT${NC}"
    echo "  Note: This might mean the server isn't running correctly"
else
    echo -e "${GREEN}✓ Process listening on port $PORT:${NC}"
    echo "  $PORT_USAGE"
fi
echo

# Step 6: Test CORS headers
echo -e "${BLUE}Step 6: Testing CORS headers...${NC}"
CORS_HEADERS=$(curl -s -I --connect-timeout 5 http://localhost:$PORT/api/status | grep -i 'Access-Control')

if [ -z "$CORS_HEADERS" ]; then
    echo -e "${YELLOW}! No CORS headers found${NC}"
    echo "  Note: This might cause issues with browser access"
else
    echo -e "${GREEN}✓ CORS headers are present:${NC}"
    echo "  $CORS_HEADERS"
fi
echo

# Step 7: Check recent logs
echo -e "${BLUE}Step 7: Getting recent server logs (if available)...${NC}"
if [ -f "/var/log/syslog" ]; then
    LOG_FILE="/var/log/syslog"
elif [ -f "/var/log/system.log" ]; then
    LOG_FILE="/var/log/system.log"
else
    LOG_FILE=""
fi

if [ -n "$LOG_FILE" ]; then
    LOGS=$(sudo grep -a "api_server" "$LOG_FILE" | tail -n 10)
    if [ -n "$LOGS" ]; then
        echo -e "${GREEN}✓ Recent logs found:${NC}"
        echo "$LOGS"
    else
        echo -e "${YELLOW}! No recent logs found for api_server${NC}"
    fi
else
    echo -e "${YELLOW}! Could not find system log file${NC}"
fi
echo

# Summary
echo -e "${BLUE}Summary and Recommendations:${NC}"
echo "----------------------------------------"

if [ -z "$SERVER_PROCESS" ]; then
    echo -e "${RED}✘ Server is not running${NC} - Start with ./fix_and_run.sh"
elif [ -z "$LOCALHOST_RESPONSE" ] && [ -z "$LOOPBACK_RESPONSE" ]; then
    echo -e "${RED}✘ Server is running but not accessible${NC} - Check port configuration"
    echo "   The server might be configured to use a different port"
elif [ -z "$IP_RESPONSE" ]; then
    echo -e "${YELLOW}! Server is accessible locally but not from network${NC} - Check:"
    echo "   1. Firewall settings - ensure port $PORT is allowed"
    echo "   2. Server is bound to 0.0.0.0 and not just 127.0.0.1"
    echo "   3. Both devices are on the same network"
else
    echo -e "${GREEN}✓ Server appears to be properly configured and accessible${NC}"
    echo -e "   You can access the server at: ${GREEN}http://localhost:$PORT/${NC}"
    
    if [ -n "$IPS" ]; then
        echo "   From other devices, try:"
        for IP in $IPS; do
            echo -e "   - ${GREEN}http://$IP:$PORT/${NC}"
        done
    fi
fi
echo
