#!/bin/bash

# This script displays information about how to access the neural network server

echo "Neural Network Server Access Information"
echo "========================================"
echo

# Get the machine's IP addresses
echo "Your machine's IP addresses:"
ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}'
echo

echo "You can access the neural network application using any of these URLs:"
echo "- http://localhost:8000/ (from this machine)"
ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print "- http://"$2":8000/ (from another device on the same network)"}'
echo

echo "If you're having trouble accessing the application, try:"
echo "1. Make sure the server is running (use ./fix_and_run.sh)"
echo "2. Check your firewall settings to ensure port 8000 is open"
echo "3. If accessing from another device, make sure both devices are on the same network"
echo
