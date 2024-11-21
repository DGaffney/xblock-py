#!/bin/bash

echo "Current directory:"
pwd

# Check if PRONTO_MODE is set to true
if [ "$PRONTO_MODE" = "true" ]; then
    echo "PRONTO_MODE is enabled. Starting pronto_server.py..."
    python3 pronto_server.py
else
    echo "PRONTO_MODE is not enabled. Starting server.py..."
    python3 server.py
fi
