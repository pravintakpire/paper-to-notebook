#!/bin/bash

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY is not set. You will need to provide it in the web interface."
fi

# Stop any existing instance
if [ -f server.pid ]; then
    echo "Stopping existing server..."
    ./stop.sh
fi

echo "Starting Paper to Notebook server..."
source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app:app --reload --port 8000 > server.log 2>&1 &
echo $! > server.pid

echo "Server started with PID $(cat server.pid)"
echo "Logs are being written to server.log"
echo "Open http://localhost:8000 in your browser"
