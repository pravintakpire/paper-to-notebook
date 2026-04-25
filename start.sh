#!/bin/bash

# Optional: set provider keys before starting
if [ -z "$OPENAI_API_KEY" ] && [ -z "$GEMINI_API_KEY" ]; then
    echo "Info: No API keys set in env. You can provide them in the web interface."
fi

# Stop any existing instance
if [ -f server.pid ]; then
    echo "Stopping existing server..."
    ./stop.sh
fi

echo "Starting Paper to Code server..."
source venv/bin/activate
pip install -r requirements.txt -q
python -m uvicorn app:app --reload --port 8000 > server.log 2>&1 &
echo $! > server.pid

echo "Server started with PID $(cat server.pid)"
echo "Logs: server.log"
echo "Open http://localhost:8000 in your browser"
