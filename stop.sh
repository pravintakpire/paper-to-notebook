#!/bin/bash

if [ -f server.pid ]; then
    PID=$(cat server.pid)
    if ps -p $PID > /dev/null; then
        echo "Stopping server with PID $PID..."
        kill $PID
        rm server.pid
        echo "Server stopped."
    else
        echo "Process $PID not found. Cleaning up pid file."
        rm server.pid
    fi
else
    echo "No server.pid file found."
fi
