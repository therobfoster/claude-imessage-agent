#!/bin/bash
# Stop the Claude iMessage Agent

AGENT_DIR="$HOME/Code/Agent"
PID_FILE="$AGENT_DIR/.agent.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    kill "$PID" 2>/dev/null
    rm "$PID_FILE"
    echo "Agent stopped (PID: $PID)"
else
    pkill -f "python.*agent.py" 2>/dev/null
    echo "Agent stopped"
fi
