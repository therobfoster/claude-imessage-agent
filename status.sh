#!/bin/bash
# Check status of the Claude iMessage Agent

AGENT_DIR="$HOME/Code/Agent"
PID_FILE="$AGENT_DIR/.agent.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Agent is running (PID: $PID)"
    else
        echo "Agent is not running (stale PID file)"
    fi
else
    if pgrep -f "python.*agent.py" > /dev/null; then
        echo "Agent is running (no PID file)"
    else
        echo "Agent is not running"
    fi
fi
