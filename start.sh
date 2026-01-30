#!/bin/bash
# Start the Claude iMessage Agent (kills any existing instances first)

AGENT_DIR="$HOME/Code/Agent"
PID_FILE="$AGENT_DIR/.agent.pid"
LOG_FILE="$AGENT_DIR/logs/agent.log"

# Kill ALL existing agent processes to prevent duplicates
pkill -f "python.*agent.py" 2>/dev/null
sleep 1

# Ensure log directory exists
mkdir -p "$AGENT_DIR/logs"

echo "Starting Claude iMessage Agent..."
nohup python3 "$AGENT_DIR/agent.py" >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "Agent started (PID: $(cat $PID_FILE))"
echo "Logs: $LOG_FILE"
