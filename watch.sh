#!/bin/bash
# Watch for changes to agent.py and restart automatically

AGENT_DIR="$HOME/Code/Agent"

echo "Watching for changes to agent.py..."
echo "Press Ctrl+C to stop"

# Start the agent initially
"$AGENT_DIR/start.sh"

# Watch for changes using fswatch
fswatch -o "$AGENT_DIR/agent.py" | while read f; do
    echo "Change detected, restarting agent..."
    "$AGENT_DIR/stop.sh"
    sleep 1
    "$AGENT_DIR/start.sh"
done
