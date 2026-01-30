# Claude iMessage Agent

A personal AI assistant that responds to your iMessages using Claude. Features semantic memory, self-modification capabilities, web search, image processing, and more.

## Features

- **iMessage Integration** - Responds to texts automatically
- **Semantic Memory** - Remembers context using vector search (ChromaDB + Gemini embeddings)
- **Smart Context** - Only includes relevant history, not just "last 5 messages"
- **Auto-Compaction** - Summarizes old conversations to stay within token limits
- **Self-Modification** - Can update its own code when asked
- **Web Search** - DuckDuckGo integration
- **Image Processing** - Analyze, edit, and generate images via Gemini
- **Browser Automation** - Screenshot pages, fill forms (optional, requires Playwright)
- **Reminders** - Set one-time or recurring reminders
- **Build Projects** - Create multi-file projects on request

## Requirements

- macOS (for iMessage access)
- Python 3.9+
- Full Disk Access permission (to read iMessage database)
- Gemini API key (free at https://makersuite.google.com/app/apikey)

## Installation

### 1. Clone the repository

Clone with your preferred agent name as the directory:

```bash
# Example: name it "jarvis", "friday", "max", etc.
git clone https://github.com/therobfoster/claude-imessage-agent.git jarvis
cd jarvis
```

Or use the default name:
```bash
git clone https://github.com/therobfoster/claude-imessage-agent.git
cd claude-imessage-agent
```

### 2. Run the setup script

```bash
python3 setup.py
```

This will:
- Install Python dependencies (tiktoken, chromadb, google-generativeai)
- Ask for your Gemini API key
- Configure allowed senders (phone numbers/emails that can message the agent)
- Set permission levels (what needs approval vs. auto-approve)
- Create necessary directories

### 3. Grant Full Disk Access

The agent needs to read `~/Library/Messages/chat.db`:

1. Open **System Preferences** > **Privacy & Security** > **Full Disk Access**
2. Click the **+** button
3. Add **Terminal** (or your terminal app: iTerm, VS Code, etc.)
4. Restart your terminal

### 4. Start the agent

```bash
python3 agent.py
```

Or run in debug mode for verbose logging:

```bash
AGENT_DEBUG=1 python3 agent.py
```

## Usage

Once running, send an iMessage from an allowed sender. The agent will:

1. Detect your message
2. Search semantic memory for relevant context
3. Generate a response using Claude
4. Send the reply via iMessage

### Example Commands

| Message | What Happens |
|---------|--------------|
| "Remember my wifi password is hunter2" | Saves to permanent memory |
| "Remind me to call mom tomorrow at 2pm" | Sets a reminder |
| "What's the weather in NYC?" | Searches the web |
| "Build me a Python script that..." | Creates files (needs approval by default) |
| "Update yourself to add a new feature" | Self-modifies (needs approval by default) |
| "What's trending on TikTok?" | Searches TikTok trends |

### Action Tags

The agent uses XML tags internally. You don't need to use these directly:

- `<REMEMBER>` - Save facts
- `<REMINDER>` - Set reminders
- `<WEB_SEARCH>` - Search the web
- `<BUILD>` - Create projects
- `<MODIFY_SELF>` - Update agent code
- `<IMAGE_ANALYZE>` / `<IMAGE_EDIT>` / `<IMAGE_GENERATE>` - Image processing

## Configuration

### config/user_config.json

```json
{
  "agent_name": "Jarvis",
  "allowed_senders": ["+1234567890", "you@icloud.com"],
  "poll_interval": 10,
  "permissions": {
    "auto_approve_commands": false,
    "auto_approve_builds": false,
    "auto_approve_self_modify": false,
    "auto_approve_file_writes": false
  }
}
```

### Permission Levels

| Permission | Default | Description |
|------------|---------|-------------|
| `auto_approve_commands` | false | Shell commands (ls, cat, etc.) |
| `auto_approve_builds` | false | Multi-file project creation |
| `auto_approve_self_modify` | false | Agent updating its own code |
| `auto_approve_file_writes` | false | Writing files outside workspace |

Set to `true` to skip approval prompts.

### config/secrets.json

```json
{
  "gemini_api_key": "your-api-key-here"
}
```

## CLI Options

```bash
python3 agent.py              # Run the agent
python3 agent.py --stats      # Show memory system statistics
python3 agent.py --reindex    # Rebuild vector store from scratch
python3 agent.py --help       # Show help
```

## Logs

Logs are saved to `logs/agent.log` with automatic rotation (10MB max, 5 backups).

View live logs:
```bash
tail -f logs/agent.log
```

## File Structure

```
Agent/
├── agent.py              # Main agent
├── token_counter.py      # Token estimation
├── memory_manager.py     # Vector storage + semantic search
├── memory_flush.py       # Pre-compaction memory save
├── compaction.py         # Auto-summarization
├── setup.py              # Interactive setup
├── config/
│   ├── user_config.json  # Your settings (gitignored)
│   ├── secrets.json      # API keys (gitignored)
│   ├── personality.md    # System prompt (customizable)
│   └── FEATURES.md       # Agent capabilities manifest
├── memory/
│   ├── knowledge.json    # Permanent facts
│   ├── reminders.json    # Active reminders
│   ├── conversation_log.jsonl  # Chat history
│   └── chroma/           # Vector database
├── logs/
│   └── agent.log
└── workspace/            # Files created by the agent
```

## Troubleshooting

### "Another agent instance is already running"

Only one instance can run at a time. Kill the existing one:
```bash
pkill -f "python.*agent.py"
rm -f .agent.pid
```

### Messages not being detected

1. Check Full Disk Access is granted
2. Verify sender is in `allowed_senders` list
3. Check `logs/agent.log` for errors

### Memory system errors

Rebuild the vector store:
```bash
python3 agent.py --reindex
```

## Security Notes

- Only responds to `allowed_senders` - unknown numbers are ignored
- API keys stored in `config/secrets.json` (gitignored)
- Dangerous commands require approval by default
- Self-modification requires approval by default

## License

MIT
