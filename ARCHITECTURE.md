# Claude iMessage Agent - Technical Architecture

Comprehensive technical documentation for the Claude iMessage Agent.

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Core Components](#core-components)
4. [Memory System](#memory-system)
5. [Message Processing Pipeline](#message-processing-pipeline)
6. [Action Tag System](#action-tag-system)
7. [Self-Modification](#self-modification)
8. [iMessage Integration](#imessage-integration)
9. [Configuration System](#configuration-system)
10. [Security Model](#security-model)
11. [File Structure](#file-structure)
12. [Dependencies](#dependencies)

---

## System Overview

The Claude iMessage Agent is a macOS-native AI assistant that monitors iMessage conversations and responds using Claude. It features semantic memory with RAG (Retrieval-Augmented Generation), self-modification capabilities, and automatic context management.

### Key Characteristics

| Property | Value |
|----------|-------|
| Platform | macOS only (requires Messages.app) |
| LLM Backend | Claude Code CLI |
| Embedding Model | Gemini text-embedding-004 |
| Vector Database | ChromaDB (local, persistent) |
| Context Window | 100,000 tokens |
| Polling Interval | Configurable (default: 10s) |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        macOS Environment                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐ │
│  │   Messages   │     │   chat.db    │     │    Terminal/Shell    │ │
│  │     App      │────▶│  (SQLite)    │◀────│    (Full Disk Acc)   │ │
│  └──────────────┘     └──────────────┘     └──────────────────────┘ │
│         ▲                    │                        │              │
│         │                    ▼                        ▼              │
│         │            ┌──────────────────────────────────────┐       │
│         │            │           agent.py                    │       │
│         │            │  ┌────────────────────────────────┐  │       │
│         │            │  │        Main Poll Loop          │  │       │
│         │            │  │  • Check for new messages      │  │       │
│         │            │  │  • Filter by allowed_senders   │  │       │
│         │            │  │  • Check reminders             │  │       │
│         │            │  └────────────────────────────────┘  │       │
│         │            │                 │                     │       │
│         │            │                 ▼                     │       │
│         │            │  ┌────────────────────────────────┐  │       │
│         │            │  │     Message Processing         │  │       │
│         │            │  │  • Load context (RAG)          │  │       │
│         │            │  │  • Build prompt                │  │       │
│         │            │  │  • Lazy-load tools             │  │       │
│         │            │  └────────────────────────────────┘  │       │
│         │            │                 │                     │       │
│         │            │                 ▼                     │       │
│         │            │  ┌────────────────────────────────┐  │       │
│  AppleScript         │  │      Claude Code CLI           │  │       │
│  send_imessage()     │  │  claude -p "prompt" --flags    │  │       │
│         │            │  └────────────────────────────────┘  │       │
│         │            │                 │                     │       │
│         │            │                 ▼                     │       │
│         │            │  ┌────────────────────────────────┐  │       │
│         │            │  │     Response Processing        │  │       │
│         │            │  │  • Parse action tags (XML)     │  │       │
│         │            │  │  • Execute actions             │  │       │
│         │            │  │  • Save to conversation log    │  │       │
│         │            │  │  • Index to vector store       │  │       │
│         │            │  └────────────────────────────────┘  │       │
│         │            │                 │                     │       │
│         │◀───────────┼─────────────────┘                     │       │
│         │            └──────────────────────────────────────┘       │
│         │                              │                             │
│         │                              ▼                             │
│         │            ┌──────────────────────────────────────┐       │
│         │            │         Memory Subsystem              │       │
│         │            │  ┌─────────┐  ┌─────────┐  ┌───────┐ │       │
│         │            │  │ChromaDB │  │MEMORY.md│  │*.json │ │       │
│         │            │  │(vectors)│  │(flushed)│  │(state)│ │       │
│         │            │  └─────────┘  └─────────┘  └───────┘ │       │
│         │            └──────────────────────────────────────┘       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### agent.py (~1,800 lines)

The main agent file containing all core logic.

| Function | Line | Purpose |
|----------|------|---------|
| `run_agent()` | ~1700 | Main entry point and polling loop |
| `process_message()` | ~1450 | Handle incoming message end-to-end |
| `invoke_claude()` | ~1165 | Build prompt and call Claude CLI |
| `process_actions()` | ~1230 | Parse XML tags and execute actions |
| `get_tools_prompt()` | ~1098 | Lazy-load tool descriptions based on keywords |
| `send_imessage()` | ~350 | Send reply via AppleScript |
| `acquire_lock()` | ~159 | Single-instance enforcement via PID file |
| `restart_self()` | ~195 | In-place restart using os.execv |

### token_counter.py

Token estimation using tiktoken (cl100k_base encoding).

```python
CONTEXT_WINDOW = 100_000    # Claude's max context
SOFT_THRESHOLD = 0.8        # 80% = warning
HARD_THRESHOLD = 0.95       # 95% = must compact

# Key functions
estimate_tokens(text) -> int
check_limit(tokens) -> dict  # {ok, warning, critical, usage_pct}
format_token_status(tokens) -> str
```

### memory_manager.py

Vector storage and semantic search using ChromaDB + Gemini embeddings.

```python
CHUNK_SIZE = 400            # tokens per chunk
CHUNK_OVERLAP = 80          # overlap between chunks
EMBEDDING_MODEL = "models/text-embedding-004"

# Key classes
GeminiEmbedder        # Generate embeddings via Gemini API
MemoryManager         # Manages ChromaDB collection

# Key functions
search_hybrid(query, top_k=5, vector_weight=0.7) -> list[SearchResult]
get_relevant_context(query, top_k=5) -> str
```

### memory_flush.py

Pre-compaction extraction of important information.

```python
FLUSH_THRESHOLD = ~85,000 tokens

# Key functions
should_flush(current_tokens) -> bool
run_memory_flush(conversation_context) -> Optional[str]
```

### compaction.py

Auto-summarization when context approaches limits.

```python
BASE_CHUNK_RATIO = 0.4      # Compact 40% of oldest messages
HARD_LIMIT = 95,000 tokens  # Trigger threshold

# Key functions
should_compact(current_tokens) -> bool
run_compaction(force=False) -> CompactionResult
```

---

## Memory System

The agent uses a multi-layered memory system for context management.

### Layer 1: Conversation Log

- **File:** `memory/conversation_log.jsonl`
- **Format:** JSON Lines (one entry per exchange)
- **Contents:** timestamp, sender, user_message, agent_response

```json
{"timestamp": "2026-01-30T09:15:00", "sender": "+1234567890", "user_message": "Hello", "agent_response": "Hi there!"}
```

### Layer 2: Vector Store (Semantic Memory)

- **Storage:** ChromaDB at `memory/chroma/`
- **Embeddings:** Gemini text-embedding-004 (768 dimensions)
- **Chunking:** 400 tokens with 80-token overlap
- **Search:** Hybrid (70% vector + 30% keyword)

**Indexing flow:**
1. New conversation entry received
2. Text chunked into 400-token segments
3. Each chunk embedded via Gemini API
4. Chunks stored in ChromaDB with metadata

**Search flow:**
1. Query embedded via Gemini (retrieval_query task type)
2. Vector similarity search returns top candidates
3. Keyword search supplements with exact matches
4. Results merged with weighted scoring

### Layer 3: Knowledge Base

- **File:** `memory/knowledge.json`
- **Purpose:** Permanent facts stored via `<REMEMBER>` tag
- **Format:** JSON array of fact objects

```json
{
  "facts": [
    {"fact": "User prefers dark mode", "timestamp": "2026-01-30T08:00:00"}
  ]
}
```

### Layer 4: Memory Flush (MEMORY.md)

- **File:** `memory/MEMORY.md`
- **Trigger:** ~85,000 tokens
- **Purpose:** AI-extracted important information before compaction
- **Process:** Claude summarizes key decisions, preferences, TODOs

### Layer 5: Compaction

- **Trigger:** ~95,000 tokens
- **Process:**
  1. Split messages: 40% oldest → compact, 60% recent → keep
  2. Summarize old messages via Claude
  3. Replace compacted messages with summary entry
  4. Log compaction to `memory/compaction_log.jsonl`

### Token Thresholds

```
0 ─────────────────────────────────────────────────────── 100k
|                                                          |
|   OK (0-80%)    |  WARNING (80-85%)  |  CRITICAL (95%+) |
|                 |                     |                  |
|                 ▼                     ▼                  |
|           Memory Flush          Compaction              |
|            (~85k)                (~95k)                 |
```

---

## Message Processing Pipeline

```
┌─────────────────┐
│  New iMessage   │
│   Detected      │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Filter: sender  │──No──▶ Ignore
│ in allowed_list │
└────────┬────────┘
         │ Yes
         ▼
┌─────────────────┐
│ Check special   │──Yes──▶ Handle (restart, status)
│ commands        │
└────────┬────────┘
         │ No
         ▼
┌─────────────────┐
│ Check pending   │──Yes──▶ Process approval
│ approvals       │
└────────┬────────┘
         │ No
         ▼
┌─────────────────┐
│ Load relevant   │
│ context (RAG)   │
│ • Vector search │
│ • Knowledge.json│
│ • Reminders     │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Build prompt    │
│ • Personality   │
│ • Context       │
│ • Tools (lazy)  │
│ • Message       │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Invoke Claude   │
│ CLI with prompt │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Parse response  │
│ Extract XML tags│
└────────┬────────┘
         ▼
┌─────────────────┐
│ Execute actions │
│ • REMEMBER      │
│ • REMINDER      │
│ • RUN_COMMAND   │
│ • BUILD         │
│ • MODIFY_SELF   │
│ • etc.          │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Send response   │
│ via AppleScript │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Save & index    │
│ conversation    │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Check token     │
│ limits          │
│ • Flush?        │
│ • Compact?      │
└─────────────────┘
```

---

## Action Tag System

The agent uses XML tags in Claude's responses to trigger actions.

### Always Available Tags

| Tag | Purpose | Example |
|-----|---------|---------|
| `<REMEMBER>` | Save permanent fact | `<REMEMBER>User's birthday is March 15</REMEMBER>` |
| `<REMINDER>` | Set reminder | `<REMINDER due="2026-02-01 14:00">Call dentist</REMINDER>` |
| `<COMPLETE>` | Mark reminder done | `<COMPLETE>reminder-123</COMPLETE>` |
| `<RUN_COMMAND>` | Execute shell command | `<RUN_COMMAND>ls -la</RUN_COMMAND>` |
| `<CREATE_FILE>` | Write file | `<CREATE_FILE path="hello.py">print("hi")</CREATE_FILE>` |

### Lazy-Loaded Tags (Token Optimization)

These tool descriptions are only included when keywords suggest they're needed:

**Build Tools** (keywords: build, create, make, project, generate, develop)
- `<BUILD>` - Create multi-file projects
- `<MODIFY_SELF>` - Self-modification

**Browser Tools** (keywords: browse, website, screenshot, webpage, click, navigate)
- `<BROWSER action="read|screenshot|click|type">` - Browser automation

**Search Tools** (keywords: search, look up, find, google, what is, who is)
- `<WEB_SEARCH>` - DuckDuckGo search
- `<TIKTOK_SEARCH>` - TikTok content search
- `<TIKTOK_TRENDS>` - TikTok trending topics

**Image Tools** (keywords: image, picture, photo, analyze, edit, generate, draw)
- `<IMAGE_ANALYZE>` - Describe/analyze images
- `<IMAGE_EDIT>` - Edit images
- `<IMAGE_GENERATE>` - Generate images

### Token Savings

Without lazy loading: ~2,500 tokens per request
With lazy loading: ~150-800 tokens (varies by detected need)
**Savings: 68-94%**

---

## Self-Modification

The agent can modify its own code via the `<MODIFY_SELF>` tag.

### Flow

```
User: "Add a new feature to..."
            │
            ▼
┌─────────────────────┐
│ Claude generates    │
│ <MODIFY_SELF>       │
│ description         │
│ </MODIFY_SELF>      │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Check permissions   │
│ auto_approve_self_  │
│ modify?             │
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     │           │
    Yes          No
     │           │
     ▼           ▼
┌─────────┐ ┌─────────────┐
│ Execute │ │ Queue for   │
│ directly│ │ approval    │
└────┬────┘ └──────┬──────┘
     │             │
     │      User says "yes"
     │             │
     └──────┬──────┘
            ▼
┌─────────────────────┐
│ run_approved_self_  │
│ modify()            │
│                     │
│ • Load FEATURES.md  │
│ • Call Claude Code  │
│ • --dangerously-    │
│   skip-permissions  │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Send confirmation   │
│ message to user     │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ restart_self()      │
│                     │
│ os.execv() replaces │
│ process in-place    │
│ (same PID, new code)│
└─────────────────────┘
```

### Auto-Restart Mechanism

After self-modification, the agent automatically restarts using `os.execv()`:

```python
def restart_self():
    release_lock()          # Release PID lock
    logging.shutdown()      # Flush logs
    os.execv(sys.executable, [sys.executable] + sys.argv)
```

This replaces the current process in-place:
- Same PID
- Same terminal
- Fresh Python interpreter
- Modified code loaded

---

## iMessage Integration

### Reading Messages

The agent reads from the macOS Messages SQLite database:

```
~/Library/Messages/chat.db
```

**Query pattern:**
```sql
SELECT ROWID, text, datetime(date/1000000000 + 978307200, 'unixepoch')
FROM message
WHERE ROWID > ? AND is_from_me = 0
ORDER BY ROWID ASC
```

**Requirements:**
- Full Disk Access permission for Terminal/shell
- Messages app signed into iCloud

### Sending Messages

Messages are sent via AppleScript through the Messages app:

```applescript
tell application "Messages"
    set targetService to 1st account whose service type = iMessage
    set targetBuddy to participant "{recipient}" of targetService
    send "{message}" to targetBuddy
end tell
```

**Key points:**
- Uses whatever iCloud account is signed into Messages
- No configuration needed - if Messages works manually, agent works
- Recipient can be phone number (+1234567890) or iCloud email

---

## Configuration System

### config/user_config.json

```json
{
  "agent_name": "Jarvis",
  "allowed_senders": ["+1234567890", "user@icloud.com"],
  "poll_interval": 10,
  "permissions": {
    "auto_approve_commands": true,
    "auto_approve_builds": false,
    "auto_approve_self_modify": false,
    "auto_approve_file_writes": false,
    "safe_commands": ["ls", "pwd", "date", "whoami", "echo", "cat", "head", "tail", "wc", "grep", "find"]
  }
}
```

### config/secrets.json

```json
{
  "gemini_api_key": "your-api-key"
}
```

### config/personality.md

System prompt that defines the agent's personality and behavior.

### Permission Levels

| Permission | Default | Description |
|------------|---------|-------------|
| `auto_approve_commands` | false | Shell commands |
| `auto_approve_builds` | false | Multi-file projects |
| `auto_approve_self_modify` | false | Code self-modification |
| `auto_approve_file_writes` | false | Files outside workspace |

---

## Security Model

### Access Control

1. **Sender Whitelist** - Only responds to `allowed_senders`
2. **Permission System** - Dangerous actions require approval by default
3. **Command Filtering** - Blocks dangerous patterns (rm -rf, sudo, etc.)
4. **Workspace Isolation** - File writes default to `workspace/` directory

### Dangerous Command Patterns (Blocked)

```python
DANGEROUS_PATTERNS = [
    r"rm\s+-rf\s+[/~]",
    r"sudo\s+rm",
    r">\s*/dev/",
    r"mkfs",
    r"dd\s+if=",
    r"chmod\s+-R\s+777\s+/",
    r"curl.*\|.*sh",
    r"wget.*\|.*sh",
    r"eval",
    r"exec"
]
```

### Single Instance Lock

PID file mechanism prevents duplicate agents:

```python
PID_FILE = AGENT_DIR / ".agent.pid"

def acquire_lock():
    if PID_FILE.exists():
        old_pid = int(PID_FILE.read_text())
        os.kill(old_pid, 0)  # Check if running
        return False  # Another instance running
    PID_FILE.write_text(str(os.getpid()))
    return True
```

---

## File Structure

```
Agent/
├── agent.py                 # Main agent (~1,800 lines)
├── token_counter.py         # Token estimation
├── memory_manager.py        # Vector storage + search
├── memory_flush.py          # Pre-compaction save
├── compaction.py            # Auto-summarization
├── setup.py                 # Interactive setup script
├── requirements.txt         # Python dependencies
├── README.md                # User documentation
├── ARCHITECTURE.md          # This file
├── CLAUDE.md                # Development context
│
├── config/
│   ├── user_config.json     # User settings (gitignored)
│   ├── secrets.json         # API keys (gitignored)
│   ├── personality.md       # System prompt
│   └── FEATURES.md          # Capabilities manifest
│
├── memory/
│   ├── knowledge.json       # Permanent facts
│   ├── reminders.json       # Active reminders
│   ├── conversation_log.jsonl  # Chat history
│   ├── compaction_log.jsonl # Compaction history
│   ├── MEMORY.md            # AI-extracted info
│   ├── .content_hashes.json # Change detection
│   └── chroma/              # Vector database
│
├── workspace/               # Agent-created files
├── tasks/                   # Scheduled task definitions
├── logs/
│   └── agent.log            # Rotating log (10MB x 5)
│
└── .agent.pid               # Single-instance lock
```

---

## Dependencies

### Required

| Package | Version | Purpose |
|---------|---------|---------|
| tiktoken | >=0.5.0 | Token counting (cl100k_base) |
| chromadb | >=0.4.0 | Vector database |
| google-generativeai | >=0.3.0 | Gemini embeddings |

### Optional

| Package | Purpose |
|---------|---------|
| playwright | Browser automation |

### External

| Tool | Purpose |
|------|---------|
| Claude Code CLI | LLM backend (`npm install -g @anthropic-ai/claude-code`) |
| Node.js | Required for Claude Code CLI |

---

## CLI Reference

```bash
# Run the agent
python3 agent.py

# Debug mode (verbose logging)
AGENT_DEBUG=1 python3 agent.py

# Show memory statistics
python3 agent.py --stats

# Rebuild vector store
python3 agent.py --reindex

# Show help
python3 agent.py --help
```

---

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `AGENT_DEBUG` | Enable debug logging | (unset) |

---

## Logging

- **File:** `logs/agent.log`
- **Rotation:** 10MB max, 5 backups
- **Levels:** INFO (normal), DEBUG (with AGENT_DEBUG=1)

**Log format:**
```
2026-01-30 09:22:36,140 - INFO - Claude iMessage Agent starting...
```
