# Agent Features Manifest

This document describes all current capabilities. Consult before making modifications.

## Core Architecture

**Main file:** `agent.py` (~1,800 lines)
**Entry point:** `run_agent()` - Main polling loop
**Message handling:** `process_message()` → `invoke_claude()` → `process_actions()`

## Memory System

### Token Counting (`token_counter.py`)
- `estimate_tokens(text)` - Count tokens using tiktoken
- `check_limit(tokens)` - Check against context window (100k)
- Warns at 80%, critical at 95%

### Vector Storage (`memory_manager.py`)
- ChromaDB at `memory/chroma/`
- Gemini embeddings (`text-embedding-004`)
- `chunk_text()` - 400 tokens, 80 overlap
- `index_conversation()` / `index_file()`
- `search_hybrid()` - 70% vector + 30% keyword

### Memory Flush (`memory_flush.py`)
- Triggers at ~85k tokens
- Extracts key info to `memory/MEMORY.md`
- `should_flush()` / `run_memory_flush()`

### Auto-Compaction (`compaction.py`)
- Triggers at ~95k tokens
- Summarizes old messages via Claude
- `should_compact()` / `run_compaction()`

## Action Tags (XML in responses)

### Always Available
- `<REMEMBER>fact</REMEMBER>` - Save to knowledge.json
- `<REMINDER due="YYYY-MM-DD HH:MM" recurring="pattern">text</REMINDER>`
- `<COMPLETE>id</COMPLETE>` - Mark reminder done
- `<RUN_COMMAND>cmd</RUN_COMMAND>` - Execute shell command
- `<CREATE_FILE path="name">content</CREATE_FILE>`

### Build Tools (lazy-loaded)
- `<BUILD>description</BUILD>` - Multi-file projects
- `<MODIFY_SELF file="agent.py">changes</MODIFY_SELF>` - Self-modification

### Browser Tools (lazy-loaded)
- `<BROWSER action="read|screenshot|click|type">target</BROWSER>`

### Search Tools (lazy-loaded)
- `<WEB_SEARCH query="terms">context</WEB_SEARCH>` - DuckDuckGo
- `<TIKTOK_SEARCH query="terms" max="20">context</TIKTOK_SEARCH>`
- `<TIKTOK_TRENDS region="US" count="20"></TIKTOK_TRENDS>`

### Image Tools (lazy-loaded)
- `<IMAGE_ANALYZE prompt="describe">which</IMAGE_ANALYZE>`
- `<IMAGE_EDIT prompt="instructions">which</IMAGE_EDIT>`
- `<IMAGE_GENERATE prompt="description" aspect="1:1"></IMAGE_GENERATE>`

## Data Files

### Memory Directory (`memory/`)
- `knowledge.json` - Permanent facts
- `reminders.json` - Active reminders
- `conversation_log.jsonl` - All conversations
- `context.md` - User-provided context
- `MEMORY.md` - AI-extracted important info
- `chroma/` - Vector database
- `.content_hashes.json` - Change detection cache

### Config Directory (`config/`)
- `personality.md` - System prompt
- `secrets.json` - API keys (Gemini)
- `FEATURES.md` - This file

## CLI Options

```bash
python agent.py           # Run agent (with single-instance lock)
python agent.py --stats   # Show memory system statistics
python agent.py --reindex # Rebuild vector store from scratch
python agent.py --help    # Usage info
```

## Key Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `invoke_claude()` | agent.py:1165 | Build prompt, call Claude CLI |
| `process_actions()` | agent.py:1230 | Parse and execute XML tags |
| `get_tools_prompt()` | agent.py:1098 | Lazy-load tool descriptions |
| `save_conversation()` | agent.py:1047 | Log + index to vector store |
| `acquire_lock()` | agent.py:131 | Single-instance enforcement |

## Recent Additions (2026-01-30)

1. Memory integration (Features 1-6)
2. Token optimization (lazy tool loading, 76% reduction)
3. Single-instance lock (PID file)
4. Log rotation (10MB, 5 backups)
5. Debug mode (`AGENT_DEBUG=1`)

## Dependencies

```
tiktoken>=0.5.0        # Token counting
chromadb>=0.4.0        # Vector database
google-generativeai    # Gemini embeddings
playwright             # Browser automation (optional)
```

## Modification Guidelines

When modifying the agent:
1. Check this manifest for existing features
2. Avoid duplicating functionality
3. Maintain lazy-loading for token efficiency
4. Add new features to appropriate sections
5. Update this manifest after changes
6. Test with `--stats` to verify memory system
