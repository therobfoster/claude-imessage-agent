# Agent Project Context

## What This Is
iMessage Agent that polls `~/Library/Messages/chat.db`, processes messages with Claude, and responds via AppleScript. Memory integration complete with RAG and auto-compaction.

## Memory Integration - COMPLETE

All 6 features implemented. See `specs/memory-integration.spec.md` for original spec.

### Feature Status
1. **Token Counting** ✓ (branch: `feat/token-counting`)
2. **Vector Storage** ✓ (branch: `feat/vector-storage`)
3. **Hybrid Search** ✓ (branch: `feat/hybrid-search`)
4. **Memory Flush** ✓ (branch: `feat/memory-flush`)
5. **Auto-Compaction** ✓ (branch: `feat/auto-compaction`)
6. **Integration** ✓ (branch: `feat/integration`)

### New Files
- `token_counter.py` - Token estimation using tiktoken
- `memory_manager.py` - ChromaDB + Gemini embeddings, hybrid search
- `memory_flush.py` - Pre-compaction memory save to MEMORY.md
- `compaction.py` - Auto-summarization when context exceeds limits

### CLI Options
```bash
python agent.py           # Run agent
python agent.py --stats   # Show memory system statistics
python agent.py --reindex # Rebuild vector store
python agent.py --help    # Usage info
```

### How It Works
1. **Token counting** - Logs usage on every Claude call, warns at 80%
2. **Semantic search** - Replaces naive "last 5" with hybrid vector+keyword
3. **Memory flush** - At 85k tokens, extracts key info to MEMORY.md
4. **Auto-compaction** - At 95k tokens, summarizes old messages

## Key Files
- `agent.py` - Main agent (~1,800 lines now)
- `memory/` - knowledge.json, reminders.json, conversation_log.jsonl, chroma/
- `specs/memory-integration.spec.md` - Implementation spec
- `config/secrets.json` - API keys (gitignored)

## Git State
All features on separate branches. Merge to main when ready.
