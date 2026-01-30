# Agent Project Context

## What This Is
iMessage Agent that polls `~/Library/Messages/chat.db`, processes messages with Claude, and responds via AppleScript. Currently implementing a memory upgrade based on OpenClaw's architecture.

## Current Work: Memory Integration
We're adding RAG (vector search) and auto-compaction to prevent context rot. See `specs/memory-integration.spec.md` for full details.

### Feature Chunks (in order)
1. **Token Counting** ✓ DONE (branch: `feat/token-counting`)
2. **Vector Storage** ← NEXT (ChromaDB + Gemini embeddings)
3. Hybrid Search (replace `get_recent_history(5)`)
4. Memory Flush (pre-compaction save)
5. Auto-Compaction (smart summarization)
6. Integration & Polish

### Feature 1 Deliverables ✓ COMPLETE
- [x] Create `requirements.txt` with tiktoken
- [x] Create `token_counter.py` with `estimate_tokens()` and `check_limit()`
- [x] Modify `agent.py` `invoke_claude()` to log token counts
- [x] Add warning at 80% context window (80k tokens)

## Key Files
- `agent.py` - Main agent (1,630 lines)
- `memory/` - knowledge.json, reminders.json, conversation_log.jsonl
- `specs/memory-integration.spec.md` - Full implementation spec
- `config/secrets.example.json` - Template for API keys

## Reference Code
- OpenClaw memory system: `~/Code/openclaw/src/memory/`
- Planning doc: `~/Code/MEMORY_INTEGRATION.md`

## Git State
- `main`: Initial commit + CLAUDE.md
- `feat/token-counting`: Feature 1 complete (token counting)
