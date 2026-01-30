# Memory Integration Spec — Agent RAG + Compaction

Goal: Integrate OpenClaw-style memory management into the iMessage Agent to prevent context rot. Replace naive "last 5 exchanges" with semantic retrieval, add token awareness, and implement auto-compaction when approaching context limits.

Reference: OpenClaw at `~/Code/openclaw/src/memory/`

## Requirements
- Semantic search over conversation history (vector similarity)
- Hybrid retrieval: 70% vector + 30% keyword matching
- Token counting before sending prompts to Claude
- Pre-compaction memory flush to preserve important context
- Auto-compaction when exceeding token threshold
- Durable memory file (`MEMORY.md`) for AI-written persistent knowledge
- Chunking: ~400 tokens per chunk, 80 token overlap
- Use existing Gemini API for embeddings (already configured)
- ChromaDB for vector storage (simpler than sqlite-vec for Python)
- Gradual migration: keep existing files, layer new features on top

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     agent.py (modified)                     │
│  invoke_claude() now calls:                                 │
│    1. token_counter.estimate_tokens(prompt)                 │
│    2. memory_manager.search(query, top_k=5)                 │
│    3. compaction.check_and_compact() if over threshold      │
└─────────────────────────────────────────────────────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ token_counter.py│  │memory_manager.py│  │  compaction.py  │
│                 │  │                 │  │                 │
│ - tiktoken      │  │ - ChromaDB      │  │ - Summarize old │
│ - estimate size │  │ - Gemini embed  │  │ - Adaptive chunk│
│ - check limits  │  │ - Hybrid search │  │ - Memory flush  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │ memory/         │
                     │ ├─ chroma/      │ (vector DB)
                     │ ├─ MEMORY.md    │ (durable)
                     │ ├─ *.jsonl      │ (existing)
                     │ └─ *.json       │ (existing)
                     └─────────────────┘
```

## Feature Chunks

### Feature 1: Foundation & Token Counting
**Branch:** `feat/token-counting`

Add token estimation to understand prompt sizes before sending.

Files:
- `token_counter.py` (new)
- `requirements.txt` (new)
- `agent.py` (modify `invoke_claude`)

Deliverables:
- [ ] Create `requirements.txt` with dependencies
- [ ] Implement `TokenCounter` class using tiktoken
- [ ] Add `estimate_tokens(text) -> int` function
- [ ] Add `check_limit(tokens, max=100_000) -> bool` function
- [ ] Log token counts in `invoke_claude()` before sending
- [ ] Add warning when approaching 80% of context window

Constants:
```python
CONTEXT_WINDOW = 100_000      # Claude's context limit
SOFT_THRESHOLD = 0.8          # 80% = warning threshold
HARD_THRESHOLD = 0.95         # 95% = must compact
```

---

### Feature 2: Vector Storage & Embeddings
**Branch:** `feat/vector-storage`

Set up ChromaDB and Gemini embeddings for semantic search.

Files:
- `memory_manager.py` (new)
- `requirements.txt` (update)

Deliverables:
- [ ] Initialize ChromaDB persistent collection at `memory/chroma/`
- [ ] Implement `GeminiEmbedder` using existing API key
- [ ] Implement `chunk_text(text, chunk_size=400, overlap=80)`
- [ ] Implement `index_conversation(entry: dict)` to embed and store
- [ ] Implement `index_file(path: str)` for MEMORY.md and context.md
- [ ] Add hash-based change detection (skip unchanged content)
- [ ] Backfill existing `conversation_log.jsonl` on first run

Constants:
```python
CHUNK_SIZE = 400              # tokens per chunk
CHUNK_OVERLAP = 80            # overlap between chunks
EMBEDDING_MODEL = "models/text-embedding-004"  # Gemini
COLLECTION_NAME = "agent_memory"
```

---

### Feature 3: Hybrid Search
**Branch:** `feat/hybrid-search`

Replace `get_recent_history(5)` with semantic + keyword search.

Files:
- `memory_manager.py` (extend)
- `agent.py` (modify `invoke_claude`, `load_context`)

Deliverables:
- [ ] Implement `search_vector(query, top_k=10) -> List[Result]`
- [ ] Implement `search_keyword(query, top_k=10) -> List[Result]`
- [ ] Implement `search_hybrid(query, top_k=5)` with 70/30 weighting
- [ ] Create `get_relevant_context(user_message, top_k=5)` wrapper
- [ ] Replace `get_recent_history(5)` call in `invoke_claude()`
- [ ] Include source attribution (file:line or conversation timestamp)

Search result structure:
```python
@dataclass
class SearchResult:
    text: str
    score: float
    source: str           # "conversation" | "memory" | "context"
    timestamp: str        # ISO format
    start_line: int | None
    end_line: int | None
```

---

### Feature 4: Memory Flush (Pre-Compaction)
**Branch:** `feat/memory-flush`

Before compaction, give Claude a chance to save important information.

Files:
- `memory_flush.py` (new)
- `agent.py` (integrate flush trigger)

Deliverables:
- [ ] Implement `should_flush(current_tokens, context_window) -> bool`
- [ ] Implement `run_memory_flush(conversation_context) -> str`
- [ ] Create flush prompt that asks Claude to extract key information
- [ ] Write extracted info to `memory/MEMORY.md` with timestamps
- [ ] Track flush count per session (only flush once before compaction)
- [ ] Index new MEMORY.md content into vector store

Flush threshold:
```python
FLUSH_THRESHOLD = CONTEXT_WINDOW - RESERVE_TOKENS - SOFT_BUFFER
# Example: 100k - 10k - 5k = 85k tokens triggers flush
```

---

### Feature 5: Auto-Compaction
**Branch:** `feat/auto-compaction`

Summarize old messages when context exceeds limits.

Files:
- `compaction.py` (new)
- `agent.py` (integrate compaction trigger)

Deliverables:
- [ ] Implement `should_compact(current_tokens) -> bool`
- [ ] Implement `split_messages_by_tokens(messages, ratio=0.4)`
- [ ] Implement `summarize_chunk(messages) -> str` using Claude
- [ ] Implement `run_compaction(conversation_log) -> CompactionResult`
- [ ] Update `conversation_log.jsonl` with compacted entries
- [ ] Preserve: decisions, TODOs, constraints, open questions
- [ ] Progressive fallback if summarization fails

Compaction constants (from OpenClaw):
```python
BASE_CHUNK_RATIO = 0.4        # 40% of context for chunks
MIN_CHUNK_RATIO = 0.15        # minimum 15%
SAFETY_MARGIN = 1.2           # 20% buffer for token estimation
```

---

### Feature 6: Integration & Polish
**Branch:** `feat/integration`

Wire everything together and add observability.

Files:
- `agent.py` (final integration)
- All modules (refinements)

Deliverables:
- [ ] Full flow: token check → search → flush? → compact? → send
- [ ] Add `--reindex` CLI flag to rebuild vector store
- [ ] Add memory stats to `status` command output
- [ ] Graceful degradation if ChromaDB fails (fall back to last 5)
- [ ] Add logging throughout memory pipeline
- [ ] Performance: cache embeddings for repeated queries
- [ ] Write migration guide for existing users

---

## File Structure (Final)

```
Agent/
├── agent.py                 # Main agent (modified)
├── token_counter.py         # Token estimation
├── memory_manager.py        # Vector storage + search
├── memory_flush.py          # Pre-compaction save
├── compaction.py            # Auto-summarization
├── requirements.txt         # Dependencies
├── memory/
│   ├── knowledge.json       # (existing)
│   ├── reminders.json       # (existing)
│   ├── context.md           # (existing)
│   ├── conversation_log.jsonl # (existing, may be compacted)
│   ├── chroma/              # (new) Vector database
│   └── MEMORY.md            # (new) AI-written durable memory
├── config/
│   └── personality.md       # (existing)
└── specs/
    └── memory-integration.spec.md  # This file
```

## Dependencies

```
# requirements.txt
tiktoken>=0.5.0              # Token counting
chromadb>=0.4.0              # Vector database
google-generativeai>=0.3.0   # Gemini embeddings (may already have)
```

## Out of Scope (Initial)
- Session-based memory tiers (just use single collection)
- Embedding provider fallback (Gemini only for now)
- Batch embedding API (sequential is fine for conversation scale)
- File watching for live MEMORY.md updates
- sqlite-vec (ChromaDB is simpler for Python)

## Open Decisions
- Should compacted summaries be stored separately or replace original entries?
- How to handle attachments/images in conversation history?
- Should MEMORY.md be human-editable or AI-only?
- Keyword search implementation: simple substring or full FTS?

## Testing Strategy
- Unit tests for each module (token_counter, memory_manager, etc.)
- Integration test: send message → verify vector indexed
- Compaction test: simulate large context, verify summarization
- Regression: ensure existing functionality unchanged

---

*Spec created: 2026-01-30*
*Reference: ~/Code/MEMORY_INTEGRATION.md (planning doc)*
*Reference: ~/Code/openclaw/src/memory/ (implementation)*
