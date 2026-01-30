# Memory Integration: OpenClaw → Agent

## Overview

This document outlines a plan to integrate OpenClaw's memory management system into this iMessage Agent. The goal is to prevent "context rot" - the gradual loss of important information as conversations grow longer.

## Current Memory System (Agent)

| Component | File | How It Works |
|-----------|------|--------------|
| Knowledge store | `memory/knowledge.json` | Simple JSON with facts, preferences, projects, people |
| Reminders | `memory/reminders.json` | Reminder list with due dates and recurring patterns |
| Static context | `memory/context.md` | Manually maintained context file |
| Conversation log | `memory/conversation_log.jsonl` | Appends every exchange as JSON lines |
| History loading | `get_recent_history(5)` | Loads last 5 conversations into prompt |

### Current Problems

1. **No intelligent retrieval**: Only loads last 5 exchanges regardless of relevance
2. **Growing log file**: `conversation_log.jsonl` grows indefinitely but isn't searchable
3. **No compaction**: When context gets too large, older information is simply lost
4. **No token awareness**: No estimation of prompt size before sending to Claude

## OpenClaw's Memory Architecture

OpenClaw solves these problems with four key systems:

### 1. RAG with Vector Storage
- Converts text into mathematical vectors (embeddings)
- Stores in sqlite-vec database
- Searches by *meaning* not just keywords
- Hybrid search: 70% vector similarity + 30% keyword matching
- Chunks text into ~400 token pieces with 80 token overlap

**Key files in OpenClaw:**
- `src/memory/manager.ts` - Main memory manager (75KB, very comprehensive)
- `src/memory/embeddings.ts` - Embedding generation
- `src/memory/hybrid.ts` - Hybrid search implementation
- `src/memory/sqlite-vec.ts` - Vector database integration

### 2. Tiered Memory
- **Session memory**: Current conversation context
- **Durable memory**: Files like `MEMORY.md` that persist across sessions
- **Long-term memory**: Vector database for semantic search

### 3. Pre-Compaction Memory Flush
When tokens approach a threshold, the system prompts the AI to save important information *before* compaction occurs.

**Key file:** `src/auto-reply/reply/memory-flush.ts`

Formula: `threshold = contextWindow - reserveTokensFloor - softThresholdTokens`

When `totalTokens >= threshold`, triggers a silent turn asking the AI to write durable memories.

### 4. Auto-Compaction with Adaptive Chunking
When context overflows, instead of just truncating:
- Summarizes old messages intelligently
- Uses adaptive chunk ratios based on message sizes
- Tracks tool failures and file operations in summaries
- Progressive fallback if summarization fails

**Key files:**
- `src/agents/compaction.ts` - Core compaction logic
- `src/agents/pi-extensions/compaction-safeguard.ts` - Safeguard extension

## Proposed Integration

### New Components to Add

```
Agent/
├── memory/
│   ├── knowledge.json      (existing)
│   ├── reminders.json      (existing)
│   ├── context.md          (existing)
│   ├── conversation_log.jsonl (existing)
│   ├── vectors.db          (NEW - sqlite/chromadb)
│   └── MEMORY.md           (NEW - durable AI-written memory)
├── memory_manager.py       (NEW - vector search, chunking)
├── token_counter.py        (NEW - estimate token usage)
├── compaction.py           (NEW - summarization logic)
└── agent.py                (MODIFIED - integrate new systems)
```

### Flow Change

**Before:**
```
[Load full context] → [Load last 5 exchanges] → [Send to Claude]
```

**After:**
```
[Estimate tokens]
→ [Search vector DB for relevant memories]
→ [Check if near context limit]
→ [If yes: trigger memory flush]
→ [If over limit: run compaction]
→ [Send to Claude with relevant context only]
```

### Python Libraries Needed

| Library | Purpose | Notes |
|---------|---------|-------|
| `chromadb` | Vector database | Easier setup than sqlite-vec |
| `tiktoken` | Token counting | OpenAI's tokenizer, works for Claude estimates |
| `sentence-transformers` | Local embeddings | Or use Gemini/OpenAI API |

## Questions to Resolve Before Implementation

### 1. Embedding Provider
You already have Gemini configured. Options:
- **Gemini embeddings** - Already have API key, good quality
- **OpenAI embeddings** - Requires separate API key, industry standard
- **Local (sentence-transformers)** - No API costs, runs on your machine

**Recommendation:** Start with Gemini since you already have it configured.

### 2. Scope of Integration
Options:
- **Full integration** - RAG + compaction + memory flush (most complex)
- **RAG only** - Just add vector search for relevant memories
- **Compaction only** - Just add summarization when context is full
- **Incremental** - Start with one, add others over time

**Recommendation:** Start with RAG (vector search), then add compaction.

### 3. Database Choice
- **ChromaDB** - Pure Python, easy setup, good for prototyping
- **sqlite-vec** - What OpenClaw uses, more setup but lighter weight
- **FAISS** - Facebook's library, very fast but more complex

**Recommendation:** ChromaDB for simplicity.

### 4. Migration Strategy
- **Replace** - Swap out current system entirely
- **Parallel** - Run both systems, compare results
- **Gradual** - Keep existing, layer new features on top

**Recommendation:** Gradual - keep existing `knowledge.json` and `conversation_log.jsonl`, add vector search as an enhancement.

## Reference: Key OpenClaw Code Sections

### Chunking Configuration (from manager.ts)
```typescript
const CHUNK_SIZE = 400;      // tokens per chunk
const CHUNK_OVERLAP = 80;    // overlap between chunks
```

### Hybrid Search Weights (from hybrid.ts)
```typescript
const VECTOR_WEIGHT = 0.7;   // 70% semantic similarity
const KEYWORD_WEIGHT = 0.3;  // 30% keyword matching
```

### Compaction Thresholds (from compaction.ts)
```typescript
const BASE_CHUNK_RATIO = 0.4;    // 40% of context for chunks
const MIN_CHUNK_RATIO = 0.15;    // minimum 15%
const SAFETY_MARGIN = 0.8;       // 20% buffer for token estimation
```

## Next Steps

1. Answer the questions above
2. Set up ChromaDB and test embedding generation
3. Create `memory_manager.py` with chunking and search
4. Modify `agent.py` to use semantic search instead of just last 5
5. Add token counting and compaction triggers
6. Test thoroughly before deploying

---

*Document created: January 30, 2026*
*OpenClaw repository: https://github.com/openclaw/openclaw*
