"""
Auto-compaction - summarize old messages when context exceeds limits.

Implements progressive summarization to maintain context quality while
staying within token limits.
"""
from __future__ import annotations

import json
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

from token_counter import (
    estimate_tokens,
    CONTEXT_WINDOW,
    HARD_LIMIT,
    get_counter
)

logger = logging.getLogger(__name__)

# Constants (from OpenClaw)
BASE_CHUNK_RATIO = 0.4        # 40% of context for chunks to summarize
MIN_CHUNK_RATIO = 0.15        # Minimum 15%
SAFETY_MARGIN = 1.2           # 20% buffer for token estimation

# Paths
AGENT_DIR = Path(__file__).parent.resolve()  # Directory where this file is located
MEMORY_DIR = AGENT_DIR / "memory"
CONVERSATION_LOG = MEMORY_DIR / "conversation_log.jsonl"
COMPACTION_LOG = MEMORY_DIR / "compaction_log.jsonl"


@dataclass
class CompactionResult:
    """Result of a compaction operation."""
    success: bool
    messages_compacted: int
    original_tokens: int
    summary_tokens: int
    summary: str
    error: Optional[str] = None


def should_compact(current_tokens: int) -> bool:
    """
    Check if compaction is needed.

    Args:
        current_tokens: Current token count.

    Returns:
        True if compaction should be triggered.
    """
    return current_tokens >= HARD_LIMIT


def load_conversations() -> list[dict]:
    """Load all conversations from the log file."""
    if not CONVERSATION_LOG.exists():
        return []

    entries = []
    with open(CONVERSATION_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def save_conversations(entries: list[dict]):
    """Save conversations back to the log file."""
    with open(CONVERSATION_LOG, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')


def split_messages_by_tokens(
    messages: list[dict],
    target_ratio: float = BASE_CHUNK_RATIO
) -> tuple[list[dict], list[dict]]:
    """
    Split messages into old (to compact) and recent (to keep).

    Args:
        messages: List of conversation entries.
        target_ratio: Target ratio of messages to compact.

    Returns:
        Tuple of (messages_to_compact, messages_to_keep).
    """
    if len(messages) <= 2:
        return [], messages

    # Calculate total tokens
    total_tokens = 0
    message_tokens = []
    for msg in messages:
        text = f"{msg.get('user_message', '')} {msg.get('agent_response', '')}"
        tokens = estimate_tokens(text)
        message_tokens.append(tokens)
        total_tokens += tokens

    # Find split point at target_ratio
    target_tokens = int(total_tokens * target_ratio)
    running_total = 0
    split_index = 0

    for i, tokens in enumerate(message_tokens):
        running_total += tokens
        if running_total >= target_tokens:
            split_index = i + 1
            break

    # Ensure we keep at least 2 recent messages
    split_index = min(split_index, len(messages) - 2)
    split_index = max(split_index, 1)  # Compact at least 1 message

    return messages[:split_index], messages[split_index:]


def format_messages_for_summary(messages: list[dict]) -> str:
    """Format messages for the summarization prompt."""
    lines = []
    for msg in messages:
        ts = msg.get('timestamp', '')[:10]  # Just date
        user = msg.get('user_message', '')
        agent = msg.get('agent_response', '')
        lines.append(f"[{ts}] User: {user}")
        lines.append(f"[{ts}] Agent: {agent}")
    return '\n'.join(lines)


def get_summarization_prompt(messages_text: str) -> str:
    """Generate the summarization prompt."""
    return f"""Summarize the following conversation history, preserving:
1. **Key Decisions** - Any important choices or conclusions
2. **Open Tasks/TODOs** - Anything that needs follow-up
3. **User Preferences** - Stated preferences or requirements
4. **Important Context** - Technical details, names, dates
5. **Constraints** - Any limitations mentioned

Be concise but complete. Format as a brief narrative summary.

## Conversation to Summarize
{messages_text}

## Summary
"""


def summarize_chunk(messages: list[dict]) -> Optional[str]:
    """
    Summarize a chunk of messages using Claude.

    Args:
        messages: Messages to summarize.

    Returns:
        Summary text, or None if failed.
    """
    messages_text = format_messages_for_summary(messages)
    prompt = get_summarization_prompt(messages_text)

    try:
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=60
        )

        summary = result.stdout.strip()
        if summary:
            return summary
        return None

    except subprocess.TimeoutExpired:
        logger.error("Summarization timed out")
        return None
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return None


def run_compaction(force: bool = False) -> CompactionResult:
    """
    Run compaction on the conversation log.

    Args:
        force: Force compaction even if under threshold.

    Returns:
        CompactionResult with details of the operation.
    """
    messages = load_conversations()

    if not messages:
        return CompactionResult(
            success=False,
            messages_compacted=0,
            original_tokens=0,
            summary_tokens=0,
            summary="",
            error="No messages to compact"
        )

    # Calculate current token count
    total_text = ' '.join(
        f"{m.get('user_message', '')} {m.get('agent_response', '')}"
        for m in messages
    )
    original_tokens = estimate_tokens(total_text)

    # Check if compaction needed
    if not force and original_tokens < HARD_LIMIT:
        return CompactionResult(
            success=True,
            messages_compacted=0,
            original_tokens=original_tokens,
            summary_tokens=0,
            summary="",
            error="Compaction not needed"
        )

    # Split messages
    to_compact, to_keep = split_messages_by_tokens(messages)

    if not to_compact:
        return CompactionResult(
            success=False,
            messages_compacted=0,
            original_tokens=original_tokens,
            summary_tokens=0,
            summary="",
            error="No messages to compact"
        )

    logger.info(f"Compacting {len(to_compact)} messages, keeping {len(to_keep)}")

    # Summarize old messages
    summary = summarize_chunk(to_compact)

    if not summary:
        # Fallback: just keep the last N messages without summary
        logger.warning("Summarization failed, using fallback (discard old)")
        save_conversations(to_keep)
        return CompactionResult(
            success=True,
            messages_compacted=len(to_compact),
            original_tokens=original_tokens,
            summary_tokens=0,
            summary="[Summarization failed - old messages discarded]",
            error="Summarization failed, used fallback"
        )

    summary_tokens = estimate_tokens(summary)

    # Create compacted entry
    compacted_entry = {
        "timestamp": datetime.now().isoformat(),
        "sender": "SYSTEM",
        "user_message": "[COMPACTED HISTORY]",
        "agent_response": summary,
        "compaction_metadata": {
            "messages_compacted": len(to_compact),
            "original_tokens": original_tokens,
            "summary_tokens": summary_tokens,
            "compacted_at": datetime.now().isoformat()
        }
    }

    # Log the compaction
    with open(COMPACTION_LOG, 'a') as f:
        f.write(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "messages_compacted": len(to_compact),
            "original_tokens": original_tokens,
            "summary_tokens": summary_tokens
        }) + '\n')

    # Save new conversation log: compacted entry + kept messages
    new_messages = [compacted_entry] + to_keep
    save_conversations(new_messages)

    logger.info(f"Compaction complete: {original_tokens} -> {summary_tokens} tokens")

    return CompactionResult(
        success=True,
        messages_compacted=len(to_compact),
        original_tokens=original_tokens,
        summary_tokens=summary_tokens,
        summary=summary
    )


def check_and_compact() -> Optional[CompactionResult]:
    """
    Check if compaction needed and run if so.

    Returns:
        CompactionResult if compaction was performed, None otherwise.
    """
    messages = load_conversations()
    if not messages:
        return None

    total_text = ' '.join(
        f"{m.get('user_message', '')} {m.get('agent_response', '')}"
        for m in messages
    )
    current_tokens = estimate_tokens(total_text)

    if should_compact(current_tokens):
        logger.info(f"Compaction triggered at {current_tokens} tokens")
        return run_compaction()

    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print(f"Hard limit: {HARD_LIMIT:,} tokens")
    print(f"Should compact at 90k? {should_compact(90_000)}")
    print(f"Should compact at 96k? {should_compact(96_000)}")

    # Load and analyze current conversations
    messages = load_conversations()
    print(f"\nCurrent messages: {len(messages)}")

    if messages:
        total_text = ' '.join(
            f"{m.get('user_message', '')} {m.get('agent_response', '')}"
            for m in messages
        )
        tokens = estimate_tokens(total_text)
        print(f"Current tokens: {tokens:,}")

        # Test split
        to_compact, to_keep = split_messages_by_tokens(messages)
        print(f"Would compact: {len(to_compact)}, would keep: {len(to_keep)}")
