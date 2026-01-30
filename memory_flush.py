"""
Memory flush - save important context before compaction.

Gives Claude a chance to extract and persist key information
before the conversation history gets summarized.
"""
from __future__ import annotations

import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from token_counter import CONTEXT_WINDOW, SOFT_LIMIT

logger = logging.getLogger(__name__)

# Constants
RESERVE_TOKENS = 10_000       # Reserve for response generation
SOFT_BUFFER = 5_000           # Buffer before soft threshold
FLUSH_THRESHOLD = CONTEXT_WINDOW - RESERVE_TOKENS - SOFT_BUFFER  # ~85k

# Paths
AGENT_DIR = Path.home() / "Code" / "Agent"
MEMORY_DIR = AGENT_DIR / "memory"
MEMORY_FILE = MEMORY_DIR / "MEMORY.md"

# Track flush state per session
_flush_count = 0
_last_flush_tokens = 0


def should_flush(current_tokens: int) -> bool:
    """
    Check if we should flush memory before compaction.

    Only flushes once per threshold crossing to avoid repeated flushes.

    Args:
        current_tokens: Current token count.

    Returns:
        True if flush should be triggered.
    """
    global _last_flush_tokens

    if current_tokens < FLUSH_THRESHOLD:
        return False

    # Only flush if we've grown significantly since last flush
    if _last_flush_tokens > 0 and current_tokens < _last_flush_tokens + 10_000:
        return False

    return True


def get_flush_prompt(conversation_context: str) -> str:
    """Generate the prompt for memory extraction."""
    return f"""You are reviewing a conversation to extract important information that should be remembered long-term.

## Conversation Context
{conversation_context}

## Instructions
Extract and summarize the following types of information from this conversation:

1. **User Preferences** - Any stated preferences, likes, dislikes
2. **Key Decisions** - Important choices or decisions made
3. **Open Tasks/TODOs** - Things that need to be done or followed up
4. **Important Facts** - Names, dates, technical details worth remembering
5. **Constraints** - Any limitations or requirements mentioned

Format your response as markdown with clear headers. Only include information that would be useful to remember in future conversations. Be concise but complete.

If there's nothing significant to remember, respond with just: "No significant information to persist."
"""


def run_memory_flush(conversation_context: str) -> Optional[str]:
    """
    Run memory flush to extract important information.

    Args:
        conversation_context: The current conversation context.

    Returns:
        Extracted information, or None if flush failed/unnecessary.
    """
    global _flush_count, _last_flush_tokens

    prompt = get_flush_prompt(conversation_context)

    try:
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=60
        )

        response = result.stdout.strip()

        if not response or "No significant information" in response:
            logger.info("Memory flush: nothing significant to persist")
            return None

        # Append to MEMORY.md
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"\n\n---\n\n## Memory Flush [{timestamp}]\n\n{response}"

        # Create file if it doesn't exist
        if not MEMORY_FILE.exists():
            MEMORY_FILE.write_text(f"# Agent Memory\n\nDurable memory extracted from conversations.\n{entry}")
        else:
            with open(MEMORY_FILE, "a") as f:
                f.write(entry)

        _flush_count += 1
        logger.info(f"Memory flush #{_flush_count} complete, wrote to MEMORY.md")

        # Try to index the updated file
        try:
            from memory_manager import get_manager
            manager = get_manager()
            manager.index_file(MEMORY_FILE, "memory")
        except Exception as e:
            logger.warning(f"Failed to index MEMORY.md: {e}")

        return response

    except subprocess.TimeoutExpired:
        logger.error("Memory flush timed out")
        return None
    except Exception as e:
        logger.error(f"Memory flush failed: {e}")
        return None


def mark_flushed(tokens: int):
    """Mark that a flush occurred at given token count."""
    global _last_flush_tokens
    _last_flush_tokens = tokens


def get_flush_stats() -> dict:
    """Get flush statistics for current session."""
    return {
        "flush_count": _flush_count,
        "last_flush_tokens": _last_flush_tokens,
        "flush_threshold": FLUSH_THRESHOLD,
        "memory_file": str(MEMORY_FILE),
        "memory_exists": MEMORY_FILE.exists()
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test flush detection
    print(f"Flush threshold: {FLUSH_THRESHOLD:,} tokens")
    print(f"Should flush at 80k? {should_flush(80_000)}")
    print(f"Should flush at 86k? {should_flush(86_000)}")

    # Test with sample context
    sample_context = """
    User: My name is Rob and I prefer dark mode in all apps.
    Agent: Got it, I'll remember you prefer dark mode!

    User: Can you remind me to call the dentist tomorrow at 2pm?
    Agent: I've set a reminder for you to call the dentist tomorrow at 2pm.

    User: I'm working on a Python project using FastAPI and PostgreSQL.
    Agent: Nice stack! FastAPI with PostgreSQL is great for building APIs.
    """

    print("\n--- Testing flush prompt ---")
    print(get_flush_prompt(sample_context)[:500] + "...")
