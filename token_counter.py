"""
Token counting utilities for Claude context management.

Uses tiktoken (OpenAI's tokenizer) which provides a reasonable approximation
for Claude's tokenization. Claude uses a similar BPE tokenizer.
"""
from __future__ import annotations

import tiktoken
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Claude's context window limits
CONTEXT_WINDOW = 100_000      # Claude's max context (tokens)
SOFT_THRESHOLD = 0.8          # 80% = warning threshold
HARD_THRESHOLD = 0.95         # 95% = must compact

# Pre-computed thresholds
SOFT_LIMIT = int(CONTEXT_WINDOW * SOFT_THRESHOLD)   # 80,000 tokens
HARD_LIMIT = int(CONTEXT_WINDOW * HARD_THRESHOLD)   # 95,000 tokens


class TokenCounter:
    """Estimates token counts for text using tiktoken."""

    def __init__(self, model: str = "cl100k_base"):
        """
        Initialize the token counter.

        Args:
            model: tiktoken encoding name. cl100k_base is used by GPT-4
                   and provides a reasonable approximation for Claude.
        """
        self._encoding = tiktoken.get_encoding(model)

    def count(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: The text to count tokens for.

        Returns:
            Number of tokens.
        """
        if not text:
            return 0
        return len(self._encoding.encode(text))

    def count_messages(self, messages: list) -> int:
        """
        Count tokens across multiple message dicts.

        Args:
            messages: List of dicts with 'user_message' and 'agent_response' keys.

        Returns:
            Total token count.
        """
        total = 0
        for msg in messages:
            if 'user_message' in msg:
                total += self.count(msg['user_message'])
            if 'agent_response' in msg:
                total += self.count(msg['agent_response'])
        return total


# Module-level singleton for convenience
_counter: Optional[TokenCounter] = None


def get_counter() -> TokenCounter:
    """Get or create the singleton TokenCounter."""
    global _counter
    if _counter is None:
        _counter = TokenCounter()
    return _counter


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Args:
        text: The text to count tokens for.

    Returns:
        Estimated number of tokens.
    """
    return get_counter().count(text)


def check_limit(tokens: int, context_window: int = CONTEXT_WINDOW) -> dict:
    """
    Check if token count is within acceptable limits.

    Args:
        tokens: Current token count.
        context_window: Max context window size (default: 100k).

    Returns:
        Dict with:
            - ok: bool, True if under soft threshold
            - warning: bool, True if between soft and hard threshold
            - critical: bool, True if over hard threshold
            - usage_pct: float, percentage of context used
            - tokens: int, the input token count
            - remaining: int, tokens remaining until hard limit
    """
    soft = int(context_window * SOFT_THRESHOLD)
    hard = int(context_window * HARD_THRESHOLD)
    usage_pct = (tokens / context_window) * 100

    return {
        "ok": tokens < soft,
        "warning": soft <= tokens < hard,
        "critical": tokens >= hard,
        "usage_pct": round(usage_pct, 1),
        "tokens": tokens,
        "remaining": max(0, hard - tokens),
        "context_window": context_window,
    }


def format_token_status(tokens: int, context_window: int = CONTEXT_WINDOW) -> str:
    """
    Format token status as a human-readable string.

    Args:
        tokens: Current token count.
        context_window: Max context window size.

    Returns:
        Formatted status string.
    """
    status = check_limit(tokens, context_window)

    if status["critical"]:
        level = "CRITICAL"
    elif status["warning"]:
        level = "WARNING"
    else:
        level = "OK"

    return (
        f"[{level}] Tokens: {tokens:,} / {context_window:,} "
        f"({status['usage_pct']}%) - {status['remaining']:,} remaining"
    )


if __name__ == "__main__":
    # Quick test
    test_text = "Hello, this is a test message to count tokens."
    tokens = estimate_tokens(test_text)
    print(f"Test text: {test_text!r}")
    print(f"Token count: {tokens}")
    print()

    # Test limits
    for test_tokens in [50_000, 80_000, 90_000, 96_000]:
        print(format_token_status(test_tokens))
