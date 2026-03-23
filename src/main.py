"""
Interactive CLI entry point for the About-Me Bot.

Usage
-----
    python -m src.main

Or, after installing the package:
    python src/main.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Load .env file if present (before importing bot so the key is available)
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ImportError:
    pass  # python-dotenv is listed in requirements.txt; skip if missing in tests

from src.bot import AboutMeBot

# ---------------------------------------------------------------------------
# ANSI colour helpers (gracefully degrade on non-ANSI terminals)
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"


def _c(text: str, *codes: str) -> str:
    if not sys.stdout.isatty():
        return text
    return "".join(codes) + text + _RESET


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_BANNER = """
╔══════════════════════════════════════════════════════╗
║          About-Me Bot  (powered by Claude + RAG)     ║
╚══════════════════════════════════════════════════════╝

Type your question and press Enter.
Commands:
  /reset  – clear conversation history
  /quit   – exit the bot
"""

_COMMANDS = {"/quit", "/exit", "/q", "/reset"}


def _print_banner(doc_count: int) -> None:
    print(_c(_BANNER, _CYAN, _BOLD))
    print(_c(f"  Knowledge base loaded: {doc_count} chunks indexed.\n", _GREEN))


def run() -> None:
    """Start the interactive CLI session."""
    print(_c("  Initialising – loading knowledge base and embedding model…", _YELLOW))
    try:
        bot = AboutMeBot()
    except ValueError as exc:
        print(_c(f"\n  Error: {exc}", _RED))
        sys.exit(1)
    except FileNotFoundError as exc:
        print(_c(f"\n  Error: {exc}", _RED))
        sys.exit(1)

    _print_banner(bot.document_count)

    while True:
        try:
            user_input = input(_c("You: ", _BOLD, _CYAN)).strip()
        except (EOFError, KeyboardInterrupt):
            print(_c("\n  Goodbye!", _GREEN))
            break

        if not user_input:
            continue

        if user_input.lower() in {"/quit", "/exit", "/q"}:
            print(_c("  Goodbye!", _GREEN))
            break

        if user_input.lower() == "/reset":
            bot.reset()
            print(_c("  Conversation history cleared.\n", _YELLOW))
            continue

        print()
        try:
            response = bot.chat(user_input)
        except Exception as exc:  # noqa: BLE001
            print(_c(f"  Error calling Claude API: {exc}\n", _RED))
            continue

        print(_c("Bot: ", _BOLD, _GREEN) + response + "\n")


if __name__ == "__main__":
    run()
