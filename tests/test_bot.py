"""
Unit tests for the About-Me Bot (src/bot.py).

All external dependencies (Anthropic API, sentence-transformers) are mocked
so no API key or internet access is required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.bot import AboutMeBot, Message


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_bot(tmp_path: Path) -> AboutMeBot:
    """Return an AboutMeBot with all external calls stubbed out."""
    md = tmp_path / "bio.md"
    md.write_text(
        "# About Me\n\nI am a software engineer.\n\nI love Python and AI.\n\nI value honesty.",
        encoding="utf-8",
    )

    # Stub the Anthropic client
    mock_anthropic_cls = MagicMock()
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Hello, I am a software engineer.")]
    mock_client.messages.create.return_value = mock_response

    # Stub the Embedder so no model download is needed
    mock_embedder = MagicMock()
    mock_embedder.embed.side_effect = lambda texts: np.random.default_rng(0).random(
        (len(texts), 8)
    ).astype(np.float32)

    with (
        patch("src.bot.anthropic.Anthropic", mock_anthropic_cls),
        patch("src.rag.Embedder", return_value=mock_embedder),
    ):
        bot = AboutMeBot(biodata_path=md, api_key="test-key")

    # Replace the internal client with our mock after construction
    bot._client = mock_client
    return bot


# ---------------------------------------------------------------------------
# Message tests
# ---------------------------------------------------------------------------


class TestMessage:
    def test_valid_roles(self) -> None:
        u = Message("user", "hello")
        a = Message("assistant", "hi")
        assert u.role == "user"
        assert a.role == "assistant"

    def test_invalid_role_raises(self) -> None:
        with pytest.raises(ValueError):
            Message("system", "content")

    def test_to_dict(self) -> None:
        m = Message("user", "test")
        assert m.to_dict() == {"role": "user", "content": "test"}


# ---------------------------------------------------------------------------
# AboutMeBot construction tests
# ---------------------------------------------------------------------------


class TestAboutMeBotInit:
    def test_missing_api_key_raises(self, tmp_path: Path) -> None:
        md = tmp_path / "bio.md"
        md.write_text("Some content.", encoding="utf-8")

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = np.zeros((1, 4), dtype=np.float32)

        with (
            patch("src.rag.Embedder", return_value=mock_embedder),
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="API key"),
        ):
            AboutMeBot(biodata_path=md, api_key=None)

    def test_document_count_positive(self, tmp_path: Path) -> None:
        bot = _make_bot(tmp_path)
        assert bot.document_count > 0


# ---------------------------------------------------------------------------
# Chat tests
# ---------------------------------------------------------------------------


class TestAboutMeBotChat:
    def test_chat_returns_string(self, tmp_path: Path) -> None:
        bot = _make_bot(tmp_path)
        reply = bot.chat("Who are you?")
        assert isinstance(reply, str)
        assert len(reply) > 0

    def test_empty_message_returns_empty(self, tmp_path: Path) -> None:
        bot = _make_bot(tmp_path)
        reply = bot.chat("   ")
        assert reply == ""

    def test_history_grows(self, tmp_path: Path) -> None:
        bot = _make_bot(tmp_path)
        bot.chat("First question")
        assert len(bot.history) == 2  # user + assistant

    def test_reset_clears_history(self, tmp_path: Path) -> None:
        bot = _make_bot(tmp_path)
        bot.chat("First question")
        bot.reset()
        assert bot.history == []

    def test_history_trim(self, tmp_path: Path) -> None:
        """History should never exceed max_history * 2 messages."""
        bot = _make_bot(tmp_path)
        bot._max_history = 2
        for _ in range(6):
            bot.chat("Question")
        assert len(bot.history) <= 4  # 2 turns × 2 messages

    def test_history_returns_copy(self, tmp_path: Path) -> None:
        bot = _make_bot(tmp_path)
        bot.chat("Hi")
        h1 = bot.history
        h1.clear()
        assert len(bot.history) == 2  # original not mutated

    def test_chat_calls_anthropic(self, tmp_path: Path) -> None:
        bot = _make_bot(tmp_path)
        bot.chat("Tell me about yourself")
        bot._client.messages.create.assert_called_once()
