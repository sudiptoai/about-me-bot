"""
Claude-powered About-Me Bot.

Uses :class:`RAGPipeline` to retrieve relevant context from a personal
knowledge base, then calls the Anthropic Claude API to generate an answer.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import anthropic

from src.rag import RAGPipeline

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a personal AI assistant that knows everything about the person described \
in the knowledge base below. You answer questions about their life, background, \
skills, work history, personality, and values. You can also make reasonable \
predictions about how they would act in a given situation, based on their \
documented personality traits and past behaviour.

Guidelines:
- Answer in first person on behalf of the person (e.g. "I studied at…").
- Be concise, warm, and professional.
- If the knowledge base does not contain enough information to answer \
confidently, say so honestly rather than making things up.
- When predicting behaviour, cite the relevant personality traits or values \
that inform the prediction.

Knowledge base (retrieved context):
{context}
"""

# ---------------------------------------------------------------------------
# Conversation history entry
# ---------------------------------------------------------------------------


class Message:
    """A single turn in a multi-turn conversation."""

    def __init__(self, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError(f"Role must be 'user' or 'assistant', got {role!r}")
        self.role = role
        self.content = content

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


# ---------------------------------------------------------------------------
# AboutMeBot
# ---------------------------------------------------------------------------


class AboutMeBot:
    """Claude-backed chatbot with RAG over a personal knowledge base.

    Parameters
    ----------
    biodata_path:
        Path to the Markdown knowledge-base file.  Defaults to
        ``data/biodata.md`` relative to the working directory.
    api_key:
        Anthropic API key.  Falls back to the ``ANTHROPIC_API_KEY``
        environment variable when not supplied.
    model:
        Claude model identifier.  Falls back to the ``CLAUDE_MODEL``
        environment variable, then to ``claude-3-5-sonnet-20241022``.
    embedding_model:
        sentence-transformers model for RAG embeddings.
    top_k:
        Number of retrieved chunks injected into each prompt.
    max_history:
        Maximum number of conversation turns kept in memory.
    """

    def __init__(
        self,
        biodata_path: str | Path | None = None,
        api_key: str | None = None,
        model: str | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        max_history: int = 10,
    ) -> None:
        resolved_path = self._resolve_biodata_path(biodata_path)
        embedding_model = os.getenv("EMBEDDING_MODEL", embedding_model)
        top_k = int(os.getenv("TOP_K", top_k))

        self._rag = RAGPipeline(
            biodata_path=resolved_path,
            embedding_model=embedding_model,
            top_k=top_k,
        )

        resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Anthropic API key not found. "
                "Set the ANTHROPIC_API_KEY environment variable or pass api_key=<your_key>."
            )
        self._client = anthropic.Anthropic(api_key=resolved_key)
        self._model = model or os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
        self._max_history = max_history
        self._history: List[Message] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """Send *user_message* and return the assistant's response."""
        user_message = user_message.strip()
        if not user_message:
            return ""

        context = self._rag.get_context(user_message)
        system = _SYSTEM_PROMPT.format(context=context)

        self._history.append(Message("user", user_message))
        self._trim_history()

        messages_payload = [m.to_dict() for m in self._history]

        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=messages_payload,
        )

        assistant_text = response.content[0].text
        self._history.append(Message("assistant", assistant_text))
        self._trim_history()

        return assistant_text

    def reset(self) -> None:
        """Clear the conversation history."""
        self._history.clear()

    @property
    def history(self) -> List[Message]:
        """Return a copy of the current conversation history."""
        return list(self._history)

    @property
    def document_count(self) -> int:
        """Number of indexed document chunks in the RAG pipeline."""
        return self._rag.document_count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_biodata_path(path: str | Path | None) -> Path:
        if path is not None:
            return Path(path)
        env_path = os.getenv("BIODATA_PATH")
        if env_path:
            return Path(env_path)
        # Default: data/biodata.md relative to the repository root
        return Path(__file__).parent.parent / "data" / "biodata.md"

    def _trim_history(self) -> None:
        """Keep at most *max_history* turns (each turn = user + assistant)."""
        max_messages = self._max_history * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]
