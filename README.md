# About-Me Bot

A **Claude-powered personal AI assistant** with **Retrieval-Augmented Generation (RAG)** that answers questions about your life, career, personality, and can even predict how you would behave in a given situation.

---

## How it works

```
User question
     │
     ▼
 Embedder (sentence-transformers)
     │  embed question
     ▼
 VectorStore (NumPy cosine similarity)
     │  retrieve top-k chunks
     ▼
 Claude (Anthropic API)
     │  generate answer with retrieved context
     ▼
 Response
```

1. **Knowledge base** – your personal information lives in `data/biodata.md` as structured Markdown.
2. **RAG pipeline** – the file is split into overlapping chunks, embedded with `sentence-transformers`, and stored in an in-memory vector store.
3. **Retrieval** – at query time the question is embedded and the most semantically similar chunks are retrieved.
4. **Generation** – the retrieved context is injected into a Claude system prompt and the model produces a grounded, first-person answer.

---

## Project structure

```
about-me-bot/
├── data/
│   └── biodata.md          # ← Edit this with YOUR information
├── src/
│   ├── __init__.py
│   ├── rag.py              # DocumentLoader, Embedder, VectorStore, RAGPipeline
│   ├── bot.py              # AboutMeBot (Claude + RAG)
│   └── main.py             # Interactive CLI
├── tests/
│   ├── test_rag.py
│   └── test_bot.py
├── .env.example
├── requirements.txt
└── README.md
```

---

## Quick start

### 1 – Clone & install

```bash
git clone https://github.com/sudiptoai/about-me-bot.git
cd about-me-bot
pip install -r requirements.txt
```

### 2 – Configure

```bash
cp .env.example .env
# Open .env and set your Anthropic API key:
#   ANTHROPIC_API_KEY=sk-ant-...
```

### 3 – Personalise the knowledge base

Edit `data/biodata.md` and replace the placeholder content with your own information:
- Personal details
- Education & work history
- Technical skills & projects
- Personality traits, values, and decision-making style
- Behavioural patterns (how you react in various situations)

The richer and more detailed you make this file, the more accurate and personal the bot's responses will be.

### 4 – Chat

```bash
python -m src.main
```

Example session:

```
  Initialising – loading knowledge base and embedding model…

╔══════════════════════════════════════════════════════╗
║          About-Me Bot  (powered by Claude + RAG)     ║
╚══════════════════════════════════════════════════════╝

  Knowledge base loaded: 14 chunks indexed.

You: Tell me about your professional background.
Bot: I'm a Senior AI Engineer at Acme Corp, where I lead the design …

You: How would you handle a production outage?
Bot: I stay calm and immediately assess the blast radius …

You: /reset
  Conversation history cleared.

You: /quit
  Goodbye!
```

### CLI commands

| Command | Description |
|---------|-------------|
| `/reset` | Clear conversation history |
| `/quit` or `/exit` | Exit the bot |

---

## Configuration

All settings can be overridden via environment variables (or in your `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(required)* | Your Anthropic API key |
| `CLAUDE_MODEL` | `claude-3-5-sonnet-20241022` | Claude model to use |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `BIODATA_PATH` | `data/biodata.md` | Path to your knowledge base |
| `TOP_K` | `5` | Number of retrieved chunks per query |

---

## Running tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Using the bot programmatically

```python
from src.bot import AboutMeBot

bot = AboutMeBot(
    biodata_path="data/biodata.md",
    api_key="sk-ant-...",        # or set ANTHROPIC_API_KEY env var
)

reply = bot.chat("What are your strongest technical skills?")
print(reply)

# Multi-turn conversation is supported automatically
reply2 = bot.chat("Which of those do you enjoy most?")
print(reply2)

# Reset conversation history
bot.reset()
```

---

## License

MIT
