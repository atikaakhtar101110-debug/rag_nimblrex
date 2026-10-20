# Document Q&A — a small RAG prototype (Streamlit + FAISS + OpenAI)

Upload PDF/TXT files, ask questions, and get answers grounded in your documents, with source file names and page numbers. If the documents don't contain the answer, the app says so instead of guessing.

## How it works

```
Upload PDF/TXT ─► ingestion.py ─► text per page ─► ~1000-char overlapping chunks
                                                         │
                         OpenAI embeddings (text-embedding-3-small)
                                                         ▼
                                       vector_store.py (in-memory FAISS)
Question ─► embed ─► top-4 similar chunks ─► rag_pipeline.py ─► gpt-4o-mini
                       (score below 0.20 dropped)           answers ONLY from the chunks
```

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI, key loading, chat, sources display |
| `ingestion.py` | PDF/TXT reading and chunking (keeps page numbers) |
| `vector_store.py` | Tiny FAISS cosine-similarity store |
| `rag_pipeline.py` | Embeddings, retrieval, grounded prompt, friendly API errors |

The index lives in memory per browser session. Restarting the app or closing the tab clears it (no database, by design).

## Cost (OpenAI API — you pay OpenAI directly)

Models are set at the top of `rag_pipeline.py`. Approximate list prices at the time of writing (**verify at https://openai.com/api/pricing**, they change):

- Embeddings, `text-embedding-3-small`: about $0.02 per 1M tokens. A 100-page PDF is roughly 50k tokens, i.e. about **$0.001**.
- Answers, `gpt-4o-mini`: about $0.15 per 1M input and $0.60 per 1M output tokens. One question sends ~4 chunks (~1–1.5k tokens) and returns at most 500 tokens, i.e. well under **$0.001 per question**.
- Questions with no relevant chunks skip the answer call entirely.
- Safety caps: 10 MB per file and 3000 chunks per indexing action. Set a monthly spending limit in your OpenAI account for extra protection.
- Streamlit Community Cloud hosting is free for public apps.

## Local setup (Windows Command Prompt)

Requires Python 3.11 or 3.12 (recommended) from python.org.

```bat
cd path\to\rag-prototype
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
notepad .env
```

In Notepad, replace `your-openai-api-key-here` with your key from https://platform.openai.com/api-keys, save, then:

```bat
streamlit run app.py
```

The app opens at http://localhost:8501. Stop it with `Ctrl+C`. Next time, only run `.venv\Scripts\activate` and `streamlit run app.py`.

## Testing it yourself

1. Start the app. Without a key you should see a clear "No OpenAI API key found" message.
2. Upload a PDF and a TXT file and click **Process documents**. You should see "✅ file: N chunks indexed" for each.
3. Ask a question whose answer is in a file. You should get an answer citing `[1]`, `[2]`, and a **Sources** expander showing file names (and page numbers for PDFs).
4. Ask something unrelated (e.g. "What is the price of bitcoin?"). You should get "I couldn't find enough information in the uploaded documents…" and no sources.
5. Upload an unsupported or scanned (image-only) PDF. You should get a readable error, not a crash.
6. Click **Clear chat** (keeps documents) and **Clear documents & chat** (resets everything).
7. Put a wrong key in `.env`, restart, and ask a question: you should see "The OpenAI API key was rejected."

## Deploy to Streamlit Community Cloud

1. **Check nothing secret is committed.** `.env` and `.streamlit/secrets.toml` are in `.gitignore`. Only `.env.example` (placeholder) is committed.
2. **Create a GitHub repository** (public is simplest on the free tier) and push the project:
   ```bat
   git init
   git add .
   git status
   git commit -m "RAG prototype"
   git branch -M main
   git remote add origin https://github.com/YOUR-USER/YOUR-REPO.git
   git push -u origin main
   ```
   Run `git status` before committing and confirm `.env` is **not** listed.
3. Go to https://share.streamlit.io and sign in with GitHub.
4. Click **Create app** → **Deploy a public app from GitHub**.
5. Choose your repository, branch `main`, and main file path `app.py`.
6. Click **Advanced settings**:
   - Python version: **3.12** (or 3.11).
   - Secrets: paste exactly (with your real key):
     ```toml
     OPENAI_API_KEY = "sk-your-real-key"
     ```
7. Click **Deploy**. The first build installs `requirements.txt` and takes a few minutes.
8. To change the key later: app menu → **Settings** → **Secrets**. Pushing to `main` redeploys automatically.

Streamlit's menus can change slightly over time; if a label differs, see https://docs.streamlit.io/deploy/streamlit-community-cloud.

Anyone with the app URL can use your API key's budget. Keep the app private (Share settings) or set an OpenAI spending limit.

## Known limitations

- No OCR: scanned/image-only PDFs are rejected with a message.
- Single-turn answers: each question is answered independently (no chat memory), which keeps cost low.
- Index is per session and in memory.
- Answer quality depends on retrieval; tune `CHUNK_SIZE`, `TOP_K`, `MIN_SCORE` if needed.
