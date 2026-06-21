# 📄 PDF Q&A Chatbot (RAG)

A beginner-friendly AI chatbot that lets you **ask questions about any PDF** using Retrieval-Augmented Generation (RAG).

---

## What is RAG?

RAG = **Retrieval Augmented Generation**

Instead of asking an LLM to "know" your document, you:
1. Split the document into small chunks
2. Convert each chunk into a vector (numbers representing meaning)
3. Store vectors in a database (FAISS)
4. At query time, find the most relevant chunks
5. Send *only those chunks* to the LLM as context

**Result:** The LLM answers from *your document*, not from hallucination.

---

## Tech Stack

| Tool | Role |
|------|------|
| `LangChain` | Framework that connects all components |
| `HuggingFace (MiniLM-L6)` | Converts text to embeddings (free, local) |
| `FAISS` | Vector database for similarity search |
| `Groq (LLaMA 3.1)` | Free, fast LLM inference |
| `Streamlit` | Web UI in ~5 lines of Python |

---

## Setup (2 minutes)

### 1. Clone/download the project
```bash
cd pdf-qa-chatbot
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Get free Groq API key
- Go to https://console.groq.com
- Sign up (free)
- Create API key → copy it

### 5. Run the app
```bash
streamlit run app.py
```

Open browser at `http://localhost:8501`

---

## How to Use

1. Paste your Groq API key in the sidebar
2. Upload any PDF (textbook, notes, research paper)
3. Wait ~30 seconds for indexing (only first time)
4. Ask questions in the chat box!

---

## Interview Talking Points

> *"I built a RAG-based chatbot that lets users query any PDF document. The system uses semantic similarity search over FAISS vector embeddings to retrieve the top 3 most relevant chunks before passing them as context to a Groq-hosted LLaMA model — this grounding prevents hallucination and keeps answers accurate to the document."*

**Concepts you can explain:**

- **RAG vs fine-tuning:** RAG is better for dynamic, domain-specific docs. Fine-tuning is expensive and static.
- **Why embeddings?** Semantic similarity — "ML" and "machine learning" have close vectors even though the words differ.
- **Why FAISS?** It does approximate nearest-neighbor search in milliseconds, even with millions of vectors.
- **chunk_overlap:** Prevents answers from being cut off mid-sentence at chunk boundaries.
- **temperature=0.1:** Low temperature makes LLM answers factual and deterministic, not creative.

---

## Project Structure

```
pdf-qa-chatbot/
├── app.py              ← Main application (all logic here)
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## Bonus: Deploy Free

Deploy on Streamlit Cloud:
1. Push to GitHub
2. Go to https://share.streamlit.io
3. Connect your repo → Deploy

Your chatbot will be live at a public URL!
