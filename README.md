# Basic Vector RAG – Cybersecurity Knowledge Assistant

A **Naive (Basic) Retrieval-Augmented Generation** system that answers cybersecurity questions using a small document collection. Documents are split into chunks, converted into embeddings, stored in a **FAISS** vector index, and the most similar chunks are passed to an LLM (Llama on **Groq**) to generate a grounded answer with source citations.

## Approach

```
 data/*.txt ──► Chunking ──► Embeddings ──► FAISS index (saved to index/)
                (120 words,   (all-MiniLM-L6-v2,
                 30 overlap)   384-dim, normalised)
                                                    │
 Question ──► Question embedding ──► Top-k cosine search ┘
                                          │
                       Retrieved chunks + question ──► LLM (Groq) ──► Answer with [sources]
```

| Step | What happens | Tool |
|---|---|---|
| 1. Load | Reads every `.txt` file in `data/` | Python |
| 2. Chunk | Fixed-size word chunks (120 words) with 30-word overlap so ideas are not cut in half | Python |
| 3. Embed | Each chunk becomes a 384-dimension vector | `sentence-transformers/all-MiniLM-L6-v2` |
| 4. Index | Vectors stored in a FAISS `IndexFlatIP` (inner product on normalised vectors = cosine similarity) and saved to disk | `faiss-cpu` |
| 5. Retrieve | The question is embedded and the top-k most similar chunks are returned | FAISS |
| 6. Generate | Chunks are inserted into a prompt that tells the LLM to answer **only** from the context and cite sources | Groq API (`openai/gpt-oss-120b`) |

**Strengths:** simple, fast, understands meaning (e.g. "lock my files and ask for money" finds the ransomware document).
**Limitations:** pure semantic search can miss exact keywords and acronyms; no re-ranking of results.

## Document collection

`data/` contains 7 short cybersecurity documents written for this project: phishing, malware, ransomware, DDoS, SQL injection, brute force attacks and incident response.

## Project structure

```
rag-basic-vector/
├── data/               # document collection (.txt)
├── rag_basic.py        # full RAG pipeline
├── requirements.txt    # dependencies
├── .env.example        # template for the API key
└── README.md
```

## Dependencies

- Python 3.9+
- `sentence-transformers`, `faiss-cpu`, `numpy`, `groq`, `python-dotenv`

## Setup

1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/rag-basic-vector.git
   cd rag-basic-vector
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Get a free API key from https://console.groq.com/keys, then copy `.env.example` to `.env` and paste the key:
   ```
   GROQ_API_KEY=your_key_here
   ```
   Without a key the program still runs and shows the retrieved chunks.

## Execution

Ask a single question:
```bash
python rag_basic.py --question "How can an organisation protect itself against ransomware?"
```

Interactive mode:
```bash
python rag_basic.py
```

Options:

| Flag | Meaning |
|---|---|
| `--question`, `-q` | Question to ask |
| `--top_k` | Number of chunks to retrieve (default 3) |
| `--rebuild` | Rebuild the FAISS index after changing files in `data/` |

### Run in Google Colab

```python
!git clone https://github.com/<your-username>/rag-basic-vector.git
%cd rag-basic-vector
!pip install -q -r requirements.txt

import os
from google.colab import userdata          # add GROQ_API_KEY in the Colab "Secrets" (key icon) panel
os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")

!python rag_basic.py -q "What are the warning signs of a phishing email?"
```

## Example output (illustrative – exact scores will differ)

```
[1] Loaded 7 documents from 'data/'
[2] Split into 21 chunks (120 words, 30 overlap)
[3] Embedded chunks with 'sentence-transformers/all-MiniLM-L6-v2' -> dimension 384
[4] FAISS index built with 21 vectors

[5] Top-3 retrieved chunks:
    0.71  ransomware.txt#2
    0.58  ransomware.txt#1
    ...
[6] Answer (openai/gpt-oss-120b):
The most important defence is a reliable backup strategy following the 3-2-1 rule ... [ransomware.txt]
```
