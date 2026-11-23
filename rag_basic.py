"""
Basic (Naive) Vector RAG
========================
Pipeline:
  1. Load documents      -> read .txt files from ./data
  2. Chunk documents     -> split into overlapping word chunks
  3. Embed chunks        -> sentence-transformers (all-MiniLM-L6-v2)
  4. Index               -> FAISS (cosine similarity via inner product)
  5. Retrieve            -> top-k most similar chunks for the question
  6. Generate            -> LLM (Groq, Llama) answers using only the retrieved context

Usage:
  python rag_basic.py --question "How can I protect against ransomware?"
  python rag_basic.py            # interactive mode
"""

import argparse
import glob
import json
import os

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

DATA_DIR = "data"
INDEX_DIR = "index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
CHUNK_SIZE = 120     # words per chunk
CHUNK_OVERLAP = 30   # words shared between neighbouring chunks


# ---------------------------------------------------------------- 1. Load
def load_documents(data_dir=DATA_DIR):
    docs = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            docs.append({"source": os.path.basename(path), "text": f.read()})
    print(f"[1] Loaded {len(docs)} documents from '{data_dir}/'")
    return docs


# ---------------------------------------------------------------- 2. Chunk
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += size - overlap
    return chunks


def chunk_documents(docs):
    chunks = []
    for doc in docs:
        for i, piece in enumerate(chunk_text(doc["text"])):
            chunks.append({"id": f"{doc['source']}#{i}", "source": doc["source"], "text": piece})
    print(f"[2] Split into {len(chunks)} chunks ({CHUNK_SIZE} words, {CHUNK_OVERLAP} overlap)")
    return chunks


# ---------------------------------------------------------------- 3 + 4. Embed & index
def build_index(chunks, embedder):
    vectors = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True,
                              show_progress_bar=False)
    vectors = np.asarray(vectors, dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])  # inner product on unit vectors = cosine
    index.add(vectors)
    print(f"[3] Embedded chunks with '{EMBED_MODEL}' -> dimension {vectors.shape[1]}")
    print(f"[4] FAISS index built with {index.ntotal} vectors")
    return index


def save_index(index, chunks):
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    print(f"    Index saved to '{INDEX_DIR}/'")


def load_index():
    index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "chunks.json"), encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"[1-4] Loaded saved index from '{INDEX_DIR}/' ({index.ntotal} vectors)")
    return index, chunks


# ---------------------------------------------------------------- 5. Retrieve
def retrieve(question, index, chunks, embedder, top_k=3):
    q = embedder.encode([question], normalize_embeddings=True)
    scores, ids = index.search(np.asarray(q, dtype="float32"), top_k)
    results = [{**chunks[i], "score": float(s)} for s, i in zip(scores[0], ids[0])]
    print(f"\n[5] Top-{top_k} retrieved chunks:")
    for r in results:
        print(f"    {r['score']:.3f}  {r['id']}")
    return results


# ---------------------------------------------------------------- 6. Generate
def generate(question, contexts):
    context_text = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in contexts)
    prompt = (
        "You are a cybersecurity assistant. Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know. "
        "Cite the source file names in square brackets.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
    )

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return ("(GROQ_API_KEY not set - showing retrieved context only)\n\n" + context_text)

    from groq import Groq
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="Basic Vector RAG")
    parser.add_argument("--question", "-q", help="question to ask (omit for interactive mode)")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--rebuild", action="store_true", help="rebuild the index from ./data")
    args = parser.parse_args()

    embedder = SentenceTransformer(EMBED_MODEL)

    if args.rebuild or not os.path.exists(os.path.join(INDEX_DIR, "faiss.index")):
        chunks = chunk_documents(load_documents())
        index = build_index(chunks, embedder)
        save_index(index, chunks)
    else:
        index, chunks = load_index()

    def ask(question):
        contexts = retrieve(question, index, chunks, embedder, args.top_k)
        print(f"\n[6] Answer ({LLM_MODEL}):\n")
        print(generate(question, contexts))

    if args.question:
        ask(args.question)
    else:
        print("\nInteractive mode - type 'exit' to quit.")
        while True:
            question = input("\nQuestion: ").strip()
            if question.lower() in {"exit", "quit", ""}:
                break
            ask(question)


if __name__ == "__main__":
    main()
