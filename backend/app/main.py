# backend\app\main.py
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import fitz  # pymupdf
import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np
from dotenv import load_dotenv
import traceback

load_dotenv()

app = FastAPI(title="Simple RAG (No OpenAI)")

# Allow frontend (local) to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to specific origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize embedding model (local)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
print("Loading embedding model:", EMBED_MODEL)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # uses local/auto HF model cache

# Initialize Chroma client
# On ordinary installs the default in-memory DB will be used.
# If you plan to persist to disk you can pass persist_directory param to Settings.
chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet"))

# Ensure collection exists (or create)
COLLECTION_NAME = "pdf_store"
try:
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    print("Loaded existing Chroma collection:", COLLECTION_NAME)
except Exception:
    collection = chroma_client.create_collection(name=COLLECTION_NAME)
    print("Created new Chroma collection:", COLLECTION_NAME)

# Simple in-memory variable to keep raw PDF text if you want to inspect later
latest_pdf_text = ""
latest_pdf_chunks = []  # parallel list of chunk strings (keeps order)

# Utility: extract text from PDF bytes (returns concatenated text)
def extract_text_from_pdf_bytes(file_bytes: bytes):
    texts = []
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                texts.append(page.get_text())
    except Exception as e:
        print("Error extracting PDF text:", e)
        traceback.print_exc()
    return "\n".join(texts)

# Chunking function (naive by characters, preserves order)
def chunk_text_text(text: str, chunk_size=800, overlap=200):
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(L, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # advance start with overlap
        if end >= L:
            break
        start = end - overlap if (end - overlap) > start else end
    return chunks

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile):
    """
    Upload a single PDF. This endpoint:
    - extracts text
    - chunks it
    - computes embeddings for chunks
    - stores chunks into Chroma collection (in-memory)
    """
    global latest_pdf_text, latest_pdf_chunks, collection

    # read bytes
    content = await file.read()
    if not content:
        return {"status": "error", "message": "Empty file"}

    # extract text
    extracted = extract_text_from_pdf_bytes(content)
    if not extracted.strip():
        return {"status": "error", "message": "No text extracted from PDF."}

    latest_pdf_text = extracted
    # chunk
    chunks = chunk_text_text(extracted, chunk_size=800, overlap=200)
    if not chunks:
        return {"status": "error", "message": "No chunks produced."}

    latest_pdf_chunks = chunks

    # compute embeddings
    embeddings = embedding_model.encode(chunks, show_progress_bar=False)
    # ensure embeddings are lists for chroma
    embeddings_list = [np.array(e).astype(float).tolist() for e in embeddings]

    # clear previous collection content (we keep only latest PDF for simplicity)
    try:
        # If you want to keep multiple uploads, remove this clear block
        ids = [doc["id"] for doc in collection.get(include=["metadatas", "documents", "ids"])["ids"]]
        if ids:
            collection.delete(ids=ids)
    except Exception:
        # ignore if collection empty or method unsupported
        pass

    # add chunks to collection: add accepts lists
    # create unique ids
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": file.filename, "chunk_index": i} for i in range(len(chunks))]

    # add to collection
    try:
        collection.add(documents=chunks, ids=ids, metadatas=metadatas, embeddings=embeddings_list)
    except Exception as e:
        # If chroma client expects embedding_function, you could alternatively provide embedding_fn instead of embeddings.
        print("Error adding to Chroma collection:", e)
        traceback.print_exc()
        return {"status": "error", "message": "Failed to add chunks to vector store."}

    return {"status": "ok", "message": f"PDF processed: {len(chunks)} chunks added.", "chunks": len(chunks)}


@app.post("/ask_question/")
async def ask_question(query: str = Form(...)):
    """
    Query endpoint (Option A fallback):
    - Retrieves top-k relevant chunks from Chroma
    - Returns combined human-readable excerpts as the answer plus structured sources
    """
    try:
        # query collection; n_results chooses how many nearest chunks to return
        n_results = 3
        results = collection.query(query_texts=[query], n_results=n_results, include=["documents", "ids", "metadatas"])

        docs = results.get("documents", [[]])[0]
        ids = results.get("ids", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        if not docs:
            return {"answer": "No relevant text found in uploaded PDF.", "sources": []}

        # Build snippets/excerpts: try to pick a sentence that contains a query token, else first few chars
        q_tokens = [t.lower() for t in query.split() if len(t) > 2]
        snippets = []
        for i, doc in enumerate(docs):
            excerpt = None
            # split into sentences heuristically
            sentences = [s.strip() for s in doc.replace("\n", " ").split(".") if s.strip()]
            for s in sentences:
                low = s.lower()
                if any(tok in low for tok in q_tokens):
                    excerpt = s
                    break
            if not excerpt:
                excerpt = doc[:350].strip()
            snippet = {
                "id": ids[i] if i < len(ids) else f"chunk_{i}",
                "excerpt": excerpt,
                "full": doc,
                "metadata": metas[i] if i < len(metas) else {}
            }
            snippets.append(snippet)

        # build a readable "answer" by concatenating small excerpts
        combined_answer = "Most relevant excerpts from the uploaded PDF:\n\n"
        for idx, s in enumerate(snippets, start=1):
            source_info = f"Source {idx}"
            if s["metadata"] and "source" in s["metadata"]:
                source_info += f" ({s['metadata'].get('source')})"
            combined_answer += f"{source_info}: {s['excerpt'].strip()}...\n\n"

        return {"answer": combined_answer, "sources": snippets}

    except Exception as e:
        print("Error in ask_question:", e)
        traceback.print_exc()
        return {"answer": "An error occurred while processing the query.", "sources": []}
