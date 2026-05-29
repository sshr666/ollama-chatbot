"""
utils/rag.py — Full RAG (Retrieval-Augmented Generation) pipeline
=================================================================
This file is the entire intelligence layer that sits between the user's
question and the Ollama model.  It handles:

  1. PDF ingestion     → extract raw text from uploaded PDF bytes
  2. Chunking          → split text into overlapping chunks (LangChain)
  3. Embedding         → convert chunks to vectors (SentenceTransformers, local)
  4. Storage           → persist vectors in ChromaDB (local disk)
  5. Retrieval         → find the top-K most relevant chunks for a query
  6. Prompt injection  → wrap retrieved chunks into a context block

WHY each library was chosen:
  pypdf              — pure-Python PDF parser, no system dependencies
  LangChain          — provides RecursiveCharacterTextSplitter (battle-tested
                       chunking logic that respects sentence boundaries)
  sentence-transformers — downloads and runs embedding models locally via
                          HuggingFace; all-MiniLM-L6-v2 is only 90 MB and
                          very fast on CPU
  ChromaDB           — embedded vector DB (no server needed), persists to disk,
                       exposes a simple similarity_search interface

DATA FLOW:
  PDF bytes
    └─► extract_text_from_pdf()   → raw text string
          └─► chunk_text()         → list[str] of overlapping chunks
                └─► embed + store in ChromaDB (add_documents_to_chroma())
                      └─► retrieve_context(query) → top-K chunks as one string
                            └─► build_rag_prompt() → final prompt for Ollama

COLLECTION NAMING:
  Each uploaded PDF gets its own ChromaDB collection named after its filename
  (sanitised).  This lets the user switch between documents without re-embedding.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Lazy imports (only loaded when RAG is actually used) ──────────────────────
# This keeps startup fast when the user hasn't uploaded any PDFs yet.

def _get_splitter():
    """Return a LangChain text splitter configured for RAG."""
    try:
        # langchain >= 0.2 moved the splitter to its own package
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        # fallback for older langchain installs
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter(
        # chunk_size: target character count per chunk.
        # 800 chars ≈ ~200 tokens — fits well inside most LLM context windows
        # while still containing enough meaning to be useful.
        chunk_size=500,

        # chunk_overlap: how many characters the next chunk shares with the
        # previous one.  Overlap prevents answers that span a chunk boundary
        # from being missed.
        chunk_overlap=90,

        # These separators are tried in order: paragraph → sentence → word → char.
        # LangChain will split on the first one that produces chunks under chunk_size.
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _get_embedding_model():
    """
    Load the local SentenceTransformer embedding model.

    all-MiniLM-L6-v2:
      - Size: ~90 MB (downloaded once to ~/.cache/huggingface/)
      - Speed: fast on CPU (~500 sentences/sec)
      - Output: 384-dimensional float vectors
      - Works completely offline after first download
    """
    from sentence_transformers import SentenceTransformer
    # The model is cached locally after first download — fully offline after that.
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def _get_chroma_client(persist_dir: str = "./chroma_db"):
    """
    Return a persistent ChromaDB client.

    persist_dir: where ChromaDB stores its SQLite + vector files on disk.
    Data survives app restarts — you only embed a PDF once.
    """
    import chromadb
    return chromadb.PersistentClient(path=persist_dir)


# ── Step 1: PDF text extraction ───────────────────────────────────────────────

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract all text from a PDF given its raw bytes.

    Parameters
    ----------
    pdf_bytes : bytes
        The binary content of the uploaded PDF file.

    Returns
    -------
    str
        All extracted text, pages joined with double newlines.
    """
    import io
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages: list[str] = []

    for page_num, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text)
        except Exception as e:
            logger.warning(f"Could not extract page {page_num}: {e}")
            continue

    if not pages:
        raise ValueError(
            "No text could be extracted from this PDF. "
            "It may be a scanned image — OCR support is not included in this version."
        )

    full_text = "\n\n".join(pages)
    logger.info(f"Extracted {len(full_text)} characters from {len(pages)} pages.")
    return full_text


# ── Step 2: Chunking ──────────────────────────────────────────────────────────

def _clean_extracted_text(text: str) -> str:
    """
    Clean raw pypdf output before chunking.

    pypdf struggles with multi-column research PDFs — it extracts text in
    reading order across columns, which causes sentences to interleave and
    figure captions / email addresses / page numbers to appear mid-paragraph.

    This function removes the worst offenders so chunks are cleaner.
    """
    lines = text.split("\n")
    cleaned: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Skip blank lines (we'll re-join with \n\n later)
        if not stripped:
            cleaned.append("")
            continue

        # Skip lines that are purely email addresses (common in paper headers)
        if re.match(r"^[\w\.\-]+@[\w\.\-]+\s*([\w\.\-]+@[\w\.\-]+\s*)*$", stripped):
            continue

        # Skip lines that look like page numbers or lone digits
        if re.match(r"^\d{1,3}$", stripped):
            continue

        # Skip very short lines that are likely headers/captions with no substance
        # e.g. "Figure 5:", "Table 2", "Input-Input Layer5"
        if len(stripped) < 40 and re.match(r"^(Figure|Table|Input|Output|Layer|Equation)\b", stripped, re.IGNORECASE):
            continue

        cleaned.append(stripped)

    rejoined = "\n".join(cleaned)

    # Collapse runs of 3+ newlines into a paragraph break
    rejoined = re.sub(r"\n{3,}", "\n\n", rejoined)

    # Remove repeated consecutive sentences — pypdf sometimes duplicates a
    # sentence when it spans a column boundary.
    # Strategy: split on ". " and deduplicate adjacent identical sentences.
    sentences = re.split(r"(?<=[.!?])\s+", rejoined)
    deduped: list[str] = []
    prev = ""
    for s in sentences:
        if s.strip() and s.strip() != prev.strip():
            deduped.append(s)
        prev = s
    rejoined = " ".join(deduped)

    return rejoined


def chunk_text(text: str) -> list[str]:
    """
    Split raw text into overlapping chunks suitable for embedding.

    Applies noise cleaning first to handle the messy output that pypdf
    produces from multi-column research PDFs (figures, captions, emails,
    duplicated sentences from column-boundary artifacts).

    Returns
    -------
    list[str]
        A list of text chunks, each ~500 characters with 90-char overlap.
    """
    # Clean before splitting — garbage in = garbage embeddings
    text = _clean_extracted_text(text)

    splitter = _get_splitter()
    chunks = splitter.split_text(text)

    clean: list[str] = []
    seen: set[str] = set()

    for chunk in chunks:
        c = chunk.strip()

        # Drop too-short chunks (headers, lone captions, etc.)
        if len(c) < 80:
            continue

        # Drop chunks that are mostly non-alphabetic (tables, reference lists,
        # strings of numbers/symbols from figure data)
        alpha_ratio = sum(ch.isalpha() for ch in c) / max(len(c), 1)
        if alpha_ratio < 0.55:
            continue

        # Deduplicate near-identical chunks (can happen with overlapping windows
        # on repeated boilerplate text like license headers)
        fingerprint = re.sub(r"\s+", " ", c[:120]).lower()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)

        clean.append(c)

    logger.info(f"Split text into {len(clean)} clean chunks (from raw split).")
    return clean


# ── Step 3 + 4: Embed and store in ChromaDB ───────────────────────────────────

def _sanitise_collection_name(filename: str) -> str:
    """
    ChromaDB collection names must be 3-63 chars, alphanumeric + hyphens only.
    We derive a stable name from the filename.
    """
    # Remove extension, replace non-alphanumeric with hyphens
    name = Path(filename).stem
    name = re.sub(r"[^a-zA-Z0-9]", "-", name)
    name = re.sub(r"-+", "-", name).strip("-").lower()

    # Ensure minimum length
    if len(name) < 3:
        name = "doc-" + name

    # Ensure maximum length (ChromaDB limit is 63)
    if len(name) > 55:
        # Hash the overflow to keep it unique
        name = name[:50] + "-" + hashlib.md5(name.encode()).hexdigest()[:4]

    return name


def collection_exists(filename: str, persist_dir: str = "./chroma_db") -> bool:
    """
    Check if a PDF has already been embedded and stored.
    Used to skip re-embedding on app restarts.
    """
    try:
        client = _get_chroma_client(persist_dir)
        collection_name = _sanitise_collection_name(filename)
        existing = [c.name for c in client.list_collections()]
        return collection_name in existing
    except Exception:
        return False


def embed_and_store(
    chunks: list[str],
    filename: str,
    persist_dir: str = "./chroma_db",
) -> str:
    """
    Embed a list of text chunks and store them in ChromaDB.

    Parameters
    ----------
    chunks : list[str]
        Text chunks from chunk_text().
    filename : str
        Original PDF filename — used to name the ChromaDB collection.
    persist_dir : str
        Where ChromaDB persists data on disk.

    Returns
    -------
    str
        The ChromaDB collection name (used later for retrieval).
    """
    model = _get_embedding_model()
    client = _get_chroma_client(persist_dir)
    collection_name = _sanitise_collection_name(filename)

    # Delete existing collection if re-uploading the same filename
    try:
        client.delete_collection(collection_name)
        logger.info(f"Deleted existing collection '{collection_name}' for re-embedding.")
    except Exception:
        pass  # Collection didn't exist yet

    collection = client.create_collection(
        name=collection_name,
        # cosine distance is standard for semantic similarity with MiniLM
        metadata={"hnsw:space": "cosine"},
    )

    # Embed all chunks in one batch (SentenceTransformers is optimised for this)
    logger.info(f"Embedding {len(chunks)} chunks with all-MiniLM-L6-v2…")
    embeddings = model.encode(chunks, show_progress_bar=False).tolist()

    # Store chunks + their vectors in ChromaDB
    # IDs must be unique strings — we use "chunk-0", "chunk-1", etc.
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk-{i}" for i in range(len(chunks))],
        metadatas=[{"source": filename, "chunk_index": i} for i in range(len(chunks))],
    )

    logger.info(f"Stored {len(chunks)} chunks in collection '{collection_name}'.")
    return collection_name


# ── Step 5: Retrieval ─────────────────────────────────────────────────────────

def retrieve_context(
    query: str,
    collection_name: str,
    top_k: int = 4,
    persist_dir: str = "./chroma_db",
) -> list[str]:
    """
    Find the top-K most semantically similar chunks to the user's query.

    HOW IT WORKS:
      1. The query is embedded using the same model used during ingestion.
      2. ChromaDB computes cosine similarity between the query vector and
         every stored chunk vector using an HNSW index (fast approximate search).
      3. The top-K closest chunks are returned.

    Parameters
    ----------
    query : str
        The user's question, exactly as typed.
    collection_name : str
        Which ChromaDB collection to search (one per PDF).
    top_k : int
        How many chunks to retrieve.  4 is usually the sweet spot:
        enough context without overflowing the prompt.
    persist_dir : str
        Path to ChromaDB data directory.

    Returns
    -------
    list[str]
        The top-K retrieved text chunks, most relevant first.
    """
    model = _get_embedding_model()
    client = _get_chroma_client(persist_dir)
    collection = client.get_collection(collection_name)

    # Embed the query with the same model used for chunks
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),  # can't request more than we have
        include=["documents"],
    )

    # results["documents"] is a list-of-lists (one list per query)
    chunks: list[str] = results["documents"][0] if results["documents"] else []
    logger.info(f"Retrieved {len(chunks)} chunks for query: '{query[:60]}…'")
    return chunks


# ── Step 6: Prompt construction ───────────────────────────────────────────────

def build_rag_prompt(user_query: str, context_chunks: list[str]) -> str:
    """
    Inject retrieved context chunks into the user's prompt.

    The resulting string REPLACES the plain user query when RAG is active.
    The original question is preserved at the bottom so the model stays
    focused on answering it.

    WHY this format:
      - Clear XML-like delimiters help the model distinguish context from question.
      - "Based only on the context" reduces hallucination (the model is told
        not to use background knowledge).
      - Keeping the original question at the end mirrors how humans read:
        context first, then the question to answer.

    Parameters
    ----------
    user_query : str
        The original question the user typed.
    context_chunks : list[str]
        Retrieved chunks from retrieve_context().

    Returns
    -------
    str
        The augmented prompt to send to Ollama instead of the raw query.
    """
    if not context_chunks:
        # No relevant context found — fall back to plain query
        return user_query

    # Number the chunks so the model can reference them ("As stated in chunk 2…")
    numbered_chunks = "\n\n".join(
        f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )

    augmented_prompt = f"""You are a helpful assistant answering questions about an uploaded document.

Use the retrieved context below to answer the question as fully as you can.
- Prioritize relevant information from the retrieved chunks.
- If the exact answer is not explicitly stated, infer the most likely answer from the available context.
- Synthesize information across multiple chunks when needed.
- Avoid unnecessary apologies or disclaimers.
- Only say the information is unavailable if the retrieved context is completely unrelated.


User question: {user_query}

Answer:"""

    return augmented_prompt


# ── High-level convenience function ───────────────────────────────────────────

def process_pdf_upload(
    pdf_bytes: bytes,
    filename: str,
    persist_dir: str = "./chroma_db",
) -> tuple[str, int]:
    """
    Full ingestion pipeline: bytes → ChromaDB collection.

    Called once when the user uploads a PDF.  Subsequent questions just
    call retrieve_context() — no re-embedding needed.

    Returns
    -------
    tuple[str, int]
        (collection_name, number_of_chunks_stored)
    """
    text = extract_text_from_pdf(pdf_bytes)
    chunks = chunk_text(text)
    collection_name = embed_and_store(chunks, filename, persist_dir)
    return collection_name, len(chunks)