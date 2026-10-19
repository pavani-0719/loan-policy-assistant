
import os
import pickle
from pathlib import Path

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ==========================================
# SETTINGS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DOCUMENT_FOLDER = BASE_DIR / "documents"
INDEX_FILE = BASE_DIR / "loan_index.faiss"
DATA_FILE = BASE_DIR / "loan_data.pkl"


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded.")


# ==========================================
# READ PDF FILES
# ==========================================

def load_pdfs():
    documents = []

    if not DOCUMENT_FOLDER.exists():
        raise FileNotFoundError(
            f"Documents folder not found: {DOCUMENT_FOLDER}"
        )

    for filename in os.listdir(DOCUMENT_FOLDER):
        if filename.lower().endswith(".pdf"):
            file_path = DOCUMENT_FOLDER / filename

            print(f"Reading: {filename}")

            reader = PdfReader(str(file_path))
            full_text = ""

            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"

            if full_text.strip():
                documents.append({
                    "filename": filename,
                    "text": full_text
                })

    return documents


# ==========================================
# SPLIT TEXT INTO CHUNKS
# ==========================================

def create_chunks(text, chunk_size=500, overlap=100):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ==========================================
# CREATE VECTOR DATABASE
# ==========================================

def create_database():
    print("Reading loan policy documents...")

    documents = load_pdfs()

    if not documents:
        raise ValueError(
            "No readable PDF documents found in the documents folder."
        )

    all_chunks = []
    metadata = []

    for document in documents:
        chunks = create_chunks(document["text"])

        for chunk in chunks:
            all_chunks.append(chunk)
            metadata.append({
                "filename": document["filename"],
                "text": chunk
            })

    if not all_chunks:
        raise ValueError(
            "No text could be extracted from the PDF documents."
        )

    print(f"Created {len(all_chunks)} text chunks.")
    print("Creating embeddings...")

    embeddings = model.encode(
        all_chunks,
        show_progress_bar=True
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # FAISS expects a string file path
    faiss.write_index(index, str(INDEX_FILE))

    with open(DATA_FILE, "wb") as file:
        pickle.dump(metadata, file)

    print("DATABASE CREATED SUCCESSFULLY")
    print(f"PDF files: {len(documents)}")
    print(f"Text chunks: {len(all_chunks)}")
    print(f"Vector database: {INDEX_FILE}")


# ==========================================
# SEARCH DOCUMENTS
# ==========================================

def search_documents(query, top_k=5):
    if not INDEX_FILE.is_file():
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_FILE}"
        )

    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {DATA_FILE}"
        )

    # FIX: FAISS expects a string path, not a Path object
    index = faiss.read_index(str(INDEX_FILE))

    with open(DATA_FILE, "rb") as file:
        metadata = pickle.load(file)

    query_embedding = model.encode(
        [query]
    ).astype("float32")

    k = min(top_k, index.ntotal)

    if k == 0:
        return []

    distances, indices = index.search(query_embedding, k)

    results = []

    for i in indices[0]:
        if 0 <= i < len(metadata):
            results.append(metadata[i])

    return results


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    create_database()
