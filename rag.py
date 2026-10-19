import os
import pickle
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ==========================================
# SETTINGS
# ==========================================


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DOCUMENT_FOLDER = BASE_DIR / "documents"
INDEX_FILE = BASE_DIR / "loan_index.faiss"
DATA_FILE = BASE_DIR / "loan_data.pkl"


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ==========================================
# READ ALL PDF FILES
# ==========================================

def load_pdfs():

    documents = []

    for filename in os.listdir(DOCUMENT_FOLDER):

        if filename.lower().endswith(".pdf"):

            file_path = os.path.join(
                DOCUMENT_FOLDER,
                filename
            )

            print(f"Reading: {filename}")

            reader = PdfReader(file_path)

            full_text = ""

            for page in reader.pages:

                text = page.extract_text()

                if text:
                    full_text += text + "\n"

            documents.append({
                "filename": filename,
                "text": full_text
            })

    return documents


# ==========================================
# SPLIT TEXT INTO CHUNKS
# ==========================================

def create_chunks(
    text,
    chunk_size=500,
    overlap=100
):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():

            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ==========================================
# CREATE VECTOR DATABASE
# ==========================================

def create_database():

    print("\n==============================")
    print("Reading loan policy documents")
    print("==============================\n")

    documents = load_pdfs()

    if not documents:

        print("ERROR: No PDF files found.")

        return


    all_chunks = []

    metadata = []


    # --------------------------------------
    # Create chunks
    # --------------------------------------

    for document in documents:

        chunks = create_chunks(
            document["text"]
        )

        for chunk in chunks:

            all_chunks.append(chunk)

            metadata.append({
                "filename": document["filename"],
                "text": chunk
            })


    print(
        f"\nCreated {len(all_chunks)} text chunks."
    )


    # --------------------------------------
    # Create embeddings
    # --------------------------------------

    print("\nCreating embeddings...")

    embeddings = model.encode(
        all_chunks,
        show_progress_bar=True
    )

    embeddings = embeddings.astype(
        "float32"
    )


    # --------------------------------------
    # Create FAISS index
    # --------------------------------------

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)


    # --------------------------------------
    # Save FAISS index
    # --------------------------------------

    faiss.write_index(
        index,
        INDEX_FILE
    )


    # --------------------------------------
    # Save metadata
    # --------------------------------------

    with open(
        DATA_FILE,
        "wb"
    ) as file:

        pickle.dump(
            metadata,
            file
        )


    print("\n==============================")
    print("DATABASE CREATED SUCCESSFULLY")
    print("==============================")

    print(
        f"PDF files: {len(documents)}"
    )

    print(
        f"Text chunks: {len(all_chunks)}"
    )

    print(
        f"Vector database: {INDEX_FILE}"
    )


# ==========================================
# SEARCH DOCUMENTS
# ==========================================

def search_documents(
    query,
    top_k=5
):

    index = faiss.read_index(
        INDEX_FILE
    )


    with open(
        DATA_FILE,
        "rb"
    ) as file:

        metadata = pickle.load(file)


    # Convert question into embedding

    query_embedding = model.encode(
        [query]
    )

    query_embedding = query_embedding.astype(
        "float32"
    )


    # Search FAISS

    distances, indices = index.search(
        query_embedding,
        top_k
    )


    results = []


    for i in indices[0]:

        results.append(
            metadata[i]
        )


    return results


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    create_database()
    