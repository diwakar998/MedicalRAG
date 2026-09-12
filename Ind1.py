# Import of the required libraries
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "legitmentoor"

# Named vectors used for hybrid search (dense = semantic, sparse = keyword/BM25-style)
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DENSE_VECTOR_SIZE = 3072  # text-embedding-3-large output dimension

# Directory containing documents to index
DOCUMENTS_DIR = Path(r"D:\PB_Medical_RAG\Documents")

# Manifest file: tracks what's already indexed (path -> hash) so we only
# process new/changed files on subsequent runs instead of the whole folder.
MANIFEST_PATH = Path(__file__).parent / "index_manifest.json"


# ------------------------------------------------------------------
# Manifest helpers
# ------------------------------------------------------------------
def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict) -> None:
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def file_hash(path: Path) -> str:
    """SHA-256 hash of file contents, read in chunks so large PDFs are fine."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------
# Qdrant helpers
# ------------------------------------------------------------------
def ensure_hybrid_collection(client: QdrantClient) -> None:
    """Create the collection with both dense + sparse vector configs if it
    doesn't exist yet. If it already exists but only has a dense vector
    (e.g. from before hybrid search was added), this will NOT retrofit old
    points with sparse vectors - see the note printed below.
    """
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        info = client.get_collection(COLLECTION_NAME)
        has_sparse = bool(getattr(info.config.params, "sparse_vectors", None))
        if not has_sparse:
            print(
                f"⚠️  Collection '{COLLECTION_NAME}' exists without a sparse vector config. "
                "Adding sparse vector support now, but PRE-EXISTING points will not have "
                "sparse vectors until they are re-indexed. Delete index_manifest.json "
                "(or recreate the collection) to force a full re-embed for true hybrid search."
            )
            client.update_collection(
                collection_name=COLLECTION_NAME,
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: qmodels.SparseVectorParams(
                        index=qmodels.SparseIndexParams()
                    )
                },
            )
        return

    print(f"Creating collection '{COLLECTION_NAME}' with dense + sparse vectors...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: qmodels.VectorParams(
                size=DENSE_VECTOR_SIZE, distance=qmodels.Distance.COSINE
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: qmodels.SparseVectorParams(
                index=qmodels.SparseIndexParams()
            )
        },
    )


def delete_points_for_source(client: QdrantClient, source_path: str) -> None:
    """Delete all vectors previously indexed for a given source file.

    Requires that each point's payload includes {"source": <file path>}.
    Safe to call even if the collection doesn't exist yet or has nothing
    for this source.
    """
    try:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="metadata.source",
                            match=qmodels.MatchValue(value=source_path),
                        )
                    ]
                )
            ),
        )
    except Exception as e:
        # Collection may not exist yet on first run - that's fine.
        print(f"  (skip delete for {source_path}: {e})")


def load_single_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    docs: list[Document] = []
    if suffix == ".pdf":
        loader = PyPDFLoader(file_path=str(path))
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = str(path)
    elif suffix in (".txt", ".md"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        docs = [Document(page_content=text, metadata={"source": str(path)})]
    else:
        print(f"Skipping unsupported file type: {path}")
    return docs


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    if not QDRANT_URL:
        raise ValueError("QDRANT_URL is not set in the .env file")
    if not QDRANT_API_KEY:
        raise ValueError("QDRANT_API_KEY is not set in the .env file")

    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(f"Documents folder not found: {DOCUMENTS_DIR}")

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
    print(client.get_collections())
    ensure_hybrid_collection(client)

    manifest = load_manifest()
    current_files = {
        str(p): p
        for p in sorted(DOCUMENTS_DIR.rglob("*"))
        if p.is_file() and p.suffix.lower() in (".pdf", ".txt", ".md")
    }

    new_files, changed_files, unchanged_files = [], [], []
    for path_str, path in current_files.items():
        h = file_hash(path)
        prev = manifest.get(path_str)
        if prev is None:
            new_files.append((path_str, path, h))
        elif prev["hash"] != h:
            changed_files.append((path_str, path, h))
        else:
            unchanged_files.append(path_str)

    deleted_files = [p for p in manifest if p not in current_files]

    print(f"New: {len(new_files)} | Changed: {len(changed_files)} | "
          f"Unchanged (skipped): {len(unchanged_files)} | Deleted: {len(deleted_files)}")

    # Nothing to do
    if not new_files and not changed_files and not deleted_files:
        print("✅ No changes detected. Index is already up to date.")
        return

    # Remove vectors for deleted files
    for path_str in deleted_files:
        print(f"Removing deleted file from index: {path_str}")
        delete_points_for_source(client, path_str)
        del manifest[path_str]

    # Remove old vectors for changed files (about to be re-added)
    for path_str, _, _ in changed_files:
        print(f"Removing stale vectors for changed file: {path_str}")
        delete_points_for_source(client, path_str)

    # Load + chunk only new/changed files
    to_process = new_files + changed_files
    all_docs: list[Document] = []
    for path_str, path, _ in to_process:
        try:
            docs = load_single_file(path)
            all_docs.extend(docs)
            print(f"Loaded: {path_str} ({len(docs)} pages/sections)")
        except Exception as e:
            print(f"Error loading {path_str}: {e}")

    if all_docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.split_documents(documents=all_docs)

        embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
        # BM25-style sparse embeddings, computed locally via fastembed - no API calls.
        sparse_embedding_model = FastEmbedSparse(model_name="Qdrant/bm25")

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embedding_model,
            sparse_embedding=sparse_embedding_model,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name=DENSE_VECTOR_NAME,
            sparse_vector_name=SPARSE_VECTOR_NAME,
        )

        print(f"Embedding + indexing {len(split_docs)} chunks (dense + sparse)...")
        vector_store.add_documents(split_docs)
        print("✅ Documents successfully indexed.")

    # Update manifest only after successful indexing
    for path_str, _, h in to_process:
        manifest[path_str] = {"hash": h}

    save_manifest(manifest)
    print(f"✅ Manifest updated: {MANIFEST_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error during indexing: {e}")
        