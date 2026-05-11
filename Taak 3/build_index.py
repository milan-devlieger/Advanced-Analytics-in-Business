"""
Build (or rebuild) the ChromaDB vector index from the Steam SQLite database.

Run once before starting the Flask app:
    uv run python build_index.py
"""
from __future__ import annotations

import os
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from steam_sqlite import load_games_from_sqlite

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("RAGLOOKER_DB_PATH", BASE_DIR / "steam_games_reviews_25.sqlite"))
CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "steam_games"
MODEL_NAME = "all-MiniLM-L6-v2"
MAX_GAMES = 5000
BATCH_SIZE = 512  # ChromaDB upsert limit is well above this; keep batches small for RAM


def build_game_text(raw: dict) -> str:
    """Concatenate the fields most useful for semantic search into one string."""
    name = raw.get("name", "")

    genres = raw.get("genres", [])
    if isinstance(genres, list):
        genres_str = ", ".join(genres)
    else:
        genres_str = str(genres)

    tags = raw.get("tags", {})
    if isinstance(tags, dict):
        tags_str = ", ".join(list(tags.keys())[:15])
    elif isinstance(tags, list):
        tags_str = ", ".join(tags[:15])
    else:
        tags_str = ""

    desc = raw.get("short_description", "")

    parts = [f"Name: {name}"]
    if genres_str:
        parts.append(f"Genres: {genres_str}")
    if tags_str:
        parts.append(f"Tags: {tags_str}")
    if desc:
        parts.append(f"Description: {desc}")

    return " | ".join(parts)


def main() -> None:
    print(f"Loading up to {MAX_GAMES} games from {DB_PATH} …")
    rows = load_games_from_sqlite(DB_PATH, MAX_GAMES)
    print(f"  Loaded {len(rows)} games.")

    print(f"Loading embedder: {MODEL_NAME} …")
    embedder = SentenceTransformer(MODEL_NAME)

    print(f"Opening ChromaDB at {CHROMA_PATH} …")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    # Delete & recreate so a re-run always produces a clean index
    try:
        client.delete_collection(COLLECTION_NAME)
        print("  Deleted existing collection.")
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for app_id, raw in rows:
        ids.append(app_id)
        documents.append(build_game_text(raw))
        metadatas.append({"app_id": app_id, "name": raw.get("name", "")})

    total = len(ids)
    print(f"Embedding and indexing {total} documents in batches of {BATCH_SIZE} …")

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch_docs = documents[start:end]
        batch_ids = ids[start:end]
        batch_meta = metadatas[start:end]

        embeddings = embedder.encode(batch_docs, show_progress_bar=False).tolist()

        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=embeddings,
            metadatas=batch_meta,
        )
        print(f"  [{end}/{total}] indexed")

    print(f"\nDone. Collection '{COLLECTION_NAME}' contains {collection.count()} documents.")
    print(f"Index persisted at: {CHROMA_PATH}")


if __name__ == "__main__":
    main()
