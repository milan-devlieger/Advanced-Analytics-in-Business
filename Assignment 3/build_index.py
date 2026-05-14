"""
Build (or rebuild) the ChromaDB vector index from the Steam SQLite database.
Now includes player review summaries in the embedded text for richer retrieval.

Run once before starting the Flask app:
    PYTHONPATH=. uv run python build_index.py
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from steam_sqlite import get_review_summaries, load_games_from_sqlite

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("RAGLOOKER_DB_PATH", BASE_DIR / "steam_games_reviews_25.sqlite"))
CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "steam_games"
MODEL_NAME = "all-MiniLM-L6-v2"
MAX_GAMES = 5000
BATCH_SIZE = 512
# all-MiniLM-L6-v2 hard-truncates at 256 tokens; keep combined text ≤ ~400 words
MAX_REVIEW_WORDS = 300


def build_game_text(raw: dict, review_summary: str = "") -> str:
    """
    Build the document text that gets embedded into ChromaDB.
    Combines store metadata (name, genres, tags, description) with a
    concise player review summary so that semantic search captures both
    official copy AND real player opinions.
    """
    name = raw.get("name", "")

    genres = raw.get("genres", [])
    genres_str = ", ".join(genres) if isinstance(genres, list) else str(genres)

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

    base_text = " | ".join(parts)

    if review_summary:
        # Guard combined length: truncate review block if necessary
        base_words = len(base_text.split())
        review_words = review_summary.split()
        budget = max(0, MAX_REVIEW_WORDS - base_words)
        if budget > 20:  # only add if we have meaningful room
            if len(review_words) > budget:
                review_summary = " ".join(review_words[:budget]) + "…"
            base_text = f"{base_text} | Reviews: {review_summary}"

    return base_text


def main() -> None:
    t_start = time.time()

    print(f"Loading up to {MAX_GAMES} games from {DB_PATH} …")
    rows = load_games_from_sqlite(DB_PATH, MAX_GAMES)
    print(f"  Loaded {len(rows)} games.")

    print("Loading player review summaries (single SQL query with window functions) …")
    t_reviews = time.time()
    review_summaries = get_review_summaries(DB_PATH, MAX_GAMES)
    games_with_reviews = sum(1 for aid, _ in rows if str(aid) in review_summaries)
    avg_len = (
        sum(len(s) for s in review_summaries.values()) / len(review_summaries)
        if review_summaries else 0
    )
    print(f"  Review summaries loaded in {time.time() - t_reviews:.1f}s")
    print(f"  Games with review data : {games_with_reviews} / {len(rows)}")
    print(f"  Avg review_summary length: {avg_len:.0f} chars")

    print(f"\nLoading embedder: {MODEL_NAME} …")
    embedder = SentenceTransformer(MODEL_NAME)

    print(f"Opening ChromaDB at {CHROMA_PATH} …")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
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
        summary = review_summaries.get(str(app_id), "")
        ids.append(app_id)
        documents.append(build_game_text(raw, summary))
        metadatas.append({"app_id": app_id, "name": raw.get("name", "")})

    total = len(ids)
    print(f"\nEmbedding and indexing {total} documents in batches of {BATCH_SIZE} …")
    t_embed = time.time()

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        embeddings = embedder.encode(
            documents[start:end], show_progress_bar=False
        ).tolist()
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings,
            metadatas=metadatas[start:end],
        )
        print(f"  [{end}/{total}] indexed")

    elapsed = time.time() - t_start
    print(f"\n{'='*55}")
    print(f"Done.  Collection '{COLLECTION_NAME}': {collection.count()} documents.")
    print(f"Games with reviews embedded : {games_with_reviews} / {total}")
    print(f"Avg review_summary length   : {avg_len:.0f} chars")
    print(f"Total indexing time         : {elapsed:.1f}s")
    print(f"Index persisted at          : {CHROMA_PATH}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
