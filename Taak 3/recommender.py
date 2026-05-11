from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import ollama
from sentence_transformers import CrossEncoder, SentenceTransformer

from steam_sqlite import load_games_from_sqlite

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("RAGLOOKER_DB_PATH", BASE_DIR / "steam_games_reviews_25.sqlite"))
CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "steam_games"
MAX_GAMES = 5000
DEFAULT_MATCH_COUNT = 5

EMBEDDER_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "gemma2:2b"


def create_search_engine() -> "GameSearchEngine":
    return GameSearchEngine(DB_PATH)


@dataclass
class GameRecord:
    app_id: str
    raw: dict[str, Any]

    @property
    def name(self) -> str:
        return self.raw.get("name", "Unknown title")

    @property
    def short_description(self) -> str:
        return self.raw.get("short_description", "")

    def to_result(self, score: float) -> dict[str, Any]:
        return {
            "app_id": self.app_id,
            "name": self.name,
            "score": round(score, 4),
            "short_description": self.short_description,
            "genres": self.raw.get("genres", []),
            "tags": self._normalize_tags(self.raw.get("tags")),
            "price": self.raw.get("price"),
            "release_date": self.raw.get("release_date"),
            "header_image": self.raw.get("header_image"),
            "store_page": f"https://store.steampowered.com/app/{self.app_id}",
            "platforms": {
                "windows": bool(self.raw.get("windows")),
                "mac": bool(self.raw.get("mac")),
                "linux": bool(self.raw.get("linux")),
            },
        }

    @staticmethod
    def _normalize_tags(tags: Any) -> list[str]:
        if isinstance(tags, dict):
            return list(tags.keys())[:8]
        if isinstance(tags, list):
            return tags[:8]
        return []


def _build_game_text(raw: dict) -> str:
    """Same logic as build_index.py — used for cross-encoder scoring."""
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

    return " | ".join(parts)


def _check_ollama(model: str) -> None:
    """Verify Ollama is running and the required model is available."""
    try:
        available = [m.model for m in ollama.list().models]
    except Exception as exc:
        print(
            f"\n[ERROR] Cannot reach Ollama. Make sure it is running:\n"
            f"  brew services start ollama   (or: ollama serve)\n"
            f"  Details: {exc}\n",
            file=sys.stderr,
        )
        raise RuntimeError("Ollama is not reachable") from exc

    # Strip ':latest' suffix for loose comparison
    normalised = [m.split(":")[0] for m in available]
    target = model.split(":")[0]
    if target not in normalised:
        print(
            f"\n[ERROR] Model '{model}' is not pulled yet.\n"
            f"  Run: ollama pull {model}\n"
            f"  Available models: {available}\n",
            file=sys.stderr,
        )
        raise RuntimeError(f"Ollama model '{model}' not found")


class GameSearchEngine:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

        # ----- models -----
        print(f"Loading embedder ({EMBEDDER_MODEL}) …")
        self.embedder = SentenceTransformer(EMBEDDER_MODEL)

        print(f"Loading reranker ({RERANKER_MODEL}) …")
        self.reranker = CrossEncoder(RERANKER_MODEL)

        # ----- Chroma -----
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        try:
            self.collection = chroma_client.get_collection(COLLECTION_NAME)
        except Exception as exc:
            raise RuntimeError(
                f"ChromaDB collection '{COLLECTION_NAME}' not found at {CHROMA_PATH}.\n"
                "Run `uv run python build_index.py` first to build the index."
            ) from exc

        # ----- game lookup (app_id → GameRecord) -----
        print(f"Loading game records from {db_path} …")
        rows = load_games_from_sqlite(db_path, MAX_GAMES)
        self._lookup: dict[str, GameRecord] = {
            app_id: GameRecord(app_id=app_id, raw=raw) for app_id, raw in rows
        }

        # ----- Ollama -----
        _check_ollama(LLM_MODEL)

        print("GameSearchEngine ready.")

    # ------------------------------------------------------------------
    # Public API (signatures must stay stable for app.py / Flask)
    # ------------------------------------------------------------------

    def search(self, query: str) -> dict[str, Any]:
        candidates = self.retrieve_candidates(query)
        ranked_matches = self.rank_candidates(query, candidates)
        results = [record.to_result(score) for record, score in ranked_matches]

        return {
            "matches": results,
            "answer": self.generate_answer(query, ranked_matches),
            "meta": {
                "retrieval_mode": "embedding+cross-encoder+llm",
                "indexed_games": self.collection.count(),
            },
        }

    def retrieve_candidates(self, query: str, k: int = 30) -> list[GameRecord]:
        """Embed the query and return the top-k nearest games from Chroma."""
        query_vec = self.embedder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=min(k, self.collection.count()),
            include=["metadatas"],
        )

        records: list[GameRecord] = []
        for meta in results["metadatas"][0]:
            app_id = meta["app_id"]
            if app_id in self._lookup:
                records.append(self._lookup[app_id])

        return records

    def rank_candidates(
        self, query: str, candidates: list[GameRecord], k: int = 10
    ) -> list[tuple[GameRecord, float]]:
        """Score every (query, game_text) pair with the cross-encoder, return top-k."""
        if not candidates:
            return []

        pairs = [(query, _build_game_text(record.raw)) for record in candidates]
        scores: list[float] = self.reranker.predict(pairs).tolist()

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def generate_answer(self, query: str, ranked: list[tuple[GameRecord, float]]) -> str:
        """Call gemma2:2b via Ollama with the top games as context."""
        if not ranked:
            return "No games were found that match your request."

        game_lines = []
        for i, (record, _score) in enumerate(ranked[:10], 1):
            genres = record.raw.get("genres", [])
            genres_str = ", ".join(genres) if isinstance(genres, list) else ""
            line = f"{i}. {record.name}"
            if genres_str:
                line += f" [{genres_str}]"
            if record.short_description:
                line += f" — {record.short_description}"
            game_lines.append(line)

        context = "\n".join(game_lines)

        prompt = (
            f"You are a helpful Steam game recommender.\n\n"
            f"A user is looking for: \"{query}\"\n\n"
            f"Here are the top matching games from the database:\n{context}\n\n"
            f"Write a 2-3 sentence recommendation explaining which of these games best fit "
            f"the user's request and why. Be specific about titles and genres."
        )

        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.message.content.strip()
