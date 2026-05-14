from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import ollama
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from steam_sqlite import get_review_summaries, load_games_from_sqlite

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("RAGLOOKER_DB_PATH", BASE_DIR / "steam_games_reviews_25.sqlite"))
CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "steam_games"
MAX_GAMES = 5000
DEFAULT_MATCH_COUNT = 5

EMBEDDER_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "gemma2:2b"

# Reciprocal Rank Fusion constant — higher k reduces the impact of top ranks
RRF_K = 60


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

    @property
    def review_summary(self) -> str:
        """Player review summary injected after index load; empty string if none."""
        return self.raw.get("review_summary", "")

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


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _build_game_text(raw: dict) -> str:
    """Reconstruct the document text for cross-encoder scoring (metadata only)."""
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


def _build_bm25_text(raw: dict) -> str:
    """
    Extended text for BM25 indexing: same as _build_game_text but also
    includes the review summary so keyword matches can hit player language.
    """
    base = _build_game_text(raw)
    summary = raw.get("review_summary", "")
    return f"{base} {summary}" if summary else base


def _tokenize(text: str) -> list[str]:
    """Simple whitespace-lowercased tokenizer used by BM25."""
    return text.lower().split()


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "…"


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def _reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = RRF_K
) -> list[str]:
    """
    Merge multiple ranked lists of app_ids using Reciprocal Rank Fusion.
    Returns app_ids sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, app_id in enumerate(ranked_list):
            scores[app_id] = scores.get(app_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


def _sigmoid(x: float) -> float:
    """Map a raw logit to (0, 1)."""
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# Ollama check
# ---------------------------------------------------------------------------

def _check_ollama(model: str) -> None:
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


# ---------------------------------------------------------------------------
# Search engine
# ---------------------------------------------------------------------------

class GameSearchEngine:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

        # ----- embedding + reranking models -----
        print(f"Loading embedder ({EMBEDDER_MODEL}) …")
        self.embedder = SentenceTransformer(EMBEDDER_MODEL)

        print(f"Loading reranker ({RERANKER_MODEL}) …")
        self.reranker = CrossEncoder(RERANKER_MODEL)

        # ----- ChromaDB -----
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        try:
            self.collection = chroma_client.get_collection(COLLECTION_NAME)
        except Exception as exc:
            raise RuntimeError(
                f"ChromaDB collection '{COLLECTION_NAME}' not found at {CHROMA_PATH}.\n"
                "Run `PYTHONPATH=. uv run python build_index.py` first."
            ) from exc

        # ----- game records (app_id → GameRecord) -----
        print(f"Loading game records from {db_path} …")
        rows = load_games_from_sqlite(db_path, MAX_GAMES)
        indexed_count = self.collection.count()
        if indexed_count != len(rows):
            raise RuntimeError(
                f"ChromaDB index contains {indexed_count} games, but SQLite has "
                f"{len(rows)} games for the current configuration. "
                "Run `PYTHONPATH=. uv run python build_index.py` to rebuild the index."
            )

        print("Loading review summaries …")
        review_summaries = get_review_summaries(db_path, MAX_GAMES)

        self._lookup: dict[str, GameRecord] = {}
        for app_id, raw in rows:
            raw["review_summary"] = review_summaries.get(str(app_id), "")
            self._lookup[app_id] = GameRecord(app_id=app_id, raw=raw)

        # ----- BM25 index (built once over all games) -----
        print("Building BM25 index …")
        # Keep a stable ordered list so BM25 result indices map to app_ids
        self._bm25_ids: list[str] = list(self._lookup.keys())
        corpus = [
            _tokenize(_build_bm25_text(self._lookup[aid].raw))
            for aid in self._bm25_ids
        ]
        try:
            self._bm25 = BM25Okapi(corpus)
            print(f"  BM25 index ready ({len(self._bm25_ids)} documents).")
        except Exception as exc:
            print(f"[WARNING] BM25 index build failed: {exc}. Will use dense-only retrieval.", file=sys.stderr)
            self._bm25 = None

        # ----- Ollama -----
        _check_ollama(LLM_MODEL)

        print("GameSearchEngine ready.")

    # ------------------------------------------------------------------
    # Public API — signatures and return shape are frozen
    # ------------------------------------------------------------------

    def search(self, query: str) -> dict[str, Any]:
        candidates = self.retrieve_candidates(query)
        ranked_matches = self.rank_candidates(query, candidates)
        results = [record.to_result(score) for record, score in ranked_matches]

        return {
            "matches": results,
            "answer": self.generate_answer(query, ranked_matches),
            "meta": {
                "retrieval_mode": "hybrid-bm25+embedding+cross-encoder+llm",
                "indexed_games": self.collection.count(),
                "features": [
                    "embeddings",
                    "bm25-hybrid",
                    "cross-encoder-reranking",
                    "llm-generation",
                    "review-augmented",
                ],
            },
        }

    def retrieve_candidates(self, query: str, k: int = 30) -> list[GameRecord]:
        """
        Hybrid retrieval: merge ChromaDB dense search and BM25 keyword search
        using Reciprocal Rank Fusion, then return the top-k GameRecords.
        Falls back to dense-only if BM25 is unavailable.
        """
        # --- Dense retrieval (ChromaDB) ---
        query_vec = self.embedder.encode(query).tolist()
        chroma_results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=min(k, self.collection.count()),
            include=["metadatas"],
        )
        dense_ids: list[str] = [
            meta["app_id"]
            for meta in chroma_results["metadatas"][0]
            if meta["app_id"] in self._lookup
        ]

        # --- BM25 retrieval ---
        bm25_ids: list[str] = []
        if self._bm25 is not None:
            try:
                tokens = _tokenize(query)
                bm25_scores = self._bm25.get_scores(tokens)
                # Get top-k indices sorted by score descending
                top_indices = sorted(
                    range(len(bm25_scores)),
                    key=lambda i: bm25_scores[i],
                    reverse=True,
                )[:k]
                bm25_ids = [
                    self._bm25_ids[i]
                    for i in top_indices
                    if bm25_scores[i] > 0  # skip zero-score results
                ]
            except Exception as exc:
                print(f"[WARNING] BM25 retrieval failed: {exc}. Using dense-only.", file=sys.stderr)

        # --- Merge with RRF ---
        ranked_lists = [dense_ids]
        if bm25_ids:
            ranked_lists.append(bm25_ids)

        fused_ids = _reciprocal_rank_fusion(ranked_lists)[:k]

        return [self._lookup[aid] for aid in fused_ids if aid in self._lookup]

    def rank_candidates(
        self, query: str, candidates: list[GameRecord], k: int = 10
    ) -> list[tuple[GameRecord, float]]:
        """
        Score every (query, game_text) pair with the cross-encoder.
        Raw logits are mapped through sigmoid → scores are always in (0, 1).
        Returns top-k sorted descending.
        """
        if not candidates:
            return []

        pairs = [(query, _build_game_text(record.raw)) for record in candidates]
        raw_scores: list[float] = self.reranker.predict(pairs).tolist()

        # Sigmoid: logit → (0, 1)
        sig_scores = [_sigmoid(s) for s in raw_scores]

        # Min-max scale to 0–10 so the best match ≈ 10, worst ≈ 0
        min_s = min(sig_scores)
        max_s = max(sig_scores)
        if max_s > min_s:
            normalised = [10.0 * (s - min_s) / (max_s - min_s) for s in sig_scores]
        else:
            normalised = [10.0 for _ in sig_scores]

        ranked = sorted(zip(candidates, normalised), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def generate_answer(self, query: str, ranked: list[tuple[GameRecord, float]]) -> str:
        """
        Call gemma2:2b via Ollama with store metadata + player review snippets
        for the top-5 games as context.
        """
        if not ranked:
            return "No games were found that match your request."

        game_lines: list[str] = []
        for i, (record, _score) in enumerate(ranked[:5], 1):
            genres = record.raw.get("genres", [])
            genres_str = ", ".join(genres) if isinstance(genres, list) else ""

            line = f"{i}. {record.name}"
            if genres_str:
                line += f" [{genres_str}]"
            if record.short_description:
                line += f"\n   Description: {record.short_description}"

            summary = record.review_summary
            if summary:
                condensed = _truncate_words(summary, 40)
                line += f"\n   Player feedback: {condensed}"

            game_lines.append(line)

        context = "\n\n".join(game_lines)

        prompt = (
            f"You are a helpful Steam game recommender.\n\n"
            f"A user is looking for: \"{query}\"\n\n"
            f"Here are the top matching games, with official descriptions and real player feedback:\n\n"
            f"{context}\n\n"
            f"Write a 2-3 sentence recommendation explaining which of these games best fit "
            f"the user's request and why. Use both the game descriptions AND player feedback "
            f"to support your picks. Mention what players liked or disliked when relevant."
        )

        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.message.content.strip()
