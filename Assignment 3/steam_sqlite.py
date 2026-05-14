from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def load_games_from_sqlite(
    db_path: Path, limit: int | None = None
) -> list[tuple[str, dict[str, Any]]]:
    query = """
        SELECT
            appid,
            name,
            short_description,
            about_the_game,
            detailed_description,
            release_date,
            price,
            header_image,
            windows,
            mac,
            linux,
            developers_json,
            publishers_json,
            categories_json,
            genres_json,
            tags_json
        FROM games
        WHERE name IS NOT NULL AND TRIM(name) != ''
        ORDER BY appid
    """
    params: tuple[Any, ...] = ()
    if limit is not None:
        query += "\n        LIMIT ?"
        params = (limit,)

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query, params).fetchall()

    records: list[tuple[str, dict[str, Any]]] = []
    for row in rows:
        raw = {
            "name": row["name"] or "Unknown title",
            "short_description": row["short_description"] or "",
            "about_the_game": row["about_the_game"] or "",
            "detailed_description": row["detailed_description"] or "",
            "release_date": row["release_date"],
            "price": row["price"],
            "header_image": row["header_image"],
            "windows": bool(row["windows"]),
            "mac": bool(row["mac"]),
            "linux": bool(row["linux"]),
            "developers": _load_json_value(row["developers_json"], []),
            "publishers": _load_json_value(row["publishers_json"], []),
            "categories": _load_json_value(row["categories_json"], []),
            "genres": _load_json_value(row["genres_json"], []),
            "tags": _load_json_value(row["tags_json"], {}),
        }
        records.append((str(row["appid"]), raw))

    return records


def get_review_summaries(
    db_path: Path, limit: int | None = 5000
) -> dict[str, str]:
    """
    Return a dict mapping app_id (str) → review_summary (str).

    For each game we select:
      - top 3 most-helpful English positive reviews  (voted_up = 1)
      - top 1 most-helpful English negative review   (voted_up = 0)
    ranked by votes_up DESC.  Reviews shorter than 20 characters are skipped.
    Each selected review is truncated to 150 words.

    A single window-function query is used so we never pull more than
    ~4 rows per game into Python.
    """
    game_filter = """
              AND appid IN (
                  SELECT appid
                  FROM games
                  WHERE name IS NOT NULL AND TRIM(name) != ''
              )
    """
    params: tuple[Any, ...] = ()
    if limit is not None:
        game_filter = """
              AND appid IN (
                  SELECT appid
                  FROM games
                  WHERE name IS NOT NULL AND TRIM(name) != ''
                  ORDER BY appid
                  LIMIT ?
              )
        """
        params = (limit,)

    query = f"""
        SELECT appid, review, voted_up
        FROM (
            SELECT
                appid,
                review,
                voted_up,
                ROW_NUMBER() OVER (
                    PARTITION BY appid, voted_up
                    ORDER BY votes_up DESC
                ) AS rn
            FROM reviews
            WHERE language = 'english'
              AND review IS NOT NULL
              AND LENGTH(TRIM(review)) > 20
{game_filter}
        )
        WHERE (voted_up = 1 AND rn <= 3)
           OR (voted_up = 0 AND rn <= 1)
        ORDER BY appid, voted_up DESC
    """

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()

    # Group by app_id
    from collections import defaultdict
    positives: dict[str, list[str]] = defaultdict(list)
    negatives: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        app_id = str(row["appid"])
        snippet = _truncate_words(row["review"].strip(), 150)
        if row["voted_up"]:
            positives[app_id].append(snippet)
        else:
            negatives[app_id].append(snippet)

    summaries: dict[str, str] = {}
    all_ids = set(positives.keys()) | set(negatives.keys())
    for app_id in all_ids:
        parts: list[str] = []
        pos = positives.get(app_id, [])
        neg = negatives.get(app_id, [])
        if pos:
            parts.append("What players like: " + " ".join(pos))
        if neg:
            parts.append("What players dislike: " + " ".join(neg))
        summaries[app_id] = " ".join(parts)

    return summaries


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "…"


def _load_json_value(payload: str | None, default: Any) -> Any:
    if not payload:
        return default

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return default
