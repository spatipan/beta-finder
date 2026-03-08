"""
feedback.py — User feedback collection for CLIP fine-tuning

Stores (query, result, thumbs_up/down) pairs in data/feedback.json.
These pairs are used as training data for Phase 2 CLIP fine-tuning.

Schema per entry:
  id              — uuid4 string
  timestamp       — ISO 8601
  session_id      — short hash per search session
  query_filename  — uploaded photo temp path
  result_filename — matched image local path
  result_shortcode— Instagram post shortcode
  result_gym      — gym key
  result_rank     — position in results (1-indexed)
  result_score    — cosine similarity score
  feedback        — "positive" | "negative"
  model_used      — CLIP model name
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

from src.config import get_path
from src.logger import setup_logger

log = setup_logger(__name__)

FEEDBACK_GOAL = 200  # Phase 2 target: 200 labeled pairs for fine-tuning


def _feedback_path() -> Path:
    return get_path("data_dir").parent / "feedback.json"


def load_feedback() -> list:
    """Load all feedback entries from feedback.json"""
    path = _feedback_path()
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        log.warning("feedback.json unreadable, starting fresh")
        return []


def save_feedback(
    result: dict,
    feedback: str,
    session_id: str,
    query_filename: str,
    model_used: str = "ViT-B-32",
) -> None:
    """
    Append one feedback entry to feedback.json.

    Args:
        result: A result dict from search() — must have filename, shortcode,
                gym, rank, score keys.
        feedback: "positive" (same route) or "negative" (different route).
        session_id: Short identifier for the search session.
        query_filename: Path to the uploaded query image.
        model_used: CLIP model name used for this search.
    """
    entries = load_feedback()

    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": session_id,
        "query_filename": str(query_filename),
        "result_filename": result.get("filename", ""),
        "result_shortcode": result.get("filename", "").split("/")[-1].split("_")[0],
        "result_gym": result.get("gym", ""),
        "result_rank": result.get("rank", 0),
        "result_score": round(float(result.get("score", 0)), 6),
        "feedback": feedback,
        "model_used": model_used,
    }

    entries.append(entry)

    path = _feedback_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    log.info(f"Feedback saved: {feedback} for rank {entry['result_rank']} ({entry['result_shortcode']})")


def get_feedback_stats() -> dict:
    """
    Return counts of feedback entries.

    Returns:
        {total, positive, negative, goal, pct_complete}
    """
    entries = load_feedback()
    positive = sum(1 for e in entries if e.get("feedback") == "positive")
    negative = sum(1 for e in entries if e.get("feedback") == "negative")
    total = len(entries)
    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "goal": FEEDBACK_GOAL,
        "pct_complete": min(100.0, round(100 * total / FEEDBACK_GOAL, 1)),
    }


def get_training_pairs() -> list:
    """
    Build (anchor, positive, negative) triplets from collected feedback.

    Strategy:
      - For each session: group by session_id
      - Positive pairs: query × thumbs-up results
      - Negative pairs: query × thumbs-down results
      - Triplet: (query, any positive result, any negative result)

    Returns list of dicts:
      {anchor, positive, negative}  — all are local image file paths
    """
    entries = load_feedback()

    # Group by session_id
    sessions: dict[str, dict] = {}
    for e in entries:
        sid = e["session_id"]
        if sid not in sessions:
            sessions[sid] = {"query": e["query_filename"], "positive": [], "negative": []}
        if e["feedback"] == "positive":
            sessions[sid]["positive"].append(e["result_filename"])
        elif e["feedback"] == "negative":
            sessions[sid]["negative"].append(e["result_filename"])

    triplets = []
    for sid, data in sessions.items():
        query = data["query"]
        for pos in data["positive"]:
            for neg in data["negative"]:
                triplets.append({
                    "anchor": query,
                    "positive": pos,
                    "negative": neg,
                })

    log.info(f"Training triplets: {len(triplets)} from {len(sessions)} sessions")
    return triplets
