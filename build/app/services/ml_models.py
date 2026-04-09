"""
services/ml_models.py
Lightweight ML post-processing layer applied on top of Gemma's extracted beliefs.

Responsibilities:
  1. TF-IDF topic clustering — group similar user beliefs across the conversation.
  2. Conversation value scoring — composite score (0–10) for the scoring team.
  3. Enrich content tags from clustered themes.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.schemas import BeliefEntry, ConversationMessage, DownstreamScores

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SENTIMENT_WEIGHTS = {"positive": 1.0, "neutral": 0.5, "negative": 0.2}

# High-value topics get a bonus to the conversation score
HIGH_VALUE_TOPICS = {"health", "autonomy", "family", "identity", "community"}

# Similarity threshold for merging duplicate beliefs in clustering
SIMILARITY_THRESHOLD = 0.75


# ---------------------------------------------------------------------------
# Topic clustering
# ---------------------------------------------------------------------------


def cluster_beliefs(beliefs: list[BeliefEntry]) -> list[list[int]]:
    """
    Group belief indices by textual similarity using TF-IDF cosine similarity.
    Returns a list of clusters (each cluster is a list of indices into `beliefs`).
    """
    if len(beliefs) <= 1:
        return [[i] for i in range(len(beliefs))]

    texts = [f"{b.topic} {b.belief}" for b in beliefs]
    try:
        vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
        tfidf_matrix = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf_matrix)
    except Exception as exc:  # noqa: BLE001
        logger.warning("TF-IDF clustering failed: %s — returning singleton clusters", exc)
        return [[i] for i in range(len(beliefs))]

    visited = set()
    clusters: list[list[int]] = []

    for i in range(len(beliefs)):
        if i in visited:
            continue
        cluster = [i]
        visited.add(i)
        for j in range(i + 1, len(beliefs)):
            if j not in visited and sim_matrix[i, j] >= SIMILARITY_THRESHOLD:
                cluster.append(j)
                visited.add(j)
        clusters.append(cluster)

    return clusters


# ---------------------------------------------------------------------------
# Conversation value scoring
# ---------------------------------------------------------------------------


def compute_value_score(
    beliefs: list[BeliefEntry],
    messages: list[ConversationMessage],
) -> float:
    """
    Heuristic conversation value score (0–10).

    Factors:
      - Number of distinct high-value topics covered       (max 3 pts)
      - Average LLM confidence across beliefs              (max 2 pts)
      - Sentiment richness (mix of + / - / neutral)        (max 2 pts)
      - Conversation length relative to 10-message norm    (max 2 pts)
      - Presence of explicit personal disclosure           (max 1 pt)
    """
    if not beliefs:
        return 0.0

    # 1. High-value topic coverage
    covered = {b.topic for b in beliefs if b.topic in HIGH_VALUE_TOPICS}
    topic_score = min(len(covered), 3)

    # 2. Average confidence
    avg_confidence = float(np.mean([b.confidence for b in beliefs]))
    confidence_score = avg_confidence * 2.0

    # 3. Sentiment richness
    sentiments = {b.sentiment for b in beliefs}
    richness = len(sentiments)
    sentiment_score = min(richness, 2) * (2.0 / 2)

    # 4. Conversation length (relative to 10-message benchmark)
    user_msgs = [m for m in messages if m.ref_user_id != 1]
    length_score = min(len(user_msgs) / 10.0, 1.0) * 2.0

    # 5. Personal disclosure bonus (look for first-person identity keywords in beliefs)
    disclosure_keywords = {"i feel", "i believe", "i think", "i am", "my life", "i have", "i was"}
    has_disclosure = any(
        any(kw in b.belief.lower() for kw in disclosure_keywords) for b in beliefs
    )
    disclosure_score = 1.0 if has_disclosure else 0.0

    total = topic_score + confidence_score + sentiment_score + length_score + disclosure_score
    return round(min(total, 10.0), 2)


# ---------------------------------------------------------------------------
# Tag enrichment
# ---------------------------------------------------------------------------

TOPIC_TO_TAGS: dict[str, list[str]] = {
    "health": ["mental-health", "physical-wellbeing", "medical-support"],
    "family": ["family-support", "relationships", "caregiving"],
    "autonomy": ["independence", "self-care", "aging-in-place"],
    "technology": ["digital-literacy", "tech-support", "app-help"],
    "community": ["social-connection", "community-events", "peer-support"],
    "identity": ["personal-growth", "self-reflection", "life-story"],
    "leisure": ["hobbies", "entertainment", "creative-activities"],
    "spirituality": ["faith", "mindfulness", "wellness"],
    "finances": ["financial-planning", "benefits", "cost-of-living"],
    "other": ["general-interest"],
}


def enrich_tags(
    beliefs: list[BeliefEntry], existing_tags: list[str]
) -> list[str]:
    """
    Merge LLM-generated tags with topic-derived tags. De-duplicate and sort.
    """
    enriched = set(t.lower().strip() for t in existing_tags)
    for belief in beliefs:
        topic_tags = TOPIC_TO_TAGS.get(belief.topic, TOPIC_TO_TAGS["other"])
        enriched.update(topic_tags)
    return sorted(enriched)


# ---------------------------------------------------------------------------
# Main scoring entrypoint
# ---------------------------------------------------------------------------


def build_downstream_scores(
    beliefs: list[BeliefEntry],
    messages: list[ConversationMessage],
    meta: dict[str, Any],
) -> DownstreamScores:
    """
    Combine value scoring, topic clustering, and tag enrichment into
    DownstreamScores for the three consumer teams.
    """
    value_score = compute_value_score(beliefs, messages)
    enriched_tags = enrich_tags(beliefs, meta.get("content_tags", []))

    return DownstreamScores(
        conversation_value_score=value_score,
        dominant_theme=meta.get("dominant_theme", "unknown"),
        suggested_follow_up=meta.get("suggested_follow_up", ""),
        content_tags=enriched_tags,
    )
