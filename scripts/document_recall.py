"""Candidate document recall for B-list questions without doc_ids."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

try:
    from .bm25_index import Chunk
    from .keyword_rules import build_question_query, extract_signals, keyword_score, normalize_text
except ImportError:
    from bm25_index import Chunk
    from keyword_rules import build_question_query, extract_signals, keyword_score, normalize_text


class DocumentRecall:
    """Recall likely documents before paragraph retrieval.

    The scorer aggregates chunk-level keyword matches into doc-level scores.
    It favors exact entity/amount/date/clause hits and can be restricted by
    domain when the question provides one.
    """

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        self.doc_text: Dict[str, str] = defaultdict(str)
        self.doc_domain: Dict[str, str] = {}
        for chunk in self.chunks:
            self.doc_text[chunk.doc_id] += " " + chunk.text[:800]
            if chunk.domain:
                self.doc_domain[chunk.doc_id] = chunk.domain

    def recall(self, question: dict, top_k: int = 8) -> List[dict]:
        domain = question.get("domain")
        query = build_question_query(question, include_options=True)
        signals = extract_signals(query, domain)
        scores: List[dict] = []
        explicit_doc_ids = question.get("doc_ids") or []
        if explicit_doc_ids:
            return [
                {"doc_id": str(doc_id), "score": 999.0, "source": "provided_doc_id", "domain": domain}
                for doc_id in explicit_doc_ids
            ]

        for doc_id, text in self.doc_text.items():
            doc_domain = self.doc_domain.get(doc_id, "")
            if domain and doc_domain and doc_domain != domain:
                continue
            score = keyword_score(text, signals, domain)
            if doc_id and doc_id in query:
                score += 5.0
            if score > 0:
                scores.append(
                    {
                        "doc_id": doc_id,
                        "score": round(score, 6),
                        "source": "document_keyword_recall",
                        "domain": doc_domain,
                    }
                )
        scores.sort(key=lambda item: item["score"], reverse=True)
        return scores[:top_k]


def recall_doc_ids(chunks: Sequence[Chunk], question: dict, top_k: int = 8) -> List[str]:
    return [item["doc_id"] for item in DocumentRecall(chunks).recall(question, top_k=top_k)]

