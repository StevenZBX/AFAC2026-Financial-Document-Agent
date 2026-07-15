"""Hybrid evidence retrieval for A/B questions.

Strategy:
- A-list: use provided doc_ids as the primary document filter.
- B-list: recall candidate documents first, then retrieve chunks.
- Search each option separately, then merge with question-level retrieval.
- Combine BM25, keyword, clause/number/date exact matches.
- Deduplicate and truncate top-k evidence for downstream Qwen Agent context.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

try:
    from .bm25_index import (
        BM25Index,
        Chunk,
        build_chunks_from_processed_roots,
        load_chunks,
        write_chunks_json,
    )
    from .document_recall import DocumentRecall
    from .keyword_rules import (
        build_question_query,
        extract_signals,
        keyword_score,
        normalize_text,
        option_queries,
    )
except ImportError:
    from bm25_index import BM25Index, Chunk, build_chunks_from_processed_roots, load_chunks, write_chunks_json
    from document_recall import DocumentRecall
    from keyword_rules import (
        build_question_query,
        extract_signals,
        keyword_score,
        normalize_text,
        option_queries,
    )


class EvidenceRetriever:
    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        self.index = BM25Index(self.chunks)
        self.doc_recall = DocumentRecall(self.chunks)

    @classmethod
    def from_chunks_json(cls, path: str | Path) -> "EvidenceRetriever":
        return cls(load_chunks(path))

    def retrieve(
        self,
        question: dict,
        top_k: int = 8,
        bm25_k: int = 30,
        candidate_doc_k: int = 8,
    ) -> dict:
        domain = question.get("domain")
        doc_ids = self._candidate_doc_ids(question, candidate_doc_k)

        candidates: List[dict] = []
        query = build_question_query(question, include_options=True)
        candidates.extend(
            self._search_query(
                query=query,
                question=question,
                doc_ids=doc_ids,
                domain=domain,
                top_k=bm25_k,
                source_label="question",
            )
        )

        for option_label, option_query in option_queries(question).items():
            option_hits = self._search_query(
                query=option_query,
                question=question,
                doc_ids=doc_ids,
                domain=domain,
                top_k=max(5, bm25_k // 2),
                source_label=f"option_{option_label}",
            )
            for hit in option_hits:
                hit["option"] = option_label
            candidates.extend(option_hits)

        evidence = self._dedupe_rank(candidates, top_k=top_k)
        return {
            "qid": question.get("qid"),
            "domain": domain,
            "split": question.get("split"),
            "candidate_doc_ids": doc_ids,
            "evidence": evidence,
        }

    def _candidate_doc_ids(self, question: dict, candidate_doc_k: int) -> Optional[List[str]]:
        doc_ids = question.get("doc_ids") or []
        if doc_ids:
            return [str(doc_id) for doc_id in doc_ids]
        recalled = self.doc_recall.recall(question, top_k=candidate_doc_k)
        return [item["doc_id"] for item in recalled] or None

    def _search_query(
        self,
        query: str,
        question: dict,
        doc_ids: Optional[Iterable[str]],
        domain: Optional[str],
        top_k: int,
        source_label: str,
    ) -> List[dict]:
        signals = extract_signals(query, domain)
        bm25_hits = self.index.search(query, top_k=top_k, doc_ids=doc_ids, domain=domain)

        allowed_doc_ids = {str(x) for x in doc_ids} if doc_ids else None
        keyword_hits: List[dict] = []
        for chunk in self.chunks:
            if allowed_doc_ids is not None and chunk.doc_id not in allowed_doc_ids:
                continue
            if domain and chunk.domain and chunk.domain != domain:
                continue
            score = keyword_score(chunk.text, signals, domain)
            if score <= 0:
                continue
            keyword_hits.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "domain": chunk.domain,
                    "text": chunk.text,
                    "score": round(score, 6),
                    "source": "keyword",
                    "metadata": chunk.metadata,
                }
            )
        keyword_hits.sort(key=lambda item: item["score"], reverse=True)

        merged: List[dict] = []
        for hit in [*bm25_hits, *keyword_hits[:top_k]]:
            hit = dict(hit)
            hit["query_source"] = source_label
            hit["final_score"] = self._combined_score(hit, signals)
            hit["matched_terms"] = self._matched_terms(hit["text"], signals.all_terms())
            merged.append(hit)
        return merged

    def _combined_score(self, hit: dict, signals) -> float:
        score = float(hit.get("score", 0.0))
        if hit.get("source") == "bm25":
            score *= 1.0
        elif hit.get("source") == "keyword":
            score *= 1.2
        text = hit.get("text", "")
        score += min(len(self._matched_terms(text, signals.clauses)), 3) * 3.0
        score += min(len(self._matched_terms(text, signals.numbers)), 5) * 2.0
        score += min(len(self._matched_terms(text, signals.dates)), 3) * 1.0
        return round(score, 6)

    @staticmethod
    def _matched_terms(text: str, terms: Sequence[str]) -> List[str]:
        compact_text = normalize_text(text).replace(" ", "")
        matched = []
        for term in terms:
            compact_term = normalize_text(term).replace(" ", "")
            if compact_term and compact_term in compact_text:
                matched.append(term)
        return matched[:20]

    @staticmethod
    def _dedupe_rank(candidates: Sequence[dict], top_k: int) -> List[dict]:
        best: Dict[str, dict] = {}
        for item in candidates:
            text = normalize_text(item.get("text", ""))
            key = item.get("chunk_id") or f"{item.get('doc_id')}:{text[:80]}"
            if key not in best or item.get("final_score", 0) > best[key].get("final_score", 0):
                best[key] = item

        ranked = sorted(best.values(), key=lambda item: item.get("final_score", 0), reverse=True)
        output = []
        seen_text = set()
        for item in ranked:
            text = normalize_text(item.get("text", ""))
            text_key = text[:120]
            if text_key in seen_text:
                continue
            seen_text.add(text_key)
            output.append(
                {
                    "doc_id": item.get("doc_id"),
                    "chunk_id": item.get("chunk_id"),
                    "domain": item.get("domain"),
                    "text": normalize_text(item.get("metadata", {}).get("original_text") or text)[:1200],
                    "score": item.get("final_score", item.get("score")),
                    "source": item.get("source"),
                    "query_source": item.get("query_source"),
                    "option": item.get("option"),
                    "matched_terms": item.get("matched_terms", []),
                    "metadata": item.get("metadata", {}),
                }
            )
            if len(output) >= top_k:
                break
        return output


def load_questions(path: str | Path) -> List[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return payload.get("questions") or payload.get("data") or []
    raise ValueError(f"Unsupported question format: {path}")


def generate_evidence(
    chunks_path: str | Path,
    question_paths: Sequence[str | Path],
    output_path: str | Path,
    top_k: int = 15,
) -> List[dict]:
    retriever = EvidenceRetriever.from_chunks_json(chunks_path)
    results: List[dict] = []
    for question_path in question_paths:
        for question in load_questions(question_path):
            results.append(retriever.retrieve(question, top_k=top_k))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evidence.json for questions.")
    parser.add_argument("--chunks", default="retriever/chunks.json", help="Path to chunks.json")
    parser.add_argument(
        "--processed-roots",
        nargs="*",
        help="Processed data roots containing document_index.json and documents/*.json",
    )
    parser.add_argument(
        "--questions",
        nargs="+",
        required=True,
        help="Question JSON files",
    )
    parser.add_argument("--output", default="retriever/evidence.json")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--chunk-max-chars", type=int, default=1400)
    args = parser.parse_args()

    if args.processed_roots:
        chunks = build_chunks_from_processed_roots(
            args.processed_roots,
            max_chars=args.chunk_max_chars,
        )
        if not chunks:
            roots = ", ".join(str(root) for root in args.processed_roots)
            raise ValueError(
                "No chunks found in the processed roots. Expected documents/*.json "
                f"containing either chunks[] or pages[].segments[]: {roots}"
            )
        write_chunks_json(chunks, args.chunks)
        print(f"wrote {len(chunks)} chunks to {args.chunks}")

    if not Path(args.chunks).exists():
        raise FileNotFoundError(
            f"{args.chunks} not found. Provide --processed-roots or generate chunks before retrieval."
        )
    results = generate_evidence(args.chunks, args.questions, args.output, top_k=args.top_k)
    print(f"wrote evidence for {len(results)} questions to {args.output}")


if __name__ == "__main__":
    main()

