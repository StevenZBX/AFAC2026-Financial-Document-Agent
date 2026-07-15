"""Lightweight BM25 index over paragraph chunks."""

from __future__ import annotations

import json
import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

try:
    from .keyword_rules import normalize_text, tokenize_for_bm25
except ImportError:  # Allows running this file directly.
    from keyword_rules import normalize_text, tokenize_for_bm25


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    domain: str
    text: str
    metadata: dict


def load_chunks(path: str | Path) -> List[Chunk]:
    """Load chunks from a JSON file.

    Accepted formats:
    - a list of chunk dictionaries
    - {"chunks": [...]} or {"data": [...]}

    Common field aliases are supported to make this compatible with future
    preprocessing scripts.
    """

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        rows = payload.get("chunks")
        if rows is None:
            rows = payload.get("data")
    else:
        rows = payload
    if not isinstance(rows, list):
        raise ValueError(f"Unsupported chunks format in {path}")

    chunks: List[Chunk] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        text = normalize_text(
            row.get("text")
            or row.get("content")
            or row.get("chunk_text")
            or row.get("paragraph")
            or ""
        )
        if not text:
            continue
        doc_id = str(row.get("doc_id") or row.get("document_id") or row.get("source") or "")
        chunk_id = str(row.get("chunk_id") or row.get("id") or f"{doc_id}#{idx}")
        domain = str(row.get("domain") or row.get("category") or "")
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                domain=domain,
                text=text,
                metadata={k: v for k, v in row.items() if k not in {"text", "content", "chunk_text"}},
            )
        )
    return chunks


def _split_long_text(text: str, max_chars: int = 1400, overlap: int = 120) -> List[str]:
    text = normalize_text(text)
    if len(text) <= max_chars:
        return [text] if text else []

    parts: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            split_at = max(
                text.rfind("。", start, end),
                text.rfind("；", start, end),
                text.rfind("\n", start, end),
            )
            if split_at > start + max_chars * 0.5:
                end = split_at + 1
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return [part for part in parts if part]


def build_chunks_from_processed_roots(
    processed_roots: Sequence[str | Path],
    max_chars: int = 1400,
) -> List[dict]:
    """Convert processed document JSON directories into chunk dictionaries.

    Each processed root is expected to look like:
    root/
      document_index.json
      documents/*.json

    The generated chunk text includes title/hierarchy/keywords plus the segment
    body so keyword and BM25 retrieval can match both fields and正文内容.
    """

    rows: List[dict] = []
    for root in processed_roots:
        root = Path(root)
        index_path = root / "document_index.json"
        documents_dir = root / "documents"
        index = {}
        if index_path.exists():
            with index_path.open("r", encoding="utf-8") as f:
                index = json.load(f)
        doc_paths = sorted(documents_dir.glob("*.json"))
        for doc_path in doc_paths:
            with doc_path.open("r", encoding="utf-8") as f:
                doc = json.load(f)
            metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
            doc_id = str(doc.get("doc_id") or metadata.get("doc_id") or doc_path.stem)
            domain = str(doc.get("domain") or metadata.get("domain") or root.name)
            doc_meta = index.get(doc_id, {}) if isinstance(index, dict) else {}

            # Markdown preprocessing emits retrieval-ready chunks directly,
            # whereas the PDF pipeline emits pages containing segments.
            for chunk_idx, chunk in enumerate(doc.get("chunks", [])):
                if not isinstance(chunk, dict):
                    continue
                search_text = normalize_text(chunk.get("retrieval_text") or chunk.get("text", ""))
                if not search_text:
                    continue
                row = dict(chunk)
                row.update(
                    {
                        "chunk_id": str(chunk.get("chunk_id") or f"{doc_id}:c{chunk_idx}"),
                        "doc_id": str(chunk.get("doc_id") or doc_id),
                        "domain": str(chunk.get("domain") or domain),
                        "text": search_text,
                        "original_text": chunk.get("text", ""),
                        "source_path": (
                            doc_meta.get("source_path")
                            or doc_meta.get("source_markdown_path")
                            or metadata.get("source_markdown_path")
                        ),
                        "processed_path": str(doc_path),
                    }
                )
                rows.append(row)

            for page_obj in doc.get("pages", []):
                page = page_obj.get("page")
                page_type = page_obj.get("page_type")
                for seg_idx, segment in enumerate(page_obj.get("segments", [])):
                    body = normalize_text(segment.get("text", ""))
                    if not body:
                        continue
                    title = normalize_text(segment.get("title", ""))
                    hierarchy = segment.get("hierarchy") or {}
                    keywords = segment.get("keywords") or []
                    hierarchy_text = " ".join(
                        normalize_text(value) for value in hierarchy.values() if value
                    )
                    keyword_text = " ".join(normalize_text(value) for value in keywords if value)
                    search_text = normalize_text(
                        "\n".join(part for part in [hierarchy_text, title, keyword_text, body] if part)
                    )
                    for part_idx, part in enumerate(_split_long_text(search_text, max_chars=max_chars)):
                        rows.append(
                            {
                                "chunk_id": f"{doc_id}:p{page}:s{seg_idx}:c{part_idx}",
                                "doc_id": doc_id,
                                "domain": domain,
                                "text": part,
                                "page": page,
                                "page_type": page_type,
                                "title": title,
                                "hierarchy": hierarchy,
                                "keywords": keywords,
                                "source_path": doc_meta.get("source_path") or doc.get("path"),
                                "processed_path": str(doc_path),
                                "original_text": body,
                            }
                        )
    return rows


def write_chunks_json(chunks: Sequence[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"chunks": list(chunks)}, f, ensure_ascii=False, indent=2)


class BM25Index:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        tokenizer: Callable[[str], List[str]] = tokenize_for_bm25,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.chunks = list(chunks)
        self.tokenizer = tokenizer
        self.k1 = k1
        self.b = b
        self.doc_tokens: List[List[str]] = [tokenizer(chunk.text) for chunk in self.chunks]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_freq: Dict[str, int] = defaultdict(int)
        for tokens in self.doc_tokens:
            for term in set(tokens):
                self.doc_freq[term] += 1
        self.idf = {
            term: math.log(1 + (len(self.chunks) - freq + 0.5) / (freq + 0.5))
            for term, freq in self.doc_freq.items()
        }

    @classmethod
    def from_json(cls, path: str | Path) -> "BM25Index":
        return cls(load_chunks(path))

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with Path(path).open("rb") as f:
            return pickle.load(f)

    def search(
        self,
        query: str,
        top_k: int = 20,
        doc_ids: Optional[Iterable[str]] = None,
        domain: Optional[str] = None,
    ) -> List[dict]:
        allowed_doc_ids = {str(x) for x in doc_ids} if doc_ids else None
        query_tokens = self.tokenizer(query)
        scores: List[tuple[float, int]] = []
        for idx, chunk in enumerate(self.chunks):
            if allowed_doc_ids is not None and chunk.doc_id not in allowed_doc_ids:
                continue
            if domain and chunk.domain and chunk.domain != domain:
                continue
            score = self._score(query_tokens, idx)
            if score > 0:
                scores.append((score, idx))
        scores.sort(key=lambda item: item[0], reverse=True)
        return [self._result(idx, score, "bm25") for score, idx in scores[:top_k]]

    def _score(self, query_tokens: Sequence[str], doc_idx: int) -> float:
        if not query_tokens or not self.avgdl:
            return 0.0
        freqs = self.term_freqs[doc_idx]
        length = self.doc_len[doc_idx]
        score = 0.0
        for term in query_tokens:
            tf = freqs.get(term, 0)
            if not tf:
                continue
            denom = tf + self.k1 * (1 - self.b + self.b * length / self.avgdl)
            score += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / denom
        return score

    def _result(self, idx: int, score: float, source: str) -> dict:
        chunk = self.chunks[idx]
        return {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "domain": chunk.domain,
            "text": chunk.text,
            "score": round(float(score), 6),
            "source": source,
            "metadata": chunk.metadata,
        }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build or query a BM25 chunk index.")
    parser.add_argument("--chunks", default="chunks.json", help="Path to chunks.json")
    parser.add_argument("--save", help="Optional pickle output path")
    parser.add_argument("--query", help="Optional query text")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    index = BM25Index.from_json(args.chunks)
    if args.save:
        index.save(args.save)
    if args.query:
        print(json.dumps(index.search(args.query, top_k=args.top_k), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
