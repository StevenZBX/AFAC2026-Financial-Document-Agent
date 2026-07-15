#!/usr/bin/env python3
"""Recursively convert MinerU Markdown collections to retrieval-ready JSON."""

import argparse
import hashlib
import html
import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Optional


@dataclass
class Config:
    source_root: Path
    output_root: Path
    max_chars: int = 1400
    min_chars: int = 350
    overlap_chars: int = 220


class TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self.row: Optional[list[str]] = None
        self.cell: Optional[list[str]] = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag == "tr":
            self.row = []
        elif tag in {"td", "th"}:
            self.cell = []

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.cell is not None:
            if self.row is not None:
                self.row.append(normalize("".join(self.cell)))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if any(self.row):
                self.rows.append(self.row)
            self.row = None


def normalize(text: str) -> str:
    text = html.unescape(text or "").replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"<sup>.*?</sup>", "", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def compact_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def table_text(match: re.Match[str]) -> str:
    parser = TableParser()
    parser.feed(match.group(0))
    return "\n".join(" | ".join(cell for cell in row if cell) for row in parser.rows)


def read_markdown(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig")
    text = re.sub(r"<table.*?</table>", table_text, text, flags=re.I | re.S)
    return normalize(text)


def safe_doc_id(relative_path: Path) -> str:
    # Regulatory text IDs must preserve original brackets and punctuation.
    if relative_path.parent.name == "txt":
        return relative_path.stem
    raw = relative_path.stem
    clean = re.sub(r"[^\w.-]+", "_", raw, flags=re.UNICODE).strip("_")
    if len(clean) <= 140:
        return clean.replace("/", "__")
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return clean[:125].replace("/", "__") + "__" + digest


def title_from(text: str, fallback: str) -> str:
    for line in text.splitlines():
        match = re.match(r"^#{1,6}\s+(.+)$", line.strip())
        if match:
            return normalize(match.group(1))
    for line in text.splitlines():
        line = normalize(line)
        if 2 <= len(line) <= 120:
            return line
    return fallback


def split_sections(text: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = []
    lines: list[str] = []
    heading = {"level": 0, "title": "document"}
    title_path: list[str] = []
    start = 1

    def flush(end: int) -> None:
        body = normalize("\n".join(lines))
        if body:
            sections.append({"heading": heading, "title_path": title_path[:], "line_start": start, "line_end": end, "text": body})

    for number, line in enumerate(text.splitlines(), 1):
        match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if match:
            flush(number - 1)
            heading = {"level": len(match.group(1)), "title": normalize(match.group(2))}
            stack = [item for item in stack if item["level"] < heading["level"]]
            stack.append(heading)
            title_path = [item["title"] for item in stack]
            lines = []
            start = number
        lines.append(line)
    flush(len(text.splitlines()))
    return sections


def tail(text: str, chars: int) -> str:
    value = normalize(text)
    return value[-chars:]


def split_section(section: dict[str, Any], config: Config) -> list[dict[str, Any]]:
    if compact_len(section["text"]) <= config.max_chars:
        return [section]
    units = [x.strip() for x in re.split(r"\n\s*\n|(?<=[。；;！？!?])", section["text"]) if x.strip()]
    expanded: list[str] = []
    for unit in units:
        if compact_len(unit) <= config.max_chars:
            expanded.append(unit)
        else:
            expanded.extend(unit[i:i + config.max_chars] for i in range(0, len(unit), config.max_chars))
    result: list[dict[str, Any]] = []
    current: list[str] = []
    for unit in expanded:
        candidate = normalize("\n".join([*current, unit]))
        if current and compact_len(candidate) > config.max_chars:
            value = normalize("\n".join(current))
            result.append({**section, "text": value})
            current = [tail(value, config.overlap_chars), unit]
        else:
            current.append(unit)
    if current:
        result.append({**section, "text": normalize("\n".join(current))})
    return result


def merge_short(chunks: list[dict[str, Any]], minimum: int) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for chunk in chunks:
        if compact_len(chunk["text"]) < minimum and merged:
            merged[-1]["text"] = normalize(merged[-1]["text"] + "\n" + chunk["text"])
            merged[-1]["line_end"] = chunk["line_end"]
        else:
            merged.append(chunk)
    if len(merged) > 1 and compact_len(merged[0]["text"]) < minimum:
        merged[1]["text"] = normalize(merged[0]["text"] + "\n" + merged[1]["text"])
        merged[1]["line_start"] = merged[0]["line_start"]
        merged.pop(0)
    return merged


def process_document(path: Path, domain_root: Path, domain: str, config: Config) -> dict[str, Any]:
    text = read_markdown(path)
    relative = path.relative_to(domain_root)
    doc_id = safe_doc_id(relative)
    title = title_from(text, path.stem)
    raw = []
    for section in split_sections(text):
        raw.extend(split_section(section, config))
    raw = merge_short(raw, config.min_chars)
    chunks = []
    previous = ""
    for index, item in enumerate(raw, 1):
        context = tail(previous, config.overlap_chars) if previous else ""
        heading_path = item["title_path"] or [title]
        retrieval = normalize("\n".join([f"文档标题：{title}", f"标题路径：{' > '.join(heading_path)}", f"上文：{context}" if context else "", item["text"]]))
        chunks.append({"chunk_id": f"{domain}::{doc_id}::chunk_{index:04d}", "doc_id": doc_id, "chunk_index": index, "title_path": heading_path, "section_title": item["heading"]["title"], "context_before": context, "text": item["text"], "retrieval_text": retrieval, "line_start": item["line_start"], "line_end": item["line_end"], "char_count": compact_len(item["text"])})
        previous = item["text"]
    metadata = {"doc_id": doc_id, "domain": domain, "title": title, "source_markdown_path": str(path), "source_relative_path": relative.as_posix()}
    return {"metadata": metadata, "chunk_count": len(chunks), "char_count": sum(x["char_count"] for x in chunks), "chunks": chunks}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run(config: Config) -> dict[str, Any]:
    domains = sorted(path for path in config.source_root.iterdir() if path.is_dir())
    summary: dict[str, Any] = {"source_root": str(config.source_root), "output_root": str(config.output_root), "domains": {}}
    for domain_root in domains:
        domain = domain_root.name
        files = sorted(domain_root.rglob("*.md"), key=lambda p: p.relative_to(domain_root).as_posix())
        documents: dict[str, Any] = {}
        all_chunks: list[dict[str, Any]] = []
        for path in files:
            document = process_document(path, domain_root, domain, config)
            meta = document["metadata"]
            output_path = config.output_root / domain / "documents" / f"{meta['doc_id']}.json"
            write_json(output_path, document)
            documents[meta["doc_id"]] = {**meta, "output_path": str(output_path), "chunk_count": document["chunk_count"], "char_count": document["char_count"]}
            all_chunks.extend(document["chunks"])
        process_data = {"domain": domain, "schema_version": "all_markdown_processed_v1", "source_dir": str(domain_root), "strategy": {"split_by": "markdown_heading_then_length", "recursive": True, "max_chars": config.max_chars, "min_chars": config.min_chars, "overlap_chars": config.overlap_chars, "retrieval_field": "retrieval_text"}, "document_count": len(documents), "chunk_count": len(all_chunks), "documents": documents, "chunks": all_chunks}
        write_json(config.output_root / domain / "document_index.json", documents)
        write_json(config.output_root / domain / "process_data.json", process_data)
        summary["domains"][domain] = {"document_count": len(documents), "chunk_count": len(all_chunks), "char_count": sum(x["char_count"] for x in all_chunks)}
        print(f"{domain}: {len(documents)} documents, {len(all_chunks)} chunks")
    summary["document_count"] = sum(x["document_count"] for x in summary["domains"].values())
    summary["chunk_count"] = sum(x["chunk_count"] for x in summary["domains"].values())
    write_json(config.output_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-chars", type=int, default=1400)
    parser.add_argument("--min-chars", type=int, default=350)
    parser.add_argument("--overlap-chars", type=int, default=220)
    args = parser.parse_args()
    run(Config(args.source_root.resolve(), args.output_root.resolve(), args.max_chars, args.min_chars, args.overlap_chars))


if __name__ == "__main__":
    main()
