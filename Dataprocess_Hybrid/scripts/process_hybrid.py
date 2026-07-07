import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pypdf import PdfReader

try:
    import pdfplumber
except ImportError:  # pragma: no cover - handled at runtime
    pdfplumber = None


@dataclass
class ProcessConfig:
    raw_dir: Path
    question_file: Path
    output_dir: Path
    prefer_txt: bool = True
    table_mode: str = "append_markdown"
    low_text_threshold: int = 30
    ocr_mode: str = "flag_only"
    limit_docs: Optional[int] = None

    @property
    def split_name(self) -> str:
        return self.question_file.parent.name

    @property
    def domain_name(self) -> str:
        suffix = "_questions"
        stem = self.question_file.stem
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
        return stem

    @property
    def output_subdir(self) -> Path:
        return self.output_dir / self.split_name / self.domain_name

    def validate(self) -> None:
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"raw_dir not found: {self.raw_dir}")
        if not self.question_file.exists():
            raise FileNotFoundError(f"question_file not found: {self.question_file}")
        self.output_subdir.mkdir(parents=True, exist_ok=True)


class Processor:
    def __init__(self, config: ProcessConfig) -> None:
        self.config = config
        self.heading_patterns = [
            ("chapter", re.compile(r"^(第[一二三四五六七八九十百零〇0-9]+章)\s*(.+)?$")),
            ("section", re.compile(r"^(第[一二三四五六七八九十百零〇0-9]+节)\s*(.+)?$")),
            ("article", re.compile(r"^(第[一二三四五六七八九十百零〇0-9]+条)\s*(.+)?$")),
            ("part", re.compile(r"^([一二三四五六七八九十]+、)\s*(.+)$")),
            ("item", re.compile(r"^(（[一二三四五六七八九十]+）)\s*(.+)$")),
            ("subitem", re.compile(r"^(\d+[、\.])\s*(.+)$")),
        ]

    def _split_lines(self, text: str) -> List[str]:
        return [line.strip() for line in text.splitlines() if line.strip()]

    def _join_lines(self, lines: List[str]) -> str:
        return "\n".join(lines).strip()

    def _is_page_marker(self, line: str) -> bool:
        normalized = re.sub(r"\s+", "", line)
        return bool(
            re.fullmatch(r"\d+(?:-\d+){1,}", normalized)
            or re.fullmatch(r"[IVXLCivxlc]+", normalized)
            or re.fullmatch(r"第?\d+页", normalized)
        )

    def _is_generic_heading(self, line: str) -> bool:
        normalized = re.sub(r"\s+", "", line)
        if not normalized or len(normalized) > 20:
            return False
        if self._is_page_marker(normalized):
            return False
        if re.search(r"[，。；：,.!?]", normalized):
            return False
        if re.search(r"\d", normalized):
            return False
        return bool(re.fullmatch(r"[\u4e00-\u9fffA-Za-z（）()“”‘’《》\-]+", normalized))

    def clean_text(self, text: str) -> str:
        text = text.replace("\u3000", " ")
        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _has_meaningful_text(self, text: str) -> bool:
        normalized = re.sub(r"\s+", "", text)
        return len(normalized) >= 2

    def _compact_cell(self, value: Any) -> str:
        return self.clean_text(str(value or "")).replace("\n", " ")

    def _compact_table(self, table: List[List[Any]]) -> List[List[str]]:
        rows = [[self._compact_cell(cell) for cell in row] for row in table if row]
        rows = [row for row in rows if any(cell for cell in row)]
        if not rows:
            return []

        max_cols = max(len(row) for row in rows)
        padded = [row + [""] * (max_cols - len(row)) for row in rows]
        keep_cols = [
            idx
            for idx in range(max_cols)
            if any(row[idx] for row in padded)
        ]
        compact = [[row[idx] for idx in keep_cols] for row in padded]

        merged: List[List[str]] = []
        for row in compact:
            non_empty = [cell for cell in row if cell]
            if (
                merged
                and len(non_empty) == 1
                and len(row) > 1
                and not row[0]
            ):
                merged[-1][0] = (merged[-1][0] + " " + non_empty[0]).strip()
            else:
                merged.append(row)
        return merged

    def _table_to_markdown(self, table: List[List[str]]) -> str:
        if not table:
            return ""
        width = max(len(row) for row in table)
        rows = [row + [""] * (width - len(row)) for row in table]
        header = rows[0]
        sep = ["---"] * width
        body = rows[1:]

        def fmt(row: List[str]) -> str:
            escaped = [cell.replace("|", "\\|") for cell in row]
            return "| " + " | ".join(escaped) + " |"

        return "\n".join([fmt(header), fmt(sep), *[fmt(row) for row in body]])

    def _extract_pdfplumber_page_assets(self, path: Path) -> Dict[int, Dict[str, Any]]:
        assets: Dict[int, Dict[str, Any]] = {}
        if pdfplumber is None:
            return assets

        try:
            with pdfplumber.open(str(path)) as pdf:
                for page_index, page in enumerate(pdf.pages, start=1):
                    raw_tables = page.extract_tables() or []
                    tables = []
                    markdown_parts = []
                    for table_index, raw_table in enumerate(raw_tables):
                        compact = self._compact_table(raw_table)
                        markdown = self._table_to_markdown(compact)
                        if not markdown:
                            continue
                        tables.append(
                            {
                                "table_index": table_index,
                                "rows": compact,
                                "markdown": markdown,
                            }
                        )
                        markdown_parts.append(
                            f"[TABLE {table_index + 1} START]\n{markdown}\n[TABLE {table_index + 1} END]"
                        )
                    assets[page_index] = {
                        "tables": tables,
                        "table_markdown": "\n\n".join(markdown_parts),
                        "image_count": len(page.images or []),
                    }
        except Exception as exc:
            assets[-1] = {"error": f"{type(exc).__name__}: {exc}"}
        return assets

    def _diagnose_page(
        self,
        text: str,
        table_count: int,
        image_count: int,
        parse_warnings: List[str],
    ) -> Dict[str, Any]:
        compact_len = len(re.sub(r"\s+", "", text))
        has_many_numbers = len(re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?|[0-9]+%", text)) >= 12
        table_keywords = bool(
            re.search(
                r"单位[:：]|资产负债表|利润表|现金流量表|主要会计数据|财务指标|发行概况|保险责任|责任免除",
                text,
            )
        )
        needs_ocr = compact_len < self.config.low_text_threshold and image_count > 0
        page_type = "text"
        if table_count:
            page_type = "table"
        elif needs_ocr:
            page_type = "needs_ocr"
        elif image_count:
            page_type = "image_or_chart"
        elif compact_len < self.config.low_text_threshold:
            page_type = "low_text"

        risk_reasons = []
        if table_count:
            risk_reasons.append("table_detected")
        if image_count:
            risk_reasons.append("image_or_chart_detected")
        if needs_ocr:
            risk_reasons.append("low_text_with_image")
        if has_many_numbers:
            risk_reasons.append("many_numbers")
        if table_keywords:
            risk_reasons.append("table_keywords")
        risk_reasons.extend(parse_warnings)

        return {
            "text_len": compact_len,
            "table_count": table_count,
            "image_count": image_count,
            "has_many_numbers": has_many_numbers,
            "table_keywords": table_keywords,
            "needs_ocr": needs_ocr,
            "ocr_mode": self.config.ocr_mode,
            "page_type_guess": page_type,
            "risk_reasons": risk_reasons,
        }

    def read_txt(self, path: Path) -> str:
        return self.clean_text(path.read_text(encoding="utf-8"))

    def read_pdf_pages(self, path: Path) -> List[Dict[str, Any]]:
        reader = PdfReader(str(path))
        assets = self._extract_pdfplumber_page_assets(path)
        asset_error = assets.get(-1, {}).get("error")
        pages: List[Dict[str, Any]] = []

        for idx, page in enumerate(reader.pages, start=1):
            extracted = page.extract_text() or ""
            cleaned = self.clean_text(extracted)
            page_assets = assets.get(idx, {})
            table_markdown = page_assets.get("table_markdown", "")
            parse_warnings = []
            if asset_error:
                parse_warnings.append(f"pdfplumber_error:{asset_error}")

            final_text = cleaned
            if self.config.table_mode == "append_markdown" and table_markdown:
                final_text = self.clean_text(f"{cleaned}\n\n[EXTRACTED_TABLES]\n{table_markdown}")

            diagnosis = self._diagnose_page(
                final_text,
                table_count=len(page_assets.get("tables", [])),
                image_count=int(page_assets.get("image_count", 0)),
                parse_warnings=parse_warnings,
            )
            pages.append(
                {
                    "page": idx,
                    "text": final_text,
                    "base_text": cleaned,
                    "tables": page_assets.get("tables", []),
                    "diagnosis": diagnosis,
                }
            )
        return pages

    def extract_pages(self, path: Path) -> List[Dict[str, Any]]:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            text = self.read_txt(path)
            diagnosis = self._diagnose_page(text, table_count=0, image_count=0, parse_warnings=[])
            return [{"page": 1, "text": text, "base_text": text, "tables": [], "diagnosis": diagnosis}]
        if suffix == ".pdf":
            return self.read_pdf_pages(path)
        raise ValueError(f"Unsupported file type: {path}")

    def _candidate_files(self, domain_dir: Path) -> List[Path]:
        return [
            path
            for path in sorted(domain_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in {".txt", ".pdf"}
        ]

    def _select_preferred_sources(self, paths: List[Path]) -> Dict[str, Path]:
        selected: Dict[str, Path] = {}
        for path in paths:
            doc_id = path.stem
            current = selected.get(doc_id)
            if current is None:
                selected[doc_id] = path
                continue
            current_is_pdf = current.suffix.lower() == ".pdf"
            candidate_is_txt = path.suffix.lower() == ".txt"
            if self.config.prefer_txt and current_is_pdf and candidate_is_txt:
                selected[doc_id] = path
        return selected

    def _collect_repeated_edge_lines(
        self,
        pages: List[List[str]],
        from_start: bool,
        window: int = 3,
    ) -> set[str]:
        counts: Dict[str, int] = {}
        if len(pages) < 3:
            return set()

        for lines in pages[1:]:
            edge_lines = lines[:window] if from_start else lines[-window:]
            for line in edge_lines:
                counts[line] = counts.get(line, 0) + 1

        min_count = max(2, len(pages) // 3)
        return {
            line
            for line, count in counts.items()
            if count >= min_count and len(line) >= 6
        }

    def _strip_page_edges(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        page_lines = [self._split_lines(page["text"]) for page in pages]
        repeated_headers = self._collect_repeated_edge_lines(page_lines, from_start=True)
        repeated_footers = self._collect_repeated_edge_lines(page_lines, from_start=False)

        cleaned_pages: List[Dict[str, Any]] = []
        for page, lines in zip(pages, page_lines):
            while lines and (lines[0] in repeated_headers or self._is_page_marker(lines[0])):
                lines.pop(0)
            while lines and (lines[-1] in repeated_footers or self._is_page_marker(lines[-1])):
                lines.pop()
            cleaned = dict(page)
            cleaned["text"] = self._join_lines(lines)
            cleaned_pages.append(cleaned)
        return cleaned_pages

    def _is_toc_line(self, line: str) -> bool:
        normalized = re.sub(r"\s+", "", line)
        return bool(
            re.search(r"[\.。…]{3,}", line) and re.search(r"\d+$", normalized)
            or re.search(r"(第[一二三四五六七八九十百零〇0-9]+[章节条]|[一二三四五六七八九十]+、).+\d+$", normalized)
        )

    def _is_toc_page(self, lines: List[str]) -> bool:
        if not lines:
            return False
        toc_hits = sum(1 for line in lines if self._is_toc_line(line))
        has_catalog_title = any(line == "目录" for line in lines[:5])
        return has_catalog_title or toc_hits >= 5

    def _parse_toc_entries(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for page in pages:
            lines = self._split_lines(page["text"])
            if not self._is_toc_page(lines):
                continue
            for line in lines:
                if line == "目录" or not self._is_toc_line(line):
                    continue
                normalized = re.sub(r"\s+", " ", line).strip()
                match = re.search(r"(.+?)[\.。…\s]+(\d+)$", normalized)
                if match:
                    entries.append(
                        {
                            "title": match.group(1).strip(),
                            "target_page": int(match.group(2)),
                            "source_page": page["page"],
                        }
                    )
        return entries

    def _match_heading(self, line: str) -> Optional[tuple[str, str]]:
        for level, pattern in self.heading_patterns:
            match = pattern.match(line)
            if not match:
                continue
            prefix = match.group(1).strip()
            suffix = match.group(2).strip() if match.lastindex and match.lastindex > 1 and match.group(2) else ""
            return level, f"{prefix} {suffix}".strip()
        return None

    def _context_keywords(self, context: Dict[str, str]) -> List[str]:
        return [value for value in context.values() if value]

    def _build_page_segments(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        context = {
            "heading": "",
            "chapter": "",
            "section": "",
            "article": "",
            "part": "",
            "item": "",
            "subitem": "",
        }
        segmented_pages: List[Dict[str, Any]] = []

        for page in pages:
            if not self._has_meaningful_text(page["text"]):
                segmented_pages.append(
                    {
                        "page": page["page"],
                        "segments": [],
                        "page_type": "empty",
                        "diagnosis": page.get("diagnosis", {}),
                        "tables": page.get("tables", []),
                    }
                )
                continue

            lines = self._split_lines(page["text"])
            if self._is_toc_page(lines):
                segmented_pages.append(
                    {
                        "page": page["page"],
                        "segments": [],
                        "page_type": "toc",
                        "diagnosis": page.get("diagnosis", {}),
                        "tables": page.get("tables", []),
                    }
                )
                continue

            segments: List[Dict[str, Any]] = []
            current_lines: List[str] = []
            current_title = ""
            current_context = context.copy()

            def flush_segment() -> None:
                if not current_lines:
                    return
                segment_text = self._join_lines(current_lines)
                if not segment_text:
                    return
                segments.append(
                    {
                        "title": current_title,
                        "text": segment_text,
                        "hierarchy": {k: v for k, v in current_context.items() if v},
                        "keywords": self._context_keywords(current_context),
                    }
                )

            for line in lines:
                heading = self._match_heading(line)
                if not heading and not current_lines and self._is_generic_heading(line):
                    heading = ("heading", line.strip())
                if heading:
                    flush_segment()
                    level, title = heading
                    if level == "heading":
                        context.update({"heading": title})
                    elif level == "chapter":
                        context.update({"chapter": title, "section": "", "article": "", "part": "", "item": "", "subitem": ""})
                    elif level == "section":
                        context.update({"section": title, "article": "", "part": "", "item": "", "subitem": ""})
                    elif level == "article":
                        context.update({"article": title, "part": "", "item": "", "subitem": ""})
                    elif level == "part":
                        context.update({"part": title, "item": "", "subitem": ""})
                    elif level == "item":
                        context.update({"item": title, "subitem": ""})
                    elif level == "subitem":
                        context.update({"subitem": title})

                    current_context = context.copy()
                    current_title = title
                    current_lines = [line]
                    continue

                if not current_lines:
                    current_context = context.copy()
                    current_title = next(
                        (value for value in reversed(list(current_context.values())) if value),
                        "",
                    )
                current_lines.append(line)

            flush_segment()
            segmented_pages.append(
                {
                    "page": page["page"],
                    "segments": segments,
                    "page_type": "content",
                    "diagnosis": page.get("diagnosis", {}),
                    "tables": page.get("tables", []),
                }
            )

        return segmented_pages

    def build_document_records(self, domain: str) -> Dict[str, Dict[str, Any]]:
        domain_dir = self.config.raw_dir / domain
        if not domain_dir.exists():
            raise FileNotFoundError(f"domain raw directory not found: {domain_dir}")

        source_paths = self._candidate_files(domain_dir)
        selected_sources = self._select_preferred_sources(source_paths)
        if self.config.limit_docs is not None:
            selected_sources = dict(list(selected_sources.items())[: self.config.limit_docs])

        documents: Dict[str, Dict[str, Any]] = {}
        for doc_id, path in selected_sources.items():
            pages = self.extract_pages(path)
            pages = self._strip_page_edges(pages)
            toc = self._parse_toc_entries(pages)
            structured_pages = self._build_page_segments(pages)
            documents[doc_id] = {
                "doc_id": doc_id,
                "domain": domain,
                "path": str(path),
                "page_count": len(structured_pages),
                "toc": toc,
                "pages": structured_pages,
                "char_count": sum(
                    len(segment["text"])
                    for page in structured_pages
                    for segment in page["segments"]
                ),
                "source_type": path.suffix.lower().lstrip("."),
                "parse_profile": {
                    "base_text_engine": "pypdf",
                    "table_engine": "pdfplumber" if pdfplumber is not None else "unavailable",
                    "table_mode": self.config.table_mode,
                    "ocr_mode": self.config.ocr_mode,
                },
            }
        return documents


class ProcessingPipeline:
    def __init__(self, config: ProcessConfig) -> None:
        self.config = config
        self.processor = Processor(config)

    def write_json(self, data: Any, filename: str) -> Path:
        output_path = self.config.output_subdir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return output_path

    def write_document_files(
        self,
        documents: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        documents_dir = self.config.output_subdir / "documents"
        documents_dir.mkdir(parents=True, exist_ok=True)

        document_index: Dict[str, Dict[str, Any]] = {}
        for doc_id, document in documents.items():
            output_path = documents_dir / f"{doc_id}.json"
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(document, f, ensure_ascii=False, indent=2)

            document_index[doc_id] = {
                "doc_id": document["doc_id"],
                "domain": document["domain"],
                "source_path": document["path"],
                "output_path": str(output_path),
                "page_count": document["page_count"],
                "char_count": document["char_count"],
                "source_type": document["source_type"],
                "parse_profile": document["parse_profile"],
            }
        return document_index

    def build_page_diagnosis(self, documents: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for doc_id, document in documents.items():
            for page in document["pages"]:
                diagnosis = dict(page.get("diagnosis") or {})
                rows.append(
                    {
                        "doc_id": doc_id,
                        "domain": document["domain"],
                        "page": page["page"],
                        "page_type": page["page_type"],
                        **diagnosis,
                    }
                )
        return rows

    def run(self) -> Dict[str, Dict[str, Any]]:
        self.config.validate()
        documents = self.processor.build_document_records(self.config.domain_name)
        document_index = self.write_document_files(documents)
        self.write_json(document_index, "document_index.json")
        self.write_json(self.build_page_diagnosis(documents), "page_diagnosis.json")
        return document_index


def default_question_files(base: Path) -> List[Path]:
    question_dir = base / "questions" / "group_a"
    return sorted(question_dir.glob("*_questions.json"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid PDF preprocessing: pypdf text + pdfplumber tables.")
    parser.add_argument("--raw-dir", type=Path, default=Path("raw"))
    parser.add_argument("--questions", nargs="*", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("processed_data_hybrid"))
    parser.add_argument("--prefer-txt", action="store_true", default=True)
    parser.add_argument("--chunk-table-mode", choices=["append_markdown", "ignore"], default="append_markdown")
    parser.add_argument("--ocr-mode", choices=["flag_only"], default="flag_only")
    parser.add_argument("--limit-docs", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    question_files = args.questions or default_question_files(Path("."))
    if not question_files:
        raise FileNotFoundError("No question files found. Pass --questions explicitly.")

    for question_file in question_files:
        config = ProcessConfig(
            raw_dir=args.raw_dir,
            question_file=question_file,
            output_dir=args.output_dir,
            prefer_txt=args.prefer_txt,
            table_mode=args.chunk_table_mode,
            ocr_mode=args.ocr_mode,
            limit_docs=args.limit_docs,
        )
        pipeline = ProcessingPipeline(config)
        document_index = pipeline.run()
        print(f"Processed {len(document_index)} documents for {config.split_name}/{config.domain_name}")
        print(f"Outputs written to: {config.output_subdir}")


if __name__ == "__main__":
    main()
