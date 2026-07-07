import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ProcessConfig:
    """
    Configuration for processing data:
        1. raw_path, the path of original dataset
        2. question file, the path of the questions
        3. output_dir, the path for storing processed data
    """
    raw_dir: Path
    question_file: Path
    output_dir: Path
    prefer_txt: bool = True

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
    """
    Main Response:
        1. transform the pdf text layer into json file
        2. unify the format of json file
        3. preserve text/table content only, without OCR on images
        
        The structure of the json includes:
            page: the page of the text belong
            title: the main topic of the text
            keyword: the keyword of the text
            text: the main content
    """
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
        repeated = {
            line
            for line, count in counts.items()
            if count >= min_count and len(line) >= 6
        }
        return repeated

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

            cleaned_pages.append(
                {
                    "page": page["page"],
                    "text": self._join_lines(lines),
                }
            )

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

    def _match_heading(self, line: str) -> tuple[str, str] | None:
        for level, pattern in self.heading_patterns:
            match = pattern.match(line)
            if not match:
                continue
            prefix = match.group(1).strip()
            suffix = match.group(2).strip() if match.lastindex and match.lastindex > 1 and match.group(2) else ""
            title = f"{prefix} {suffix}".strip()
            return level, title
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
                    current_title = next((value for value in reversed(list(current_context.values())) if value), "")
                current_lines.append(line)

            flush_segment()

            segmented_pages.append(
                {
                    "page": page["page"],
                    "segments": segments,
                    "page_type": "content",
                }
            )

        return segmented_pages

    def read_txt(self, path: Path) -> str:
        text = path.read_text(encoding="utf-8")
        return self.clean_text(text)

    def read_pdf_pages(self, path: Path) -> List[Dict[str, Any]]:
        # Only read the PDF text layer. This intentionally does not run OCR
        # or attempt to extract text from embedded images.
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError(
                "PDF parsing requires pypdf. Install with: pip install pypdf"
            ) from exc

        reader = PdfReader(str(path))
        pages: List[Dict[str, Any]] = []

        for idx, page in enumerate(reader.pages, start=1):
            extracted = page.extract_text() or ""
            cleaned = self.clean_text(extracted)
            pages.append(
                {
                    "page": idx,
                    "text": cleaned,
                }
            )

        return pages

    def extract_pages(self, path: Path) -> List[Dict[str, Any]]:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            return [
                {
                    "page": 1,
                    "text": self.read_txt(path),
                }
            ]
        if suffix == ".pdf":
            return self.read_pdf_pages(path)
        raise ValueError(f"Unsupported file type: {path}")

    def _candidate_files(self, domain_dir: Path) -> List[Path]:
        candidates: List[Path] = []
        for path in sorted(domain_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".txt", ".pdf"}:
                continue
            candidates.append(path)
        return candidates

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

    def build_document_records(self, domain: str) -> Dict[str, Dict[str, Any]]:
        domain_dir = self.config.raw_dir / domain
        if not domain_dir.exists():
            raise FileNotFoundError(f"domain raw directory not found: {domain_dir}")

        source_paths = self._candidate_files(domain_dir)
        selected_sources = self._select_preferred_sources(source_paths)

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
            }

        return documents


class ProcessingPipeline:
    """
    API for processing all files
    """
    def __init__(self, config: ProcessConfig) -> None:
        self.config = config
        self.processor = Processor(config)

    def write_json(self, data: Any, filename: str) -> Path:
        output_path = self.config.output_subdir / filename
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
            }

        return document_index

    def run(self) -> Dict[str, Dict[str, Any]]:
        self.config.validate()

        documents = self.processor.build_document_records(self.config.domain_name)
        document_index = self.write_document_files(documents)
        self.write_json(document_index, "document_index.json")

        return document_index


if __name__ == "__main__":
    config = ProcessConfig(
        raw_dir=Path("public_dataset_upload/raw"),
        question_file=Path("public_dataset_upload/questions/group_a/research_questions.json"),
        output_dir=Path("processed_data"),
        prefer_txt=True,
    )

    pipeline = ProcessingPipeline(config)
    document_index = pipeline.run()
    print(f"Processed {len(document_index)} documents.")
    print(f"Outputs written to: {config.output_subdir}")
