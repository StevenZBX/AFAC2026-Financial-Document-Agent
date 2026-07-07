# Hybrid PDF Process

This folder is for the preprocessing step only. It does not generate retrieval evidence.

## Strategy

- Plain text pages: keep the existing `pypdf` text extraction path.
- Table pages: use `pdfplumber.extract_tables()`, clean rows/columns, convert tables to Markdown, and append the Markdown to page text.
- Image/chart pages: detect image-heavy or low-text pages and mark them as `needs_ocr` or `image_or_chart` in `page_diagnosis.json`.
- OCR: currently `flag_only`. Add PaddleOCR, MinerU, or another OCR engine after manual sampling confirms which pages need it.

This is the competition-friendly hybrid approach: cheap parsing first, stronger parsing only where the page looks risky.

## Run A Small Test

From the repository root:

```bash
./run_sample.sh
```

## Run All Group A Domains

```bash
./run_all_group_a.sh
```

The run scripts read data from:

```text
/Users/limeixuan/Desktop/public_dataset_upload
```

Override it with:

```bash
DATA_ROOT=/path/to/public_dataset_upload ./run_sample.sh
```

## Outputs

For each domain:

```text
processed_data_hybrid/group_a/<domain>/
├── document_index.json
├── page_diagnosis.json
└── documents/
    └── <doc_id>.json
```

Use `page_diagnosis.json` for manual sampling. Prioritize pages with:

- `table_detected`
- `image_or_chart_detected`
- `low_text_with_image`
- `many_numbers`
- `table_keywords`
