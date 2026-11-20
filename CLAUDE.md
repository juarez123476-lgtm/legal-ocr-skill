# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository is a **Claude Code skill** (`legal-ocr`) that extracts text from scanned Brazilian legal PDFs (sentenças, acórdãos, petições) using OCR. It is a single-purpose Python pipeline, not an application with multiple modules — almost all logic lives in one file, `pipeline_ocr.py`. It's designed to be integrated into the SIGEDEC judicial-decision system as an ingestion step feeding a Qdrant/RAG pipeline (see the "Integração com SIGEDEC" sections in `README.md` / `SKILL.md`), but this repo has no dependency on SIGEDEC code itself.

## Commands

### Install
```bash
chmod +x setup.sh
./setup.sh          # installs deps, downloads PaddleOCR/EasyOCR pt models, creates logs/
```
`setup.sh` installs CUDA or CPU PyTorch depending on whether `nvidia-smi` is detected, then `pip install -r requirements.txt`, then downloads the Portuguese PaddleOCR and EasyOCR models (EasyOCR download is ~400MB). Manual equivalent: `pip install -r requirements.txt`.

### Run the pipeline
```bash
python pipeline_ocr.py documento.pdf                              # writes documento_ocr.json
python pipeline_ocr.py documento.pdf -o out.json --quality high --dpi 400
python pipeline_ocr.py documento.pdf --no-gpu --no-fallback --confidence-threshold 0.2
```
CLI flags map 1:1 to `LegalDocumentOCRPipeline.__init__` / `process_legal_document` args — see `main()` at the bottom of `pipeline_ocr.py`.

### Tests
```bash
pip install pytest pytest-cov   # not installed by default in this environment
pytest                          # config in pytest.ini: testpaths=tests, python_files=test_*.py
pytest tests/test_pipeline_ocr.py::TestPostProcessText::test_specific_case  # single test
pytest -k "fuzzy"               # run tests matching a keyword
pytest --cov=pipeline_ocr        # coverage (pytest-cov is a listed dependency)
```
There is no linter or formatter configured (no flake8/ruff/black/pyproject.toml in the repo) — don't invent a lint step.

### Sanity checks (no test PDF needed)
```bash
python -c "from pipeline_ocr import LegalDocumentOCRPipeline; print('OK')"
python -c "import torch; print(torch.cuda.is_available())"
```

## Architecture

### Everything is one class: `LegalDocumentOCRPipeline`
`pipeline_ocr.py` defines a single pipeline class whose methods form a strict sequential stage chain, driven by `process_legal_document()`:

1. `pdf_to_images` — PyMuPDF (`fitz`) renders each PDF page to a numpy array at the given DPI (chosen over `pdf2image` for speed; `pdf2image`/Poppler is still listed in requirements as a fallback but unused in code).
2. `preprocess_image` — OpenCV pipeline: grayscale → Hough-transform skew correction → median blur denoise → CLAHE contrast → (in `quality='high'` mode only) local brightness flattening via morphological division → adaptive-threshold binarization → dilation to reconnect broken strokes.
3. `ocr_image` — tries PaddleOCR (`lang='pt'`) first; if it errors, returns empty, or its average confidence is below `confidence_threshold`, falls back to EasyOCR (only if `use_fallback=True` and EasyOCR initialized). The two engines return different result shapes, both handled explicitly here and in `extract_text_from_ocr`.
4. `extract_text_from_ocr` — normalizes PaddleOCR's `[[box, (text, conf)], ...]` and EasyOCR's `[(box, text, conf), ...]` shapes into plain text.
5. `post_process_text` — corrects OCR text word-by-word against `legal_dictionary.json`: exact match first, then `difflib.SequenceMatcher` fuzzy match (ratio > 0.85) against dictionary keys.
6. `identify_document_structure` — line-by-line state machine that classifies text into `header/preambulo/relatorio/fundamentacao/dispositivo/assinaturas` by matching Portuguese legal section markers (e.g. "vistos", "é o relatório", "pelo exposto").
7. `validate_quality` — heuristic confidence score (starts at 100, deducted for O/0, l/1, S/5 confusion ratios, missing expected terms like "juiz"/"tribunal"/"processo", and abnormally short average word length); `requires_review` is `True` below 70.

`process_legal_document` runs all of the above per-page in a loop, tracks which pages fell back to EasyOCR, then aggregates full text, document-level structure, and a `quality_summary` (`avg_confidence` → `_quality_rating` → excellent/good/acceptable/poor).

### `legal_dictionary.json` is a flat dict, not just term pairs
It's loaded directly via `json.load()` into `self.legal_dict` and contains `_comment`, `_version`, `_last_updated` metadata keys alongside the actual `misspelling: correction` term pairs. Both the exact-match check and the fuzzy-match loop in `post_process_text` iterate over `self.legal_dict.keys()`/values directly, so these metadata keys are silently included in fuzzy-match candidates. Keep this in mind when adding new terms or metadata keys to the dictionary — don't assume every key is a real legal term. If the file is missing, `_load_legal_dictionary` falls back to a small hardcoded dict of ~8 terms.

### Test suite mocks all heavy/native dependencies
`tests/test_pipeline_ocr.py` stubs `cv2`, `fitz`, `torch`, `PIL`, `paddleocr`, `easyocr` into `sys.modules` (via `setdefault`, so a real install is preferred when present) **before** importing `pipeline_ocr`, so the suite runs without GPU drivers, model weights, or OpenCV/Poppler installed. When adding tests for new pipeline stages, follow this same stubbing pattern rather than requiring the real ML/CV stack. Tests are organized by pipeline method (post-processing, quality validation, structure detection, OCR format normalization, engine fallback) and use realistic Brazilian legal-document fixtures (sentenças, LGPD contracts, CLT rescission, franchise/LPI clauses).

### Duplicate skill manifest
`SKILL.md` (repo root) and `.claude/skills/legal-ocr.md` both define this skill for Claude Code but have diverged (different `description`/`allowed-tools` frontmatter). If you update one, check whether the other also needs updating.

### Known gaps vs. docs
`README.md`/`SKILL.md` reference a `reference.md` file for technical details — it does not exist in this repo. Don't assume it's there.
