"""
Per-file-type parser nodes. Each returns a partial PipelineState update
with a `documents` list of {page_content, metadata} dicts, matching the
shape the chunker expects.

Ported from the original auto-embedder notebook, with these fixes:
  - No hardcoded API keys (LLM comes from app.pipeline.llm, config-driven).
  - No interactive input() calls — file_path/url + project_id are passed
    in explicitly by the API layer.
  - The web parser's external extraction service URL is configurable
    (not hardcoded to a local port), and falls back to a built-in
    httpx + BeautifulSoup extractor when unset/unreachable.
"""

import re
from pathlib import Path
from typing import Any, List

import docx
import httpx
import pdfplumber
import pptx
import pytesseract
from PIL import Image as PILImage

from app.core.config import get_settings
from app.pipeline.llm import summarize_table_markdown
from app.pipeline.state import PipelineState

FileTypes = {
    ".pdf": "pdf",
    ".pptx": "pptx",
    ".docx": "docx",
    ".md": "markdown",
    ".txt": "text",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
}


def detect_file_type(path: Path) -> str:
    return FileTypes.get(path.suffix.lower(), "unsupported")


def validator_node(state: PipelineState) -> PipelineState:
    if state.get("url"):
        return {"valid": True, "file_type": "web", "error": None}

    file_path = state.get("file_path")

    if not file_path:
        return {
            "valid": False,
            "error": "No file_path or url provided.",
            "file_type": "unsupported",
        }

    p = Path(file_path).expanduser().resolve()

    if not p.exists() or not p.is_file():
        return {
            "valid": False,
            "error": f"Invalid file path: {p}",
            "file_type": "unsupported",
        }

    ftype = detect_file_type(p)

    if ftype == "unsupported":
        return {
            "valid": False,
            "error": f"Unsupported file type: {p.suffix}",
            "file_type": "unsupported",
        }

    return {"valid": True, "file_type": ftype, "error": None}


def table_to_markdown(table: List[List[Any]]) -> str:
    """Converts a pdfplumber table (list of lists) into a Markdown string."""
    if not table or not table[0]:
        return ""

    md = ""
    for i, row in enumerate(table):
        cleaned_row = [
            str(cell).replace("\n", " ").strip() if cell is not None else ""
            for cell in row
        ]
        md += "| " + " | ".join(cleaned_row) + " |\n"

        if i == 0:
            md += "|" + "|".join(["---"] * len(cleaned_row)) + "|\n"

    return md


def docx_table_to_markdown(table) -> str:
    if not table.rows:
        return ""

    md = ""
    for i, row in enumerate(table.rows):
        cleaned_row = [cell.text.replace("\n", " ").strip() for cell in row.cells]
        md += "| " + " | ".join(cleaned_row) + " |\n"

        if i == 0:
            md += "|" + "|".join(["---"] * len(cleaned_row)) + "|\n"

    return md


def pptx_table_to_markdown(table) -> str:
    if not hasattr(table, "rows") or len(table.rows) == 0:
        return ""

    md = ""
    for i, row in enumerate(table.rows):
        cleaned_row = [cell.text.replace("\n", " ").strip() for cell in row.cells]
        md += "| " + " | ".join(cleaned_row) + " |\n"

        if i == 0:
            md += "|" + "|".join(["---"] * len(cleaned_row)) + "|\n"

    return md


def is_two_column(page) -> bool:
    """
    Analyzes word coordinates to detect a two-column PDF page layout by
    checking whether a vertical "gutter" runs down the middle of the page.
    """
    words = page.extract_words()
    if not words:
        return False

    page_width = page.width
    mid_x = page_width / 2

    gutter_width = page_width * 0.04
    gutter_left = mid_x - (gutter_width / 2)
    gutter_right = mid_x + (gutter_width / 2)

    crossing_words = 0
    for word in words:
        if word["x0"] < gutter_right and word["x1"] > gutter_left:
            crossing_words += 1

    crossing_ratio = crossing_words / len(words)
    return crossing_ratio < 0.15


def pdf_parser_node(state: PipelineState) -> PipelineState:
    file_path = state["file_path"]
    documents = []

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            two_column = is_two_column(page)

            if two_column:
                width = page.width
                height = page.height

                left_bbox = (0, 0, width * 0.52, height)
                right_bbox = (width * 0.48, 0, width, height)

                left_text = page.crop(left_bbox).extract_text(x_tolerance=1.5)
                right_text = page.crop(right_bbox).extract_text(x_tolerance=1.5)

                text_blocks = [t for t in (left_text, right_text) if t]
                text = "\n".join(text_blocks)
            else:
                text = page.extract_text(x_tolerance=2.0)

            if text and text.strip():
                documents.append(
                    {
                        "page_content": text,
                        "metadata": {
                            "source": file_path,
                            "page_number": page_num,
                            "content_type": "text",
                            "layout_detected": (
                                "two_column" if two_column else "single_column"
                            ),
                        },
                    }
                )

            for table in page.extract_tables():
                md_table = table_to_markdown(table)
                if md_table.strip():
                    summary = summarize_table_markdown(md_table)

                    documents.append(
                        {
                            "page_content": summary,
                            "metadata": {
                                "source": file_path,
                                "page_number": page_num,
                                "content_type": "table_summary",
                                "original_table_markdown": md_table,
                            },
                        }
                    )

    return {"documents": documents}


def txt_parser_node(state: PipelineState) -> PipelineState:
    file_path = state["file_path"]
    documents = []

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    if text.strip():
        documents.append(
            {
                "page_content": text,
                "metadata": {
                    "source": file_path,
                    "page_number": 1,
                    "content_type": "text",
                    "layout_detected": "plain_text",
                },
            }
        )

    return {"documents": documents}


def docx_parser_node(state: PipelineState) -> PipelineState:
    file_path = state["file_path"]
    documents = []

    doc = docx.Document(file_path)

    full_text = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    if full_text:
        documents.append(
            {
                "page_content": "\n".join(full_text),
                "metadata": {
                    "source": file_path,
                    "page_number": 1,
                    "content_type": "text",
                    "layout_detected": "docx_native",
                },
            }
        )

    for table in doc.tables:
        md_table = docx_table_to_markdown(table)
        if md_table.strip():
            summary = summarize_table_markdown(md_table)

            documents.append(
                {
                    "page_content": summary,
                    "metadata": {
                        "source": file_path,
                        "page_number": 1,
                        "content_type": "table_summary",
                        "original_table_markdown": md_table,
                    },
                }
            )

    return {"documents": documents}


def markdown_parser_node(state: PipelineState) -> PipelineState:
    file_path = state["file_path"]
    documents = []

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    table_pattern = re.compile(r"(?:^\|.*\|\s*)+", re.MULTILINE)
    extracted_tables = table_pattern.findall(text)

    for md_table in extracted_tables:
        if md_table.strip():
            summary = summarize_table_markdown(md_table.strip())

            documents.append(
                {
                    "page_content": summary,
                    "metadata": {
                        "source": file_path,
                        "page_number": 1,
                        "content_type": "table_summary",
                        "original_table_markdown": md_table.strip(),
                    },
                }
            )

    text_without_tables = table_pattern.sub(
        "\n\n[TABLE SUMMARIZED SEPARATELY]\n\n", text
    )

    if text_without_tables.strip():
        documents.append(
            {
                "page_content": text_without_tables,
                "metadata": {
                    "source": file_path,
                    "page_number": 1,
                    "content_type": "text",
                    "layout_detected": "markdown",
                },
            }
        )

    return {"documents": documents}


def pptx_parser_node(state: PipelineState) -> PipelineState:
    file_path = state["file_path"]
    documents = []

    prs = pptx.Presentation(file_path)

    for i, slide in enumerate(prs.slides):
        slide_num = i + 1
        slide_text = []

        for shape in slide.shapes:
            if shape.has_text_frame:
                if shape.text.strip():
                    slide_text.append(shape.text.strip())

            elif shape.has_table:
                md_table = pptx_table_to_markdown(shape.table)
                if md_table.strip():
                    summary = summarize_table_markdown(md_table)

                    documents.append(
                        {
                            "page_content": summary,
                            "metadata": {
                                "source": file_path,
                                "page_number": slide_num,
                                "content_type": "table_summary",
                                "original_table_markdown": md_table,
                            },
                        }
                    )

        if slide_text:
            documents.append(
                {
                    "page_content": "\n\n".join(slide_text),
                    "metadata": {
                        "source": file_path,
                        "page_number": slide_num,
                        "content_type": "text",
                        "layout_detected": "pptx_slide",
                    },
                }
            )

    return {"documents": documents}


def image_parser_node(state: PipelineState) -> PipelineState:
    file_path = state["file_path"]
    documents = []

    settings = get_settings()
    if settings.TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD

    image = PILImage.open(file_path)
    extracted_text = pytesseract.image_to_string(image)

    if extracted_text and extracted_text.strip():
        documents.append(
            {
                "page_content": extracted_text.strip(),
                "metadata": {
                    "source": file_path,
                    "page_number": 1,
                    "content_type": "text",
                    "layout_detected": "image_ocr",
                },
            }
        )

    return {"documents": documents}


def _extract_via_external_service(url: str) -> dict:
    settings = get_settings()

    base = settings.WEB_EXTRACTION_SERVICE_URL.rstrip("/")
    api_url = f"{base}{settings.WEB_EXTRACTION_SERVICE_PATH}"

    with httpx.Client(timeout=settings.WEB_FETCH_TIMEOUT_SECONDS) as client:
        response = client.post(api_url, json={"url": url})
        response.raise_for_status()
        return response.json()


def _extract_via_builtin_fallback(url: str) -> dict:
    """
    Minimal built-in extractor used when no external extraction service
    is configured (or it's unreachable): fetch the page and strip it
    down to readable text with BeautifulSoup.
    """
    from bs4 import BeautifulSoup

    settings = get_settings()

    with httpx.Client(
        timeout=settings.WEB_FETCH_TIMEOUT_SECONDS,
        follow_redirects=True,
        headers={
            "User-Agent": settings.WEB_FETCH_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    ) as client:
        response = client.get(url)
        response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else url
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{2,}", "\n\n", text).strip()

    return {
        "text": text,
        "metadata": {"url": url, "title": title},
        "confidence": 0.6,
    }


def web_parser_node(state: PipelineState) -> PipelineState:
    url = state["url"]
    documents = []

    settings = get_settings()

    extracted_data = None

    if settings.WEB_EXTRACTION_SERVICE_URL:
        try:
            extracted_data = _extract_via_external_service(url)
        except Exception:
            extracted_data = None

    if extracted_data is None:
        extracted_data = _extract_via_builtin_fallback(url)

    content = extracted_data.get("text", "")
    metadata = extracted_data.get("metadata", {})
    confidence = extracted_data.get("confidence", 1.0)

    if content.strip():
        documents.append(
            {
                "page_content": content,
                "metadata": {
                    "source": metadata.get("url", url),
                    "title": metadata.get("title", "Unknown Title"),
                    "page_number": 1,
                    "content_type": "text",
                    "layout_detected": "web_html",
                    "confidence_score": (
                        confidence.get("score", 1.0)
                        if isinstance(confidence, dict)
                        else float(confidence)
                    ),
                },
            }
        )

    return {"documents": documents}


PARSER_NODES = {
    "pdf": pdf_parser_node,
    "text": txt_parser_node,
    "docx": docx_parser_node,
    "markdown": markdown_parser_node,
    "pptx": pptx_parser_node,
    "image": image_parser_node,
    "web": web_parser_node,
}