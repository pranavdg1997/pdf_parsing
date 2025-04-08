Optional libraries are auto‑detected; the script degrades gracefully when Camelot or OpenCV are absent.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import pdfplumber
from PyPDF2 import PdfReader

# Third‑party (optional) -------------------------------------------------------
try:
    import camelot  # type: ignore
    HAS_CAMELOT = True
except Exception:
    HAS_CAMELOT = False

import pandas as pd  # always required (used for CSV writing)
from PIL import Image  # noqa: F401  (used indirectly by pdfplumber)

# ----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s – %(levelname)s – %(message)s")
logger = logging.getLogger(__name__)

###############################################################################
# Helper functions                                                             #
###############################################################################

def is_page_scanned(page: pdfplumber.page.Page) -> bool:
    """Heuristic to decide whether *page* is image‑based (scanned) or text‑based."""
    text = page.extract_text() or ""

    if not text.strip():
        return True  # no digital text at all

    if len(text) < 100 and page.images:
        return True  # very little text but images exist

    if page.images:
        text_chars_per_image = len(text) / len(page.images)
        if text_chars_per_image < 200:
            return True

    # crude OCR artifact check
    artefacts = re.findall(r"[^\w\s,.;:!?()\[\]{}\-]{3,}", text)
    if len(artefacts) > 3:
        return True

    return False


def save_page_as_image(pdf_path: str, page_num: int, out_dir: str = "./images") -> str:
    """Rasterise *page_num* (0‑indexed) of *pdf_path* to PNG, return filepath."""
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    out_path = os.path.join(out_dir, f"{stem}_page_{page_num + 1}.png")
    try:
        with pdfplumber.open(pdf_path) as pl:
            pl.pages[page_num].to_image(resolution=300).save(out_path, format="PNG")
        return out_path
    except Exception as exc:
        logger.error("Failed to rasterise page %s: %s", page_num + 1, exc)
        return ""

###############################################################################
# Heading / section utilities                                                 #
###############################################################################

def extract_headings(text: str) -> List[Dict[str, Any]]:
    """Return a list of detected headings with hierarchy level."""
    patterns = [
        (r"^[\s]*([A-Z][A-Z\s]+)[\s]*$", 1),  # ALL‑CAPS
        (r"^[\s]*(\d+\.\s+.+)[\s]*$", 2),     # 1. Intro
        (r"^[\s]*([A-Z][A-Za-z\s]+):[\s]*$", 2),  # Background:
        (r"^[\s]*([A-Z][a-z\s]{2,})[\s]*$", 3),   # Sentence case
        (r"^[\s]*[•\-*]\s+([A-Z].+?)[\s]*$", 3),  # bullet
    ]

    headings: List[Dict[str, Any]] = []
    ctx: dict[int, List[str]] = {}
    prev_lvl = 0
    prev_text: Optional[str] = None

    for ln, line in enumerate(text.split("\n"), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        for pat, base_lvl in patterns:
            m = re.match(pat, stripped)
            if not m:
                continue
            htext = m.group(1).strip()
            if len(htext) < 3:
                break
            lvl = base_lvl
            if htext.isupper() and len(htext) > 5:
                lvl = 1
            elif htext.isupper():
                lvl = min(lvl, 2)
            indent = len(line) - len(line.lstrip())
            if indent > 4:
                lvl += 1
            if prev_text and lvl > prev_lvl:
                ctx.setdefault(lvl, []).append(prev_text)
            parent = ctx.get(lvl - 1, [None])[-1] if lvl > 1 else None
            headings.append({"text": htext, "line_number": ln, "level": lvl, "parent": parent})
            prev_lvl, prev_text = lvl, htext
            break

    return headings


def find_section_boundaries(text: str, headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Slice *text* into sections using *headings*."""
    lines = text.split("\n")
    total = len(lines)
    sections: List[Dict[str, Any]] = []
    for idx, h in enumerate(headings):
        start = h["line_number"]
        end = headings[idx + 1]["line_number"] - 1 if idx + 1 < len(headings) else total
        sections.append({
            **h,
            "start_line": start,
            "end_line": end,
            "content": "\n".join(lines[start:end]).strip()
        })
    return sections

###############################################################################
# Table extraction                                                            #
###############################################################################

def _looks_like_table(df: pd.DataFrame) -> bool:
    """Return *True* if *df* is considered a real table (≥2 rows & ≥2 cols)."""
    return df.shape[0] >= 2 and df.shape[1] >= 2


def extract_tables(page: pdfplumber.page.Page, pdf_stem: str, page_num: int,
                   out_dir: str = "./tables") -> List[Dict[str, Any]]:
    """Extract *true* tables with pdfplumber, write CSV, return metadata list."""
    os.makedirs(out_dir, exist_ok=True)
    tables: List[Dict[str, Any]] = []
    try:
        for idx, raw in enumerate(page.extract_tables() or []):
            df = pd.DataFrame(raw)
            if not _looks_like_table(df):
                continue  # skip word‑lists / single‑column junk
            csv_name = f"{pdf_stem}_page_{page_num + 1}_table_{idx + 1}.csv"
            df.to_csv(os.path.join(out_dir, csv_name), index=False, header=False)
            tables.append({
                "table_number": idx + 1,
                "csv_file": csv_name,
                "rows": df.shape[0],
                "columns": df.shape[1],
                "extraction_method": "pdfplumber"
            })
    except Exception as exc:
        logger.warning("pdfplumber table extraction failed on page %s: %s", page_num + 1, exc)
    return tables


def extract_tables_camelot(pdf_path: str, page_num: int, pdf_stem: str,
                           out_dir: str = "./tables") -> List[Dict[str, Any]]:
    """Use Camelot to extract *true* tables from *page_num* (0‑based)."""
    if not HAS_CAMELOT:
        return []
    os.makedirs(out_dir, exist_ok=True)
    page_label = page_num + 1  # Camelot is 1‑based
    tables: List[Dict[str, Any]] = []
    try:
        for flavor in ("lattice", "stream"):
            for idx, tbl in enumerate(camelot.read_pdf(pdf_path, pages=str(page_label),
                                                       flavor=flavor, suppress_stdout=True)):
                df = tbl.df
                if not _looks_like_table(df):
                    continue
                csv_name = f"{pdf_stem}_page_{page_label}_table_{flavor}_{idx + 1}.csv"
                df.to_csv(os.path.join(out_dir, csv_name), index=False, header=False)
                tables.append({
                    "table_number": idx + 1,
                    "csv_file": csv_name,
                    "rows": df.shape[0],
                    "columns": df.shape[1],
                    "accuracy": getattr(tbl, "accuracy", None),
                    "whitespace": getattr(tbl, "whitespace", None),
                    "extraction_method": f"camelot-{flavor}"
                })
    except Exception as exc:
        logger.warning("Camelot failed on page %s: %s", page_label, exc)
    return tables

###############################################################################
# Word‑level spatial extraction                                               #
###############################################################################

def extract_words_to_csv(page: pdfplumber.page.Page, pdf_stem: str, page_num: int,
                         out_dir: str = "./words") -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    words = page.extract_words() or []
    if not words:
        return {"words_csv_file": None, "words_list": []}

    records = [
        [
            w.get("text", ""),
            w.get("x0", 0),
            w.get("x1", 0),
            w.get("top", 0),
            w.get("bottom", 0),
            w.get("width", 0),
            w.get("height", 0)
        ]
        for w in words
    ]
    df = pd.DataFrame(records, columns=["text", "x0", "x1", "y0", "y1", "width", "height"])
    csv_name = f"{pdf_stem}_page_{page_num + 1}_words.csv"
    df.to_csv(os.path.join(out_dir, csv_name), index=False)
    return {
        "words_csv_file": csv_name,
        "words_list": df["text"].tolist()
    }

###############################################################################
# Per‑page processing                                                         #
###############################################################################

def process_page(pdf_path: str, page_num: int, img_dir: str, tbl_dir: str,
                 wdir: str) -> Dict[str, Any]:
    pdf_stem = os.path.splitext(os.path.basename(pdf_path))[0]
    page_info: Dict[str, Any] = {
        "page_number": page_num + 1,
        "is_scanned": False,
        "image_path": None,
        "text": None,
        "headings": [],
        "sections": [],
        "tables": [],
        "words_csv_file": None,
        "words_list": [],
        "width": None,
        "height": None
    }

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        page_info["width"], page_info["height"] = page.width, page.height

        if is_page_scanned(page):
            page_info["is_scanned"] = True
            page_info["image_path"] = save_page_as_image(pdf_path, page_num, img_dir)
            return page_info  # nothing else to do

        text = page.extract_text() or ""
        page_info["text"] = text
        if text:
            page_info["headings"] = extract_headings(text)
            page_info["sections"] = find_section_boundaries(text, page_info["headings"])

        # --- table extraction -------------------------------------------------
        page_info["tables"] = extract_tables(page, pdf_stem, page_num, tbl_dir)
        camelot_tables = extract_tables_camelot(pdf_path, page_num, pdf_stem, tbl_dir)
        if camelot_tables and (
            len(camelot_tables) > len(page_info["tables"]) or
            any(t.get("accuracy", 0) and t["accuracy"] > 80 for t in camelot_tables)
        ):
            page_info["tables"] = camelot_tables
        else:
            # merge unique ones
            existing = {(t["csv_file"] or "") for t in page_info["tables"]}
            for t in camelot_tables:
                if t["csv_file"] not in existing:
                    page_info["tables"].append(t)

        # --- word coordinates -------------------------------------------------
        word_meta = extract_words_to_csv(page, pdf_stem, page_num, wdir)
        page_info.update(word_meta)

    return page_info

###############################################################################
# Document‑level orchestration                                                #
###############################################################################

def extract_metadata(reader: PdfReader, path: str) -> Dict[str, Any]:
    meta = reader.metadata or {}
    return {
        "filename": os.path.basename(path),
        "path": path,
        "pages": len(reader.pages),
        "title": meta.get("/Title"),
        "author": meta.get("/Author"),
        "subject": meta.get("/Subject"),
        "keywords": meta.get("/Keywords"),
        "creation_date": meta.get("/CreationDate"),
        "modification_date": meta.get("/ModDate")
    }


def summarise(doc: Dict[str, Any]) -> Dict[str, Any]:
    summary = {
        "title": doc.get("title") or "Untitled",
        "author": doc.get("author") or "Unknown",
        "total_pages": doc.get("pages", 0),
        "digital_pages": 0,
        "scanned_pages": 0,
        "tables_count": 0,
        "headings_count": 0,
        "document_type": "Unknown",
        "creation_date": doc.get("creation_date")
    }
    all_heads = []
    for p in doc.get("pages", []):
        if p["is_scanned"]:
            summary["scanned_pages"] += 1
        else:
            summary["digital_pages"] += 1
        summary["tables_count"] += len(p["tables"])
        all_heads.extend(p["headings"])
    summary["headings_count"] = len(all_heads)
    if any(h["text"].lower().startswith("abstract") for h in all_heads):
        summary["document_type"] = "Academic Paper"
    return summary


def process_pdf(pdf_path: str, img_dir: str = "./images", tbl_dir: str = "./tables",
                wdir: str = "./words") -> Dict[str, Any]:
    logger.info("Processing PDF %s", pdf_path)
    reader = PdfReader(pdf_path)
    doc_info = extract_metadata(reader, pdf_path)

    futures, pages = [], []  # type: ignore

    with ProcessPoolExecutor() as pool:
        for i in range(len(reader.pages)):
            futures.append(pool.submit(process_page, pdf_path, i, img_dir, tbl_dir, wdir))
        for fut in as_completed(futures):
            pages.append(fut.result())

    pages.sort(key=lambda p: p["page_number"])
    doc_info["pages"] = pages
    doc_info["summary"] = summarise(doc_info)

    return doc_info

###############################################################################
# I/O utilities                                                               #
###############################################################################

def write_json(doc: Dict[str, Any], out_path: Optional[str] = None) -> str:
    if not out_path:
        out_path = f"{os.path.splitext(doc['filename'])[0]}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    logger.info("JSON written to %s", out_path)
    return out_path

###############################################################################
# CLI                                                                         #
###############################################################################

def main() -> None:
    ap = argparse.ArgumentParser(description="Parse PDF to structured JSON + CSV artefacts")
    ap.add_argument("pdf_path")
    ap.add_argument("--output", "-o")
    ap.add_argument("--images-dir", "-i", default="./images")
    ap.add_argument("--tables-dir", "-t", default="./tables")
    ap.add_argument("--words-dir", "-w", default="./words")
    args = ap.parse_args()

    doc = process_pdf(args.pdf_path, args.images_dir, args.tables_dir, args.words_dir)
    json_path = write_json(doc, args.output)
    print(f"Finished. JSON saved to {json_path}")


if __name__ == "__main__":
    main()
