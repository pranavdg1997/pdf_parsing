#!/usr/bin/env python3
"""
PDF Processing Pipeline

This script processes PDF documents to extract structured information and generates 
JSON output containing document metadata, text content, and structure information.

Features:
- Extracts text from digital PDFs
- Converts scanned/image-based PDF pages to PNG images
- Identifies headings, sections, and their boundaries
- Generates structured JSON output with document metadata
- Advanced table extraction using Camelot
- Spatial analysis of PDF elements
- Enhanced metadata extraction
- Document type detection

Dependencies:
- pdfplumber: For text extraction and layout analysis
- PyPDF2: For PDF document analysis
- Camelot: For advanced table extraction (optional)
- OpenCV: For image processing (optional)
- Pillow: For image handling and manipulation
- pandas: For data manipulation with Camelot tables (optional)
- re: For pattern matching

Changes:
1) Parallel processing added where possible.
2) Storing word coordinates in a CSV instead of the JSON.
3) Extracted tables in CSV, storing the CSV file names in JSON.
"""

import os
import json
import re
import logging
import tempfile
import subprocess
from typing import Dict, List, Tuple, Any, Optional, Union
import argparse
import pdfplumber
from PyPDF2 import PdfReader
from PIL import Image
import io
import camelot
import cv2
import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def is_page_scanned(page) -> bool:
    """
    Determine if a PDF page is scanned (image-based) or contains digital text.
    """
    text = page.extract_text()

    # 1. If there's no text at all, it's definitely a scanned page
    if not text:
        return True

    # 2. If there's very little text but many images, it's likely a scanned page
    if len(text.strip()) < 100 and len(page.images) > 0:
        return True

    # 3. Check for OCR artifacts
    if text and len(page.images) > 0:
        ocr_artifacts = re.findall(r'[^\\w\\s,.;:!?()\\[\\]{}"\'-]{3,}', text)
        if len(ocr_artifacts) > 3:
            return True

    # 4. Check text to image ratio
    if len(page.images) > 0:
        text_chars = len(text.strip())
        if text_chars / len(page.images) < 200:
            return True

    return False


def save_page_as_image(pdf_path: str, page_num: int, output_dir: str = "./images") -> str:
    """
    Convert a PDF page to an image and save it to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_page_{page_num+1}.png")

    try:
        pdf = PdfReader(pdf_path)
        if page_num >= len(pdf.pages):
            logger.error(f"Page {page_num+1} does not exist in the PDF")
            return ""
        with pdfplumber.open(pdf_path) as plumb_pdf:
            page = plumb_pdf.pages[page_num]
            img = page.to_image(resolution=300)
            img.save(output_path, format="PNG")
        return output_path
    except Exception as e:
        logger.error(f"Error converting page to image: {e}")
        return ""


def extract_headings(text: str) -> List[Dict[str, Any]]:
    """
    Extract heading information from text with improved hierarchy detection.
    """
    headings = []
    heading_patterns = [
        {'pattern': r'^[\s]*([A-Z][A-Z\s]+)[\s]*$', 'level': 1},
        {'pattern': r'^[\s]*(\d+\.\s+.+)[\s]*$', 'level': 2},
        {'pattern': r'^[\s]*([A-Z][A-Za-z\s]+):[\s]*$', 'level': 2},
        {'pattern': r'^[\s]*([A-Z][a-z\s]{2,})[\s]*$', 'level': 3},
        {'pattern': r'^[\s]*[•\-\*]\s+([A-Z].+?)[\s]*$', 'level': 3}
    ]

    lines = text.split('\n')
    line_num = 0
    prev_heading_level = 0
    prev_heading = None
    heading_contexts = {}

    for line in lines:
        line_num += 1
        line = line.strip()
        if not line:
            continue

        for pattern_info in heading_patterns:
            pattern = pattern_info['pattern']
            base_level = pattern_info['level']
            matches = re.match(pattern, line)
            if matches:
                heading_text = matches.group(1).strip()
                if len(heading_text) < 3:
                    continue
                level = base_level

                # Adjust based on uppercase
                if heading_text.isupper() and len(heading_text) > 5:
                    level = min(level, 1)
                elif heading_text.isupper():
                    level = min(level, 2)

                # Indentation-based adjustment
                leading_spaces = len(line) - len(line.lstrip())
                if leading_spaces > 4:
                    level += 1

                # Handle nested headings
                if prev_heading is not None and prev_heading_level > 0 and level > prev_heading_level:
                    heading_contexts[level] = heading_contexts.get(prev_heading_level, []) + [prev_heading]

                parent = None
                if level > 1 and level-1 in heading_contexts and heading_contexts[level-1]:
                    parent = heading_contexts.get(level-1, [])[-1]

                heading_entry = {
                    'text': heading_text,
                    'line_number': line_num,
                    'level': level,
                    'parent': parent
                }

                headings.append(heading_entry)
                prev_heading = heading_text
                prev_heading_level = level
                break

    return headings


def find_section_boundaries(text: str, headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Find the start and end boundaries of sections defined by headings.
    """
    lines = text.split('\n')
    total_lines = len(lines)
    sections = []

    for i, heading in enumerate(headings):
        section = heading.copy()
        start_line = heading['line_number']
        if i < len(headings) - 1:
            end_line = headings[i + 1]['line_number'] - 1
        else:
            end_line = total_lines
        section_content = '\n'.join(lines[start_line:end_line]).strip()
        section['start_line'] = start_line
        section['end_line'] = end_line
        section['content'] = section_content
        sections.append(section)

    return sections


def extract_tables(page, pdf_name: str, page_num: int, tables_dir: str = './tables') -> List[Dict[str, Any]]:
    """
    Extract tables from a PDF page using pdfplumber, save as CSV, and return info.
    """
    tables = []
    os.makedirs(tables_dir, exist_ok=True)

    try:
        page_tables = page.extract_tables()
        for i, table_data in enumerate(page_tables):
            if not table_data or len(table_data) == 0:
                continue

            df = pd.DataFrame(table_data)
            csv_filename = f"{pdf_name}_page_{page_num+1}_table_{i+1}.csv"
            csv_path = os.path.join(tables_dir, csv_filename)
            df.to_csv(csv_path, index=False, header=False)

            tables.append({
                'table_number': i + 1,
                'csv_file': csv_filename,
                'rows': len(df),
                'columns': len(df.columns),
                'extraction_method': 'pdfplumber'
            })

    except Exception as e:
        logger.warning(f"Error extracting tables with pdfplumber: {str(e)}")

    return tables


def extract_tables_with_camelot(pdf_path: str, page_num: int, pdf_name: str, tables_dir: str = './tables') -> List[Dict[str, Any]]:
    """
    Extract tables from a PDF page using camelot, save as CSV, and return info.
    """
    tables = []
    page_index = page_num + 1
    os.makedirs(tables_dir, exist_ok=True)

    try:
        lattice_tables = camelot.read_pdf(
            pdf_path,
            pages=str(page_index),
            flavor='lattice',
            suppress_stdout=True
        )
        stream_tables = camelot.read_pdf(
            pdf_path,
            pages=str(page_index),
            flavor='stream',
            suppress_stdout=True
        )

        def save_camelot_tables(camelot_tables, flavor):
            local_tables = []
            for i, tbl in enumerate(camelot_tables):
                if not tbl.df.empty:
                    csv_filename = f"{pdf_name}_page_{page_index}_table_{flavor}_{i+1}.csv"
                    csv_path = os.path.join(tables_dir, csv_filename)
                    tbl.df.to_csv(csv_path, index=False, header=False)

                    local_tables.append({
                        'table_number': i + 1,
                        'csv_file': csv_filename,
                        'rows': len(tbl.df),
                        'columns': len(tbl.df.columns),
                        'accuracy': tbl.accuracy,
                        'whitespace': tbl.whitespace,
                        'extraction_method': f'camelot-{flavor}'
                    })
            return local_tables

        tables += save_camelot_tables(lattice_tables, 'lattice')
        tables += save_camelot_tables(stream_tables, 'stream')

    except Exception as e:
        logger.warning(f"Error extracting tables with Camelot: {str(e)}")

    return tables


def extract_spatial_elements(page, pdf_name: str, page_num: int, words_dir: str = './words') -> Dict[str, Any]:
    """
    Extract text elements with their spatial information from a PDF page, save coords in CSV.
    Return a simplified structure for JSON (just words, plus CSV filename).
    """
    os.makedirs(words_dir, exist_ok=True)
    words = page.extract_words()

    word_data = []
    for w in words:
        text = w.get('text', '')
        x0 = w.get('x0', 0)
        x1 = w.get('x1', 0)
        y0 = w.get('top', 0)
        y1 = w.get('bottom', 0)
        width = w.get('width', 0)
        height = w.get('height', 0)
        word_data.append([text, x0, x1, y0, y1, width, height])

    if word_data:
        df = pd.DataFrame(word_data, columns=["text", "x0", "x1", "y0", "y1", "width", "height"])
        csv_filename = f"{pdf_name}_page_{page_num+1}_words.csv"
        csv_path = os.path.join(words_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        words_list = df["text"].tolist()
    else:
        csv_filename = None
        words_list = []

    return {
        'words_csv_file': csv_filename,
        'words_list': words_list
    }


def process_page(pdf_path: str, page_num: int, output_dir: str, tables_dir: str, words_dir: str) -> Dict[str, Any]:
    """
    Process a single PDF page to extract information.
    """
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    page_info = {
        'page_number': page_num + 1,
        'is_scanned': False,
        'image_path': None,
        'text': None,
        'headings': [],
        'sections': [],
        'tables': [],
        'words_csv_file': None,
        'words_list': [],
        'width': None,
        'height': None
    }

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        if is_page_scanned(page):
            page_info['is_scanned'] = True
            page_info['image_path'] = save_page_as_image(pdf_path, page_num, output_dir)
        else:
            # Extract text
            text = page.extract_text()
            page_info['text'] = text

            # Extract headings and sections
            if text:
                page_info['headings'] = extract_headings(text)
                page_info['sections'] = find_section_boundaries(text, page_info['headings'])

            # Extract tables with pdfplumber
            page_info['tables'] = extract_tables(page, pdf_name, page_num, tables_dir)

            # Attempt advanced table extraction via Camelot
            try:
                camelot_tables = extract_tables_with_camelot(pdf_path, page_num, pdf_name, tables_dir)
                # If Camelot finds more/better tables, replace
                if camelot_tables and (
                    len(camelot_tables) > len(page_info['tables']) or
                    any(t.get('accuracy', 0) > 80 for t in camelot_tables)
                ):
                    page_info['tables'] = camelot_tables
                else:
                    # Or merge them in if there's anything new
                    existing_count = len(page_info['tables'])
                    for i, tinfo in enumerate(camelot_tables):
                        tinfo['table_number'] = existing_count + i + 1
                        page_info['tables'].append(tinfo)
            except Exception as e:
                logger.warning(f"Error in Camelot extraction: {e}")

            # Extract spatial elements (words), store CSV
            word_info = extract_spatial_elements(page, pdf_name, page_num, words_dir)
            page_info['words_csv_file'] = word_info['words_csv_file']
            page_info['words_list'] = word_info['words_list']

        # Page dimensions
        page_info['width'] = page.width
        page_info['height'] = page.height

    return page_info


def extract_document_metadata(pdf, pdf_path: str) -> Dict[str, Any]:
    """
    Extract metadata from a PDF document.
    """
    metadata = {
        'filename': os.path.basename(pdf_path),
        'path': pdf_path,
        'pages': len(pdf.pages),
        'title': None,
        'author': None,
        'subject': None,
        'keywords': None,
        'creation_date': None,
        'modification_date': None
    }

    pdf_metadata = pdf.metadata
    if pdf_metadata:
        if pdf_metadata.get('/Title'):
            metadata['title'] = pdf_metadata.get('/Title')
        if pdf_metadata.get('/Author'):
            metadata['author'] = pdf_metadata.get('/Author')
        if pdf_metadata.get('/Subject'):
            metadata['subject'] = pdf_metadata.get('/Subject')
        if pdf_metadata.get('/Keywords'):
            metadata['keywords'] = pdf_metadata.get('/Keywords')
        if pdf_metadata.get('/CreationDate'):
            metadata['creation_date'] = pdf_metadata.get('/CreationDate')
        if pdf_metadata.get('/ModDate'):
            metadata['modification_date'] = pdf_metadata.get('/ModDate')

    return metadata


def generate_document_summary(document_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a summary of the document based on extracted information.
    """
    summary = {
        'title': document_info.get('title', 'Untitled Document'),
        'author': document_info.get('author', 'Unknown Author'),
        'total_pages': document_info.get('pages', 0),
        'digital_pages': 0,
        'scanned_pages': 0,
        'headings_count': 0,
        'tables_count': 0,
        'creation_date': document_info.get('creation_date'),
        'document_type': 'Unknown'
    }

    pages_info = document_info.get('pages', [])
    all_headings = []

    for page in pages_info:
        if page.get('is_scanned'):
            summary['scanned_pages'] += 1
        else:
            summary['digital_pages'] += 1
        # Count tables
        tables = page.get('tables', [])
        summary['tables_count'] += len(tables)
        # Gather headings
        headings = page.get('headings', [])
        all_headings.extend(headings)

    summary['headings_count'] = len(all_headings)

    # Simple doc type detection
    if all_headings:
        main_headings_text = ' '.join([h['text'].lower() for h in all_headings if h.get('level', 0) == 1])
        if 'report' in main_headings_text or 'case' in main_headings_text:
            summary['document_type'] = 'Case Report/Medical Document'
        elif 'chapter' in main_headings_text or 'section' in main_headings_text:
            summary['document_type'] = 'Book/Publication'
        elif 'abstract' in main_headings_text or 'introduction' in main_headings_text:
            summary['document_type'] = 'Academic Paper'
        elif 'resume' in main_headings_text or 'cv' in main_headings_text:
            summary['document_type'] = 'Resume/CV'

    return summary


def process_pdf(pdf_path: str,
                output_dir: str = "./images",
                tables_dir: str = "./tables",
                words_dir: str = "./words") -> Dict[str, Any]:
    """
    Process a PDF document to extract structured information with parallel processing.
    """
    logger.info(f"Processing PDF: {pdf_path}")

    try:
        pdf_reader = PdfReader(pdf_path)
        document_info = extract_document_metadata(pdf_reader, pdf_path)

        # Parallel processing of pages
        page_count = len(pdf_reader.pages)
        futures = []
        pages_info = []
        with ProcessPoolExecutor() as executor:
            for page_num in range(page_count):
                futures.append(executor.submit(
                    process_page,
                    pdf_path,
                    page_num,
                    output_dir,
                    tables_dir,
                    words_dir
                ))
            for future in as_completed(futures):
                result = future.result()
                pages_info.append(result)

        # Sort pages by page_number
        pages_info.sort(key=lambda x: x['page_number'])
        document_info['pages'] = pages_info

        # Document summary
        document_info['summary'] = generate_document_summary(document_info)
        return document_info

    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return {
            'error': str(e),
            'filename': os.path.basename(pdf_path),
            'path': pdf_path
        }


def save_json_output(document_info: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """
    Save document information as JSON.
    """
    if output_path is None:
        if 'filename' in document_info:
            base_name = os.path.splitext(document_info['filename'])[0]
            output_path = f"{base_name}.json"
        else:
            output_path = "pdf_output.json"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    json_data = json.dumps(document_info, indent=2)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(json_data)

    logger.info(f"Saved JSON output to {output_path}")
    return output_path


def main():
    """
    Main function to process a PDF document and generate JSON output.
    """
    parser = argparse.ArgumentParser(description='Process a PDF document and generate JSON output')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--output', '-o', help='Path to save the JSON output')
    parser.add_argument('--images-dir', '-i', default='./images', help='Directory to save images for scanned pages')
    parser.add_argument('--tables-dir', '-t', default='./tables', help='Directory to save CSV files for tables')
    parser.add_argument('--words-dir', '-w', default='./words', help='Directory to save CSV files for word coords')
    args = parser.parse_args()

    document_info = process_pdf(args.pdf_path, args.images_dir, args.tables_dir, args.words_dir)
    json_path = save_json_output(document_info, args.output)
    print(f"PDF processing complete. JSON output saved to: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = f.read()
    print("\nJSON Output:")
    print(json_data)


if __name__ == "__main__":
    main()
