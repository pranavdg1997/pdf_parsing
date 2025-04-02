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
from PIL import Image, ImageDraw
import io

# Import additional libraries
import camelot
import cv2
import numpy as np
import pandas as pd


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def is_page_scanned(page) -> bool:
    """
    Determine if a PDF page is scanned (image-based) or contains digital text.
    
    Args:
        page: A pdfplumber page object
        
    Returns:
        bool: True if the page appears to be scanned, False if it contains digital text
    """
    # Extract text from the page
    text = page.extract_text()
    
    # More sophisticated scanned page detection:
    
    # 1. If there's no text at all, it's definitely a scanned page
    if not text:
        return True
        
    # 2. If there's very little text but many images, it's likely a scanned page
    if len(text.strip()) < 100 and len(page.images) > 0:
        return True
        
    # 3. Check for OCR artifacts (common in scanned PDFs)
    if text and len(page.images) > 0:
        # Look for OCR artifacts like random character sequences
        ocr_artifacts = re.findall(r'[^\w\s,.;:!?()[\]{}"\'-]{3,}', text)
        if len(ocr_artifacts) > 3:  # Multiple OCR artifacts found
            return True
    
    # 4. Check text to image ratio - if mostly images, probably scanned
    if len(page.images) > 0:
        text_chars = len(text.strip())
        # If very few characters per image, likely scanned
        if text_chars / len(page.images) < 200:
            return True
    
    return False


def save_page_as_image(pdf_path: str, page_num: int, output_dir: str = "./images") -> str:
    """
    Convert a PDF page to an image and save it to disk.
    
    Args:
        pdf_path: Path to the PDF file
        page_num: Page number to convert (0-indexed)
        output_dir: Directory to save the image
        
    Returns:
        str: Path to the saved image file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_page_{page_num+1}.png")
    
    try:
        # Open the PDF with PyPDF2
        pdf = PdfReader(pdf_path)
        
        if page_num >= len(pdf.pages):
            logger.error(f"Page {page_num+1} does not exist in the PDF")
            return ""
        
        # Convert page to image
        with pdfplumber.open(pdf_path) as plumb_pdf:
            page = plumb_pdf.pages[page_num]
            img = page.to_image(resolution=300)
            img.save(output_path, format="PNG")
            
        return output_path
    
    except Exception as e:
        pass
        return ""


def extract_headings(text: str) -> List[Dict[str, Any]]:
    """
    Extract heading information from text with improved hierarchy detection.
    
    Args:
        text: Text content to analyze
        
    Returns:
        List of dictionaries containing heading information with proper hierarchy levels
    """
    headings = []
    
    # Regular expressions for common heading patterns
    heading_patterns = [
        # Pattern for main headings in all caps (e.g., "TITLE OF CASE")
        {'pattern': r'^[\s]*([A-Z][A-Z\s]+)[\s]*$', 'level': 1},
        
        # Pattern for numbered headings (e.g., "1. Introduction")
        {'pattern': r'^[\s]*(\d+\.\s+.+)[\s]*$', 'level': 2},
        
        # Pattern for section headings with colons (e.g., "Background:")
        {'pattern': r'^[\s]*([A-Z][A-Za-z\s]+):[\s]*$', 'level': 2},
        
        # Pattern for subsection headings (e.g., "Treatment approach")
        {'pattern': r'^[\s]*([A-Z][a-z\s]{2,})[\s]*$', 'level': 3},
        
        # Pattern for bullet point headings (often subsections)
        {'pattern': r'^[\s]*[•\-\*]\s+([A-Z].+?)[\s]*$', 'level': 3}
    ]
    
    lines = text.split('\n')
    line_num = 0
    prev_heading_level = 0
    prev_heading = None  # Initialize prev_heading to avoid unbound variable error
    heading_contexts = {}  # Store contexts for nested headings
    
    for line in lines:
        line_num += 1
        line = line.strip()
        
        if not line:
            continue
        
        # Try each pattern
        for pattern_info in heading_patterns:
            pattern = pattern_info['pattern']
            base_level = pattern_info['level']
            
            matches = re.match(pattern, line)
            if matches:
                heading_text = matches.group(1).strip()
                
                # Skip very short headings (likely false positives)
                if len(heading_text) < 3:
                    continue
                
                # Determine heading level based on text properties and context
                level = base_level
                
                # Adjust level based on text characteristics
                if heading_text.isupper() and len(heading_text) > 5:
                    level = min(level, 1)  # Main headings in ALL CAPS
                elif heading_text.isupper():
                    level = min(level, 2)  # Shorter all-caps are likely level 2
                
                # Check for indentation (increase level for indented content)
                leading_spaces = len(line) - len(line.lstrip())
                if leading_spaces > 4:
                    level += 1
                
                # Handle nested headings (headings within sections)
                if prev_heading is not None and prev_heading_level > 0 and level > prev_heading_level:
                    # This is a subheading of the previous heading
                    heading_contexts[level] = heading_contexts.get(prev_heading_level, []) + [prev_heading]
                
                # Create heading entry with parent information
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
    
    Args:
        text: Full text content
        headings: List of heading dictionaries
        
    Returns:
        List of dictionaries with section information including boundaries
    """
    lines = text.split('\n')
    total_lines = len(lines)
    sections = []
    
    for i, heading in enumerate(headings):
        section = heading.copy()
        start_line = heading['line_number']
        
        # Find end line (either the next heading or end of text)
        if i < len(headings) - 1:
            end_line = headings[i + 1]['line_number'] - 1
        else:
            end_line = total_lines
        
        # Extract section content
        section_content = '\n'.join(lines[start_line:end_line]).strip()
        
        # Add section information
        section['start_line'] = start_line
        section['end_line'] = end_line
        section['content'] = section_content
        
        sections.append(section)
    
    return sections


def extract_tables(page) -> List[Dict[str, Any]]:
    """
    Extract tables from a PDF page using pdfplumber.
    
    Args:
        page: A pdfplumber page object
        
    Returns:
        List of dictionaries containing table information
    """
    tables = []
    
    try:
        # Extract tables using pdfplumber's table finder
        page_tables = page.extract_tables()
        
        for i, table_data in enumerate(page_tables):
            if not table_data or len(table_data) == 0:
                continue
                
            # Process the table data
            processed_table = []
            header_row = []
            
            # Try to identify header row (usually first row)
            if len(table_data) > 0:
                header_row = [str(cell).strip() if cell is not None else "" for cell in table_data[0]]
            
            # Process all rows
            for row in table_data:
                processed_row = [str(cell).strip() if cell is not None else "" for cell in row]
                processed_table.append(processed_row)
            
            # Calculate table boundaries on the page
            table_bbox = None
            try:
                # This might not always be available depending on how the table was extracted
                if hasattr(page, 'find_tables') and page.find_tables():
                    table_settings = {
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                        "snap_tolerance": 3,
                        "join_tolerance": 3,
                        "edge_min_length": 3,
                        "min_words_vertical": 2,
                        "min_words_horizontal": 2
                    }
                    found_tables = page.find_tables(table_settings)
                    if i < len(found_tables):
                        table_bbox = {
                            'x0': found_tables[i].bbox[0],
                            'y0': found_tables[i].bbox[1],
                            'x1': found_tables[i].bbox[2],
                            'y1': found_tables[i].bbox[3]
                        }
            except Exception as e:
                pass
            
            # Create the table entry
            table_entry = {
                'table_number': i + 1,
                'header': header_row,
                'data': processed_table,
                'rows': len(processed_table),
                'columns': len(header_row) if header_row else (len(processed_table[0]) if processed_table else 0),
                'boundaries': table_bbox,
                'extraction_method': 'pdfplumber'
            }
            
            tables.append(table_entry)
            
    except Exception as e:
        logger.warning(f"Error extracting tables with pdfplumber: {str(e)}")
    
    return tables


def extract_tables_with_camelot(pdf_path: str, page_num: int) -> List[Dict[str, Any]]:
    """
    Extract tables from a PDF page using camelot, which is more advanced
    for complex tables than pdfplumber.
    
    Args:
        pdf_path: Path to the PDF file
        page_num: Page number (0-indexed, but camelot uses 1-indexed pages)
        
    Returns:
        List of dictionaries containing table information
    """
    if not HAS_ADVANCED_LIBRARIES:
        logger.warning("Camelot library not available. Skipping advanced table extraction.")
        return []
    
    tables = []
    
    try:
        # Camelot uses 1-indexed page numbers
        camelot_page_num = page_num + 1
        
        # Try both lattice and stream methods for table detection
        lattice_tables = camelot.read_pdf(
            pdf_path, 
            pages=str(camelot_page_num), 
            flavor='lattice',
            suppress_stdout=True
        )
        
        stream_tables = camelot.read_pdf(
            pdf_path, 
            pages=str(camelot_page_num), 
            flavor='stream',
            suppress_stdout=True
        )
        
        # Process lattice tables (tables with visible borders)
        for i, table in enumerate(lattice_tables):
            if table.df.empty:
                continue
                
            # Convert pandas dataframe to list of lists
            data = table.df.values.tolist()
            header = table.df.columns.tolist() if not table.df.columns.equals(pd.RangeIndex(start=0, stop=len(table.df.columns))) else data[0] if data else []
            
            table_entry = {
                'table_number': i + 1,
                'header': header,
                'data': data,
                'rows': len(data),
                'columns': len(header) if header else (len(data[0]) if data and data[0] else 0),
                'accuracy': table.accuracy,
                'whitespace': table.whitespace,
                'extraction_method': 'camelot-lattice'
            }
            
            tables.append(table_entry)
        
        # Process stream tables (tables without visible borders)
        for i, table in enumerate(stream_tables):
            # Skip already processed tables with high accuracy
            if any(t['extraction_method'] == 'camelot-lattice' and t['accuracy'] > 90 for t in tables):
                continue
                
            if table.df.empty:
                continue
                
            # Convert pandas dataframe to list of lists
            data = table.df.values.tolist()
            header = table.df.columns.tolist() if not table.df.columns.equals(pd.RangeIndex(start=0, stop=len(table.df.columns))) else data[0] if data else []
            
            table_entry = {
                'table_number': len(tables) + 1,
                'header': header,
                'data': data,
                'rows': len(data),
                'columns': len(header) if header else (len(data[0]) if data and data[0] else 0),
                'accuracy': table.accuracy,
                'whitespace': table.whitespace,
                'extraction_method': 'camelot-stream'
            }
            
            tables.append(table_entry)
            
    except Exception as e:
        pass
    
    return tables


def extract_spatial_elements(page) -> List[Dict[str, Any]]:
    """
    Extract text elements with their spatial information from a PDF page.
    This is useful for analyzing document layout and finding text by location.
    
    Args:
        page: A pdfplumber page object
        
    Returns:
        List of dictionaries containing text elements with spatial information
    """
    elements = []
    
    try:
        # Extract words with spatial information
        words = page.extract_words()
        
        for i, word in enumerate(words):
            element = {
                'element_id': i + 1,
                'text': word.get('text', ''),
                'x0': word.get('x0', 0),
                'x1': word.get('x1', 0),
                'y0': word.get('top', 0),  # pdfplumber uses 'top' instead of 'y0'
                'y1': word.get('bottom', 0),  # pdfplumber uses 'bottom' instead of 'y1'
                'width': word.get('width', 0),
                'height': word.get('height', 0),
                'type': 'word'
            }
            
            elements.append(element)
        
        # Extract lines with spatial information
        lines = page.lines
        
        for i, line in enumerate(lines):
            element = {
                'element_id': len(elements) + i + 1,
                'x0': line.get('x0', 0),
                'x1': line.get('x1', 0),
                'y0': line.get('top', 0),
                'y1': line.get('bottom', 0),
                'width': line.get('width', 0),
                'height': line.get('height', 0),
                'type': 'line'
            }
            
            elements.append(element)
        
        # Extract rectangles/boxes with spatial information
        rects = page.rects
        
        for i, rect in enumerate(rects):
            element = {
                'element_id': len(elements) + i + 1,
                'x0': rect.get('x0', 0),
                'x1': rect.get('x1', 0),
                'y0': rect.get('top', 0),
                'y1': rect.get('bottom', 0),
                'width': rect.get('width', 0),
                'height': rect.get('height', 0),
                'type': 'rectangle'
            }
            
            elements.append(element)
            
    except Exception as e:
        pass
    
    return elements


# OCR implementation removed as per user request
# Enterprise OCR tools will be used separately instead


def process_page(page, page_num: int, pdf_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Process a single PDF page to extract information.
    
    Args:
        page: pdfplumber page object
        page_num: Page number (0-indexed)
        pdf_path: Path to the PDF file
        output_dir: Directory to save images for scanned pages
        
    Returns:
        Dictionary containing page information
    """
    page_info = {
        'page_number': page_num + 1,
        'is_scanned': False,
        'image_path': None,
        'text': None,
        'headings': [],
        'sections': [],
        'tables': [],
        'spatial_elements': []
    }
    
    # Check if the page is scanned or contains digital text
    if is_page_scanned(page):
        page_info['is_scanned'] = True
        page_info['image_path'] = save_page_as_image(pdf_path, page_num, output_dir)
    else:
        # Extract text content
        text = page.extract_text()
        page_info['text'] = text
        
        # Extract headings and sections
        if text:
            page_info['headings'] = extract_headings(text)
            page_info['sections'] = find_section_boundaries(text, page_info['headings'])
        
        # Extract basic tables with pdfplumber
        page_info['tables'] = extract_tables(page)
        
        # Extract spatial elements for better layout analysis
        page_info['spatial_elements'] = extract_spatial_elements(page)
        
        # Try advanced table extraction with Camelot if available
        if HAS_ADVANCED_LIBRARIES:
            try:
                camelot_tables = extract_tables_with_camelot(pdf_path, page_num)
                
                # Only use Camelot tables if they provide better results
                if camelot_tables and (
                    len(camelot_tables) > len(page_info['tables']) or
                    any(table.get('accuracy', 0) > 80 for table in camelot_tables)
                ):
                    # Replace basic tables with advanced tables
                    page_info['tables'] = camelot_tables
                    logger.info(f"Using advanced table extraction for page {page_num+1}")
                elif camelot_tables:
                    # Add any additional tables found by Camelot that weren't found by pdfplumber
                    existing_table_count = len(page_info['tables'])
                    for i, table in enumerate(camelot_tables):
                        table['table_number'] = existing_table_count + i + 1
                        page_info['tables'].append(table)
            except Exception as e:
                pass
    
    # Add page dimensions
    page_info['width'] = page.width
    page_info['height'] = page.height
    
    return page_info


def extract_document_metadata(pdf, pdf_path: str) -> Dict[str, Any]:
    """
    Extract metadata from a PDF document.
    
    Args:
        pdf: PyPDF2 PdfReader object
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary containing document metadata
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
    
    # Extract metadata from the PDF if available
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
    
    Args:
        document_info: Dictionary containing document information
        
    Returns:
        Dictionary containing document summary with enhanced information about
        document structure, tables, and spatial elements.
    """
    summary = {
        'title': document_info.get('title', 'Untitled Document'),
        'author': document_info.get('author', 'Unknown Author'),
        'total_pages': document_info.get('pages', 0),
        'digital_pages': 0,
        'scanned_pages': 0,
        'headings_count': len(document_info.get('headings', [])),
        'heading_hierarchy': {},
        'tables_count': 0,
        'table_extraction_methods': {},
        'spatial_elements_count': 0,
        'element_types': {},
        'creation_date': document_info.get('creation_date'),
        'document_type': 'Unknown',
        'document_structure': {
            'has_toc': False,
            'has_bibliography': False,
            'has_appendix': False,
            'section_count': 0
        }
    }
    
    # Count digital vs scanned pages
    pages_info = document_info.get('pages', [])
    for page in pages_info:
        if page.get('is_scanned', False):
            summary['scanned_pages'] += 1
        else:
            summary['digital_pages'] += 1
            
            # Count tables and track extraction methods
            tables = page.get('tables', [])
            summary['tables_count'] += len(tables)
            
            # Track table extraction methods
            for table in tables:
                method = table.get('extraction_method', 'unknown')
                summary['table_extraction_methods'][method] = summary['table_extraction_methods'].get(method, 0) + 1
            
            # Count spatial elements and track types
            spatial_elements = page.get('spatial_elements', [])
            summary['spatial_elements_count'] += len(spatial_elements)
            
            # Track element types
            for element in spatial_elements:
                element_type = element.get('type', 'unknown')
                summary['element_types'][element_type] = summary['element_types'].get(element_type, 0) + 1
    
    # Analyze heading hierarchy
    headings = document_info.get('headings', [])
    heading_levels = {}
    for heading in headings:
        level = heading.get('level', 0)
        heading_levels[level] = heading_levels.get(level, 0) + 1
    
    summary['heading_hierarchy'] = heading_levels
    
    # Count total sections (main + subsections)
    if heading_levels:
        summary['document_structure']['section_count'] = sum(heading_levels.values())
    
    # Try to determine document type based on content
    if headings:
        # Look at main headings to determine document type
        main_headings_text = ' '.join([h['text'] for h in headings if h.get('level', 0) == 1])
        main_headings_text = main_headings_text.lower()
        
        if 'report' in main_headings_text or 'case' in main_headings_text:
            summary['document_type'] = 'Case Report/Medical Document'
        elif 'chapter' in main_headings_text or 'section' in main_headings_text:
            summary['document_type'] = 'Book/Publication'
        elif 'abstract' in main_headings_text or 'introduction' in main_headings_text:
            summary['document_type'] = 'Academic Paper'
        elif 'resume' in main_headings_text or 'cv' in main_headings_text:
            summary['document_type'] = 'Resume/CV'
        
        # Check for table of contents, bibliography, appendix
        for heading in headings:
            heading_text = heading.get('text', '').lower()
            if 'content' in heading_text or 'toc' in heading_text:
                summary['document_structure']['has_toc'] = True
            elif 'bibliography' in heading_text or 'references' in heading_text:
                summary['document_structure']['has_bibliography'] = True
            elif 'appendix' in heading_text:
                summary['document_structure']['has_appendix'] = True
    
    # Additional document analysis
    text_content = ''
    for page in pages_info:
        if page.get('text'):
            text_content += page.get('text', '') + ' '
    
    # Estimate average words per page
    if summary['digital_pages'] > 0:
        word_count = len(text_content.split())
        summary['avg_words_per_page'] = round(word_count / summary['digital_pages'], 1)
    
    return summary


def process_pdf(pdf_path: str, output_dir: str = "./images") -> Dict[str, Any]:
    """
    Process a PDF document to extract structured information.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save images for scanned pages
        
    Returns:
        Dictionary containing structured document information
    """
    logger.info(f"Processing PDF: {pdf_path}")
    
    try:
        # Open the PDF with PyPDF2 for metadata
        pdf_reader = PdfReader(pdf_path)
        
        # Extract document metadata
        document_info = extract_document_metadata(pdf_reader, pdf_path)
        
        # Process each page
        pages_info = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                logger.info(f"Processing page {page_num+1} of {len(pdf.pages)}")
                page_info = process_page(page, page_num, pdf_path, output_dir)
                pages_info.append(page_info)
        
        # Combine all information into a single document structure
        document_info['pages'] = pages_info
        
        # Extract document-level headings and sections
        all_headings = []
        for page in pages_info:
            for heading in page.get('headings', []):
                heading_copy = heading.copy()
                heading_copy['page_number'] = page['page_number']
                all_headings.append(heading_copy)
        
        document_info['headings'] = all_headings
        
        # Generate document summary
        document_info['summary'] = generate_document_summary(document_info)
        
        return document_info
    
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return {
            'error': str(e),
            'filename': os.path.basename(pdf_path),
            'path': pdf_path
        }


def save_json_output(document_info: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """
    Save document information as JSON.
    
    Args:
        document_info: Dictionary containing document information
        output_path: Path to save the JSON file (optional)
        
    Returns:
        Path to the saved JSON file or JSON string if output_path is None
    """
    if output_path is None:
        # Generate output filename based on input filename
        if 'filename' in document_info:
            base_name = os.path.splitext(document_info['filename'])[0]
            output_path = f"{base_name}.json"
        else:
            output_path = "pdf_output.json"
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    
    # Convert document info to JSON
    json_data = json.dumps(document_info, indent=2)
    
    # Save JSON to file
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
    args = parser.parse_args()
    
    # Process the PDF document
    document_info = process_pdf(args.pdf_path, args.images_dir)
    
    # Save the JSON output
    json_path = save_json_output(document_info, args.output)
    
    print(f"PDF processing complete. JSON output saved to: {json_path}")
    
    # Print the JSON output to console
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = f.read()
    print("\nJSON Output:")
    print(json_data)


if __name__ == "__main__":
    main()
