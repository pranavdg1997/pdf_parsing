# PDF Processing Pipeline

A comprehensive Python-based PDF processing pipeline that extracts structured information from PDF documents using advanced parsing techniques. This tool analyzes PDF files to extract text, headings, tables, and document structure information, outputting the results in a structured JSON format.

## Features

- **Digital Text Extraction**: Extract and analyze text content from PDF documents
- **Scanned Page Detection**: Identify scanned/image-based pages and save them as PNG files
- **Heading Detection**: Advanced heading identification with hierarchy analysis
- **Section Boundary Detection**: Identify boundaries between document sections
- **Advanced Table Extraction**: Extract tables using both simple and advanced methods
- **Spatial Element Analysis**: Extract positional information of text elements for layout analysis
- **Document Structure Analysis**: Detect document structure including TOC, bibliography, and appendices
- **Document Type Detection**: Automatic classification of document types (e.g., case reports, academic papers)
- **Metadata Extraction**: Extract and process document metadata information

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r dependencies.txt
```

Key dependencies:
- pdfplumber: For text extraction and layout analysis
- PyPDF2: For PDF document analysis
- camelot-py: For advanced table extraction
- opencv-python-headless: For image processing
- Pillow: For image handling and manipulation
- pandas: For data manipulation with tables
- ghostscript: Required by camelot for PDF processing

## Usage

### Basic Usage
```bash
python pdf_processor.py path/to/your/document.pdf
```

This will process the PDF and generate a JSON output file with the extracted information.

### Command-line Arguments
```
usage: pdf_processor.py [-h] [--output OUTPUT] [--images-dir IMAGES_DIR] pdf_path

Process a PDF document and generate JSON output

positional arguments:
  pdf_path              Path to the PDF file

options:
  -h, --help            show this help message and exit
  --output OUTPUT, -o OUTPUT
                        Path to save the JSON output
  --images-dir IMAGES_DIR, -i IMAGES_DIR
                        Directory to save images for scanned pages
```

### View Document Summary
To view a summary of the processed document:

```bash
python view_summary.py path/to/output.json
```

### Test Scan Detection
To test the scan detection functionality:

```bash
python test_scan_detection.py path/to/your/document.pdf
```

## Output Structure

The JSON output includes:

- **metadata**: Document metadata (title, author, creation date, etc.)
- **pages**: Array of information for each page, including:
  - Text content
  - Tables
  - Headings
  - Spatial elements
  - Scanned/digital status
- **headings**: Document-level heading information with hierarchy
- **summary**: Comprehensive document summary including:
  - Document type
  - Content statistics
  - Structure analysis

## Advanced Features

### Table Extraction
Tables are extracted using two methods:
1. **Basic**: Using pdfplumber for simple tables
2. **Advanced**: Using camelot for complex tables with lattice and stream methods

### Spatial Analysis
The tool extracts spatial information about text elements, including:
- Words with their coordinates and dimensions
- Lines and rectangles
- Hierarchical layout information

### Heading Hierarchy Analysis
The tool analyzes heading structures to determine:
- Heading levels (1-3)
- Parent-child relationships
- Section boundaries

### Document Type Detection
The tool attempts to classify documents into categories:
- Case Report/Medical Document
- Book/Publication
- Academic Paper
- Resume/CV

## OCR Integration

This tool identifies scanned pages and saves them as PNG images suitable for OCR processing. The OCR implementation is left to separate enterprise OCR tools and is not included in this pipeline.

## Limitations

- OCR for scanned pages requires external tools
- Some complex PDF structures may not be parsed correctly
- PDF files with security restrictions may not be fully parsed

## License

This project is open-source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.