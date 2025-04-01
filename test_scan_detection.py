#!/usr/bin/env python3
"""
Test script for scanned page detection and image extraction
"""

import os
import pdfplumber
from pdf_processor import is_page_scanned, save_page_as_image

def test_scan_detection(pdf_path: str):
    """Test the scan detection and image extraction functionality"""
    print(f"Testing scan detection on: {pdf_path}")
    os.makedirs("./test_images", exist_ok=True)
    
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            is_scanned = is_page_scanned(page)
            print(f"Page {i+1}: {'SCANNED' if is_scanned else 'DIGITAL'}")
            
            # Save a test image for the first page anyway for debugging
            if i == 0 or is_scanned:
                image_path = save_page_as_image(pdf_path, i, "./test_images")
                if image_path:
                    print(f"  Saved image: {image_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <pdf_path>")
        sys.exit(1)
        
    test_scan_detection(sys.argv[1])