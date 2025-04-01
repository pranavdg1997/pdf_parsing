#!/usr/bin/env python3
import json
import sys
from collections import OrderedDict

def main():
    """Extract and print the summary section from a PDF processor output JSON file."""
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <json_file>")
        sys.exit(1)
        
    json_file = sys.argv[1]
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if 'summary' not in data:
            print("No summary section found in the JSON file.")
            sys.exit(1)
            
        summary = data['summary']
        
        print("\n====== PDF DOCUMENT SUMMARY ======")
        print(f"Title: {summary.get('title', 'Unknown')}")
        print(f"Author: {summary.get('author', 'Unknown')}")
        print(f"Document Type: {summary.get('document_type', 'Unknown')}")
        print(f"Creation Date: {summary.get('creation_date', 'Unknown')}")
        
        print("\n--- Document Structure ---")
        print(f"Total Pages: {summary.get('total_pages', 0)}")
        print(f"Digital Pages: {summary.get('digital_pages', 0)}")
        print(f"Scanned Pages: {summary.get('scanned_pages', 0)}")
        if 'avg_words_per_page' in summary:
            print(f"Average Words Per Page: {summary.get('avg_words_per_page', 0)}")
            
        doc_structure = summary.get('document_structure', {})
        print(f"Total Sections: {doc_structure.get('section_count', 0)}")
        print(f"Has Table of Contents: {doc_structure.get('has_toc', False)}")
        print(f"Has Bibliography/References: {doc_structure.get('has_bibliography', False)}")
        print(f"Has Appendix: {doc_structure.get('has_appendix', False)}")
        
        print("\n--- Content Analysis ---")
        print(f"Total Headings: {summary.get('headings_count', 0)}")
        print(f"Tables Count: {summary.get('tables_count', 0)}")
        print(f"Spatial Elements: {summary.get('spatial_elements_count', 0)}")
        
        print("\nHeading Hierarchy:")
        if summary.get('heading_hierarchy'):
            for level, count in sorted(summary['heading_hierarchy'].items()):
                print(f"  Level {level}: {count} headings")
        else:
            print("  No heading hierarchy information available.")
            
        # Print table extraction methods if available
        if 'table_extraction_methods' in summary and summary['table_extraction_methods']:
            print("\nTable Extraction Methods:")
            methods = summary['table_extraction_methods']
            for method, count in methods.items():
                print(f"  {method}: {count} tables")
                
        # Print element types if available
        if 'element_types' in summary and summary['element_types']:
            print("\nSpatial Element Types:")
            types = summary['element_types']
            for element_type, count in types.items():
                print(f"  {element_type}: {count} elements")
        
        print("\n==================================")
        
    except Exception as e:
        print(f"Error processing JSON file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()