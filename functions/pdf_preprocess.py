import os
import re
import requests
# import PyPDF2
# Use PyMuPDF (fitz) instead of PyPDF2 for better table detection
import fitz  # PyMuPDF

from typing import List, Dict, Union
from pathlib import Path


def download_pdf(url, output_path):
    """
    Downloads a PDF from a given URL and saves it to output_path.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded PDF: {output_path}")
    except requests.RequestException as e:
        print(f"Failed to download {url}. Error: {e}")


def get_file_paths(directory_path):
    """
    Get a list of file paths from a directory.

    Args:
        directory_path (str): Path to the directory

    Returns:
        list: List of strings where each string is a full file path
    """
    file_paths = []

    # Check if the path exists and is a directory
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        # Walk through the directory
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                # Create full file path as string
                file_path = os.path.join(root, file)  # This returns a string
                file_paths.append(file_path)
    else:
        print(f"The path {directory_path} does not exist or is not a directory.")

    return file_paths  # Returns a list of strings


def download_guidelines(urls: List[str], guideline_text: List, pdfs_path: str = 'data') -> List[str]:
    """
    Download pdf files
    """

    for url in urls:
        filename = url.split('/')[-1]
        # Output filenames
        pdf_file = f"{pdfs_path}/{filename}"
        # Initialize a list to store information about new files
        print('⛳️ Dowloading guideline...')

        # TODO (enable if some kind of url will be provided): Download each
        # download_pdf(url, pdf_file)

        process_pdf_file(guideline_text, pdf_file, url)

    return guideline_text


def process_pdf_file_bk(document_text: List,
                     pdf_path: str, url: str) -> List:
    """
    Process content of a PDF file and append information to the document_text list.

    Parameters:
    - file_info (Dict): Information about the PDF file.
    - document_text (List): List containing document information.
    - pdf_path (str): Path to the folder containing PDF files (default is 'data/').

    Returns:
    - List: Updated document_text list.
    """
    if pdf_path.split('.')[-1] == 'pdf':
        file_path = Path(pdf_path)
        file_title = file_path.stem
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        pages_amount = len(pdf_reader.pages)
        print(f'Amount of pages: {pages_amount} in {file_title}')

        for i, page in enumerate(pdf_reader.pages):
            document_text.append([file_title, url, i + 1, page.extract_text()])
    return document_text


def process_pdf_file(document_text: List, pdf_path: str, url: str) -> List:
    """
    Process content of a PDF file and append information to the document_text list,
    with improved table handling for semantic search.
    """
    if pdf_path.split('.')[-1] == 'pdf':
        file_path = Path(pdf_path)
        file_title = file_path.stem
        
        # Use PyMuPDF (fitz)
        import fitz
        
        doc = fitz.open(pdf_path)
        pages_amount = len(doc)
        print(f'Amount of pages: {pages_amount} in {file_title}')
        
        for page_num in range(pages_amount):
            page = doc[page_num]
            
            # Extract text with blocks to better identify paragraphs
            blocks = page.get_text("blocks")
            
            # Process each block (paragraph) separately
            for block_idx, block in enumerate(blocks):
                # Each block is a tuple: (x0, y0, x1, y1, "text", block_no, block_type)
                block_text = block[4]
                
                # Try to detect if this is a table-like structure
                lines = block_text.split('\n')
                if len(lines) > 1:
                    # Check if it's likely a numbered list (common in protocols)
                    is_numbered_list = all(line.strip().startswith(str(i+1) + '.') for i, line in enumerate(lines[:3])) if len(lines) >= 3 else False
                    
                    if is_numbered_list:
                        # Format numbered lists to preserve relationships
                        for line in lines:
                            line = line.strip()
                            if line:
                                # Add each instruction as a separate entry for better semantic matching
                                document_text.append([
                                    file_title,
                                    url,
                                    page_num + 1,
                                    block_idx + 1,  # Block/paragraph number
                                    line
                                ])
                    else:
                        # Check if it's likely a table based on column alignment
                        potential_table = False
                        if len(lines) > 2:
                            # Check for consistent whitespace that might indicate columns
                            spaces_per_line = [len(line) - len(line.lstrip()) for line in lines]
                            if len(set(spaces_per_line[:3])) == 1:  # Consistent indentation in first 3 lines
                                potential_table = True
                        
                        if potential_table:
                            # For table-like content, try to preserve column relationships
                            if len(lines) > 1:  # At least header and one row
                                header_line = lines[0]
                                
                                # Skip the header line - common in Swedish medical protocols
                                for data_line in lines[1:]:
                                    # Add context that this is likely part of a tabular structure
                                    entry = f"Tabelldata: {data_line} (Del av tabell med rubrik: {header_line})"
                                    document_text.append([
                                        file_title,
                                        url,
                                        page_num + 1,
                                        block_idx + 1,
                                        entry
                                    ])
                            else:
                                document_text.append([
                                    file_title,
                                    url,
                                    page_num + 1,
                                    block_idx + 1,
                                    block_text
                                ])
                        else:
                            # Regular paragraph
                            document_text.append([
                                file_title,
                                url,
                                page_num + 1,
                                block_idx + 1,
                                block_text
                            ])
                else:
                    # Single line block
                    document_text.append([
                        file_title,
                        url,
                        page_num + 1,
                        block_idx + 1,
                        block_text
                    ])
                    
    return document_text


def extract_year_regex(filename):
    pattern = re.compile(r'^(?P<start>\d{4})|(?P<end>\d{4})')
    match = pattern.search(filename)
    if match:
        # If 'start' group is matched, return it; else return 'end'
        return match.group('start') or match.group('end')
    return None
