import os
import pandas as pd

from functions.pdf_preprocess import (
    download_guidelines,
    process_pdf_file,
    extract_year_regex
)
from functions.text_preprocess import process_text_data

import config


def get_reports_df(urls):
    guideline_text = []

    download_guidelines(urls, guideline_text=guideline_text, pdfs_path=config.DOWNLOAD_PATH)
    
    # Create a DataFrame
    columns = ["file_name", "file_link", "page_number", "paragraph", "text"]
    guideline_text_df = pd.DataFrame(
        data=guideline_text,
        columns=columns,
    )

    guideline_text_df["year"] = guideline_text_df["file_name"].apply(extract_year_regex).astype(str).astype(int)
    guideline_text_df
    guideline_text_df["timestamp"] = pd.to_datetime("2024-12-31")

    # Process text data using the process_text_data function
    guideline_text_processed_df = process_text_data(guideline_text_df)
    guideline_text_processed_df["source"] = "administrative_protocols"

    # Reorder columns and rename
    guideline_text_processed_df = guideline_text_processed_df[
        ['file_name', 'file_link', 'source', 'page_number', 'paragraph', 'text', 'year', 'timestamp']]
    guideline_text_processed_df.columns = ['name', 'url', 'source', 'page_number', 'paragraph', 'text', 'year',
                                           'timestamp']

    return guideline_text_processed_df
