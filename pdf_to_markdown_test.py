
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import numpy as np
import tabula
import pdfplumber
import markdown
import streamlit as st
import os 
from datetime import datetime
import time   


def save_uploaded_file(path, uploaded_file):
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())

def is_table_present_on_page(pdf_path, page_number):
    # print(f"IDENTIFYING TABLES IN {page_number}")
    try:
        tables = tabula.read_pdf(pdf_path, pages=page_number, multiple_tables=True, lattice=True)
        # for index,table in enumerate(tables, start = 1):
        #     print(f'INDEX ------{index}')
        #     print(f'TABLE ------{table}')
        # print("-----------------------------------")
        return len(tables) > 0
    except Exception as e:
        return False

def pdf_to_markdown(pdf_path, output_path):
    # Read the PDF using pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        # Initialize an empty Markdown string
        final_markdown = ""

        # Loop through each page of the PDF
        for page_number, page in enumerate(pdf.pages, 1):
            # Check if the page contains a table
            if is_table_present_on_page(pdf_path, page_number):
                # Extract the table using tabula-py
                dfs = tabula.read_pdf(pdf_path, pages=page_number, lattice=True)
                # print(f"PAGE NUMBER : {page_number} {len(tabula.read_pdf(pdf_path, pages=page_number, lattice=True))}")
                # Replace line breaks in each cell with markdown line breaks
                for df in dfs:
                    for i, row in df.iterrows():
                        for j, cell in row.items():
                            if isinstance(cell, str):
                                df.at[i, j] = cell.replace('\n', '<br>')

                    df = df.dropna(axis=1, how='all')

                    # Convert the DataFrame to a markdown table
                    markdown_table = df.to_markdown(index=False)
                    # print(markdown_table)
                    final_markdown += markdown_table + "\n"
            else:
                # Extract text from the page
                text = page.extract_text()
                # print(text)
                final_markdown += text + "\n"

    st.write("Markdown file created!")
    st.markdown(final_markdown)

    # Write the final Markdown content to a file
    with open(output_path, "w", encoding="utf-8") as md_file:
        md_file.write(final_markdown)


def main():
    st.title("PDF Text Extractor")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None:
        # Step 2: Display Page Numbers as Checkboxes
        st.title("Uploaded PDF")
        
        temp_pdf_path = os.path.join("pdf_files", f"{os.path.splitext(uploaded_file.name)[0]}{int(time.time())}.pdf")
        save_uploaded_file(temp_pdf_path, uploaded_file)

        if st.button("Convert To Markdown"):
            markdown_pdf_path = os.path.join("markdown_files", f"{os.path.splitext(uploaded_file.name)[0]}{int(time.time())}.md")
            # print("Creating markdown file..")
            pdf_to_markdown(temp_pdf_path, markdown_pdf_path)
            



main()