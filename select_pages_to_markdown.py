from langchain import LLMChain
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownTextSplitter
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import numpy as np
import tabula
import chromadb
import pdfplumber
import markdown
import streamlit as st
from PyPDF2 import PdfReader, PdfWriter
import io
import base64
import os 
from datetime import datetime
import time   

selected_page_nums = []
selected_pages = []
# Step 1: Upload PDF
st.title("PDF Text Extractor")
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    # Step 2: Display Page Numbers as Checkboxes
    st.title("Uploaded PDF")
    
    pdf_reader = PdfReader(uploaded_file)
    total_pages = len(pdf_reader.pages)
    all_page_nums = range(1, total_pages + 1)
    
    selected_page_nums = st.multiselect("Select page numbers", all_page_nums)  # Adjust range
    
    # Step 3: Create a new PDF with Selected Pages
    if selected_page_nums:
        st.title("Selected Pages")
        
        new_pdf_bytes = io.BytesIO()
        new_pdf_writer = PdfWriter()
        selected_pages = []
        
        for page_num in selected_page_nums:
            new_pdf_writer.add_page(pdf_reader.pages[page_num - 1])
            selected_pages.append(pdf_reader.pages[page_num - 1])
            
        new_pdf_writer.write(new_pdf_bytes)
        new_pdf_bytes.seek(0)
        st.write(f'<iframe src="data:application/pdf;base64,{base64.b64encode(new_pdf_bytes.getvalue()).decode("utf-8")}" width="800" height="600"></iframe>', unsafe_allow_html=True)

    # Step 4: Add a search button to return the selected page numbers
    temp_pdf_path = os.path.join("temp_pdf_files", f"{os.path.splitext(uploaded_file.name)[0]}{int(time.time())}.pdf")
    if st.button("Search"):
        with open(temp_pdf_path, "wb") as f:
            f.write(new_pdf_bytes.getvalue())

        def callAPI(query):
            def is_table_present_on_page(pdf_path, page_number):
                try:
                    tables = tabula.read_pdf(pdf_path, pages=page_number, multiple_tables=True, lattice=True)
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
                            df = tabula.read_pdf(pdf_path, pages=page_number, lattice=True)[0]
                            # Replace line breaks in each cell with markdown line breaks
                            for i, row in df.iterrows():
                                for j, cell in row.items():
                                    if isinstance(cell, str):
                                        df.at[i, j] = cell.replace('\n', '<br>')

                            df = df.dropna(axis=1, how='all')

                            # Convert the DataFrame to a markdown table
                            markdown_table = df.to_markdown(index=False)
                            print(markdown_table)
                            final_markdown += markdown_table + "\n\n"
                        else:
                            # Extract text from the page
                            text = page.extract_text()
                            final_markdown += text + "\n\n"

                # Write the final Markdown content to a file
                with open(output_path, "w", encoding="utf-8") as md_file:
                    md_file.write(final_markdown)

            markdown_pdf_path = os.path.join("markdown_pdf_files", f"{os.path.splitext(uploaded_file.name)[0]}{int(time.time())}.md")
            print("Creating markdown file..")
            pdf_to_markdown(temp_pdf_path, markdown_pdf_path)
            print("Markdown file created!")

            #STEP 1 - Load the document
            print("Creating embeddings..")
            loader = TextLoader(markdown_pdf_path, encoding="utf-8")
            # loader = UnstructuredMarkdownLoader("./FinalOutput3.md")
            documents = loader.load()

            # STEP 2 - split the documents into chunks
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_documents(documents)

            # STEP 3 - create the vectorestore to use as the index
            embeddings = OpenAIEmbeddings()

            # STEP 4 - Create the Chroma index from the list of embeddings
            db = Chroma.from_documents(texts, embeddings)
            print("Embeddings created!")

            #STEP 4 - expose this index in a retriever interface
            retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":5})

            #STEP 5 - Define structured outout parser
            parser = StructuredOutputParser.from_response_schemas(
                response_schemas=[
                    ResponseSchema(
                        name="array",
                        description="""array of json objects in the following format: [
                                    {{ "name": string // name of the site/location', 
                                        "address": string // address of that site/location', 
                                        "acreage": string //acreage of the location,
                                        "notes" : string //any notes regarding the location/site,
                                        }}
                                    ]""",
                    ),
                ]
            )
            format_instructions = parser.get_format_instructions()

            #Step 7- define template and QA Retrieval chain
            template = """Use the following pieces of context to answer the question at the end. If you don't find releveant information keep that json field empty, don't try to make up an answer.
            {context}

            text: {question}
            Format the response as an array of JSON objects.
            """
            PROMPT = PromptTemplate(
                template=template, input_variables=["context", "question"]
            )

            chain_type_kwargs = {"prompt": PROMPT}

            qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model = "gpt-4"), chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs, return_source_documents = True)

            #STEP 8 - Send query
            # query = "Provide me a list of all the sites."
            print("Calling gpt api")
            result =  qa({"query": query})
            print(result["result"])
            # src_docs = result["source_documents"]
            # for doc in src_docs:
            #     print(doc, "\n\n\n")

            st.text_area("Output", value=result["result"], height=500)
            # final_output = (parser.parse(result["result"]))['array']
            # for item in final_output:
            #     print(item, "\n")

            # # with get_openai_callback() as cb:
            # #     qa.run(query)
            # #     print(cb)

        callAPI("Provide me a list of all the work/task that needs to be performed along with it's frequency and the locations where it needs to be done.")


