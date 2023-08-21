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
import chromadb

from dotenv import load_dotenv
load_dotenv()


#METHOD 1 -> DIRECT EMBEDDINGS FROM TEXT

#STEP 1 - Load the document
# loader = PyPDFLoader("D:\Projects\LPMS\docs\Mowing-Chemical Treatment Services.pdf")
loader = PyPDFLoader("docs\Annual Parks Bid.pdf")
documents = loader.load()
# loader = CSVLoader(file_path="table_7.csv", encoding="utf-8", csv_args={
#                 'delimiter': ','})
# documents = loader.load()

# STEP 2 - split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# STEP 3 - create the vectorestore to use as the index
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(texts, embeddings)
# collection = client.get_collection(name="langchain")
# collection.delete();

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
{format_instructions}
Format the response as an array of JSON objects.
"""
PROMPT = PromptTemplate(
    template=template, input_variables=["context", "question"], partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain_type_kwargs = {"prompt": PROMPT}

qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model = "gpt-3.5-turbo-16k-0613"), chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs, return_source_documents = True)

#STEP 8 - Send query
query = "Please provide me a list of all the locations/sites with their address, acreage, type of work and any special requirement/notes."
result =  qa({"query": query})
print(result["result"])
src_docs = result["source_documents"]
for doc in src_docs:
    print(doc, "\n\n\n")


# final_output = (parser.parse(result["result"]))['array']
# for item in final_output:
#     print(item, "\n")

# # with get_openai_callback() as cb:
# #     qa.run(query)
# #     print(cb)


#METHOD - 2 -> USING CSV AGENT

# from langchain.llms import OpenAI
# from langchain.chat_models import ChatOpenAI
# from langchain.agents.agent_types import AgentType
# from langchain.agents import create_csv_agent

# agent = create_csv_agent(
#     ChatOpenAI(temperature=0, model="gpt-4"),
#     "CSV/table_7.csv",
#     verbose=True,
#     agent_type=AgentType.OPENAI_FUNCTIONS,
# )

# # agent.run("Give me a list of all the names of the sites.")

# #agent.run("Give me a list of all the locations along with their addresses and acreage and any special notes (if mentioned).")

# agent.run("Please give me a insight about this dataset?")

#METHOD 3 -> USING MARKDOWN

# import pandas as pd
# import numpy as np
# import tabula

# df = tabula.read_pdf("D:\Projects\LPMS\docs\Annual Parks Bid.pdf", pages=[13], lattice = True, guess = False)[0]

# # Iterate through each cell in the DataFrame and replace line breaks with markdown line breaks
# for i, row in df.iterrows():
#     for j, cell in row.items():
#         if isinstance(cell, str):
#             df.at[i, j] = cell.replace('\n', '<br>')

# # Convert the DataFrame to a markdown table
# markdown_table = df.to_markdown(index=False)

# # Print the markdown table
# print(markdown_table)
# text_file = open('Output.txt', 'r')
# markdown_table = text_file.read()
# # print(markdown_table)

# PROMPT = PromptTemplate(
#         template='From the given table extract the following information:- {question} Here is the table :- {markdown_table} ', input_variables=["question", "markdown_table"])
# chain = LLMChain(llm=ChatOpenAI(model = "gpt-3.5-turbo-16k-0613", temperature=0), prompt=PROMPT)
# # print(prompt)
# result = chain.run({"question": "Extract all the locations with their address, acreage and notes", "markdown_table": markdown_table})
# print(result)