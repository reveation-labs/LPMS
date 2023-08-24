from langchain.vectorstores import Chroma
import streamlit as st
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import chromadb
from dotenv import load_dotenv
from langchain import LLMChain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import time
import json
import re
load_dotenv()


def main(query, text):
    print("starting process..")
    # print("Calling gpt api")
    # print("Api called..")
    schemaType = choose_schema(query,text)
    chain = process_request(query, text, schemaType)
    result =  chain({"query": query})
    print(result["result"])
    with get_openai_callback() as cb:
        chain.run(query)
        print(cb)
    client = chromadb.Client()
    client.delete_collection("langchain")
    print("Finished!")
    # return result["result"]


def choose_schema(query, text):

    path = "temp_files/temp.md"
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)

    docs = UnstructuredMarkdownLoader(path).load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)

    # STEP 3 - create the vectorestore to use as the index
    embeddings = OpenAIEmbeddings()

    # STEP 4 - Create the Chroma index from the list of embeddings
    db = Chroma.from_documents(texts, embeddings)
    # print("Embeddings created!")

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":5})

    schemas = StructuredOutputParser.from_response_schemas(
        response_schemas=[
            ResponseSchema(
                    name="Location_Details",
                    description="""array of json objects in the following format: [
                                {{ "name": string // name of the site/location', 
                                    "address": string // address of that site/location', 
                                    "acreage": int //acreage of the location,
                                    "notes" : string //any notes regarding the location/site,
                                    "frequency" : string //any information about frequency if mentioned
                                    }}
                                ]""",
                ),
            ResponseSchema(
                    name="Bid_Submission_Details",
                    description="""json object in the following format: 
                                {{ "deadline": string // date of deadline or due date', 
                                    "inquiry": string // contact details of person to inquire in case of doubts', 
                                    "submission_address": string //address of the location where bid needs to be submitted,
                                    "site-visit" string //information about site visits if required
                                    "miscellaneous" : string //any other important information,
                                    }}
                                """,
                ),
            
            ResponseSchema(
                    name="task_frequency_details",
                    description="""array of json objects in the following format: [
                                {{ "task": string // name of the task',
                                    "total frequency" : string //total frequency,
                                    "month wise frequency" : string //frequency for each month
                                    "quantity" : int //estimated quantity to apply if any in case of chemicals often abbreviated as Qty
                                    "area" : string //area if mentioned,
                                    "miscellaneous" : string //any extra information regarding the task
                                    }}
                                ]""",
                ),
        ]
    )
    
    # Template1 = """Use the following pieces of context to answer the question at the end. If you don't find releveant information keep the json fields empty, don't try to make up an answer.
    #     {context}

    #     text: {question}

    #     Once you find the answer, I have following schemas choose the only one which suits the answer best and format your response according to that. 
    #     Final Response should be in markdown json format with just the(if any field is missing you should keep it empty)
    #     //Includes location address and other details where the task needs to be performed.
    #     name="Location_Details",
    #                 description=array of json objects in the following format: [
    #                             '{{ "name": string // name of the site/location', 
    #                                 "address": string // address of that site/location', 
    #                                 "acreage": int //acreage of the location,
    #                                 "notes" : string //any notes regarding the location/site,
    #                                 "frequency" : string //any information about frequency if mentioned
    #                                 }}'
    #                             ],
    #     //Includes different bid related details
    #     name="Bid_Details ",
    #                 description=json object in the following format: 
    #                             '{{ "deadline": date and time // date of deadline or due date', 
    #                                 "inquiry": string // contact details of person to inquire in case of doubts', 
    #                                 "submission_address": string //address of the location where bid needs to be submitted,
    #                                 "site-visit" string //information about site visits if required
    #                                 "miscellaneous" : string //any other important information,
    #                                 }}'
    #                             ,
    #     //Includes the type of work/task that needs to be performed on different locations
    #     name="task_frequency_details ",
    #                 description=array of json objects in the following format: [
    #                             '{{ "task": string // name of the task',
    #                                 "total frequency" : int //total frequency,
    #                                 "month wise frequency" : string //frequency for each month
    #                                 "quantity" : int //estimated quantity to apply if any in case of chemicals often abbreviated as Qty
    #                                 "area" : string //area if mentioned,
    #                                 "miscellaneous" : string //any extra information regarding the task
    #                                 }}'
    #                             ]
        
    #     //Other details like general requirements, insurance requirements, terms and conditions etc.
    #     name = "other_details",
    #                 description = json object (makey your own key/values)
    

    # """

    Template1 = """Use the following pieces of context to choose one of the following schemas which suits the best to the answer to the question.
        {context}

        text: {question}

        return just the name of the schema chosen without any extra text in the following format.
        {{
            'schema' : int // the index of the schema chosen
        }}

        0-> Other_Details, 1-> Bid_Details, 2-> Task_Details, 3 -> Location_Details  
        //Includes location address and other details where the task needs to be performed.
        name="Location_Details",
                    description=array of json objects in the following format: [
                                '{{ "name": string // name of the site/location', 
                                    "address": string // address of that site/location', 
                                    "acreage": int //acreage of the location,
                                    "notes" : string //any notes regarding the location/site,
                                    "frequency" : string //any information about frequency if mentioned
                                    }}'
                                ],
        //Includes different bid related details
        name="Bid_Details ",
                    description=json object in the following format: 
                                '{{ "deadline": date and time // date of deadline or due date', 
                                    "inquiry": string // contact details of person to inquire in case of doubts', 
                                    "submission_address": string //address of the location where bid needs to be submitted,
                                    "site-visit" string //information about site visits if required
                                    "miscellaneous" : string //any other important information,
                                    }}'
                                ,
        //Includes the type of work/task that needs to be performed on different locations
        name="Task_Details ",
                    description=array of json objects in the following format: [
                                '{{ "task": string // name of the task',
                                    "total frequency" : int //total frequency,
                                    "month wise frequency" : string //frequency for each month
                                    "quantity" : int //estimated quantity to apply if any in case of chemicals often abbreviated as Qty
                                    "area" : string //area if mentioned,
                                    "miscellaneous" : string //any extra information regarding the task
                                    }}'
                                ]
        
        //Other details like general requirements, insurance requirements, terms and conditions etc.
        name = "Other_Details",
                    description = json object (makey your own key/values)
    

    """
    PROMPT = PromptTemplate(
            template=Template1, input_variables=["context", "question"]
        )


    chain_kwargs = {"prompt": PROMPT}
    chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model = "gpt-4"), chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_kwargs, return_source_documents = False)
    
    result = chain({"query": query})
    # print(f"{result['result']}")
    pattern = r'{"schema": (\d+)}'
    match = re.search(pattern, result['result'])
    if match:
        # Extract the value of the schema field
        schema_value = int(match.group(1))
        print(f"Schema value: {schema_value}")
        return schema_value
    else:
        print("Schema field not found.")
        return 0

def process_request(query, text, promptType):
    path = "temp_files/temp.md"
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)

    docs = UnstructuredMarkdownLoader(path).load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)

    # STEP 3 - create the vectorestore to use as the index
    embeddings = OpenAIEmbeddings()

    # STEP 4 - Create the Chroma index from the list of embeddings
    db = Chroma.from_documents(texts, embeddings)
    print("Embeddings created!")

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":5})

    Template = """Use the following pieces of context to answer the question at the end. If you don't find releveant information keep the json fields empty, don't try to make up an answer.
        {context}

        text: {question}
        {format_instructions}
        """
    
    if promptType == 0 :

        Template = """Given the context: "{context}", answer the following question based on the context in short by extracting only important information from the context: "{question}" Format the response as array of json object or a single json object whichever suits best.
        """

        PROMPT = PromptTemplate(
            template=Template, input_variables=["context", "question"]
        )


        chain_kwargs = {"prompt": PROMPT}
        chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model = "gpt-4"), chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_kwargs, return_source_documents = False)
    
    elif promptType == 1:

        bidDetailsParser = StructuredOutputParser.from_response_schemas(
            response_schemas=[
                ResponseSchema(
                    name="Bid_Submission_Details",
                    description="""json object in the following format: 
                                {{ "deadline": string // date of deadline or due date', 
                                    "inquiry": string // contact details of person to inquire in case of doubts', 
                                    "submission_address": string //address of the location where bid needs to be submitted,
                                    "site-visit" string //information about site visits if required
                                    "miscellaneous" : string //any other important information,
                                    }}
                                """,
                ),
            ]    
        )
        PROMPT = PromptTemplate(
            template=Template, input_variables=["context", "question"], partial_variables={"format_instructions" : bidDetailsParser.get_format_instructions()}
        )
        chain_kwargs = {"prompt": PROMPT}
        chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model = "gpt-4"), chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_kwargs, return_source_documents = False)

    elif promptType == 2:

        taskParser = StructuredOutputParser.from_response_schemas(
            response_schemas=[
                ResponseSchema(
                    name="task_frequency_details",
                    description="""array of json objects in the following format: [
                                {{ "task": string // name of the task',
                                    "total frequency" : string //total frequency,
                                    "month wise frequency" : string //frequency for each month
                                    "quantity" : int //estimated quantity to apply if any in case of chemicals often abbreviated as Qty
                                    "area" : string //area if mentioned,
                                    "miscellaneous" : string //any extra information regarding the task
                                    }}
                                ]""",
                ),

            ]    
        )

        PROMPT = PromptTemplate(
            template=Template, input_variables=["context", "question"], partial_variables={"format_instructions" : taskParser.get_format_instructions()}
        )
        chain_kwargs = {"prompt": PROMPT}
        chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model = "gpt-4"), chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_kwargs, return_source_documents = False)

    elif promptType == 3:

        locationParser = StructuredOutputParser.from_response_schemas(
            response_schemas=[
                ResponseSchema(
                    name="Location_Details",
                    description="""array of json objects in the following format: [
                                {{ "name": string // name of the site/location', 
                                    "address": string // address of that site/location', 
                                    "acreage": int //acreage of the location,
                                    "notes" : string //any notes regarding the location/site,
                                    "frequency" : string //any information about frequency if mentioned
                                    }}
                                ]""",
                )
            ]    
        )
        PROMPT = PromptTemplate(
            template=Template, input_variables=["context", "question"], partial_variables={"format_instructions" : locationParser.get_format_instructions()}
        )
        chain_kwargs = {"prompt": PROMPT}
        chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model = "gpt-4"), chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_kwargs, return_source_documents = False)
    
    # print(PROMPT)
    return chain 

main(query = "Extract locations and related details from the text.", 
     text= f"""| Longview, Texas      | Frequency 1       | Frequency 2       |     Cycles | Cost Per   | Yearly       |
|                      | November -        | March - October   |   per year | Cycle      | Cycle Cost   |
|                      | February          |                   |            |            |              |
|:---------------------|:------------------|:------------------|-----------:|:-----------|:-------------|
| 303 Evergreen        | 2 Times per Month | Weekly            |         43 | $          | $            |
| 414/418 S. Center    | 2 Times per Month | Weekly            |         43 | $          | $            |
| 3704 Teri Lyn        | 2 Times per Month | Weekly            |         43 | $          | $            |
| 105/107  Woodbine Pl | 2 Times per Month | Weekly            |         43 | $          | $            |
| 801 Pegues           | 2 Times per Month | Weekly            |         43 | $          | $            |
| 409 S. Fredonia      | 2 Times per Month | Weekly            |         43 | $          | $            |
| 425 S Main St        | 2 Times per Month | Weekly            |         43 | $          | $            |
| 7th Street Lot       | 1 Time per Month  | 2 Times per Month |         20 | $          | $            |
| 103 Branch St.       | 2 Times per Month | Weekly            |         43 | $          | $            |
| 950 N 4th Street     | 2 Times per Month | Weekly            |         43 | $          | $            |
| 1300 N 6th Street    | 2 Times per Month | Weekly            |         43 | $          | $            |
| 3770 PR 3439         | 2 Times per Month | Weekly            |         43 | $          | $            |
| Marshall, Texas        | Frequency 1       | Frequency 2       |     Cycles | Cost Per   | Yearly       |
|                        | November –        | March - October   |   per year | Cycle      | Cycle Cost   |
|                        | February          |                   |            |            |              |
|:-----------------------|:------------------|:------------------|-----------:|:-----------|:-------------|
| 401 N. Grove           | 2 Times per Month | Weekly            |         43 | $          | $            |
| 502 Rusk/ 204 N. Alamo | 2 Times per Month | Weekly            |         43 | $          | $            |
| 1500 W. Grand          | 2 Times per Month | Weekly            |         43 | $          | $            |
| 7470 State Hwy 154     | 2 Times per Month | 2 Times per Month |         24 | $          | $            |
| 1512 Indian Springs    | 2 Times per Month | Weekly            |         43 | $          | $            |
| Empty Lot Allen St.    | 1 Time per Month  | 1 Time per Month  |         12 | $          | $            |
| White Oak, Texas   | Frequency 1       | Frequency 2       |     Cycles | Cost Per   | Yearly       |
|                    | November –        | March - October   |   per year | Cycle      | Cycle Cost   |
|                    | February          |                   |            |            |              |
|:-------------------|:------------------|:------------------|-----------:|:-----------|:-------------|
| 523 S Suncamp Rd   | 2 Times per Month | 2 times per month |         20 | $          | $            |
| Henderson, Texas   | Frequency 1       | Frequency 2       |     Cycles | Cost Per   | Yearly       |
|                    | November –        | March - October   |   per year | Cycle      | Cycle Cost   |
|                    | February          |                   |            |            |              |
|:-------------------|:------------------|:------------------|-----------:|:-----------|:-------------|
| 209 N Main St      | 2 Times per Month | Weekly            |         43 | $          | $            |
| Carthage, Texas   | Frequency 1       | Frequency 2       |     Cycles | Cost Per   | Yearly       |
|                   | November –        | March - October   |   per year | Cycle      | Cycle Cost   |
|                   | February          |                   |            |            |              |
|:------------------|:------------------|:------------------|-----------:|:-----------|:-------------|
| 1701 S. Adams     | 2 Times per Month | Weekly            |         43 | $          | $            |
| Gilmer, Texas   | Frequency 1       | Frequency 2       |     Cycles | Cost Per   | Yearly       |
|                 | November –        | March - October   |   per year | Cycle      | Cycle Cost   |
|                 | February          |                   |            |            |              |
|:----------------|:------------------|:------------------|-----------:|:-----------|:-------------|
| 101 Madison St  | 2 Times per Month | Weekly            |         43 | $          | $            |
| Texarkana, Texas   | Frequency 1       | Frequency 2       |     Cycles | Cost Per   | Yearly       |
|                    | November –        | March - October   |   per year | Cycle      | Cycle Cost   |
|                    | February          |                   |            |            |              |
|:-------------------|:------------------|:------------------|-----------:|:-----------|:-------------|
| 2435 College Dr.   | 2 Times per Month | Weekly            |         43 | $          | $            |
| 1911 Galleria Oaks | 2 Times per Month | Weekly            |         43 | $          | $            |
| 4217 Hazel St.     | 1 Time  per Month | 2 Times per Month |         20 | $          | $            |
| Clarksville, Texas   | Frequency 1       | Frequency 2       |     Cycles | Cost Per   | Yearly       |
|                      | November –        | March - October   |   per year | Cycle      | Cycle Cost   |
|                      | February          |                   |            |            |              |
|:---------------------|:------------------|:------------------|-----------:|:-----------|:-------------|
| 106 N. M. L. King    | 2 Times per Month | Weekly            |         43 | $          | $            |
| Atlanta, Texas            | Frequency 1       | Frequency 2       |     Cycles | Cost Per   | Yearly       |
|                           | November –        | March - October   |   per year | Cycle      | Cycle Cost   |
|                           | February          |                   |            |            |              |
|:--------------------------|:------------------|:------------------|-----------:|:-----------|:-------------|
| Empty Lot by fire station | 2 Times per Month | 2 Times per Month |         24 | $          | $            |

""")



