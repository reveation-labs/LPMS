
from langchain.vectorstores import Chroma
from pydantic import Extra
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

from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router import MultiPromptChain
from langchain.chains.llm import LLMChain
from langchain.chains.router.base import MultiRouteChain, RouterChain
from langchain.chains.base import Chain
from langchain.chains import ConversationChain

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
    Callbacks,
)

from typing import Any, Dict, List, Optional, Union, Mapping

load_dotenv()

def main(input, text):
    
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

    #STEP 5 - Define structured outout parser
    locationParser = StructuredOutputParser.from_response_schemas(
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
                )
            ]    
    )

    bidDetailsParser = StructuredOutputParser.from_response_schemas(
            response_schemas=[
                ResponseSchema(
                    name="Bid_Submission_Details",
                    description="""json object in the following format: 
                                {{ "deadline": string // date of deadline or due date', 
                                    "inquiry": string // contact details of person to inquire in case of doubts', 
                                    "submission_address": string //address of the location where bid needs to be submitted,
                                    "miscellaneous" : string //any other important information,
                                    }}
                                """,
                ),
            ]    
    )

    taskParser = StructuredOutputParser.from_response_schemas(
            response_schemas=[
                ResponseSchema(
                    name="task_frequency",
                    description="""array of json objects in the following format: [
                                {{ "task": string // name of the task',
                                    "total frequency" : string //total frequency,
                                    "month wise frequency" : string //frequency for each month
                                    }}
                                ]""",
                ),

            ]    
    )
    location_format_instructions = locationParser.get_format_instructions()
    bid_details_format_instructions = bidDetailsParser.get_format_instructions()
    task_format_instructions = taskParser.get_format_instructions()
    
    defaultTemplate = """Use the following pieces of context to answer the question at the end. Format the output as json keys and values, don't try to make up an answer.
    {context}

    text: {input}
    """
    
    miscPrompt = PromptTemplate(
        template=defaultTemplate, input_variables=["context", "input"]
    )
    # misc_kwargs = {"prompt": miscPrompt}


    structuredTemplate = """Use the following pieces of context to answer the question at the end. If you don't find releveant information keep the json fields empty, don't try to make up an answer.
    {context}

    text: {input}
    {format_instructions}
    """

    locationPrompt = PromptTemplate(
        template=structuredTemplate, input_variables=["context", "input"], partial_variables={"format_instructions": location_format_instructions}
    )
    # location_kwargs = {"prompt": locationTemplate}
    # locationQAChain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model = "gpt-4"), chain_type="stuff", retriever=retriever, chain_type_kwargs=location_kwargs, return_source_documents = True)
    

    bidPrompt = PromptTemplate(
        template=structuredTemplate, input_variables=["context", "input"], partial_variables={"format_instructions": bid_details_format_instructions}
    )
    # bid_kwargs = {"prompt": bidTemplate}
    # bidQAChain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model = "gpt-4"), chain_type="stuff", retriever=retriever, chain_type_kwargs=bid_kwargs, return_source_documents = True)


    taskPrompt = PromptTemplate(
        template=structuredTemplate, input_variables=["context", "input"], partial_variables={"format_instructions": task_format_instructions}
    )
    # task_kwargs = {"prompt": taskTemplate}
    # taskQAChain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model = "gpt-4"), chain_type="stuff", retriever=retriever, chain_type_kwargs=task_kwargs, return_source_documents = True)
    

    prompt_infos = [
        {
            "name": "location",
            "description": "Good for extracting locations related information",
            "kwargs" : {"prompt": locationPrompt}
        },
        {
            "name": "bid",
            "description": "Good for extracting bid related information",
            "kwargs" : {"prompt": bidPrompt}
        },
        {
            "name": "task",
            "description": "Good for extracting task related information",
            "kwargs" : {"prompt": taskPrompt}
        },
        {
            "name": "miscellaneous",
            "description": "Good for extracting amy other information",
            "kwargs" : {"prompt": miscPrompt}
        }
    ]

    destination_chains = {}
    for p_info in prompt_infos:
        name = p_info["name"]
        # prompt = p_info["prompt"]
        chain_kwargs = p_info["kwargs"]
        chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model = "gpt-4"), chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_kwargs)
        # input_key_map = {"input": "input"}
        # adapted_chain = InputConverterChain()
        # adapted_chain.destination_chain = chain
        # adapted_chain.input_key_map = input_key_map
        destination_chains[name] = chain

    # print(f"DESTINATION CHAIN: {destination_chains['location']}")
    # llm = OpenAI()
    # print(destination_chains)
    # default_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model = "gpt-4"), chain_type="stuff", retriever=retriever, chain_type_kwargs=default_kwargs)
    # default_chain = LLMChain(llm=llm)
    chain_kwargs = {"prompt": miscPrompt}
    chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model = "gpt-4"), chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_kwargs)
    # input_key_map = {"input": "input"}
    # adapted_chain.destination_chain = chain
    # adapted_chain.input_key_map = input_key_map
    default_chain = chain

    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = "\n".join(destinations)

    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)

    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        output_parser=RouterOutputParser(),
    )
   
    llm = OpenAI()
    # print(router_prompt.template)

    router_chain = LLMRouterChain.from_llm(llm, router_prompt)
    # print(router_chain)
    # chain = MultiPromptChain(
    #     router_chain=router_chain,
    #     destination_chains=destination_chains,
    #     default_chain=default_chain,
    #     verbose=True,
    # )


    chain = CustomMultiPromptChain(
        router_chain=router_chain,
        destination_chains=destination_chains,
        default_chain=default_chain,
        verbose=True,
    )

    
    # print(chain({"input": input}))
    print(chain({"input" : "Extract location related information", "query" : "Extract location related information"}))
    # client = chromadb.Client()
    # client.delete_collection("langchain")


# class MultitypeDestRouteChain(MultiRouteChain) :
#     """A multi-route chain that uses an LLM router chain to choose amongst prompts."""

#     router_chain: RouterChain
#     """Chain for deciding a destination chain and the input to it."""
#     destination_chains: Mapping[str, Chain]
#     """Map of name to candidate chains that inputs can be routed to."""
#     default_chain: LLMChain
#     """Default chain to use when router doesn't map input to one of the destinations."""

#     @property
#     def output_keys(self) -> List[str]:
#         return ["text"]
    

# class CustomMultiPromptChain(MultiRouteChain):
#     # Existing properties and methods
    
#     @property
#     def input_keys(self) -> List[str]:
#         # Include the input key for the router chain ('input') and the input keys for destination chains
#         return ['input'] + ['query']
    
#     def _call(
#         self,
#         inputs: Dict[str, Any],
#         run_manager: Optional[CallbackManagerForChainRun] = None,
#     ) -> Dict[str, Any]:
#         # Extract the input for the router chain
#         router_input = {
#             "input": inputs['input'],
#         }
        
#         # # Determine the destination chain using the router chain
#         # destination_name = self.router_chain.run(router_input)['choice']
        
#         # # Get the selected destination chain
#         # selected_chain = self.destination_chains.get(destination_name, self.default_chain)
        
#         # # Call the selected destination chain with the remaining inputs
#         # result = selected_chain.run(inputs, run_manager=run_manager)
#         # Determine the destination chain using the router chain
#         # print(f"RouterChain {self.router_chain}")
#         router_output = self.router_chain(router_input)
#         # print(f"RouterOutput {router_output}")
#         destination_name = router_output['destination']
#         # next_inputs = router_output.get('next_inputs', {})  # Capture any next_inputs if available
#         next_inputs = inputs
#         # Merge any captured next_inputs back into the inputs dictionary
        
#         inputs.update(next_inputs)
#         print(inputs)
#         # Get the selected destination chain
#         selected_chain = self.destination_chains.get(destination_name, self.default_chain)
        
#         # Call the selected destination chain with the remaining inputs
#         result = selected_chain.run(inputs)
#         # Return the result
#         return result

class CustomMultiPromptChain (MultiRouteChain):

    destination_chains: Mapping[str, Chain]
    """Map of name to candidate chains that inputs can be routed to. Not restricted to LLM"""


from langchain.chains.base import Chain
from langchain.callbacks.manager import (
    AsyncCallbackManager,
    AsyncCallbackManagerForChainRun,
    CallbackManager,
    CallbackManagerForChainRun,
    Callbacks,
)

# class InputConverterChain(Chain):
#     destination_chain: Chain = None
#     input_key_map: Dict[str, Any] = None

#     # __fields_set__: bool = False

#     class Config:
#         """Configuration for this pydantic object."""
#         extra = Extra.allow
#         arbitrary_types_allowed = True

#     @property
#     def input_keys(self) -> List[str]:
#         """Will be whatever keys the prompt expects.
#         :meta private:
#         """
#         return ["input"]

#     @property
#     def output_keys(self) -> List[str]:
#         """Will always return text key.
#         :meta private:
#         """
#         return ["text"]

#     def _call(
#             self,
#             inputs: Dict[str, Any],
#             run_manager: Optional[CallbackManagerForChainRun] = None,
#     ) -> Dict[str, str]:
#         print("Inputs before mapping:", inputs)

#         for k, v in self.input_key_map.items():
#             if k in inputs.keys():
#                 inputs[v] = inputs[k]

#         print("##### inputs is now")
#         print(inputs)
#         # return self.destination_chain.run(adapted_inputs)

#         data = self.destination_chain(inputs)
#         print("Inputs after mapping:", inputs)



main(input = "Extract bid submission details.", text= "INVITATION FOR BIDS CITY OF CONROE ANNUAL PARKS MOWING BID CITY OF CONROE P.O. BOX 3066 CONROE, TEXAS 77305 BIDS DUE THURSDAY SEPTEMBER 9, 2021 @ 2:00 PM CITY OF CONROE PURCHASING DEPARTMENT")