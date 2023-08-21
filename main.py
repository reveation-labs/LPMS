
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

from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router import MultiPromptChain
from langchain.chains.llm import LLMChain
from langchain.chains.router.base import MultiRouteChain, RouterChain
from langchain.chains.base import Chain
from langchain.chains import ConversationChain


from typing import Mapping
from typing import List

load_dotenv()



def main():
    # st.title("File Upload and Storage")

    # # Create a folder to store uploaded files if it doesn't exist
    # if not os.path.exists("uploads"):
    #     os.makedirs("uploads")
    
    # # Create a folder to store markdown files if it doesn't exist
    # if not os.path.exists("markdown_files"):
    #     os.makedirs("markdown_files")

    # uploaded_file = st.file_uploader("Choose a file to upload", type=["pdf"])

    if True:
        # Display the uploaded file
        # st.text(f"Uploaded file: {uploaded_file.name}")

        # Save the uploaded file to the "uploads" folder
        # with st.spinner("Converting PDF to Markdown..."):
        #     pdf_path = save_uploaded_file(uploaded_file)
        #     markdown_path =  pdf_to_markdown(pdf_path, uploaded_file)

        markdown_path = "D:\Projects\LPMS\markdown_files\Markdown.md"
        #Read the content of the Markdown file
        with open(markdown_path, "r", encoding="utf-8") as markdown_file:
            markdown_content = markdown_file.read()
     
        sections = markdown_content.split("# Section")
        st.title("Proposal Evaluation")
        # print(type(sections))

        summary = ''
        for index, section in enumerate(sections, start = 0):
            # st.text_area(f"Section {index}", value=section, height=200)
            if(index == 0):
                continue
            st.markdown(f"## Section {index}")
            st.markdown(section)

            if f"Suggest Button {index}{index}" not in st.session_state:
                st.session_state[f"Suggest Button {index}{index}"] = False

            if st.button("Suggest Prompts", key=f"Suggest Button {index}") or st.session_state[f"Suggest Button {index}{index}"] == True:
                st.session_state[f"Suggest Button {index}{index}"] = True
                prompts = suggest_prompts(section)
                res = json.loads(prompts)
                st.session_state[f"firstIteration{index}"] = False
                selected_prompt = st.selectbox("Select a prompt:", res, key=f"Selectbox {index}")
                if st.button("Use this prompt", key=f"Prompt Button {index} {res.index(selected_prompt)}"):
                    print("calling api")
                    resp = process_request(selected_prompt, section)
                    st.header("Response:")
                    st.write(resp)
                    summary+=resp+ "\n"

            
        # selected_sections = st.multiselect("Select section numbers", range(1, len(sections) + 1))
        # selected_sections = [1,2,3,4,5,6,7,8,9]
        # response = ''
        # if st.button("Extract Key Info from these sections"):
        #     with st.spinner("Processing"):
                
        #         # query = "Extract key details from the text which are important for preparing bid."
        #         # resp = process_request(query, sections[selected_sections[3]])
        #         # response += resp + "\n"
        #         # time.sleep(2)
                
        #         # query = "Provide me a list of all the work/task that needs to be performed along with it's frequency for each month"
        #         # resp = process_request(query, sections[selected_sections[1]])
        #         # response += resp + "\n"
        #         # time.sleep(2)
              
        #         # query = "Provide me a list of all the locations along with it's address, acreage and notes. Count the number of total locations and make sure you don't miss any."
        #         # resp = process_request(query, sections[selected_sections[2]])
        #         # response += resp + "\n"
        #         # response = """{"Bid_Submission_Details": {"deadline": "SEPTEMBER 9, 2021 @ 2:00 PM", "inquiry": "Christie Spath, Purchasing, 401 Sgt. Ed Holcomb Blvd., Conroe, TX. 77304, Office: 936-522-3829", "submission_address": "USPS: City of Conroe, Soco Gorjon, City Secretary, P.O. Box 3066, Conroe, TX. 77305 or Physical: City of Conroe, Soco Gorjon, City Secretary, 300 West Davis St., Conroe, TX. 77301", "notes": "Submit electronically through Vendor Registry or three (3) copies of each proposal shall be CLEARLY MARKED \u201cBid #PK2122 \u2013 ANNUAL PARKS MOWING\u201d. Responses received later than the due date will not be accepted, and returned unopened."}, "task_frequency": [{"task": "Mow, Edge, Trim, Debris Disposal", "total frequency": "42.0", "month wise frequency": "Jan: 2, Feb: 2, Mar: 4, Apr: 4, May: 4, Jun: 5, Jul: 4, Aug: 5, Sep: 4, Oct: 4, Nov: 2, Dec: 2"}, {"task": "Undeveloped Property Mowing", "total frequency": "19.0", "month wise frequency": "Jan: 1, Feb: 1, Mar: 2, Apr: 2, May: 2, Jun: 2, Jul: 2, Aug: 2, Sep: 2, Oct: 1, Nov: 1, Dec: 1"}, {"task": "Fire Training Facility", "total frequency": "19.0", "month wise frequency": "Jan: 1, Feb: 1, Mar: 2, Apr: 2, May: 2, Jun: 2, Jul: 2, Aug: 2, Sep: 2, Oct: 1, Nov: 1, Dec: 1"}], "array": [{"name": "Candy Cane Park Complex", "address": "1202 - 1205 Candy Cane Lane / 1504 Parkwood West / 77301", "acreage": "27.0", "notes": "No ZTR at 1504 Parkwood West See map for boundaries"}, {"name": "Aquatic Center (interior)", "address": "1205 Candy Cane Lane / 77301", "acreage": "2.0", "notes": "Day & Time Constraints No ZTR or riding mowers Bagging clippings may be required"}, {"name": "Roberson Park", "address": "1301 Roberson St. / 77301", "acreage": "1.6", "notes": "Includes pathway to N. Frazier"}, {"name": "Milltown Park", "address": "600 York / 77301", "acreage": "2.3", "notes": ""}, {"name": "Conroe Founders Plaza", "address": "205 Metcalf St. / 77301", "acreage": "0.6", "notes": "No ZTR"}, {"name": "Heritage Place Park", "address": "500 Metcalf St. / 77301", "acreage": "2.9", "notes": ""}, {"name": "Stewarts Creek Park", "address": "1329 E. Dallas Street / 77301", "acreage": "9.0", "notes": "Including Entergy R.O.W. See map for boundaries"}, {"name": "Booker T. Washington Park", "address": "813 First St. / 77301", "acreage": "3.0", "notes": ""}, {"name": "Lewis Park", "address": "501 Park Place / 77301", "acreage": "5.0", "notes": ""}, {"name": "Dugan Park", "address": "719 E. Ave. G / 77301", "acreage": "0.6", "notes": "Vacant Lot"}, {"name": "Kasmiersky Park", "address": "889 Old Magnolia Rd. / 77304", "acreage": "9.0", "notes": ""}, {"name": "McDade Park", "address": "10310 FM 2854 / 77304", "acreage": "38.0", "notes": "Includes disc golf course See map for boundaries"}, {"name": "Flournoy Park", "address": "413 Tenth St. / 77301", "acreage": "4.0", "notes": ""}, {"name": "Dr. Martin Luther King, Jr. Park", "address": "1001 Dr. Martin Luther King, Jr. Place South", "acreage": "16.0", "notes": "Splash pad open seasonally (Mar \u2013 Oct)"}, {"name": "John Burge Park at Shadow Lakes", "address": "11050 John Burge Park St. / 77304", "acreage": "39.0", "notes": ""}, {"name": "Lions Park", "address": "1851 Northampton / 77303", "acreage": "3.5", "notes": ""}, {"name": "Lone Star Flag Park", "address": "212 I-45 North / 77301", "acreage": "3.5", "notes": "Limited mowing during wildflower season"}, {"name": "White Oak Point Park", "address": "3511 White Oak Point Dr. / 77304", "acreage": "2.0", "notes": ""}, {"name": "Wiggins Village Park", "address": "565 Bryant Rd. / 77303", "acreage": "12.0", "notes": ""}, {"name": "Oscar Johnson, Jr. Community Center", "address": "100 Park Place / 119 E. Ave G. / 77301", "acreage": "4.8", "notes": "Day & Time Constraints No ZTR or riding mowers in pool area. Bagging of clippings may be required in pool area See map for boundaries"}, {"name": "Hicks St. Property", "address": "NE corner San Jacinto at Hicks St. / 77301", "acreage": "0.5", "notes": "Vacant Lot"}, {"name": "Dallas Street Medians", "address": "Between Frazier St. and W. Davis", "acreage": "2.1", "notes": "Includes NW corner of Dallas @ N. Frazier"}, {"name": "Holly Hills Medians", "address": "Hillcrest @ N. Frazier St.", "acreage": "0.1", "notes": ""}, {"name": "Faith Walston Memorial", "address": "Dallas St. @ West Davis St.", "acreage": "0.2", "notes": ""},{"name": "I-45 Triangle & Medians", "address": "I-45 @ West Davis St.", "acreage": "0.9", "notes": "Beautification property"},{"name": "S. Frazier Medians", "address": "941 S. Frazier St.", "acreage": "1.2", "notes": "Beautification property includes Moore Family Memorial"},{"name": "Maurel Drive Medians", "address": "Between Longmire Rd & N. Loop 336", "acreage": "0.03", "notes": ""},{"name": "Montgomery Park Blvd.", "address": "Montgomery Park Blvd. @ N. Loop 336", "acreage": "0.3", "notes": ""},{"name": "Teas Road Medians", "address": "Teas Rd. @ FM 3083", "acreage": "0.2", "notes": ""},{"name": "Westview Blvd. Medians", "address": "Westview @ Wilson & Westview @ N. Loop 336", "acreage": "0.2", "notes": ""},{"name": "South Loop 336 Medians", "address": "1616 & 1648 S. Frazier 210 & 260 S. Loop 336", "acreage": "4.0", "notes": "Limited mowing during wildflower season See map for boundaries"},{"name": "Enterprise Row", "address": "Enterprise Row between S. Frazier & I-45 Feeder", "acreage": "8.6", "notes": "Entergy & pipeline ROW See map for boundaries"},{"name": "McDade Estates Property", "address": "1942, 1944 & 1946 O'Grady Dr. 1645, 1647, 1649 & 1651 White Oak Creek / 77304", "acreage": "2.0", "notes": "Vacant Lots"},{"name": "Artesian Lakes Property", "address": "200 Magnolia St. / 77304", "acreage": "0.6", "notes": "Vacant Lot"},{"name": "Main Street Parking", "address": "Main St. / 77301", "acreage": "0.3", "notes": "Jury Parking Lot South of Conroe Founders Plaza"},{"name": "Veterans Memorial Park", "address": "997 West Davis @ I-45 N.  / 77301", "acreage": "12.0", "notes": ""},{"name": "Conroe West Recreation Center", "address": "10245 Owen Drive / 77304", "acreage": "42.0", "notes": "See map for boundaries"},   {"name": "Conroe Tower/City Hall", "address": "300 W. Davis St. / 77301", "acreage": "0.5", "notes": "Fenced area between City Hall and Montgomery County Tax Office"},{"name": "Conroe Municipal Complex", "address": "700 Old Montgomery Road / 77304", "acreage": "4.1", "notes": ""},{"name": "Transportation Administration", "address": "202 Avenue A / 77301", "acreage": "0.4", "notes": ""},{"name": "Fire Station 1", "address": "300 Sgt. Ed Holcomb Blvd. North / 77304", "acreage": "2.3", "notes": "Restricted Access"},{"name": "Fire Station 2", "address": "425 E. Loop 336 / 77303", "acreage": "1.0", "notes": "Restricted Access"},{"name": "Fire Station 3", "address": "424 Foster Road / 77301", "acreage": "1.0", "notes": "Restricted Access"},{"name": "Fire Station 4", "address": "14901 Walter Woodson Drive. / 77384", "acreage": "3.2", "notes": "Restricted Access"},{"name": "Fire Station 5", "address": "1601 N. FM 3083 (Carter Moore Drive) / 77304", "acreage": "2.0", "notes": "Restricted Access"},    {"name": "Fire Station 6", "address": "15663 Hwy. 105 West / 77356", "acreage": "3.4", "notes": "Includes vacant property to west. See map for boundaries"},{"name": "Fire Station 7", "address": "7971 Longmire Road / 77304", "acreage": "2.0", "notes": "Restricted Access"},{"name": "Fire Training Facility", "address": "2357 Mike Meador Pkwy / 77303", "acreage": "4.8", "notes": "Restricted Access"},{"name": "Conroe Police Station", "address": "2300 Plantation Drive / 77304", "acreage": "11.5", "notes": "Includes ROW & median mowing on Plantation Blvd. Restricted Access See map for boundaries"},{"name": "Fire Arms Training Facility", "address": "2300 Sgt. Ed Holcomb Blvd. South / 77304", "acreage": "6.0", "notes": "Restricted Access Day & Time Constraints Does not include berm slopes"}]}"""
            
        #     st.header("Key Information Extracted:")
        #     st.write(response) 
        #     summary += response     
        #     # compare_with_criteria(summary)


def compare_with_criteria(summary):
   
    criteria = f"1. The locations/site should not be outside Texas. 2.Total frequency for any task should be less than 50. 3.Bid submission deadline should be after August. 4.It should not have any equipment insurance requirement. 5. Should not have work in month of May. "
    template = """I  have this criteria, compare it with the below summary to see if it matches all the criteria.
    If you don't find relevant information to compare with any criteria respond back with "No relevant information available for evaluation"
    Respond back in this format :- 
    write the criteria, then a tick mark if it matches and cross mark if it doesn't followed by a reason.
    Criteria : {criteria}
    Summary : {summary}
    """ 
    PROMPT = PromptTemplate(
        template=template, input_variables=["criteria", "summary"]
    )

    chain = LLMChain(llm=ChatOpenAI(model_name="gpt-4"),prompt=PROMPT)
    input = {'criteria':criteria,'summary':summary}
    ans = chain.run(input)
    print(ans)
    st.header('Evaluation:')
    st.write(ans)

@st.cache_data
def suggest_prompts(text):
    print("New REQUEST!!!!")
    template = """I have some text which is a part of Request For Proposal Document. Suggest around 2 to 5 maximum prompts to extract the key/important information from the text which will be useful for preparing the Bid. Return an array of prompts. The prompt should be as if directing the AI to extract following information.
    Text : {text}
    """ 
    PROMPT = PromptTemplate(
        template=template, input_variables=["text"]
    )

    chain = LLMChain(llm=ChatOpenAI(model_name="gpt-4"),prompt=PROMPT)
    input = {'text':text}
    prompts = chain.run(input)
    return prompts
    # print(PROMPT)
    # print(ans)

    # path = f"temp_files/temp_{int(time.time())}.md"
    # with open(path, "w", encoding="utf-8") as file:
    #     file.write(text)

    # docs = UnstructuredMarkdownLoader(path).load()
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # texts = text_splitter.split_documents(docs)

    # # STEP 3 - create the vectorestore to use as the index
    # embeddings = OpenAIEmbeddings()

    # # STEP 4 - Create the Chroma index from the list of embeddings
    # db = Chroma.from_documents(texts, embeddings)
    # print("Embeddings created!")

    # retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":5})
    # qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model = "gpt-4"), chain_type="stuff", retriever=retriever, return_source_documents = False)
    # query = "Use the following pieces of context which are a part of Request For Proposal Document. Suggest around 2 to 5 maximum prompts to extract the key/important information from the text which will be useful for preparing the Bid. Return an array of prompts. The prompt should be as if directing the AI to extract following information."
    # result = qa({"query": query})
    # with get_openai_callback() as cb:
    #     qa.run(query)
    #     print(cb)

    # # print(result["result"])
    # st.text_area("Suggested Prompts:", result["result"], height = 600)

def process_request(query, text):
    print("starting process..")

    chain = create_chain(text)
    print("Calling gpt api")
    print("Api called..")
    result =  chain({"query": query})
    # with get_openai_callback() as cb:
    #     chain.run(query)
    #     print(cb)
    print(result["result"])
    client = chromadb.Client()
    client.delete_collection("langchain")
    return result["result"]


def create_chain(text):

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
    parser = StructuredOutputParser.from_response_schemas(
            response_schemas=[
                ResponseSchema(
                    name="Location_Details",
                    description="""array of json objects in the following format: [
                                {{ "name": string // name of the site/location', 
                                    "address": string // address of that site/location', 
                                    "acreage": string //acreage of the location,
                                    "notes" : string //any notes regarding the location/site,
                                    }}
                                ]""",
                ),
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
                ResponseSchema(
                    name="Task_Details",
                    description="""array of json objects in the following format: [
                                {{ "task": string // name of the task',
                                    "total frequency" : string //total frequency,
                                    "month wise frequency" : string //frequency for each month
                                    }}
                                ]""",
                ),
            ]    
    )

    # locationParser = StructuredOutputParser.from_response_schemas(
    #         response_schemas=[
    #             ResponseSchema(
    #                 name="Location_Details",
    #                 description="""array of json objects in the following format: [
    #                             {{ "name": string // name of the site/location', 
    #                                 "address": string // address of that site/location', 
    #                                 "acreage": string //acreage of the location,
    #                                 "notes" : string //any notes regarding the location/site,
    #                                 }}
    #                             ]""",
    #             )
    #         ]    
    # )

    # bidDetailsParser = StructuredOutputParser.from_response_schemas(
    #         response_schemas=[
    #             ResponseSchema(
    #                 name="Bid_Submission_Details",
    #                 description="""json object in the following format: 
    #                             {{ "deadline": string // date of deadline or due date', 
    #                                 "inquiry": string // contact details of person to inquire in case of doubts', 
    #                                 "submission_address": string //address of the location where bid needs to be submitted,
    #                                 "miscellaneous" : string //any other important information,
    #                                 }}
    #                             """,
    #             ),
    #         ]    
    # )

    # taskParser = StructuredOutputParser.from_response_schemas(
    #         response_schemas=[
    #             ResponseSchema(
    #                 name="task_frequency_details",
    #                 description="""array of json objects in the following format: [
    #                             {{ "task": string // name of the task',
    #                                 "total frequency" : string //total frequency,
    #                                 "month wise frequency" : string //frequency for each month
    #                                 }}
    #                             ]""",
    #             ),

    #         ]    
    # )

    # location_format_instructions = locationParser.get_format_instructions()
    # bid_details_format_instructions = bidDetailsParser.get_format_instructions()
    # task_format_instructions = taskParser.get_format_instructions()

    # format_instructions = [location_format_instructions, bid_details_format_instructions, task_format_instructions]

    Template = """Use the following pieces of context to answer the question at the end.
    Format your response as json objects.
    {context}

    text: {question}
    """

    # PROMPT = PromptTemplate(
    #     template=Template, input_variables=["context", "question"], partial_variables={"format_instructions_1": location_format_instructions, "format_instructions_2" : bid_details_format_instructions, "format_instructions_3" : task_format_instructions}
    # )
    PROMPT = PromptTemplate(
        template=Template, input_variables=["context", "question"]
    )


    chain_kwargs = {"prompt": PROMPT}
    chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model = "gpt-4"), chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_kwargs, return_source_documents = False)
    # print(PROMPT)
    return chain

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
    


# process_request(query = "Extract bid submission details.", text= "INVITATION FOR BIDS CITY OF CONROE ANNUAL PARKS MOWING BID CITY OF CONROE P.O. BOX 3066 CONROE, TEXAS 77305 BIDS DUE THURSDAY SEPTEMBER 9, 2021 @ 2:00 PM CITY OF CONROE PURCHASING DEPARTMENT")

main()



