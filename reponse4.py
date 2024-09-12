import os
import warnings
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import google.generativeai as genai
from langchain_core._api.deprecation import LangChainDeprecationWarning
from langchain.prompts import PromptTemplate
def resp(input):
    persist_directory = 'db'
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
    load_dotenv()
    api_key = os.getenv("API_KEY")
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = vectordb.as_retriever()
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=1.0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    custom_prompt_template = """You are a plant medical disease bot that retrieves information or management or soution of {query} from a database. 


    Human: {query}
    medbot: """
    PROMPT = PromptTemplate(
    template=custom_prompt_template, input_variables=["query"]
    )
    qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT, "document_variable_name": "query"}
    )
    def process_llm_response(llm_response):
        return llm_response['result']
    query =input
    llm_response = qa_chain({"query": query})  # pass query
    db_info = process_llm_response(llm_response)
    return db_info
