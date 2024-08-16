import os

import streamlit as st
from langchain.agents import Tool, initialize_agent

#from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool

#from dotenv import load_dotenv
#from langchain_anthropic import ChatAnthropic
#from langchain_openai import ChatOpenAI, OpenAIEmbeddings
#from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.llms import Ollama  # Import Ollama for llama3
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from PyPDF2 import PdfReader

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#embeddings = SpacyEmbeddings(model_name="en_core_web_sm")
#embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
embeddings = OllamaEmbeddings(base_url="http://192.168.2.39:11434", model="nomic-embed-text")

def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text



def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def vector_store(text_chunks):
    
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")


def get_conversational_chain(tools, ques, chat_history):
    #os.environ["ANTHROPIC_API_KEY"]=os.getenv["ANTHROPIC_API_KEY"]
    #llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0, api_key=os.getenv("ANTHROPIC_API_KEY"),verbose=True)
    #llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key="")
    # Initialize Ollama with the llama3 model
    llm = Ollama(base_url='http://192.168.2.39:11434', model="llama3.1")
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant. Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer""",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
    tool=[tools]
    #agent = create_tool_calling_agent(llm, tool, prompt)
    #agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=True)
    #response=agent_executor.invoke({"input": ques})

    # Initialize the agent with the tools
    agent = initialize_agent(tool, llm, agent="conversational-react-description", verbose=True)

    #response = agent({"input": ques})
    
    # Ensure all required keys are provided
    inputs = {
        "chat_history": "",
        "input": ques,
        "agent_scratchpad": ""
    }
    
    # Invoke the agent with the user question 
    try:
        response = agent(inputs)
        if isinstance(response, dict) and 'output' in response:
            print(response)           
            st.write("Reply: ", response['output'])
            return response['output']
        else:
            st.error("Unexpected response format.")
            return "Unexpected response format."
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return f"An error occurred: {e}"


def user_input(user_question):
    
    
    
    new_db = FAISS.load_local("faiss_db", embeddings,allow_dangerous_deserialization=True)
    
    retriever=new_db.as_retriever()
    retrieval_chain= create_retriever_tool(retriever,"pdf_extractor","This tool is to give answer to queries from the pdf")
    
    # Use the chat history stored in session state
    chat_history = st.session_state.get('chat_history', "")
    
    response = get_conversational_chain(retrieval_chain, user_question, chat_history)
    
    # Update the chat history with the latest question and response
    st.session_state['chat_history'] = chat_history + f"\nHuman: {user_question}\nAssistant: {response}"

    
    return response





def main():
    st.set_page_config("Chat PDF")
    st.header("RAG based Chat with PDF")
    
    # Initialize session state if not already present
    if 'history' not in st.session_state:
        st.session_state['history'] = []
        st.session_state['chat_history'] = ""

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        with st.spinner("Processing your question..."):
            resp = user_input(user_question)
            ### Show sidebar history
        st.session_state['history'].append("😎: "+user_question)
        st.session_state['history'].append("👾: "+resp)
        


    with st.sidebar:
        pdf_doc = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = pdf_read(pdf_doc)
                text_chunks = get_chunks(raw_text)
                vector_store(text_chunks)
                st.success("Done")
                
        st.sidebar.markdown("<br />".join(st.session_state['history'])+"<br /><br />", unsafe_allow_html=True)

if __name__ == "__main__":
    main()