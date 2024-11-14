import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama  # Import Ollama for llama3
from langchain.agents import initialize_agent, Tool

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
    # Simplified version of the original function
        # Create a new FAISS index if it doesn't exist
    try:
        faiss_index = FAISS.load_local("pdf_chunks", allow_dangerous_deserialization=True)
    except ValueError:
        faiss_index = FAISS.create_local(index_name="pdf_chunks", embedding_dim=1000)

    # Add the text chunks to the vector store
    for chunk in text_chunks:
        embeddings = [chunk]  # Assuming each chunk is an embedding of size 1000
        faiss_index.add_embeddings(embeddings)


def get_conversational_chain(tools, ques, chat_history):
    llm = Ollama(base_url='http://localhost:11434', model="llama3")
    
    agent = initialize_agent([tools], llm, "conversational-react-description", verbose=True)
    
    response = agent({"input": ques})
    
    return response['output']


def user_input(user_question):
    new_db = FAISS.load_local("faiss_db", embeddings,allow_dangerous_deserialization=True)
    retriever=new_db.as_retriever()
    retrieval_chain= create_retriever_tool(retriever,"pdf_extractor")
    
    chat_history = st.session_state.get('chat_history', "")
    
    response = get_conversational_chain(retrieval_chain, user_question, chat_history)
    
    # Update the chat history with the latest question and response
    st.session_state['chat_history'] = chat_history + f"\nHuman: {user_question}\nAssistant: {response}"
    
    return response


def main():
    st.set_page_config("Chat PDF")
    st.header("RAG based Chat with PDF")
    
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        resp = user_input(user_question)
        
        # Show sidebar history
        st.session_state['history'].append("😎: "+user_question)
        st.session_state['history'].append("👾: "+resp)


    with st.sidebar:
        pdf_doc = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            raw_text = pdf_read(pdf_doc)
            text_chunks = get_chunks(raw_text)
            vector_store(text_chunks)
            
            st.success("Done")
                
        # Show sidebar history
        for hist in st.session_state['history']:
            print(hist)


if __name__ == "__main__":
    main()