import streamlit as st # create interactive web application 
from langchain_community.document_loaders import PyPDFLoader #LangChain's community document loader.
from langchain.text_splitter import RecursiveCharacterTextSplitter #Splits large text documents into smaller chunks
from langchain_google_genai import GoogleGenerativeAIEmbeddings #Converts text into embeddings 
from langchain_community.vectorstores import FAISS #Stores embeddings 
from langchain_core.prompts import ChatPromptTemplate # Helps create structured prompts for conversational AI models
from langchain.chains import RetrievalQA # Combines a retrieval mechanism (e.g., FAISS) with a language model to answer questions by finding and processing relevant documents.
from langchain_groq import ChatGroq # Allows using Groq’s high-performance AI 
from dotenv import load_dotenv #Loads environment variables from a .env file.
load_dotenv()
import os # Used to interact with environment variables 


# Set the page configuration
st.set_page_config(
    page_title="Ask My PDF",
    layout="centered"
)

# Streamlit application layout
st.title("Ask My PDF")
st.markdown("<hr style='border: 1px solid black; margin-top: -10px;'>", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state for the file uploader and vectorstore
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Define helper functions
def create_embeddings_from_pdf(file_path):
    """Load a PDF, split it into chunks, and create embeddings. Returns the vector store."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    texts = text_splitter.split_documents(documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(documents=texts, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

def answer_question_from_vectorstore(vectorstore, question):
    """Answer a question using the provided vector store."""
    prompt = ChatPromptTemplate.from_template("""
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        Question: {question}
        Context: {context}
        Answer:
    """)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever(search_type="mmr", search_kwargs={"k": 6})
    llm = ChatGroq(temperature=0.2, model_name="llama-3.1-70b-versatile", max_tokens=8000)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type_kwargs={"prompt": prompt})
    result = qa_chain.invoke({"query": question})
    return result["result"]

# PDF file uploader
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    if st.session_state.vectorstore is None:
        with open("temp_pdf.pdf", "wb") as f:
            f.write(uploaded_file.read())
        
        with st.spinner("Please wait, we are processing your PDF..."):
            # Create embeddings from the PDF
            st.session_state.vectorstore = create_embeddings_from_pdf("temp_pdf.pdf")
            st.success("PDF processed! You can now ask questions.")

if st.session_state.vectorstore:
    question = st.chat_input("Ask a question about the PDF:")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        ai_msg = answer_question_from_vectorstore(st.session_state.vectorstore, question)
        with st.chat_message("assistant"):
            st.markdown(ai_msg)
        st.session_state.messages.append({"role": "assistant", "content": ai_msg})

# Clear chat history
if st.sidebar.button("Clear"):
    st.session_state.messages = []
    st.session_state.vectorstore = None
    st.rerun()
