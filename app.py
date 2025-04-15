import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain_deepseek import ChatDeepSeek
from langchain.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import conversational_retrieval
from htmlTemplate import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    text = ""

    if not pdf_docs:
        return text

    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")

    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="deepseek-ai/DeepSeek-V3-0324")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    # llm = ChatDeepSeek()
    llm = HuggingFaceHub(
        repo_id="deepseek-ai/DeepSeek-V3-0324",
        model_kwargs={"temperature":0.5,"max_length":512}
    )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = conversational_retrieval(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response('chat_history')

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()

    # Check for required API keys
    if not os.getenv("DEEPSEEK_API_KEY"):
        st.warning("DeepSeek API token not found. Please set it in your .env file.")

    st.set_page_config(page_title="PDF Chatbot", page_icon=":books")

    st.write(css, unsafe_allow_html=True)

    # Check if the conversation is in session state of application
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs")
    user_question = st.text_input("Ask about your documents: ")

    # Add initial instructions
    if st.session_state.conversation is None:
        st.info("Please upload PDF documents using the sidebar to begin.")
    else:
        if not user_question:
            st.write("Some example questions you might ask:")
            st.write("- What are the main topics covered in these documents?")
            st.write("- Can you summarize the key points?")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your PDFs")
        pdf_docs = st.file_uploader("Upload and Process", accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("Processing"):
                # Get the pdf text
                raw_text = get_pdf_text(pdf_docs)

                # Get the text chunks
                with st.status("Creating text chunks..."):
                    text_chunks = get_text_chunks(raw_text)
                    st.write(f"Created {len(text_chunks)} text chunks")

                # Create the vector store with the embeddings
                with st.status("Building vector store..."):
                    vectorstore = get_vectorstore(text_chunks)
                    st.success("Vector store created!")

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

        if st.session_state.conversation is not None:
            if st.button("Reset Conversation"):
                st.session_state.chat_history = None
                st.session_state.conversation = None
                st.experimental_rerun()

if __name__ == '__main__':
    main()