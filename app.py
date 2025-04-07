import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfFileReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_deepseek import ChatDeepSeek

def main():
    load_dotenv()
    st.set_page_config(page_title="PDF Chatbot", page_icon=":books")

    st.header("Chat with multiple PDFs")
    st.text_input("Ask about your documents: ")

    with st.sidebar:
        st.subheader("Your PDFs")
        pdf_docs = st.file_uploader("Upload and Process", accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("Processing"):
                # Get the pdf text


                # Get the text chunks


                # Create the vector store with the embeddings




if __name__ == '__main__':
    main()