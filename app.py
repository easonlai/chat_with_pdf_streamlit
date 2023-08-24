# Import necessary libraries.
import streamlit as st
import openai
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import AzureOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

# Configure the baseline configuration of the OpenAI library for Azure OpenAI Service.
OPENAI_API_KEY = "PLEASE_ENTER_YOUR_OWNED_AOAI_SERVICE_KEY"
OPENAI_API_BASE = "https://PLESAE_ENTER_YOUR_OWNED_AOAI_RESOURCE_NAME.openai.azure.com/"
OPENAI_DEPLOYMENT_NAME = "PLEASE_ENTER_YOUR_OWNED_AOAI_TEXT_MODEL_NAME"
OPENAI_MODEL_NAME = "text-davinci-003"
OPENAI_EMBEDDING_DEPLOYMENT_NAME = "PLEASE_ENTER_YOUR_OWNED_AOAI_EMBEDDING_MODEL_NAME"
OPENAI_EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
OPENAI_API_VERSION = "2023-05-15"
OPENAI_API_TYPE = "azure"

openai.api_key = OPENAI_API_KEY
openai.api_base = OPENAI_API_BASE
openai.api_version = OPENAI_API_VERSION
openai.api_type = OPENAI_API_TYPE

# Set web page title and icon.
st.set_page_config(
    page_title="Chat 💬 with your PDF 📄",
    page_icon=":robot:"
)

# Upload the PDF file.
pdf = st.file_uploader("Upload your PDF", type=["pdf"])

if pdf is not None:
    # Extract text from the uploaded PDF file.
    pdf_reader = PdfReader(pdf)
    raw_text = ""
    for page in pdf_reader.pages:
        raw_text += page.extract_text()

    # Split the text into chunks of 1000 characters with 200 characters overlap.
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    text_chunks = text_splitter.split_text(raw_text)

    # Pass the text chunks to the Embedding Model from Azure OpenAI API to generate embeddings.
    embeddings = OpenAIEmbeddings(deployment=OPENAI_EMBEDDING_DEPLOYMENT_NAME, 
                                  openai_api_key=OPENAI_API_KEY, 
                                  model=OPENAI_EMBEDDING_MODEL_NAME, 
                                  openai_api_type=OPENAI_API_TYPE, 
                                  chunk_size=1)

    # Use FAISS to index the embeddings. This will allow us to perform a similarity search on the texts using the embeddings.
    # https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/faiss.html
    faiss_vector_store = FAISS.from_texts(text_chunks, embeddings)

    # Create a text input box for the user to ask a question about the PDF.
    user_input = st.text_input("Ask a question about your PDF:")
    if user_input:
        # Perform a similarity search on the text chunks using the user input.
        docs = faiss_vector_store.similarity_search(user_input)

        # Create a Question Answering chain using the embeddings and the similarity search.
        # https://docs.langchain.com/docs/components/chains/index_related_chains
        chain = load_qa_chain(AzureOpenAI(openai_api_key=OPENAI_API_KEY, 
                                          deployment_name=OPENAI_DEPLOYMENT_NAME, 
                                          model_name=OPENAI_MODEL_NAME, 
                                          openai_api_version=OPENAI_API_VERSION), chain_type="stuff")

        # Run the chain and get the response.
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_input)

        # Display the response.
        st.write(response)
