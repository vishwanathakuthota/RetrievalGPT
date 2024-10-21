# pip install langchain openai chromadb tiktoken unstructured

# lib imports
import os
import sys

import openai
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_chroma import Chroma

import constants

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = constants.APIKEY

# Enable to save to disk and reuse the model
PERSIST = False

query = None
if len(sys.argv) > 1:
    query = sys.argv[1]

# Create embeddings
embedding = OpenAIEmbeddings()

# Reuse index if it exists
if PERSIST and os.path.exists("persist"):
    print("Reusing index ... \n")
    vectorstore = Chroma(persist_directory="persist", embedding_function=embedding)
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    # Use PyPDFLoader for PDFs and DirectoryLoader for other files
    pdf_loader = PyPDFLoader("data/python_for_finance.pdf")  # Adjust file path as necessary
    dir_loader = DirectoryLoader("data/")  # Adjust directory as needed

    # Load documents from both PDF and other text files
    documents = pdf_loader.load() + dir_loader.load()

    # Persist or create a new index
    if PERSIST:
        vectorstore = Chroma(embedding_function=embedding, persist_directory="persist")
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        # Pass the embedding explicitly to VectorstoreIndexCreator
        index = VectorstoreIndexCreator(embedding=embedding).from_documents(documents)

# Create the ConversationalRetrievalChain with the index retriever
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

chat_history = []
while True:
    if not query:
        query = input("Prompt: ")
    if query in ['quit', 'q', 'exit']:
        sys.exit()

    result = chain({"question": query, "chat_history": chat_history})
    print(result['answer'])

    chat_history.append((query, result['answer']))
    query = None