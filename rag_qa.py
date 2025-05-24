# %%
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from config import OPENAI_API_KEY
import config
import openai
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
# %%
persitent_directory = "vector_store"
pdf_directory = "pdf_files"

# %%
def process_all_pdfs():
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    all_chunks=[]
    
    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf'):
            path = os.path.join(pdf_directory, filename)
            loader = PyPDFLoader(path)
            docs=loader.load()
            for doc in docs:
                doc.metadata['source']=os.path.basename(path)
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

            chunks=splitter.split_documents(docs)
            for chunk in chunks:
                chunk.metadata["source"] = docs[0].metadata["source"]
            all_chunks.extend(chunks)
    vectordb=Chroma.from_documents(
        documents=all_chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory=persitent_directory
    )
    vectordb.persist()



# %%
def load_vector_store():
    vectordb = Chroma(
        persist_directory=persitent_directory,
        embedding_function=OpenAIEmbeddings()
    )
    return vectordb
def get_qa_chain(vectordb):
    retriever=vectordb.as_retriever(search_kwargs={"k": 3})
    chain =RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return chain


# %%



