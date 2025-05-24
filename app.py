# %%
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from rag_qa import process_all_pdfs, load_vector_store, get_qa_chain
from qa_agent import get_sql_agent
from rag_qa import get_qa_chain
# %%
persitent_directory = "vector_store"

# %%
st.set_page_config(page_title="Multi-PDF QA App")
st.title("📄 RAG Q&A from Multiple PDFs")

mode = st.sidebar.radio("Select mode", ["PDF QA", "Trade Data Query","Alert Explainer"])

if mode=="PDF QA":

# %%
        # Upload and save files
        uploaded_files = st.file_uploader("Upload multiple PDF files", type="pdf", accept_multiple_files=True)
        print("session state",st.session_state)
        if uploaded_files:
            os.makedirs("pdf_files", exist_ok=True)
            for uploaded_file in uploaded_files:
                with open(os.path.join("pdf_files", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())

            # Reprocess all PDFs in the folder
            process_all_pdfs()
            st.success("Vector store created from all PDFs in pdf_files folder.")
            st.session_state["vector_ready"] = True

        # Load vector store and set up QA
        if os.path.exists("vector_store") or st.session_state.get("vector_ready"):

            st.subheader("🔎 Ask a question about the documents")
            query = st.text_input("Ask a question about the PDFs")
            if query:
                chain=get_qa_chain(load_vector_store())
                result = chain({"query": query})
                st.write("### 🧠 Answer:")
                st.write(result["result"])

                st.write("---")
                st.write("### 📚 Source Documents:")
                for doc in result["source_documents"]:
                    st.write("Metadata:", doc.metadata)

                    st.write(f"**Source:** {doc.metadata['source']}")
                    st.write(doc.page_content[:500] + "...")
        pass
elif mode=="Alert Explainer":
    st.header("🧮 Alert Explainer")
    db_uri = st.text_input("Enter your DB URI:", value="sqlite:///trade_data.db")
    if db_uri:
        alert_id =st.text_input("Enter the alert ID:")
        if alert_id:
            
            try:
                from alert_explainer import alert_explainer
                agent= get_sql_agent(db_uri)
                #retriever=get_qa_chain(load_vector_store())
                retriever = load_vector_store().as_retriever(search_kwargs={"k": 3})
                with st.spinner("Analysing alert..."):
                    explanation,trade_data=alert_explainer(alert_id,agent,retriever)
                    st.subheader("🔍 Explanation:")
                    st.write(explanation)
                    download_text=f"""
                    
                    Alert_ID :{alert_id}
                    Trade_Details:{trade_data}
                    Explanation: {explanation}
                    
                    """
                    st.download_button(
                        label="Download Explanation",
                        data=download_text,
                        file_name=f"explanation_{alert_id}.txt",
                        mime="text/markdown"
                    )
            except Exception as e:
                st.error(f"Error during analysing explanation:{e}")



else:
     st.header("🧮 Natural Language Trade Data Query")
     db_uri = st.text_input("Enter your DB URI:", value="sqlite:///trade_data.db")


     if db_uri:
        agent = get_sql_agent(db_uri)
# %%

        question = st.text_input("Ask a trade-related question:")
        if question:
            with st.spinner("Running SQL query..."):
                try:
                    result = agent.run(question)
                    st.markdown("### 📊 Result")
                    st.write(result)
                except Exception as e:
                    st.error(f"Error executing query: {e}")


