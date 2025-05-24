# %%


from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st

# %%
def alert_explainer(alert_id: str,agent,retriever):
    trade_data=agent.run(f"Give the trade details for the alert id {alert_id}")
    print("Trade data",trade_data)
    prompt = PromptTemplate.from_template("""
    {input}

    <context>
    {context}
    </context>

    Determine whether this trade is a potential policy breach.
    Explain your reasoning and cite relevant policies.
    """)
    qa_chain=create_stuff_documents_chain(
        llm=ChatOpenAI(temperature=0),
        prompt=prompt
    )
    rag_chain=create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=qa_chain
    )
    # combined_input = f"""Trade Alert ID: {alert_id}
    # Trade Details: {trade_data}"""

    combined_input = f"""
    Trade Alert ID: {alert_id}
    Trade Details: {trade_data}
    """

    response = rag_chain.invoke({
        "input": combined_input
    })
    source_chunks = response.get("source_documents", [])
    st.markdown("### 📚 Source Documents Used")

    if source_chunks:
        st.markdown("### 📚 Source Documents Used")
        for doc in source_chunks:
            st.markdown(f"**📄 File:** `{doc.metadata.get('source', 'Unknown')}`")
            st.write(doc.page_content[:300] + "...")

    return response["answer"],trade_data

# %%


# %%


# %%



