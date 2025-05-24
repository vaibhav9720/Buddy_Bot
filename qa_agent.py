# %%
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.chat_models import ChatOpenAI


# %%
def get_sql_agent(db_uri: str):
    
    """
    Creates and returns a LangChain SQL Agent for natural language queries.
    :param db_uri: Database URI, e.g. 'mysql+pymysql://user:pass@host:port/dbname'
    """

    db=SQLDatabase.from_uri(db_uri)
    toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(temperature=0))
    agent = create_sql_agent(llm=ChatOpenAI(temperature=0), toolkit=toolkit, verbose=True)
    return agent

# %%


# %%


# %%


# %%


# %%



