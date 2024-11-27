import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy import text
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import time

# Connect to database
def init_database(password: str, database: str) -> SQLDatabase:
  db_uri = f"postgresql://postgres.rpeohwliutyvtvmcwkwh:{password}@aws-0-us-west-1.pooler.supabase.com:6543/{database}"
  # Here I'm limiting the LLM to only 2 tables (makes things easier, gives better results)
  return SQLDatabase.from_uri(db_uri, include_tables = ['slots', 'standings', 'team_names'], view_support=True)

# TODO setup hugging face so we don't hit rate limits at Grow

# Setup HF llm
# llm = HuggingFaceEndpoint(
#     repo_id="defog/sqlcoder-7b-2",
#     task="text-generation",
#     max_new_tokens=512,
#     do_sample=False,
#     repetition_penalty=1.03,
#     huggingfacehub_api_token=st.secrets['HF_KEY']
# )

# api_key
# chat_model = ChatHuggingFace(llm=llm)

# chat_model.invoke([AIMessage('you are'), HumanMessage(content="what happens when apple falls from tree?")])

llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0, api_key=st.secrets['groq_api'], max_tokens=500)

# SQL chain setup
# My understanding is that this outputs a SQL query and passes it to the LLM below
# then the LLM translates the SQL output to a human-readable result
def get_sql_chain(db: SQLDatabase):
# def get_sql_chain(db: SQLDatabase, user_query: str, chat_history: list):

  template = """
    You are a data analyst for a fantasy football company. You're name is JC. You are interacting with a user 
    who is asking you questions about a fantasy football league database.

    Based on the table schema below, write a PostgreSQL query that would answer the user's question. 
    Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    Do not wrap table name in double quotes (").
    
    Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

    If you don't know the answer, don't try to guess.

    Don't use window functions in the WHERE clause.
    
    For example:
    
    Question: Which manager had the best record in 2008?
    
    SQL Query:
    SELECT manager
    FROM standings
    WHERE year = 2008
    ORDER BY wins DESC LIMIT 1;
    
    Question: What were the top 3 quarterbacks in 2007?
    
    SQL Query:
    SELECT year, player, SUM(points) AS season_points
    FROM slots
    WHERE year = 2007 and position = 'QB' 
    GROUP BY year, player
    ORDER BY season_points DESC LIMIT 3;
    
    Write in PostgreSQL syntax.
    
    Do not use backslashes in your query.

    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
  prompt = ChatPromptTemplate.from_template(template)
    
  def get_schema(_):
    return db.get_table_info()

  # chain = (RunnablePassthrough.assign(schema=get_schema)
  #   | prompt
  #   | llm
  #   | StrOutputParser())
  
  # return chain.invoke({"question": user_query, "chat_history": chat_history})

  return (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm
    | StrOutputParser()
  )

# response = get_sql_chain(db, user_query, st.session_state.chat_history)
# response

# Connect to LLM
def get_response(user_query: str, db: SQLDatabase, chat_history: list):
  
  sql_chain = get_sql_chain(db)
  
  template = """
    You are a data analyst for a fantasy football company.

    You're name is JC.
    
    You are interacting with a user who is asking you questions about a fantasy football league database.
    
    Based on the table schema below, question, sql query, and sql response, write a natural 
    language response.
    
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    PostgreSQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""
  
  prompt = ChatPromptTemplate.from_template(template)

  # Setup the langchain
  chain = (
    RunnablePassthrough
    .assign(query=sql_chain)
    .assign(
      schema=lambda _: db.get_table_info(),
      # NOTE needed to replace '\\' to get SQLalchemy to work correctly
      # TODO I need to find out how to check if the SQL query will be valid before running it
      response=lambda vars: db.run(vars["query"].replace("\\", "")),
    )
    # pass query, schema, response to prompt generator
    | prompt
    # pass prompt to the llm
    | llm
    # format the output
    | StrOutputParser()
  )
  
  return chain.invoke({
    "question": user_query,
    "chat_history": chat_history,
  })

# from llama_index.core.query_engine import NLSQLTableQueryEngine
# help(NLSQLTableQueryEngine)

# tmp = chain.invoke({"question": user_query, "chat_history":st.session_state.chat_history})
# try:
#    chain.invoke({"question": user_query, "chat_history":st.session_state.chat_history})
# except:
#    print("nothing")

# help(chain.invoke)

# testing
# user_query = "how many dollars did my kicker Bob get against bofo?"
# user_query = "what team does dave have in 2007?"

# Check for chat history  
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
      AIMessage(content="Hi, kindly dimwit. I'm an expert on all things Slow Learner's. Ask me anything about your league's history."),
    ]

# TODO make it like he's JC?

# Page title, etc.
st.set_page_config(page_title="Slow Learners Chat", page_icon=":football:")

# Page header
st.title("Slow Learners Database Chat")

# Add logo
st.image("assets/dalle_logo2.jpg")

# Sidebar
# with st.sidebar:
#     st.subheader("Settings")
#     st.write("This is a simple chat application using PostgreSQL. Connect to the database and start chatting.")
    
#     st.text_input("Database", value="postgres", key="Database")
#     st.text_input("Password", type="password", value="admin", key="Password")
    
#     if st.button("Connect"):
#         with st.spinner("Connecting to database..."):
#             db = init_database(
#                 st.session_state["Password"],
#                 st.session_state["Database"]
#             )
#             st.session_state.db = db
#             st.success("Connected to database!")

# Load db if not in currently in session
if 'db' not in st.session_state:
    db = init_database(st.secrets["supa_password"], "postgres")
    st.session_state.db = db

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message...")

# Printout the conversation
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    with st.chat_message("AI"):
        message_placeholder = st.empty()
        full_response = ""
        with st.spinner("Thinking..."):
           response = get_response(user_query=user_query, db=st.session_state.db, chat_history=st.session_state.chat_history)
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
        # st.markdown(response)
        
    st.session_state.chat_history.append(AIMessage(content=full_response))

# TODO add plotting capability

