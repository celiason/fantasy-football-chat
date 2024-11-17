from sqlalchemy import create_engine
from llama_index.core import SQLDatabase
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import NLSQLTableQueryEngine
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


db = SQLDatabase.from_uri('postgresql+psycopg2://chad:password@localhost:5432/football')

db.get_usable_table_names()


# Configure the LLM to use Ollama with a specific model
Settings.llm = Ollama(
    model="llama3",
    request_timeout=300.0,
)

# Configure embeddings using Hugging Face's pre-trained model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")


def llm_query(question):
    # first sql llm
    llm = ChatOllama(model="llama-sql")
    prompt = ChatPromptTemplate.from_template(" {topic}")
    # chain
    chain = prompt | llm | StrOutputParser()
    # chain invocation
    sql = chain.invoke({"topic": f"{question}"})
    sql = re.sub(r'(?:(?<=_) | (?=_))','',sql)
    # return sql query
    return sql

llm_query('hi there')

# Create the query engine for natural language SQL querying
query_engine = NLSQLTableQueryEngine(
    sql_database=db, llm=Settings.llm, response_mode="context"
)

# Define a natural language query
query_str = "How many adds are there in the events table?"
# There are 3596 adds in the events table.

query_str = "What player was on the most teams? Look at the events table and the players table. Destination is the team key where a player ends up."
# Response:
# Robbie Gould is the player who has been on the most teams with a total of 23 teams.


query_str = "IN what month are the most players added? How has that changed by year?"
# Response (cool!):
# According to the data, the month with the most players added varies from year to year. Here's a breakdown of the top months for each year:

# * 2007: September (456) and October (652)
# * 2008: September (523), October (480), and November (323)
# * 2014: September (510), October (389), and December (144)
# * 2015: November (535), October (466), and September (444)
# * 2016: October (528), November (450), and September (396)
# * 2018: October (507), November (453), and December (311)
# * 2019: December (381), October (434), and November (414)
# * 2020: November (540), October (520), and December (455)
# * 2021: November (589), October (462), and September (406)
# * 2022: November (508), October (501), and December (322)

# It's clear that the months with the most players added vary significantly from year to year. If you're looking for a consistent pattern, it seems that October and November tend to be the top months across multiple years.

query_str = "Who was the earlist player drafted in each year? (look at the events table and the players table)"


query_str = "Give me a table with total adds, drops, and trades for each year."
# Here's a synthesized response based on the query results:

# **Total Adds, Drops, and Trades for Each Year**

# Here is the breakdown of total adds, drops, and trades for each year from 2007 to 2023:

# | Year | Total Adds | Total Drops | Total Trades |
# | --- | --- | --- | --- |
# | 2007 | 365 | 370 | 44 |
# | 2008 | 272 | 272 | 12 |
# | 2014 | 254 | 251 | 26 |
# | 2015 | 325 | 320 | 37 |
# | 2016 | 314 | 306 | 17 |
# | 2018 | 367 | 363 | 26 |
# | 2019 | 356 | 355 | 9 |
# | 2020 | 345 | 336 | 4 |
# | 2021 | 352 | 352 | 3 |
# | 2022 | 335 | 331 | 6 |
# | 2023 | 311 | 314 | 0 |

# This table shows the total number of adds, drops, and trades for each year from 2007 to 2023. The years are listed in chronological order, with the most recent year (2023) at the bottom


query_str = "Is there a relationship between the number of adds/drops a team makes and the number of wins?"
# Nope. Gave a query, but it isn't right. Team_id does not match up with team_key1 or team_key2
# NOTE: I might want to rethink the database structure to make it more clear.

# SELECT 
#   t.name, 
#   COUNT(m.matchup_id) AS num_wins, 
#   COUNT(DISTINCT e.event_id) AS num_moves
# FROM 
#   teams t
#   JOIN matchups m ON t.team_id = m.team_key1 OR t.team_id = m.team_key2
#   JOIN events e ON e.player_id IN (m.team_key1, m.team_key2)
# WHERE 
#   e.trans_type IN ('add', 'drop')
# GROUP BY 
#   t.name
# ORDER BY 
#   num_wins DESC;


# Execute the query using the engine
# The title() method ensures the query string matches the capitalization format in the database

def llm_query(question, tone='professional'):

    # question = "What manager was involved in the most trades for each year?"
    
    prompt = f"""

    Question: {question}

    Respond in a {tone} tone.

    Context:

    Table: statistics
    Table: 

    - The events table has all the transactions that occur for a player
        (add to a team, drop from a team, put on an active roster 
        for a game week, etc.)
    - The games table is the set of matchups betwen teams for each week. team_key_winner and team_key_loser correspond to the team_key in teams.
    - The managers table is the list of people in the league that manage a team
    - The players table is the list of NFL players that are managed by managers
    - The teams table has all the teams managed by different managers
    - The weeks table has the date ranges of weeks for each season (year)

    Only use the SQL database.

    Don't make up answers if you don't know.
    Admit you don't know.
    Be polite.

    When I ask "How many games did Bill win in 2007"
    you should perform this type of query-
    SELECT g.year,COUNT(g.team_key_winner),m.manager_name
    FROM games g
    LEFT JOIN teams t
        on g.team_key_winner = t.team_key
    LEFT JOIN managers m
        on t.manager_id = m.manager_id
    WHERE manager_name = 'Bill' and g.year = 2007
    GROUP BY m.manager_name, g.year
    ORDER BY g.year ASC;

    Only provide the answer, don't give details about the SQL query.

    """

    response = query_engine.query(prompt)

    # Print the response
    print(response)


llm_query("What team does Bo manage?")
llm_query("What team does Shane manage?")
llm_query("What team does Chad manage?")
llm_query("What are the top 3 players on an active roster?", add_context=True)

llm_query("What manager had the best draft grade for each year?")

llm_query("How many games did manager Shane win in 2008?")
# GOOD: Manager Shane won 11 games in the year 2008.
llm_query("How many games on average did manager Shane win in a year?")
# BAD: it gave 16.4 haha

llm_query("Has any manager ever won 13 games in a year? If so, who?")

llm_query("what's the average time a player stays in event_type active? summarize by year.")
# BAD: -27 days in 2007

llm_query("What manager had the most points in 2022?")
# BAD: no response


llm_query("My manager name is Chad. How many games did I win in 2008? (just give me the result, not the SQL query).")

ALTER TABLE teams
ADD PRIMARY KEY team_key;


SELECT game_id,
    CASE WHEN points_away > points_home THEN team_key_away ELSE team_key_home END AS winner,
    CASE WHEN points_away > points_home THEN team_key_home ELSE team_key_away END AS loser
FROM games
LIMIT 5;

# I kept getting a connect error 61
# figure out I needed to download ollama first
# downloaded from here- https://github.com/ollama/ollama?tab=readme-ov-file
# then in the terminal, I did:
# ollama pull llama3


# SELECT COUNT(*) AS total_wins
# FROM matchups m
# JOIN teams t ON m.team_key1 = t.team_key OR m.team_key2 = t.team_key
# JOIN managers ma ON t.manager_id = ma.manager_id
# WHERE ma.nickname = 'Chad' AND m.year = 2008 AND points1 > points2
;


llm_query("What player had the most total points in 2007? hint: use the statistics table")

llm_query("What are the names of the players with the top-3 most total points in 2023? hint: use the statistics table")

llm_query("What manager had the best draft pick in 2023? (i.e., the player drafted had the most total points). Hint: use the statistics table")

llm_query("How many points did the first 10 players drafted score in the 2007 season?")


# Modifying tables programattically
from sqlalchemy import create_engine
engine = create_engine('postgresql+psycopg2://chad:password@localhost:5432/football', echo=True)
connection = engine.connect() 
  
table_name = 'events'

query = f'ALTER TABLE {table_name} RENAME COLUMN trans_type TO transaction_type;'
connection.execute(query)

query = 'ALTER TABLE teams RENAME COLUMN name TO team_name;'
connection.execute(query)


query = f"ALTER TABLE teams DROP COLUMN nickname;"
connection.execute(query)


