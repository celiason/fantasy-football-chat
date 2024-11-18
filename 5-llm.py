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

# Create the query engine for natural language SQL querying
query_engine = NLSQLTableQueryEngine(
    sql_database=db, llm=Settings.llm, response_mode="context"
)

def llm_query(question, tone='professional'):

    # question = "What manager was involved in the most trades for each year?"
    
    prompt = f"""

    Question: {question}

    Respond in a {tone} tone.

    Context:

    Table: transactions
        event_id - ID of the event
        year - The NFL year (or season)
        week - The NFL week
        timestamp - The timestamp of the event
        player_id - The player ID
        source - The source of the player (either a team, or free agent/waiver wire)
        team_key - The destination of the player (either a team, or free agent/waiver wire)
        type - Whether a player was drafted, added, dropped, benched, or placed on the active roster

    Table: games
        matchup_id - ID of the matchup
        year - The NFL year (or season)
        week - The NFL week
        winning_manager_id - The manager ID for the winning team (see managers table)
        losing_manager_id - The manager ID for the losing team (see managers table)
        points_winner - How many fantasy points the winner had
        points_loser - How many fantasy points the loser had

    Table: managers
        manager_id - Unique ID of the manager
        name - Nickname (or person name) of the manager

    Table: teams
        team_key - Unique team key (matches with team_key columns in games table)
        team_name - Name of the team for a given year and manager
        division_id - Division a team was in that year
        draft_grade - Draft grade for a team
        year - NFL season
        manager_id - Unique ID of the manager (see managers table)

    Table: players
        player_id - Unique player ID
        name - Player name
        position - Position of the player
    
    Table: statistics
        stat_id - Unique ID for a statistic
        week_id - ID for the week of the season (see weeks table)
        player_id - ID of the player (see players table)
        total_points - Total fantasy points in a week for that player
        ...

    Table: rosters
        index - Unique ID
        year - NFL season/year
        week - NFL week
        manager_id - Manager ID (see managers table)
        player_id - Player ID (see players table)
        selected_position - Selected position for a player

    Table: drafts
        index = Unique draft ID
        pick = Draft pick
        round = Round of the draft
        player_id = Player ID selected in a draft
        year = NFL season
        manager_id = Manager ID that selected a player in the draft

    Only use the SQL database.

    Don't make up answers if you don't know.
    Admit you don't know.
    Be polite.

    Only provide the answer, don't give details about the SQL query.

    """

    response = query_engine.query(prompt)

    # Print the response
    print(response)


# Define a natural language query
query_str = "How many adds are there in the transactions table?"
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


llm_query("What team does Bo manage?")

llm_query("What team does Shane manage?")

llm_query("What team does Chad manage?")

# llm_query("What manager had the best draft for each NFL season year?")

llm_query("How many games did Shane win in 2008?")
# GOOD: Manager Shane won 11 games in the year 2008.

llm_query("How many games on average did manager Shane win in a year?")
# BAD: it gave 16.4 haha

llm_query("Has any manager ever won 13 games in a year? If so, who?")

# llm_query("what's the average time a player stays in event_type active? summarize by year.")
# Based on the provided context and data, I can confirm that yes, there are managers who have won 13 games in a year. The refined list of managers who have achieved this feat is:
# 1. Andrew
# 2. Chad
# 3. Shane
# 4. Bo
# 5. Josiah Mclat
# 6. Aaron
# 7. Greg
# 8. Mike
# 9. Kai
# 10. David

llm_query("What manager had the most points in 2022?")
# According to the query results, the manager with the most points in 2022 is Bo, with a total of 1954.71 points.

llm_query("My manager name is Chad. How many games did I win in 2008?")

# ALTER TABLE teams
# ADD PRIMARY KEY team_key;


# SELECT game_id,
#     CASE WHEN points_away > points_home THEN team_key_away ELSE team_key_home END AS winner,
#     CASE WHEN points_away > points_home THEN team_key_home ELSE team_key_away END AS loser
# FROM games
# LIMIT 5;

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


llm_query("What players (and their positions) were on my roster in week 2 of 2008?")
# * Eli Manning (QB)
# * Torry Holt (WR)
# * Roddy White (WR)
# * Marion Barber (RB)
# * Ryan Grant (RB)
# * Kevin Boss (TE)
# * Michael Turner (W/R)
# * Andre Johnson (BN)
# * Ricky Williams (BN)
# * Jay Cutler (BN)
# * Javon Walker (BN)
# * Matt Leinart (BN)
# * Jason Hanson (K)

llm_query("What RB had the most total points in 2023?")

llm_query("What WR had the most total points in 2023?")

# llm_query("What manager had the best draft pick in 2023? (i.e., the player drafted had the most total points)")
# doesnt work

llm_query("How many points did the first 10 players drafted score in the 2007 season?")




# Modifying tables programattically
# from sqlalchemy import create_engine
# engine = create_engine('postgresql+psycopg2://chad:password@localhost:5432/football', echo=True)
# connection = engine.connect() 
  
# table_name = 'events'

# query = f'ALTER TABLE {table_name} RENAME COLUMN trans_type TO transaction_type;'
# connection.execute(query)

# query = 'ALTER TABLE teams RENAME COLUMN name TO team_name;'
# connection.execute(query)


# query = f"ALTER TABLE teams DROP COLUMN nickname;"
# connection.execute(query)
