from sqlalchemy import create_engine
from llama_index.core import SQLDatabase
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import NLSQLTableQueryEngine
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# from langchain_community.utilities.sql_database import SQLDatabase
# from langchain_ollama import OllamaLLM


db_uri = "postgresql+psycopg2://chad:password@localhost:5432/football"

# Support views!!
db = SQLDatabase.from_uri(db_uri, view_support=True)

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

# Function to run an LLM query
def llm_query(question, tone='professional', schema=True):

    if schema:
        schema_context = """
            Table: transactions
            transaction_id - ID of the transaction
            season_id - The NFL season
            timestamp - The timestamp of the event
            player_id - The player ID
            status - Whether the transaction was successful
            source - The source of the player (either a manager ID, or free agent/waiver wire)
            team_key - The destination of the player (either a manager ID, or free agent/waiver wire)
            type - Whether a player was drafted, added, or dropped

        Table: games
            game_id - ID of the game
            season_id - The season ID
            week_id - The week ID
            playoffs - Whether it's the playoffs
            manager_id1 - The manager ID for the team 1
            manager_id2 = The manager ID for the team 2
            points1 = Points for manager_id1
            points2 = Points for manager_id2
            winner_manager_id = The manager ID of the winning team (NA if it's a tie)

        Table: managers
            manager_id - Unique ID of the manager
            manager - Nickname (or person name) of the manager

        Table: teams
            team_id - Unique team key (matches with team_key columns in games table)
            team - Name of the team for a given year and manager
            number_of_moves - Number of transactions/moves made
            division_id - Division a team was in that year
            draft_grade - Draft grade for a team
            manager_id - Unique ID of the manager (see managers table)
            season_id - Season ID

        Table: players
            player_id - Unique player ID
            player - Player name
            position - Position of the player
        
        Table: stats
            stat_id - Unique ID for a statistic
            week_id - ID for the week of the season (see weeks table)
            player_id - ID of the player (see players table)
            total_points - Total fantasy points in a week for that player
            ...

        Table: rosters
            roster_id - Unique roster slot ID
            week_id - NFL week ID
            season_id - NFL season ID
            manager_id - Manager ID (see managers table)
            player_id - Player ID (see players table)
            selected_position - Selected position for a player

        Table: drafts
            draft_id = Unique draft ID
            pick = Draft pick
            round = Round of the draft
            player_id = Player ID selected in a draft
            manager_id = Manager ID that selected a player in the draft
            season_id = NFL season
        
        Table: standings
            year - Year
            manager - manager name
            games_played - Total games played that year (not including playoffs)
            games_won - Total games won
            total_points - Total points gained
            total_points_allowed - Total points allowed
        """
    else:
        schema_context = ""

    prompt = f"""

    Question: {question}

    Respond in a {tone} tone.

    Context:

    {schema_context}

    Only use the SQL database provided.

    Don't make up answers if you don't know.
    Admit you don't know.
    Be polite.

    """

    response = query_engine.query(prompt)

    # Print the response
    print(response)



# Some examples (and associated responses):

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

llm_query("What manager had the most total points in 2022? (look at standings)")
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
# ;


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
# The RB with the most total points in 2023 is Christian McCaffrey, with a 
# total of 409.7 fantasy points.

llm_query("What WR had the most total points in 2020?")
# According to the provided SQL query results, the wide receiver (WR) with the 
# most total points in 2020 is Davante Adams, who accumulated a total of 355.7 
# fantasy points during that season.

# llm_query("What manager had the best draft pick in 2023? (i.e., the player 
# drafted had the most total points)")
# doesnt work

llm_query("How many points did the first 10 players drafted score in the 2007 season?")

llm_query("What is Chad's record when he plays Shane?")
# 9-7

# llm_query("What is Shane's record when he plays Chad?")
# not working

# llm_query("How many times has Chad beaten Shane?")
# 81 times.. haha. Oops.

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


llm_query("How many adds are there in the transactions table?")
# There are 3631 adds

llm_query("What player was on the most rosters?")
# Based on the query results, it appears that the player who was on the most rosters is "Los Angeles", with a total of 194 rosters. This information is based on the analysis of the provided SQL database and the given question.

llm_query("IN what month are the most players added? How has that changed by year?")
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


llm_query("Give me a table with total adds, drops, and trades for each year.")
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

llm_query("What is the most points anyone ever scored in a game? What manager was it and in what year?")
# 

llm_query("Give me a table of RBs and the number of rosters they've been on, also the last year they were on a roster.")

llm_query("Who had the best record in 2023?")

llm_query("Give me a list of teams with best record for each year")


llm_query("What was the best free agent transaction ever made? (in terms of points that player gained for a manager/team)")

