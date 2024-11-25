#------------------------------------------------------------------------
# Setup 'football' database
#------------------------------------------------------------------------

from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
import numpy as np
import streamlit as st

password = st.secrets["supa_password"]

REBUILD_DB = False

#------------------------------------------------------------------------
# Initialize the database
#------------------------------------------------------------------------
uri = f"postgresql://postgres.rpeohwliutyvtvmcwkwh:{password}@aws-0-us-west-1.pooler.supabase.com:6543/postgres"
# engine = create_engine('postgresql+psycopg2://chad:password@localhost:5432/football', echo=True)
engine = create_engine(uri, echo=True)

#------------------------------------------------------------------------
# We will use this below
#------------------------------------------------------------------------
rosters = pd.read_csv("data/rosters.csv")

#------------------------------------------------------------------------
# Seasons
#------------------------------------------------------------------------
seasons = pd.DataFrame({'year': range(2007, 2025)})
seasons.index.name='season_id'
seasons.index = seasons.index + 1

if REBUILD_DB:
       seasons.to_sql('seasons', engine)

#------------------------------------------------------------------------
# Players table
#------------------------------------------------------------------------
players = rosters.groupby(['player_id','name']).head(1)
players['position'] = players['eligible_positions'].str.extract('(\w+)', expand=False)
players = players[['player_id','name','position_type','position']].sort_values('player_id')
players.rename({'name': 'player'}, axis=1, inplace=True)
players.set_index('player_id', inplace=True)
players.drop('position_type', axis=1, inplace=True)

if REBUILD_DB:
       players.to_sql('players', engine, if_exists='replace')

#------------------------------------------------------------------------
# Weeks table
#------------------------------------------------------------------------

# Load data
weeks = pd.read_csv("data/weeks.csv")

# Convert start/end to datetime format
weeks['start'] = pd.to_datetime(weeks['start'])
weeks['end'] = pd.to_datetime(weeks['end']) + pd.Timedelta('23:59:59')

# Just unique entries to add 'playoffs' to weeks
rost_nondup = rosters[['year','week','playoffs']].drop_duplicates()

# Merge with rost_nondup
weeks = weeks.merge(rost_nondup, on=['year','week'], how='left')

# Remove NAs (we only started doing week 17 recently)
weeks = weeks[weeks['playoffs'].notna()]

# Add season ID
# weeks = weeks.merge(seasons.reset_index(), on='year')
weeks.index.name = 'week_id'

# Want 1-based index
weeks.index = weeks.index+1

# Setup database columns
# sql_weeks = weeks.drop('year', axis=1)
sql_weeks = weeks
sql_weeks = sql_weeks[['year','week','start','end','playoffs']]

# Write to db
if REBUILD_DB:
       sql_weeks.to_sql("weeks", engine, if_exists='replace')

#------------------------------------------------------------------------
# Teams and managers tables
#------------------------------------------------------------------------

teams = pd.read_csv("data/teams.csv", index_col=0)

# Fix some missing manager names
teams.loc[(teams['name'] == 'Rubber City Galoshes'), 'nickname'] = 'matt_harbert'
teams.loc[(teams['name'] == 't-t-totally dudes'), 'nickname'] = 'josiah mclat'
teams.loc[(teams['name'] == 'The Woebegones'), 'nickname'] = 'Charles'
teams.loc[(teams['name'] == "Doinel's Destroyers!"), 'nickname'] = 'Rusty Blevins'
teams.loc[(teams['name'] == 'The Five Toes'), 'nickname'] = 'Justin Smith'
teams.loc[(teams['name'] == 'Browns West'), 'nickname'] = 'Bodad'

# Check it worked
len(teams[teams['nickname'] == '--hidden--']) == 0

#------------------------------------------------------------------------
# Managers table
#------------------------------------------------------------------------

# Create it
managers = pd.DataFrame(list(set(teams["nickname"])), columns=['nickname'])

managers.sort_values('nickname', inplace=True)
managers.reset_index(inplace=True)
managers = managers.drop('index', axis=1)

managers.index.name = 'manager_id'

# 1-value index
managers.index = managers.index+1

# Rename for SQL db
managers = managers.rename(columns={'index': 'manager_id', 'nickname': 'manager'})

# Write to db
if REBUILD_DB:
       managers.to_sql("managers", engine, if_exists='replace')

#------------------------------------------------------------------------
# Teams table
#------------------------------------------------------------------------

# Now we'll add manager ID to teams
teams = teams.merge(managers.reset_index(), left_on='nickname', right_on='manager', suffixes=['_teams', ''])
teams = teams.rename(columns={'name': 'team'})
teams = teams.merge(seasons.reset_index(), on='year')

# Setup columns for db
sql_teams = teams[['season_id','year','manager_id','team','number_of_moves','division_id','draft_grade']]
sql_teams.index.name = 'team_id'
sql_teams.index = sql_teams.index + 1

# Output to db
if REBUILD_DB:
       sql_teams.to_sql("teams", engine, if_exists='replace')

#------------------------------------------------------------------------
# Rosters table
#------------------------------------------------------------------------

rosters = rosters.merge(teams, on='team_key', suffixes=['', '_teams'])
rosters = rosters.merge(weeks.reset_index(), on=['year','week'], suffixes=['', '_teams'])

sql_rosters = rosters[['year','week','manager_id','player_id','selected_position']]
sql_rosters.index.name = 'roster_id'
sql_rosters.index = sql_rosters.index + 1

# Write to db
if REBUILD_DB:
       sql_rosters.to_sql('rosters', engine, if_exists='replace')

#------------------------------------------------------------------------
# Matchups table
#------------------------------------------------------------------------

matchups = pd.read_csv("data/matchups.csv", index_col=0)

# Sanity check
any(matchups['team_key1'] == matchups['team_key2'])

# Merges with team
matchups = matchups.merge(teams[['team_key','manager_id','year']], left_on='team_key1', right_on='team_key')
matchups.rename({'manager_id': 'manager_id1'}, axis=1, inplace=True)
matchups = matchups.merge(teams[['team_key','manager_id']], left_on='team_key2', right_on='team_key')
matchups.rename({'manager_id': 'manager_id2'}, axis=1, inplace=True)

matchups['winner_manager_id'] = np.where(matchups['points1'] > matchups['points2'], matchups['manager_id1'], matchups['manager_id2'])

# Account for ties
matchups['winner_manager_id'] = np.where(matchups['points1'] == matchups['points2'], np.nan, matchups['winner_manager_id'])

# Set to int64
matchups['winner_manager_id'] = matchups['winner_manager_id'].astype('Int64')

matchups['winner_points'] = np.where(matchups['points1'] > matchups['points2'], matchups['points1'], matchups['points2'])

# Account for ties
matchups['winner_points'] = np.where(matchups['points1'] == matchups['points2'], np.nan, matchups['winner_points'])
# We had 5 ties

# Add playoffs
matchups = matchups.merge(weeks[['year','week','playoffs']], how='left', on=['year','week'])


matchups.reset_index(inplace=True)
matchups.index.name = 'game_id'
matchups.index = matchups.index + 1

# Add week ID
matchups = matchups.merge(weeks.reset_index()[['year','week','week_id']], on=['year','week'])

# Add season
matchups = matchups.merge(seasons.reset_index(), on='year')

columns = ['year','week','playoffs','manager_id1','manager_id2','points1','points2','winner_manager_id']

# Sanity check again
any(matchups['manager_id1'] == matchups['manager_id2'])

matchups.index = matchups.index + 1
matchups.index.name = 'game_id'

# Write to db
if REBUILD_DB:
       matchups[columns].to_sql("games", engine, if_exists='replace')

#------------------------------------------------------------------------
# Statistics table
#------------------------------------------------------------------------

statistics = pd.read_csv("data/statistics.csv")

# Merge with weeks
statistics = statistics.merge(weeks.reset_index(), on=['year','week'])

statistics.columns = statistics.columns.str.replace(' ', '_')
statistics.columns = statistics.columns.str.replace('-', '_')
statistics.columns = statistics.columns.str.replace('+', '_plus')
statistics.columns = statistics.columns.str.lower()

# Merge some overlapping columns
statistics['fgm_0_19'] = statistics['fgm_0_19'].fillna(statistics['fg_0_19'])
statistics['fgm_20_29'] = statistics['fgm_20_29'].fillna(statistics['fg_20_29'])
statistics['fgm_30_39'] = statistics['fgm_30_39'].fillna(statistics['fg_30_39'])

# Drops
statistics.drop(['fg_0_19','fg_20_29','fg_30_39'], axis=1, inplace=True)

# Look at missing values by column
# import missingno as msno
# msno.bar(statistics)
# msno.heatmap(statistics)

# Make a histogram to check
# import seaborn as sns
# sns.histplot(data=statistics, x='total_points', bins=20)
# lots of zeros, makes sense. some negatives.

statistics.rename({'name': 'player'}, axis=1, inplace=True)

# Final columns in the table
columns = ['year','week','player_id', 'total_points', 'pass_yds', 'pass_td', 'int',
       'rush_yds', 'rush_td', 'rec', 'rec_yds', 'rec_td', 'ret_yds', 'ret_td',
       '2_pt', 'fum_lost', 'fum_ret_td', 'fg_40_49',
       'fg_50_plus', 'fgm_0_19', 'fgm_20_29', 'fgm_30_39', 'pat_made',
       'pat_miss', 'pts_allow', 'sack', 'fum_rec', 'td', 'safe', 'blk_kick',
       'kick_and_punt_ret_td', 'pts_allow_0', 'pts_allow_1_6',
       'pts_allow_7_13', 'pts_allow_14_20', 'pts_allow_21_27',
       'pts_allow_28_34', 'pts_allow_35_plus', 'rush_att',
       'targets', '4_dwn_stops', 'xpr', 'rec_1st_downs', 'rush_1st_downs',
       'fg_yds', 'fg_made', 'fg_miss']

stats = statistics[columns]
stats.index.name = 'stat_id'
stats.index = stats.index + 1

if REBUILD_DB:
       stats.to_sql('stats', engine, if_exists='replace')

#------------------------------------------------------------------------
# Drafts
#------------------------------------------------------------------------

drafts = pd.read_csv("data/drafts.csv", index_col=0)

# Add manager ID
drafts = drafts.merge(teams[['team_key','manager_id']], on="team_key")
drafts = drafts.drop('team_key', axis=1)
# drafts.rename({'player_id':'pid', 'manager_id':'mid'}, axis=1, inplace=True)
drafts = drafts.merge(seasons.reset_index(), on='year')
drafts = drafts.drop('year', axis=1)
drafts.index.name = 'draft_id'
drafts.index = drafts.index + 1

# Write to db
if REBUILD_DB:
       drafts.to_sql("drafts", engine, if_exists='replace')

#------------------------------------------------------------------------
# Events table
#------------------------------------------------------------------------

events = pd.read_csv("data/events.csv")

teams_lookup = teams.set_index('team_key')['manager_id'].to_dict()

events['source_manager_id'] = events['source'].replace(teams_lookup)
events['destination_manager_id'] = events['destination'].replace(teams_lookup)

# Rename and pick columns
events.rename({'trans_type':'type'}, axis=1, inplace=True)
events.index.name = 'event_id'

events['source'] = np.where(events['source_manager_id'].isin(['freeagents','waivers']), events['source_manager_id'], 'team')
events['destination'] = np.where(events['destination_manager_id'].isin(['freeagents','waivers']), events['destination_manager_id'], 'team')

events['source_manager_id'] = events['source_manager_id'].apply(lambda x: np.nan if isinstance(x, str) else x)
events['destination_manager_id'] = events['destination_manager_id'].apply(lambda x: np.nan if isinstance(x, str) else x)

events['source_manager_id'] = events['source_manager_id'].astype('Int64')
events['destination_manager_id'] = events['destination_manager_id'].astype('Int64')

# events['source_manager_id'].value_counts()
# events['destination_manager_id'].value_counts()
# events.groupby('year')['type'].value_counts()

events_sql = events[['year','week','timestamp','player_id','source_manager_id','destination_manager_id','source','destination','type','selected_position']]

events_sql = events_sql.sort_values('timestamp')

events_sql = events_sql.reset_index(drop=True)

events_sql.index.name = 'event_id'

# 1-based
events_sql.index = events_sql.index + 1

print(events_sql.head())

# Output to SQL database
if REBUILD_DB:
       events_sql.to_sql("events", engine, if_exists='replace')

#------------------------------------------------------------------------
# Transactions
#------------------------------------------------------------------------

transactions = pd.read_csv("data/transactions.csv", index_col=0)

transactions['source'] = transactions['source'].replace(teams_lookup)
transactions['destination'] = transactions['destination'].replace(teams_lookup)

# Rename and pick columns
transactions.rename({'trans_type':'type'}, axis=1, inplace=True)
transactions.index.name = 'transaction_id'

# transactions['source'] = np.where(transactions['source_mid'].isin(['freeagents','waivers']), transactions['source_mid'], 'team')
# transactions['destination'] = np.where(transactions['destination_mid'].isin(['freeagents','waivers']), transactions['destination_mid'], 'team')

# transactions['source'].value_counts()
# transactions['destination'].value_counts()

transactions['source'] = transactions['source'].apply(lambda x: np.nan if isinstance(x, str) else x)
transactions['destination'] = transactions['destination'].apply(lambda x: np.nan if isinstance(x, str) else x)

# Integer conversion
transactions['source'] = transactions['source'].astype('Int64')
transactions['destination'] = transactions['destination'].astype('Int64')

from datetime import datetime

transactions['timestamp'] = transactions['timestamp'].transform(datetime.fromtimestamp)

transactions.reset_index(inplace=True, drop=True)
transactions.index.name='transaction_id'

transactions = transactions.merge(seasons.reset_index(), on='year')

transactions = transactions[['year','player_id','timestamp','status','source','destination','type']]
transactions.index.name='transaction_id'
transactions.index = transactions.index + 1
transactions.to_sql("transactions", engine)

#------------------------------------------------------------------------
# Running queries from python!
#------------------------------------------------------------------------

from sqlalchemy import text

db = engine.connect()

# running queries from .SQL files
# psql -d football -f query.sql
# query = open("views.sql").read()
query = """select * from slots;"""
df = pd.read_sql_query(text(query), con=db)

# Checks - all these should be the same!
counts = df.groupby(['year','manager','week'])['player'].count() # OK first week good, others not..
counts.plot() # pretty good, a few weirdos with +/- 1 player on a week, but that might be normal
# NOTE this is working for most weeks, but there a few cases when the drop 
# time is before the last time of the NFL week (11:59 PM last day of week) and 
# I need to figure out what to do with that...


# Create Views


