################################################################################################
# Push to database
################################################################################################

from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2

# Initialize the database
engine = create_engine('postgresql+psycopg2://chad:password@localhost:5432/football', echo=True)

# We will use this below
rosters = pd.read_csv("data/rosters.csv")

# Players table
players = rosters.groupby(['player_id','name']).head(1)
players['position'] = players['eligible_positions'].str.extract('(\w+)', expand=False)
players = players[['player_id','name','position_type','position']].sort_values('player_id')
players.set_index('player_id', inplace=True)
players.to_sql('players', engine)

# Events table
events = pd.read_csv("data/events.csv")
events.drop('name', axis=1, inplace=True)
# Only keep add, drop and trades
events = events[events['trans_type'].isin(['add','drop','trade'])]
# Rename and pick columns
events.rename({'trans_type':'type','destination':'team_key'}, axis=1, inplace=True)
events.index.name = 'event_id'
events_sql = events[['year','week','timestamp','player_id','source','team_key','type']]
events_sql.to_sql("transactions", engine)

# Weeks table
weeks = pd.read_csv("data/weeks.csv")
weeks['start'] = pd.to_datetime(weeks['start'])
weeks['end'] = pd.to_datetime(weeks['end']) + pd.Timedelta('11:59:59')
weeks.index.name = 'week_id'
weeks.to_sql("weeks", engine)

# Teams tables
teams = pd.read_csv("data/teams.csv", index_col=0)

# Managers table
managers = pd.DataFrame(list(set(teams["nickname"])), columns=['nickname'])
managers.sort_values('nickname', inplace=True)
managers.reset_index(inplace=True)
managers_sql = managers.rename(columns={'index': 'manager_id', 'nickname': 'name'})
managers_sql.set_index('manager_id', inplace=True)
managers_sql.to_sql("managers", engine)

teams = teams.merge(managers.reset_index(), on="nickname")
# teams.reset_index(inplace=True)
teams.index.name = 'team_id'
teams = teams.rename({'index': 'manager_id'}, axis=1)
teams_sql = teams[['name','number_of_moves','division_id','draft_grade','year','manager_id']]
teams_sql.to_sql("teams", engine)


# Rosters table
rosters = rosters.merge(teams, on='team_key', suffixes=['', '_teams'])
rosters = rosters[['year','week','playoffs','manager_id','player_id','selected_position']]
rosters.to_sql('rosters', engine)

import numpy as np

# Matchups table
matchups = pd.read_csv("data/matchups.csv", index_col=0)
matchups['team_key_winner'] = np.where(matchups['points1'] > matchups['points2'], matchups['team_key1'], matchups['team_key2'])
matchups['team_key_loser'] = np.where(matchups['points1'] < matchups['points2'], matchups['team_key1'], matchups['team_key2'])
matchups['points_winner'] = np.where(matchups['points1'] > matchups['points2'], matchups['points1'], matchups['points2'])
matchups['points_loser'] = np.where(matchups['points1'] < matchups['points2'], matchups['points1'], matchups['points2'])
matchups = matchups.merge(teams, left_on='team_key_winner', right_on='team_key')
matchups.rename({'manager_id': 'winning_manager_id'}, axis=1, inplace=True)
matchups = matchups.merge(teams, left_on='team_key_loser', right_on='team_key')
matchups.rename({'manager_id': 'losing_manager_id'}, axis=1, inplace=True)
matchups.reset_index(inplace=True)
matchups.index.name = 'matchup_id'
columns = ['year_x','week','winning_manager_id','losing_manager_id','points_winner','points_loser']

matchups[columns].to_sql("games", engine)

# Statistics table
statistics = pd.read_csv("data/statistics.csv")

# Merge with weeks
statistics = statistics.merge(weeks.reset_index(), on=['year','week'])

statistics.columns = statistics.columns.str.replace(' ', '_')
statistics.columns = statistics.columns.str.replace('-', '_')
statistics.columns = statistics.columns.str.replace('+', '_plus')
statistics.columns = statistics.columns.str.lower()
statistics.columns
print(statistics.head())

statistics['fgm_0_19'] = statistics['fgm_0_19'].fillna(statistics['fg_0_19'])
statistics['fgm_20_29'] = statistics['fgm_20_29'].fillna(statistics['fg_20_29'])
statistics['fgm_30_39'] = statistics['fgm_30_39'].fillna(statistics['fg_30_39'])

statistics.drop(['fg_0_19','fg_20_29','fg_30_39'], axis=1, inplace=True)

# Look at missing values by column
import missingno as msno
msno.bar(statistics)
# msno.heatmap(statistics)

# Make a histogram to check
import seaborn as sns
sns.histplot(data=statistics, x='total_points', bins=20)
# lots of zeros, makes sense. some negatives.
columns = ['week_id', 'player_id', 'total_points', 'pass_yds', 'pass_td', 'int',
       'rush_yds', 'rush_td', 'rec', 'rec_yds', 'rec_td', 'ret_yds', 'ret_td',
       '2_pt', 'fum_lost', 'fum_ret_td', 'fg_40_49',
       'fg_50_plus', 'fgm_0_19', 'fgm_20_29', 'fgm_30_39', 'pat_made',
       'pat_miss', 'pts_allow', 'sack', 'fum_rec', 'td', 'safe', 'blk_kick',
       'kick_and_punt_ret_td', 'pts_allow_0', 'pts_allow_1_6',
       'pts_allow_7_13', 'pts_allow_14_20', 'pts_allow_21_27',
       'pts_allow_28_34', 'pts_allow_35_plus', 'week', 'year', 'rush_att',
       'targets', '4_dwn_stops', 'xpr', 'rec_1st_downs', 'rush_1st_downs',
       'fg_yds', 'fg_made', 'fg_miss']

statistics = statistics[columns]
statistics.index.name = 'stat_id'
statistics.to_sql('statistics', engine)

# Drafts
drafts = pd.read_csv("data/drafts.csv", index_col=0)
# Add manager ID
drafts = drafts.merge(teams[['team_key','manager_id']], on="team_key")
drafts = drafts.drop('team_key', axis=1)
drafts.to_sql("drafts", engine)
