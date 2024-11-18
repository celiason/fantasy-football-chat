################################################################################################
# Push to database
################################################################################################

from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2

# Initialize the database
engine = create_engine('postgresql+psycopg2://chad:password@localhost:5432/football', echo=True)

# Players table
rosters = pd.read_csv("data/rosters.csv")
players = rosters.groupby(['player_id','name']).head(1)
players = players[['player_id','name','position_type','eligible_positions']].sort_values('player_id')
players.set_index('player_id', inplace=True)
players.to_sql('players', engine)

# Events table
events = pd.read_csv("data/events.csv")
events.drop('name', axis=1, inplace=True)
events.index.name = 'event_id'
print(events.head())

teams

events.to_sql("events", engine)

# Weeks table
weeks = pd.read_csv("data/weeks.csv")
weeks['start'] = pd.to_datetime(weeks['start'])
weeks['end'] = pd.to_datetime(weeks['end']) + pd.Timedelta('11:59:59')
weeks.index.name = 'week_id'
weeks.to_sql("weeks", engine)

# Managers table
managers = pd.DataFrame(list(set(teams["nickname"])), columns=['nickname'])
managers.sort_values('nickname', inplace=True)
managers.reset_index(inplace=True)
managers.rename(columns={'index': 'manager_id'}, inplace=True)

# Teams tables
teams = pd.read_csv("data/teams.csv", index_col=0)
teams = teams.merge(managers, on="nickname")
teams.reset_index(inplace=True)
teams.drop('index', axis=1, inplace=True)
teams.index.name = 'team_id'
teams.to_sql("teams", engine)

managers.set_index('manager_id', inplace=True)
managers.to_sql("managers", engine)

# Matchups table
matchups['team_key_winner'] = np.where(matchups['points1'] > matchups['points2'], matchups['team_key1'], matchups['team_key2'])
matchups['team_key_loser'] = np.where(matchups['points1'] < matchups['points2'], matchups['team_key1'], matchups['team_key2'])
matchups['points_winner'] = np.where(matchups['points1'] > matchups['points2'], matchups['points1'], matchups['points2'])
matchups['points_loser'] = np.where(matchups['points1'] < matchups['points2'], matchups['points1'], matchups['points2'])
matchups.reset_index(inplace=True)
matchups.index.name = 'matchup_id'
columns = ['year','week','team_key_winner','team_key_loser','points_winner','points_loser']
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
