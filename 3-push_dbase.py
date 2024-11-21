#------------------------------------------------------------------------
# Push to database
#------------------------------------------------------------------------

from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
import numpy as np

#------------------------------------------------------------------------
# Initialize the database
#------------------------------------------------------------------------
engine = create_engine('postgresql+psycopg2://chad:password@localhost:5432/football', echo=True)

#------------------------------------------------------------------------
# We will use this below
#------------------------------------------------------------------------
rosters = pd.read_csv("data/rosters.csv")

#------------------------------------------------------------------------
# Players table
#------------------------------------------------------------------------
players = rosters.groupby(['player_id','name']).head(1)
players['position'] = players['eligible_positions'].str.extract('(\w+)', expand=False)
players = players[['player_id','name','position_type','position']].sort_values('player_id')
players.set_index('player_id', inplace=True)
players.to_sql('players', engine)

#------------------------------------------------------------------------
# Weeks table
#------------------------------------------------------------------------
weeks = pd.read_csv("data/weeks.csv")
weeks['start'] = pd.to_datetime(weeks['start'])
weeks['end'] = pd.to_datetime(weeks['end']) + pd.Timedelta('11:59:59')
# Just unique entries to add 'playoffs' to weeks
rost_nondup = rosters[['year','week','playoffs']].drop_duplicates()
# Merge with rost_nondup
weeks = weeks.merge(rost_nondup, on=['year','week'], how='left')
# Remove NAs (we only started doing week 17 recently)
weeks = weeks[weeks['playoffs'].notna()]
weeks.index.name = 'week_id'
weeks.to_sql("weeks", engine)

#------------------------------------------------------------------------
# Teams tables
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

managers = pd.DataFrame(list(set(teams["nickname"])), columns=['nickname'])
managers.sort_values('nickname', inplace=True)
managers.reset_index(inplace=True)
managers = managers.drop('index', axis=1)
managers.index.name = 'manager_id'
managers_sql = managers.rename(columns={'index': 'manager_id', 'nickname': 'name'})
managers_sql.to_sql("managers", engine)

# Now we'll add manager ID to teams
teams = teams.merge(managers.reset_index(), on="nickname")
teams.index.name = 'team_id'
teams_sql = teams[['name','number_of_moves','division_id','draft_grade','year','manager_id']]
teams_sql.to_sql("teams", engine)

#------------------------------------------------------------------------
# Rosters table
#------------------------------------------------------------------------

rosters = rosters.merge(teams, on='team_key', suffixes=['', '_teams'])
rosters = rosters[['year','week','playoffs','manager_id','player_id','selected_position']]
rosters.to_sql('rosters', engine)

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


columns = ['year','week','playoffs','manager_id1','manager_id2','points1','points2','winner_manager_id']

# Sanity check again
any(matchups['manager_id1'] == matchups['manager_id2'])

# Push to database
matchups[columns].to_sql("games", engine)

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
import missingno as msno
msno.bar(statistics)
# msno.heatmap(statistics)

# Make a histogram to check
import seaborn as sns
sns.histplot(data=statistics, x='total_points', bins=20)
# lots of zeros, makes sense. some negatives.

# Final columns in the table
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
statistics.to_sql('stats', engine)

#------------------------------------------------------------------------
# Drafts
#------------------------------------------------------------------------

drafts = pd.read_csv("data/drafts.csv", index_col=0)

# Add manager ID
drafts = drafts.merge(teams[['team_key','manager_id']], on="team_key")
drafts = drafts.drop('team_key', axis=1)
drafts.to_sql("drafts", engine)

#------------------------------------------------------------------------
# Events table
#------------------------------------------------------------------------

events = pd.read_csv("data/events.csv")

teams_lookup = teams.set_index('team_key')['manager_id'].to_dict()

events['source_mid'] = events['source'].replace(teams_lookup)
events['destination_mid'] = events['destination'].replace(teams_lookup)

# Rename and pick columns
events.rename({'trans_type':'type'}, axis=1, inplace=True)
events.index.name = 'event_id'

events['source'] = np.where(events['source_mid'].isin(['freeagents','waivers']), events['source_mid'], 'team')
events['destination'] = np.where(events['destination_mid'].isin(['freeagents','waivers']), events['destination_mid'], 'team')

events['source'].value_counts()
events['destination'].value_counts()

events['source_mid'] = events['source_mid'].apply(lambda x: np.nan if isinstance(x, str) else x)
events['destination_mid'] = events['destination_mid'].apply(lambda x: np.nan if isinstance(x, str) else x)

events['source_mid'] = events['source_mid'].astype('Int64')
events['destination_mid'] = events['destination_mid'].astype('Int64')

events['source_mid'].value_counts()
events['destination_mid'].value_counts()

events.groupby('year')['type'].value_counts()

events_sql = events[['year','week','timestamp','player_id','source_mid','destination_mid','source','destination','type','selected_position']]
events_sql = events_sql.reset_index(drop=True)
events_sql.index.name = 'tid'

print(events_sql.head())

events_sql.to_sql("events", engine)

rosters[(rosters['year']==2023) & (rosters['week']==1) & (rosters['manager_id']==5)]

events[(events['year']==2023) & (events['player_id']==100030)]



transactions = pd.read_csv("data/transactions.csv")
transactions.to_sql("trans_test", engine)

