import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Load data
matchups = pd.read_csv("data/matchups.csv", index_col=0)

drafts = pd.read_csv("data/drafts.csv", index_col=0)
drafts.reset_index(inplace=True)
drafts.drop(['index'], axis=1, inplace=True)
drafts['source'] = 'freeagents'

transactions = pd.read_csv("data/transactions.csv", index_col=0)

# Convert from ms timestamp to dates
transactions['timestamp'] = transactions['timestamp'].transform(datetime.fromtimestamp)

# Sorting by date for later
transactions.sort_values('timestamp', inplace=True)
transactions.reset_index(inplace=True)

teams = pd.read_csv("data/teams.csv", index_col=0)

rosters = pd.read_csv("data/rosters.csv", index_col=0)
rosters = rosters.sort_values(['year','week'])
rosters.reset_index(inplace=True)
rosters.drop('index', axis=1, inplace=True)
rosters['status'] = np.where(rosters['selected_position']=='BN', 'bench', 'active')

weeks = pd.read_csv("data/weeks.csv")
weeks['start'] = pd.to_datetime(weeks['start'])
weeks['end'] = pd.to_datetime(weeks['end']) + pd.Timedelta('11:59:59')
# I realized this was critical because sometimes people add/drop right before the end of the week

# See how many unique players we've used over the years
len(set(rosters['player_id']))  # 1090

# This function will create a path for a player (e.g., bench -> active -> bench)

# df = rosters[(rosters['year']==2007) & (rosters['player_id'] == 5109)]

# Function that defines paths a player takes in a given year
def get_path(df):
    df.sort_values('week', inplace=True)
    breakpoints = df['selected_position'].ne(df['selected_position'].shift(1)).cumsum()
    breakpoint_indices = breakpoints[breakpoints.diff().ne(0)].index.tolist()
    return df.loc[breakpoint_indices]

# Setup the moves table
moves = rosters.groupby(['year','player_id']).apply(get_path)
moves.drop(['player_id','year'], axis=1, inplace=True)
moves.reset_index(inplace=True)
moves['source'] = moves.groupby(['year','player_id'])['status'].shift(1)

# cool codes!
# BUG 
# FIXME
# NOTE
# TODO

# Last thing to figure out is the moves from BN to starting positions
# NOTE: can maybe check if any players active on week and added in previous week,
# then update the timestamp for move from BN-active as the transaction timestamp
# OR another way to think about it- if there is someone that was added Sunday night say, 
# for a MNF game, then I would need to have a small function that checks for that case and 
# adjusts the move to active right after the FA - team transition event.
# I think i figured this out- I just set the active/bench assignment to a minute
# before the end of a week.

# Adding pseudo-timestamp for moves (using the LAST day of the NFL week)
moves = moves.merge(weeks[['year','week','end']], on=['year','week'])
moves.rename({'end': 'timestamp', 'status': 'destination'}, axis=1, inplace=True)
moves['timestamp'] = moves['timestamp'] - pd.Timedelta(seconds = 1)

# Look at derrick henry for fun
print(moves[(moves['player_id']==29279) & (moves['year'] == 2023)].head())
print(moves[(moves['player_id']==549) & (moves['year'] == 2007)].head())

print(moves[(moves['player_id']==100010) & (moves['year'] == 2007)].head())



print(transactions[['timestamp','player_id','source','destination','trans_type']].head(10))


# I can sell this as...
# I'm essentially modeling the life cycle of a player. this can be applied to customer
# life cycles as well. signs up, purchaes stuff, leaves, maybe comes back, done for good.
# model with survival models for Bio?? LTV (lifetime value) I think is the model Rafael talked
# about when he was at Instacart

# Working with the drafts table

# Get first transaction times by year
first_trans_time = transactions.groupby('year')['timestamp'].min()
first_trans_time.keys()
first_trans_time.values

# TODO: make sure all drafted players should have transition from FA -> BN

drafts['timestamp'] = pd.NaT
idx = drafts.groupby('year')['pick'].idxmax()
drafts.loc[idx, 'timestamp'] = first_trans_time[drafts.loc[idx, 'year']].values

# Num picks per year as a timedelta object in minutes
draft_lengths = drafts.groupby('year').size() - 1
draft_lengths = draft_lengths.astype('timedelta64[m]')

# First pick
first_pick_times = first_trans_time - draft_lengths
idx = drafts.groupby('year')['pick'].idxmin()
drafts.loc[idx, 'timestamp'] = pd.to_datetime(first_pick_times[drafts.loc[idx, 'year']].values)

drafts['timestamp'] = pd.to_datetime(drafts['timestamp'])

# Assuming each round takes 1 minute, we'll create pseudo-timestamps for drafted players
# using the interpolate function of pandas
drafts['timestamp'] = drafts['timestamp'].interpolate()

# Some more renaming
drafts.rename({'team_key': 'destination'}, inplace=True, axis=1)
drafts['trans_type'] = 'draft'
drafts['week'] = int(0)

# check that The interpolation worked (should be ramps up for each year)
drafts.plot(x='timestamp', y='pick')

# Some renaming before we concatenate
moves['trans_type'] = moves['destination']
moves['timestamp'] = pd.to_datetime(moves['timestamp'])
moves['source'] = moves['team_key']
moves['destination'] = moves['team_key']

# Function to check if a date is in any range
def get_week(date, df_ranges):
    for _, row in df_ranges.iterrows():
        if row['start'] <= date <= row['end']:
            return int(row['week'])  # Use '%U' for week number

# Get weeks from weeks table
transactions['week'] = transactions['timestamp'].apply(lambda x: get_week(x, df_ranges=weeks))

# Fix weeks that are NA (should be = 0, meaning "preseason")
idx = transactions['week'].isna()
transactions.loc[idx, 'week'] = 0
transactions['week'] = transactions['week'].astype(int)

# Now concatenate into a big transitions dataframe
# We need 3 tables: moves, transactions, and drafts
drafts['selected_position'] = 'BN'
transactions['selected_position'] = np.where(transactions["trans_type"].isin(['add','trade']), 'BN', '')
# moves['selected_position']

events = pd.concat([moves,transactions,drafts])
events.sort_values('timestamp', inplace=True)
column_order = ['year','week','timestamp','player_id','source','destination','trans_type','selected_position']
events = events[column_order]
events.reset_index(inplace=True)
events.drop('index', axis=1, inplace=True)

print(events.head())
events['selected_position'].value_counts()

# Check for NAs
events.isna().sum() # cool, no NAs!

# NOTE: how can i see about database efficiency for transaction stream vs static weekly roster tables?

# Compare sizes
len(rosters) # 32907
len(moves) # 10356 (about 1/3 the size of the original rosters table)
len(drafts) # 1985
len(transactions) # 7429
len(events) # 20744

len(events) / (len(rosters) + len(moves) + len(drafts) + len(transactions))
# 38.7% of the original amount of data, so that's cool


# Unique players table
rosters = pd.read_csv("data/rosters.csv")
players = rosters.groupby(['player_id','name']).head(1)
players = players[['player_id','name','position_type','eligible_positions']].sort_values('player_id')

# COOL!!! working.
# events = events.merge(players, left_on='player_id', right_on='player_id')

# Derrick henry time!
# events[events['player_id']==29279]


# When is first draft pick each year?
# events.loc[events.groupby('year')['timestamp'].idxmin()]


# TODO get player stats from Yahoo (this will take a while)
# NOTE: need to loop by week, year, for each player_id



# TODO output as a database use SQLalchemy(?)


# Sample query, what was my starting roster in week 1 of 2007?
my_query = "name == 'The Owls' and year == 2007"
my_team_key = teams.query(my_query)['team_key'].values[0]

my_query = f"week == 1 and trans_type == 'active' and year == 2007 and destination == '{my_team_key}'"

my_query = f"week == 2 and year == 2007 and destination == '{my_team_key}'"
events.query(my_query)

testing = events[(events['source'] == my_team_key) | (events['destination'] == my_team_key)].sort_values([])
testing
testing.to_csv("data/testing5.csv")

events[events['trans_type']=='active']


# tmp = rosters.merge(teams, on='team_key', suffixes=["", "_teams"])

# tmp[(tmp["status"]=="active") & (tmp["week"]==1) & (tmp["year"]==2007) & (tmp["nickname"]=="Chad")]
# tmp[(tmp["status"]=="active") & (tmp["week"]==2) & (tmp["year"]==2007) & (tmp["nickname"]=="Chad")]

# TODO might need to see about having a category for BN-Active and Active moves to diff position

events.to_csv("data/events.csv", index=False)


# Visualize the data!

import seaborn as sns
plotdf = events.groupby('year')['trans_type'].value_counts().reset_index(name='count')
plotdf

sns.lineplot(data=plotdf, x='year', y='count', hue='trans_type')


print(events.head())
print(weeks.head())
print(players.head())
print(teams.head())

