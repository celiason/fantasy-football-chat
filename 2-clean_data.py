import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Load data
matchups = pd.read_csv("data/matchups.csv", index_col=0)

#------------------------------------------------------------------------
# Drafts
#------------------------------------------------------------------------

drafts = pd.read_csv("data/drafts.csv", index_col=0)
drafts.reset_index(inplace=True)
drafts.drop(['index'], axis=1, inplace=True)
drafts['source'] = 'freeagents'

#------------------------------------------------------------------------
# Transactions
#------------------------------------------------------------------------

transactions = pd.read_csv("data/transactions.csv", index_col=0)

# Convert from ms timestamp to dates
transactions['timestamp'] = transactions['timestamp'].transform(datetime.fromtimestamp)

# Sorting by date for later
transactions.sort_values('timestamp', inplace=True)
transactions.reset_index(inplace=True)
print(transactions.head())

#------------------------------------------------------------------------
# Teams
#------------------------------------------------------------------------

teams = pd.read_csv("data/teams.csv", index_col=0)

#------------------------------------------------------------------------
# Rosters
#------------------------------------------------------------------------

rosters = pd.read_csv("data/rosters.csv", index_col=0)
rosters = rosters.sort_values(['year','week'])
rosters.reset_index(inplace=True)
rosters.drop('index', axis=1, inplace=True)

# NOTE: there was a problem here. IR players aren't active.
rosters['status'] = np.where(rosters['selected_position'].isin(['BN','IR']), 'inactive', 'active')
rosters['status'].value_counts()
rosters['selected_position'].value_counts()

#------------------------------------------------------------------------
# Weeks
#------------------------------------------------------------------------

weeks = pd.read_csv("data/weeks.csv")
weeks['start'] = pd.to_datetime(weeks['start'])
weeks['end'] = pd.to_datetime(weeks['end']) + pd.Timedelta('11:59:59')
# I realized this was critical because sometimes people add/drop right before the end of the week

# See how many unique players we've used over the years
len(set(rosters['player_id']))  # 1090

#------------------------------------------------------------------------
# Setup the moves table
#------------------------------------------------------------------------

rosters = rosters.merge(weeks[['year','week','end']], on=['year','week'])
rosters.rename({'end': 'timestamp', 'status': 'destination'}, axis=1, inplace=True)
rosters['timestamp'] = rosters['timestamp'] - pd.Timedelta(seconds = 1)

# print(moves[(moves['year']==2023) & (moves['player_id']==100024)])

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


# I can sell this as...
# I'm essentially modeling the life cycle of a player. this can be applied to customer
# life cycles as well. signs up, purchaes stuff, leaves, maybe comes back, done for good.
# model with survival models for Bio?? LTV (lifetime value) I think is the model Rafael talked
# about when he was at Instacart

#------------------------------------------------------------------------
# Working with the drafts table
#------------------------------------------------------------------------

# Get first transaction times by year
first_trans_time = transactions.groupby('year')['timestamp'].min()

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
rosters['trans_type'] = rosters['destination']
rosters['timestamp'] = pd.to_datetime(rosters['timestamp'])
rosters['source'] = rosters['team_key']
rosters['destination'] = rosters['team_key']

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

# Convert to integer
transactions['week'] = transactions['week'].astype(int)

# Now concatenate into a big transitions dataframe
# We need 3 tables: moves, transactions, and drafts
drafts['selected_position'] = 'BN'
transactions['selected_position'] = np.where(transactions["trans_type"].isin(['add','trade']), 'BN', '')

# Concatenate rosters, transactions, and drafts
events = pd.concat([rosters,transactions,drafts])
events.sort_values('timestamp', inplace=True)
column_order = ['year','week','timestamp','player_id','source','destination','trans_type','selected_position']
events = events[column_order]
events.reset_index(inplace=True, drop=True)

print(events.head())

# Counts of position selections
events['selected_position'].value_counts()

# Check for NAs
events.isna().sum() # cool, no NAs!

# There was a problem with this player.
# figured out what was happening- in week 10 he was dropped before going active
# apparently in that year we could drop someone right after they played..?
rosters[(rosters['player_id']==6781) & (rosters['year']==2007)]
events[(events['player_id']==6781) & (events['year']==2007)]
transactions[(transactions['player_id']==6781) & (transactions['year']==2007)]


# last event can't be drop by a team before making that player active
# here is a function that adjusts the active time to a second before the drop time
# if a drop occurs right before a player going active
# NOTE: I'm currently setting the active time as the last minute of the last day of the NFL week
# I could also probably just link up the actual game time for a pid
# nfl database maybe?
def adjust_active_time(df):
    # Reset the index
    df = df.reset_index()
     # Iterate through the rows of the DataFrame
    for i in range(1, len(df)):  # Start from 1 to safely access i-1
        if df.loc[i, 'trans_type'] == 'active' and df.loc[i-1, 'trans_type'] == 'drop':
            # Adjust the timestamp for 'active'
            df.loc[i, 'timestamp'] = df.loc[i-1, 'timestamp'] - pd.Timedelta(seconds=1)
    return df

events = events.groupby(['year','player_id','week']).apply(adjust_active_time)
events = events.sort_values('timestamp')
events = events.drop(['index','year','week','player_id'], axis=1)
events = events.reset_index()

# This function will create a path for a player (e.g., bench -> active -> bench)
# Function that defines paths a player takes in a given year
def get_path(df):
    df.sort_values('timestamp', inplace=True)
    breakpoints = df['selected_position'].ne(df['selected_position'].shift(1)).cumsum()
    breakpoint_indices = breakpoints[breakpoints.diff().ne(0)].index.tolist()
    return df.loc[breakpoint_indices]

# BUG: fix this
# df = rosters[(rosters['year']==2023) & (rosters['player_id']==100024)]
# breakpoints = df['selected_position'].ne(df['selected_position'].shift(1)).cumsum()
# breakpoint_indices = breakpoints[breakpoints.diff().ne(0)].index.tolist()
# print(df)
# print(df.loc[breakpoint_indices])

# rosters/slots? - need manager ID, player ID, week ID, selected position

# Apply the function
moves = events.groupby(['year','player_id']).apply(get_path)
moves.drop(['player_id','year'], axis=1, inplace=True)
moves.reset_index(inplace=True)
moves

len(events) # 42321
len(moves) # 19854

# NOTE: how can i see about database efficiency for transaction stream vs static weekly roster tables?

#------------------------------------------------------------------------
# Compare sizes
#------------------------------------------------------------------------

len(rosters) # 32907
len(moves) # 19854 (about 1/3 the size of the original rosters table)
len(drafts) # 1985
len(transactions) # 7429

# Unique players table
players = rosters.groupby(['player_id','name']).head(1)
players = players[['player_id','name','position_type','eligible_positions']].sort_values('player_id')


# TODO get player stats from Yahoo (this will take a while)
# NOTE: need to loop by week, year, for each player_id


# TODO might need to see about having a category for BN-Active and Active moves to diff position


#------------------------------------------------------------------------
# Visualize the data
#------------------------------------------------------------------------

import seaborn as sns
plotdf = events.groupby('year')['trans_type'].value_counts().reset_index(name='count')
plotdf

sns.lineplot(data=plotdf, x='year', y='count', hue='trans_type')


print(events.head())
print(weeks.head())
print(players.head())
print(teams.head())




transactions[(transactions['year']==2023) & (transactions['player_id'] == 100024)]
rosters[(rosters['year']==2023) & (rosters['player_id'] == 100024)]
moves[(moves['year']==2023) & (moves['player_id'] == 100024)]

# week 9 active
# week 10 dropped
# week 13 picked back up (by same team.. I think that's where the prob is)
# week 13 started

get_path(rosters[(rosters['year']==2023) & (rosters['player_id'] == 100024)])



# Output to file
moves.to_csv("data/events.csv", index=False)

