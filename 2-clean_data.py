import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Load data
matchups = pd.read_csv("data/matchups.csv", index_col=0)

#------------------------------------------------------------------------
# Drafts
#------------------------------------------------------------------------

drafts = pd.read_csv("data/drafts.csv")
drafts.reset_index(inplace=True)
drafts.drop(['index'], axis=1, inplace=True)
drafts['source'] = 'freeagents'

#------------------------------------------------------------------------
# Transactions
#------------------------------------------------------------------------

transactions = pd.read_csv("data/transactions.csv")

# Convert from ms timestamp to dates
transactions['timestamp'] = transactions['timestamp'].transform(datetime.fromtimestamp)

# Sorting by date for later
transactions.sort_values('timestamp', inplace=True)
transactions.reset_index(inplace=True)

#------------------------------------------------------------------------
# Teams
#------------------------------------------------------------------------

teams = pd.read_csv("data/teams.csv")

#------------------------------------------------------------------------
# Rosters
#------------------------------------------------------------------------

rosters = pd.read_csv("data/rosters.csv")
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
weeks['end'] = pd.to_datetime(weeks['end']) + pd.Timedelta('23:59:59')
# I realized this was critical because sometimes people add/drop right before the end of the week
# Found a problem!! I had been using 11:59:59 as a minute before midnight..

# See how many unique players we've used over the years
len(set(rosters['player_id']))  # 1233

#------------------------------------------------------------------------
# Setup the moves table
#------------------------------------------------------------------------

rosters = rosters.merge(weeks[['year','week','end']], on=['year','week'])
rosters.rename({'end': 'timestamp', 'status': 'destination'}, axis=1, inplace=True)
rosters['timestamp'] = rosters['timestamp'] - pd.Timedelta(seconds = 1)

# cool comment codes!
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

# Initiate empty column (we'll convert to datetime later after interpolating)
drafts['timestamp'] = np.nan

# Get first transaction times by year (use as last pick time of the draft)
last_pick_times = transactions.groupby('year')['timestamp'].min()
idx_last_pick = drafts.groupby('year')['pick'].idxmax()
drafts.loc[idx_last_pick, 'timestamp'] = pd.to_numeric(last_pick_times[drafts.loc[idx_last_pick, 'year']].values)

# Assuming each round takes 1 minute, we'll create pseudo-timestamps for drafted players

# Num picks per year as a timedelta object in minutes
draft_lengths = drafts.groupby('year').size() - 1
draft_lengths = draft_lengths.astype('timedelta64[m]')

# First pick times
first_pick_times = last_pick_times - draft_lengths
idx_first_pick = drafts.groupby('year')['pick'].idxmin()
drafts.loc[idx_first_pick, 'timestamp'] = pd.to_numeric(first_pick_times[drafts.loc[idx_first_pick, 'year']].values)

# using the interpolate function of pandas
# NOTE: I had to make sure this WASNT a datetime class before lienar interpolation
drafts['timestamp'] = drafts['timestamp'].interpolate(method='linear')
drafts['timestamp'] = pd.to_datetime(drafts['timestamp'])

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
# NOTE: i know this function can be optimized, but it works for now.
def get_week(date, df_ranges):
    for _, row in df_ranges.iterrows():
        if row['start'] <= date <= row['end']:
            return int(row['week'])

# Get weeks from weeks table
transactions['week'] = transactions['timestamp'].apply(lambda x: get_week(x, df_ranges=weeks))

# Fix weeks that are NA (should be = 0, meaning "preseason")
idx = transactions['week'].isna()
transactions.loc[idx, 'week'] = 0

# Convert to integer
transactions['week'] = transactions['week'].astype(int)

# Now concatenate into a big transitions dataframe
# We need 3 tables: moves, transactions, and drafts

# Prep drafts, transactions for concatenating with rosters
drafts['selected_position'] = 'BN'
transactions['selected_position'] = np.where(transactions["trans_type"].isin(['add','trade']), 'BN', '')

# Concatenate rosters, transactions, and drafts
events = pd.concat([rosters,transactions,drafts])
events.sort_values('timestamp', inplace=True)
column_order = ['year','week','timestamp','player_id','source','destination','trans_type','selected_position']
events = events[column_order]
events.reset_index(inplace=True, drop=True)

# Counts of position selections
events['selected_position'].value_counts()

# Check for NAs
events.isna().sum() # cool, no NAs!

# last event can't be drop by a team before making that player active
# here is a function that adjusts the active time to a second before the drop time
# if a drop occurs right before a player going active
# NOTE: I'm currently setting the active time as the last minute of the last day of the NFL week
# I could also probably just link up the actual game time for a pid
# nfl database maybe?

# Tack on week start, end for function below
events = events.merge(weeks, on=['year','week'], how='left')

def adjust_active_time(df):
    # Reset the index
    df = df.reset_index()
     # Iterate through the rows of the DataFrame
    for i in range(0, len(df)-1):  # Start from 0 to safely access i+1
    # i=0
        if df.loc[i, 'trans_type'] == 'drop' and df.loc[i+1, 'trans_type'] == 'active':
            # Adjust the timestamp for the drop to following week at MIDNIGHT
            df.loc[i, 'timestamp'] = df.loc[i, 'end'] + pd.Timedelta(seconds=1)
            df.loc[i, 'week'] = df.loc[i, 'week'] + 1
    return df

# Adjust times by year, player, and week
events = events.groupby(['year','player_id','week']).apply(adjust_active_time)

# Re-sort
events = events.sort_values('timestamp')
events = events.rename({'week': 'week_adj'}, axis=1)

# Clean up
events = events.drop(['index','year','player_id'], axis=1)
events = events.reset_index()
events = events.drop(['week'], axis=1)

# Now rename back
events = events.rename({'week_adj': 'week'}, axis=1)

# Check that there aren't any NAs
any(events['week'].isna())

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

len(events) # 64615
len(moves) # 29199

# NOTE: how can i see about database efficiency for transaction stream vs static weekly roster tables?

#------------------------------------------------------------------------
# Compare sizes
#------------------------------------------------------------------------

len(rosters) # 51013
len(moves) # 29199 (about 1/3 the size of the original rosters table)
len(drafts) # 3112
len(transactions) # 10490

# Unique players table
players = rosters.groupby(['player_id','name']).head(1)
players = players[['player_id','name','position_type','eligible_positions']].sort_values('player_id')

#------------------------------------------------------------------------
# Visualize the data
#------------------------------------------------------------------------

import seaborn as sns

plotdf = moves.groupby('year')['trans_type'].value_counts().reset_index(name='count')
plotdf


fig = sns.lineplot(data=plotdf, x='year', y='count', hue='trans_type')
plt.savefig('figures/transaction_type_yearly_count_combined.png')

#------------------------------------------------------------------------
# Output to file
#------------------------------------------------------------------------

moves.to_csv("data/events.csv", index=False)
