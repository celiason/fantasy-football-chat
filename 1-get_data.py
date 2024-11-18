# Get data from yahoo

from time import sleep
import seaborn as sns
import matplotlib.pyplot as plt
from yahoo_oauth import OAuth2
import yahoo_fantasy_api as yfa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint # just discovered this! awesome way to visualize nested lists
from collections import ChainMap
from tqdm import tqdm

# Load my functions
from src.utils import get_league, pull_trade_details, pull_managers, get_transactions, get_rosters

# pprint(lg.matchups())

# Pull all Slow Learners league years
# TODO: need to ask Bo for login to get years I wasn't there (2009-2011)
all_transactions = pd.DataFrame()
for year in tqdm(range(2007, 2024), desc="Processing Years"):
    # Get league
    lg = get_league(year=year)
    # Check if empty league
    if lg is None:
        continue
    # Transactions dataset
    transactions = get_transactions(lg)
    transactions = pd.DataFrame(transactions)
    transactions['year'] = year
    all_transactions = pd.concat([all_transactions, transactions])

len(all_transactions) # 7429
# print(all_transactions.head())
# all_transactions['trans_type'].value_counts()

# Draft dataset
all_drafts = pd.DataFrame()
for year in tqdm(range(2007, 2024), desc="Processing Years"):
    draft = pd.DataFrame(lg.draft_results())
    draft['year'] = year
    all_drafts = pd.concat([all_drafts, draft])

# Teams dataset
all_teams = pd.DataFrame()
for year in tqdm(range(2007, 2024), desc="Processing Years"):
    teams = pull_managers(lg.teams())
    teams = pd.DataFrame(teams)
    teams['year'] = year
    all_teams = pd.concat([all_teams, teams])

# Save
all_teams.to_csv("data/teams.csv")
all_transactions.to_csv("data/transactions.csv")
all_drafts.to_csv("data/drafts.csv")

# A player starts in a pool
# can get drafted
# can go to bench or starting lineup
# if on starting lineup for a week, points count
# can also go to waviers, FA
# or get traded
# so really all we need to do is track movement of players

all_teams['nickname'].value_counts()

len(all_transactions) # total of 4242 transactions :)

# Plot trades over time
all_transactions[all_transactions['trans_type']=="trade"].groupby(['year']).size().plot()

# Get matchups
all_matchups = pd.DataFrame()
for year in tqdm(range(2007, 2024), desc="Processing matchups"):
    try:
        df = get_matchups(year=year)
        all_matchups = pd.concat([all_matchups, df])
    except:
        continue

# list(product(range(2007,2024), [2,3,4]))

all_matchups.to_csv("data/matchups.csv")

# Get player stats
# player_id = 40068
# lg.player_stats([player_id], req_type="season")
# lg.player_stats([player_id], req_type="week", week=1)


# Loop through and get all rosters for all years
# NB: I got data for - 2007, 2008, 2014-2017
# getting booted due to too many calls to the API I assume.
# not sure what the lockout time is.
# Redid for subset of years and merged dataframe
all_rosters = pd.DataFrame()
for year in range(2007, 2024):
    try:
        print(f"Processing data for the year {year}")
        rost = get_rosters(year=year)
        all_rosters = pd.concat([all_rosters, rost])
        all_rosters['year'] = year
    except:
        print(f"No league data for the year {year}")
        continue
    sleep(300)  # sleep for 5 minutes


# Check that what I've done is working like I think (teams should have ~14 players each week)
# df = all_rosters[all_rosters['year'] == 2023].groupby(['week', 'team_key']).size().unstack()
# sns.heatmap(df, vmin=1, vmax=16)
# yep looks ok

# Set the desired order of columns
column_order = ['year', 'week', 'playoffs', 'team_key' ,'player_id', 'name', 'position_type', 'eligible_positions', 'selected_position']

# Output to CSV file
all_rosters[column_order].to_csv("data/rosters.csv")

# Get week start and end points
weeks = []
for year in tqdm(range(2007, 2024), desc="Processing Years"):
    try:
        lg = get_league(year)
    except:
        continue
    for week in range(1, 18):
        try:
            dates = lg.week_date_range(week)
            row = {'year': year, 'week': week, 'start': dates[0], 'end': dates[1]}
            weeks.append(row)
        except:
            continue

# Convert to dataframe
df = pd.DataFrame(weeks)

# Save dataframe
df.to_csv("data/weeks.csv", index=False)


# Get player statistics (this will take a long time)

# Load rosters
rosters = pd.read_csv("data/rosters.csv")

# Create empty list for stats
all_stats = []

# Loop over years
for year in range(2007, 2024):
    try:
        print(f"Processing year {year}...")
        lg = get_league(year)
        # Get player IDs for that year
        player_ids = list(set(rosters[rosters["year"]==year]["player_id"]))
    except:
        continue
    # Loop over weeks
    for week in tqdm(range(1, 18), desc="Processing weeks"):
        try:
            # Get stats for each week
            stats = lg.player_stats(player_ids=player_ids, week=week, req_type="week")
            # Convert to data frame and added week, year details
            stats = pd.DataFrame(stats)
            stats['week'] = week
            stats['year'] = year
            # Append to main stats list
            all_stats.append(stats)
        except:
            continue

# Now concat all together and save as CSV
statistics = pd.concat(all_stats)
statistics['total_points'] = statistics['total_points'].astype(float)
statistics.to_csv("data/statistics.csv", index=False)

statistics[statistics['player_id']==28398].plot(x='year', y='total_points', kind='scatter')


# NOTE: it seems that yahoo only started getting position-level stats in 2014. Might need to backfill with NFL data

# TODO make and "update_db" kinda function that will check for changes and pull new data from Yahoo..
# that way we can stay current and look at the year we're in.

