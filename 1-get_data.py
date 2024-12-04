# Get data from yahoo

# TODO - fix problem with trades from 2009-2014 (NULL source/dest teams)

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
import streamlit as st

# Load my functions
from src.utils import get_league, pull_trade_details, pull_managers, get_transactions, get_rosters, get_matchups

# Load credentials
# YAHOO_KEY = st.secrets["yahoo_key"]
# YAHOO_SECRET = st.secrets["yahoo_secret"]

YAHOO_KEY = st.secrets["bo_client_id"]
YAHOO_SECRET = st.secrets["bo_client_secret"]

sc = OAuth2(YAHOO_KEY, YAHOO_SECRET)
# sc = OAuth2(key, secret)
gm = yfa.Game(sc, 'nfl')

# TODO make and "update_db" kinda function that will check for changes and pull new data from Yahoo..
# that way we can stay current and look at the year we're in.

# Full set of years
years = range(2007, 2024)

all_settings = pd.DataFrame()
for year in tqdm(years, desc="Processing Years"):
    try:
        lg = get_league(year=year, gm=gm)
    except:
        continue
    settings = lg.settings()
    settings_df = pd.json_normalize(settings, sep='_')
    settings_df['year'] = year
    all_settings = pd.concat([all_settings, settings_df])
all_settings.to_csv("data/settings.csv", index=False)

# Pull all Slow Learners league years
all_transactions = pd.DataFrame()
for year in tqdm(years, desc="Processing Years"):
    # Get league
    lg = get_league(year=year, gm=gm)
    # Check if empty league
    if lg is None:
        continue
    # Transactions dataset
    transactions = get_transactions(lg)
    transactions = pd.DataFrame(transactions)
    transactions['year'] = year
    all_transactions = pd.concat([all_transactions, transactions])

# Draft dataset
all_drafts = pd.DataFrame()
for year in tqdm(years, desc="Processing Years"):
    lg = get_league(year=year, gm=gm)
    draft = pd.DataFrame(lg.draft_results())
    draft['year'] = year
    all_drafts = pd.concat([all_drafts, draft])

# Teams dataset
all_teams = pd.DataFrame()
for year in tqdm(years, desc="Processing Years"):
    teams = pull_managers(year=year, gm=gm)
    teams = pd.DataFrame(teams)
    teams['year'] = year
    all_teams = pd.concat([all_teams, teams])

# Save
all_teams.to_csv("data/teams.csv")
all_transactions.to_csv("data/transactions.csv")
all_drafts.to_csv("data/drafts.csv")

# Get matchups
all_matchups = pd.DataFrame()
for year in tqdm(years, desc="Processing matchups"):
    try:
        df = get_matchups(year, gm)
        all_matchups = pd.concat([all_matchups, df])
    except:
        continue

# Reset indices
all_matchups = all_matchups.reset_index()

# Save CSV
all_matchups.to_csv("data/matchups.csv", index=False)

# Loop through and get all rosters for all years
# NB: I got data for - 2007, 2008, 2014-2017
# getting booted due to too many calls to the API I assume.
# not sure what the lockout time is.
# Redid for subset of years and merged dataframe
all_rosters = pd.DataFrame()
for year in years:
    try:
        print(f"Processing data for the year {year}")
        rost = get_rosters(year=year, gm=gm)
        rost['year'] = year
        all_rosters = pd.concat([all_rosters, rost])
    except:
        print(f"No league data for the year {year}")
        continue
    sleep(11)  # sleep for 5 minutes

# Set the desired order of columns
column_order = ['year', 'week', 'playoffs', 'team_key' ,'player_id', 'name', 'position_type', 'eligible_positions', 'selected_position']

# Output to CSV file
all_rosters[column_order].to_csv("data/rosters.csv")

# Get week start and end points
weeks = []
for year in tqdm(years, desc="Processing Years"):
    try:
        lg = get_league(year, gm)
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

# Re-load rosters
rosters = pd.read_csv("data/rosters.csv")

# Create empty list for stats
all_stats = []

# Get player statistics (this takes a few minutes)
for year in years:
    try:
        print(f"Processing year {year}...")
        lg = get_league(year, gm)
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

# NOTE: it seems that yahoo only started getting position-level stats in 2014. Might need to backfill with NFL data


# Get final season ranks

bigdf = pd.DataFrame()
for year in tqdm(years, desc='Processing years'):
    lg = get_league(year=year, gm=gm)
    teams = [x['team_key'] for x in lg.standings()]
    rank = [x['rank'] for x in lg.standings()]
    df = pd.DataFrame({'team_key': teams, 'rank': rank})
    df['year'] = year
    # concatenate
    bigdf = pd.concat([bigdf, df])

bigdf.rename(columns={'team': 'team_key'}, inplace=True)
bigdf.to_csv("data/standings.csv", index=False)

# Add rankings to teams table
teams = pd.read_csv("data/teams.csv")

teams = teams.merge(bigdf, on=['team_key', 'year'], how='left')
teams['rank'] = teams['rank'].astype(int)
# teams['rank'].value_counts()

# Resave teams df
teams.to_csv("data/teams.csv", index=False)
