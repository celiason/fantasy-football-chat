# Collection of functions to build a yahoo fantasy football private database

from time import sleep
import seaborn as sns
import matplotlib.pyplot as plt
from yahoo_oauth import OAuth2
import yahoo_fantasy_api as yfa
import pandas as pd
from HierarchiaPy import Hierarchia
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint # just discovered this! awesome way to visualize nested lists
from collections import ChainMap
from tqdm import tqdm

# Re-authorize
sc = OAuth2(None, None, from_file='secrets.yaml')
gm = yfa.Game(sc, 'nfl')
# lg = get_league(year=2024)

# Function to return a league object given a league name
def get_league(year, league_name="The Slow Learners"):
    sc = OAuth2(None, None, from_file='secrets.yaml')
    gm = yfa.Game(sc, 'nfl')
    league_ids = gm.league_ids(year=year)
    for id in league_ids:
        lg = gm.to_league(id)
        name = lg.matchups(week=1)['fantasy_content']['league'][0]['name']
        if name == league_name:
            return lg
        else:
            next

# x = trades list
# x = trades[0]
def pull_trade_details(x):
    res = []
    timestamp = x['timestamp']
    trans_id = x['transaction_id']
    players = x['players']
    status = x['status']
    for i in range(players['count']):
        p1 = dict(ChainMap(*players[str(i)]['player'][0]))
        t1 = players[str(i)]['player'][1]['transaction_data'][0]
        source = t1['source_team_key']
        destination = t1['destination_team_key']
        player_id = p1['player_id']
        row = {'trans_id': trans_id, 'timestamp': timestamp, 'trans_type': 'trade',
               'player_id': player_id, 'status': status, 'source': source, 'destination': destination}
        res.append(row)
    # Turn into dataframe
    # df = pd.DataFrame(data)
    # df['timestamp'] = timestamp
    # df['trans_id'] = trans_id
    return res

# Testing zone
# trades = lg.transactions(tran_types="trade", count=None)
# pull_trade_details(trades[0])
# pprint(trades) # sweet!!

# Function to prepare transactions for reading
def prepare_transactions(adds):
    items = []
    for item in adds:
        item
        # item = adds[22]
        trans_type = item['type']
        timestamp = item['timestamp']    
        if trans_type == 'add/drop':
            item1 = item['players']['0']['player'][0]
            item2 = item['players']['1']['player'][0]
            meta1 = item['players']['0']['player'][1]['transaction_data'][0]
            meta2 = item['players']['1']['player'][1]['transaction_data']
            items.append([{'timestamp': timestamp, 'transaction_data': meta1}] + item1)
            items.append([{'timestamp': timestamp, 'transaction_data': meta2}] + item2)
        elif trans_type == 'drop':
            meta = item['players']['0']['player'][1]['transaction_data']
            items.append([{'timestamp': timestamp, 'transaction_data': meta}] + item['players']['0']['player'][0])
        else:
            meta = item['players']['0']['player'][1]['transaction_data'][0]
            items.append([{'timestamp': timestamp, 'transaction_data': meta}] + item['players']['0']['player'][0])
    return items

def get_transactions(lg):
    res = []
    # Adds
    adds = lg.transactions(tran_types="add", count=None)
    adds = prepare_transactions(adds)
    # Trades
    trades = lg.transactions(tran_types="trade", count=None)
    for trade in trades:
        res.extend(pull_trade_details(trade))
    # Add/drops
    for i in range(len(adds)):
        timestamp = adds[i][0]['timestamp']
        trans_type = adds[i][0]['transaction_data']['type']

        if trans_type == 'add':
            source = adds[i][0]['transaction_data']['source_type']
            dest = adds[i][0]['transaction_data']['destination_team_key']
        if trans_type == 'drop':
            source = adds[i][0]['transaction_data']['source_team_key']
            dest = adds[i][0]['transaction_data']['destination_type']
        player_id = adds[i][2]['player_id']
        # Add to result
        res.append({'timestamp': timestamp, 'trans_type': trans_type,
                    'player_id': player_id, 'source': source, 'destination': dest})
    return res

# Pull manager and team data
def pull_managers(teams):
    res = []
    team_names = list(teams.keys())
    for name in team_names:
        # name=team_names[0]
        team = teams[name]
        nickname = team['managers'][0]['manager']['nickname']
        info = {key: team[key] for key in ['team_key', 'name', 'number_of_moves']}
        if 'division_id' in team.keys():
            info['division_id'] = team['division_id']
        else:
            info['division_id'] = None
        if 'draft_grade' in team.keys():
            info['draft_grade'] = team['draft_grade']
        else:
            info['draft_grade'] = None
        info['nickname'] = nickname
        res.append(info)
    return res

# Function to get winners and losers for a given league and week
def get_matchups(year):
    """
    lg = league object
    week = week to target
    """
    # Get matchups
    # year = 2007
    # week = 1
    lg = get_league(year=year)

    df_all = pd.DataFrame()

    def pull_matchup(matchups):
        # num_teams = matchups['fantasy_content']['league'][0]['num_teams']
        matchups = matchups['fantasy_content']['league'][1]['scoreboard']['0']['matchups']
        num_matchups = matchups['count']
        # Store them in a pandas dataframe object
        df = pd.DataFrame()
        for i in range(num_matchups):
            # i=0
            m = matchups[str(i)]
            team1 = m['matchup']['0']['teams']['0']
            team2 = m['matchup']['0']['teams']['1']
            # names
            key1 = team1['team'][0][0]['team_key']
            key2 = team2['team'][0][0]['team_key']
            # name1 = team1['team'][0][2]
            # name2 = team2['team'][0][2]
            # points
            points1 = team1['team'][1]['team_points']['total']
            points2 = team2['team'][1]['team_points']['total']
            # data frame output
            subdf = pd.DataFrame({'team_key1': key1, 'team_key2': key2, 'points1': points1, 'points2': points2}, index=[0])
            df = pd.concat([df, subdf], axis=0)
            df['week'] = week
            df['year'] = year
        return df

    # Get league settings
    meta = lg.settings()
    end_week = int(meta['end_week'])
    playoff_start = int(meta['playoff_start_week'])

    for week in range(1, end_week+1):
        matchups = lg.matchups(week=week)
        df = pull_matchup(matchups)
        df_all = pd.concat([df_all, df])
    
    return df_all.set_index(['year','week'])

# Testing
# get_winners(year=2007, week=1)



# Get weekly rosters
def get_rosters(year):
    # Get league
    lg = get_league(year=year)
    # Get league settings
    meta = lg.settings()
    end_week = int(meta['end_week'])
    playoff_start = int(meta['playoff_start_week'])
    # Get unique team keys for a year
    team_keys = list(lg.teams().keys())
    # Create empty data frame
    df = pd.DataFrame()
    # Loop through weeks
    for week in tqdm(range(1, end_week+1), desc="Processing weeks"):
        # Get team IDs
        for key in team_keys:
            roster = lg.to_team(key).roster(week=week)
            roster = pd.DataFrame(roster)
            roster['team_key'] = key
            roster['week'] = week
            if week >= playoff_start:
                roster['playoffs'] = 'yes'
            else:
                roster['playoffs'] = 'no'
            # Add to exiting dataframe            
            df = pd.concat([df, roster])
    return df

# check
# rost_2007 = get_rosters(year=2007)
# rost_2007.groupby(['team_key','week']).size()


