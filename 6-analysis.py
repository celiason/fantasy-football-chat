# TODO: use this later for getting combinations of 
# from itertools import product
# list(product([2005,2006], [1,2,3,4,5,6]))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from HierarchiaPy import Hierarchia

# Load data
matchups = pd.read_csv("data/matchups.csv", index_col=0)
drafts = pd.read_csv("data/drafts.csv", index_col=0)
transactions = pd.read_csv("data/transactions.csv", index_col=0)
teams = pd.read_csv("data/teams.csv", index_col=0)

picks = teams.loc[teams['nickname'].str.contains('Chad'), 'team_key'].tolist()

teams.head()
transactions.head()
drafts.head()
matchups.head()


from datetime import datetime

# Convert from ms timestamp to dates
transactions['date'] = transactions['timestamp'].transform(datetime.fromtimestamp)

# Look at transactions per week
df = transactions.groupby(pd.Grouper(key='date', freq='M')).size()
df.plot()

# What if I want to know who won the championship in a given year?


# Model features that explain who wins a championship



# How much does the draft matter?



# What is the average length of time a player stays on a roster?



# Show me a graph of who has the most points in a given year. Color the lines by team name.
# X axis should be week of the season




# Scatterplot of matchup points
colors = np.where(matchups['team_key1'].isin(picks) | matchups['team_key2'].isin(picks), 'blue', 'gray')
matchups.plot(x='points1', y='points2', kind='scatter', c=colors)
plt.plot([50, 230], [50, 230], color='red', linestyle='--', label='1:1 Line')

matchups.reset_index(inplace=True)

# Total points on the year

# Need this unique id for each matchup before wide_to_long
matchups['matchup'] = matchups.groupby(['year','week']).cumcount() + 1

# This melts the data frame- really useful 'stubs' argument
matchups_melted = pd.wide_to_long(matchups, axis=1), stubnames=['points','team_key'], i=['year','week','matchup'], j='team_id')
# matchups_melted.pivot_table(values=['points'], index=['week'], columns='team')

total_points = matchups_melted.groupby(['year','team_key'])['points'].sum()

df = matchups.merge(teams[["team_key","name","nickname"]], left_on='team_key1', right_on='team_key', suffixes=["", "_teams1"])
df = df.merge(teams[["team_key","name","nickname"]], left_on='team_key2', right_on='team_key', suffixes=["", "_teams2"])

# Create winner and loser columns for hierarchy analysis
df['winner'] = np.where(df['points1'] > df['points2'], df['nickname'], df['nickname_teams2'])
df['loser'] = np.where(df['points1'] < df['points2'], df['nickname'], df['nickname_teams2'])

# Lookup dict to change names
m = {'Buie': 'JC', 'daniel': 'Will'}

# Replace names
df['winner'].replace(m, inplace=True)
df['loser'].replace(m, inplace=True)

# Check that it worked
df['winner'].value_counts()
df['loser'].value_counts()

# Create dominance matrix
hier_df = Hierarchia(df, 'winner', 'loser')

print(hier_df.mat)
print(hier_df.indices)

# Davids score is a measure of dominance
davids_scores = hier_df.davids_score()
print(davids_scores)

# Who is "--hidden--"? Maybe Matt Harbert?
df[df['nickname']=='--hidden--']['name'].value_counts()
# name
# Doinel's Destroyers!    13
# The Woebegones           9
# Browns West              7
# Rubber City Galoshes     4
# The Five Toes            3

# Save plot
plt.bar(davids_scores.keys(), davids_scores.values())
plt.xticks(rotation=90)
plt.savefig("figures/davids.png")


def calc_david(df):
    hmat = Hierarchia(df, "winner", "loser")
    scores = hmat.davids_score()
    return scores

df.groupby('year').apply(calc_david)

plt.figure(figsize=(12, 8))

heatmap_data = teams.pivot_table(index='nickname', columns='year', values='draft_grade', aggfunc=lambda x: x.mode()[0] if not x.mode().empty else None)

import seaborn as sns

grade_map = {
        'A+': 4.3,
        'A': 4.0,
        'A-': 3.7,
        'B+': 3.3,
        'B': 3.0,
        'B-': 2.7,
        'C+': 2.3,
        'C': 2.0,
        'C-': 1.7,
        'D+': 1.3,
        'D': 1.0,
        'D-': 0.7,
        'F': 0.0
    }

heatmap_data_num = heatmap_data.replace(grade_map)


sns.heatmap(heatmap_data_num, cmap='cividis', annot=heatmap_data, fmt="")
plt.title('Heatmap of Draft Grades by Team and Year')
plt.xlabel('Year')
plt.ylabel('Team Nickname')
plt.savefig("figures/draft_grade_heatmap.png")
plt.show()


import pandas as pd
import seaborn as sns
df = pd.read_csv("test.csv")
df['differential'] = df['points_rank'] - df['draft_rank']

sns.scatterplot(df, x='points_rank', y='draft_rank', hue='position')

sns.scatterplot(df, x='position', y='differential')

sns.lineplot(df, x='draft_rank', y='differential', hue='position')

df[df['points_rank'] == df['draft_rank']]

df[df['position']=='RB']

# NOTE I want to think of some metric that captures how good someone did in a draft.

