# TODO: use this later for getting combinations of 
# from itertools import product
# list(product([2005,2006], [1,2,3,4,5,6]))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from HierarchiaPy import Hierarchia
import seaborn as sns
from datetime import datetime
import re

from sqlalchemy import text
from sqlalchemy import create_engine

engine = create_engine('postgresql+psycopg2://chad:password@localhost:5432/football', echo=True)

db = engine.connect()

# Dropoff
query = open("dropoff.sql").read()
df = pd.read_sql_query(text(query), con=db)

current_managers = ['andrew','jon','bo','josiah','shane','chad','jarrod','aaron','kai','charles','david','daniel']

# Clean up the manager names
df['manager'] = [re.split('[ _]', x)[0].lower() for x in df['manager']]

# Filter out the managers we don't care about
# df = df[df['manager'].isin(current_managers)]

# Only plot manager 'chad'
# df = df[df['manager'] == 'chad']
# sns.barplot(data=df, x='year', y='players_remaining', hue='manager')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot the dropoff
g = sns.FacetGrid(data=df.sort_values('year'), col='manager', col_wrap=5)
g.map(sns.barplot, 'year', 'players_remaining')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot stacked bar chart    
sns.barplot(data=df, x='year', y='players_remaining', hue='manager')

# Plot stacked bar chart
g = sns.FacetGrid(data=df, col='manager', col_wrap=5)
g.map(sns.barplot, 'year', 'players_remaining')

sns.barplot(data=df, x='year', y='players_remaining', hue='manager', dodge=True)


plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.ylabel('Players Remaining')
plt.title('Player Retention by Manager Over Time')
plt.tight_layout()
plt.savefig('figures/retention.png')


g = sns.FacetGrid(data=df, col='manager', hue='year', col_wrap=5)
g.map(sns.lineplot, 'week', 'count')
g.add_legend()
plt.savefig('figures/dropoff.png')

# running queries from .SQL files
# psql -d football -f query.sql
query = open("stints.sql").read()
df = pd.read_sql_query(text(query), con=db)
df['manager'] = [re.split('[ _]', x)[0].lower() for x in df['manager']]
df = df[df['manager'].isin(current_managers)]
df['stint'] = df['stint'].dt.days

# Histogram
df['stint'].hist()

df['stint'].idmax()


sns.histplot(data=df, x='stint', hue='manager', bins=10, multiple='stack')

# Average by year
df_yearly = df.groupby(['year','manager','pos'])['stint'].apply('mean').reset_index()

sns.histplot(df_yearly[df_yearly['pos'].isin(['WR','QB','RB'])], x='stint', hue='pos')

# Create the boxplot ordered by mean
df_kickers = df_yearly[df_yearly['pos']=='K']

# Calculate the mean for each category
means = df_kickers.groupby('manager')['stint'].mean().sort_values()
sns.boxplot(x='stint', y='manager', data=df_kickers, order=means.index, color='lightgreen')
plt.savefig('figures/k_stints.png')

# Create the boxplot ordered by mean
df_rb = df_yearly[df_yearly['pos']=='RB']
# Calculate the mean for each category
means = df_rb.groupby('manager')['stint'].mean().sort_values()
sns.boxplot(x='stint', y='manager', data=df_rb, order=means.index, color='lightgreen')
plt.savefig('figures/rb_stints.png')

# Create the boxplot ordered by mean
df_wr = df_yearly[df_yearly['pos']=='WR']
# Calculate the mean for each category
means = df_wr.groupby('manager')['stint'].mean().sort_values()
sns.boxplot(x='stint', y='manager', data=df_wr, order=means.index, color='lightgreen')
plt.savefig('figures/wr_stints.png')

# Create the boxplot ordered by mean
df_def = df_yearly[df_yearly['pos']=='DEF']
# Calculate the mean for each category
means = df_def.groupby('manager')['stint'].mean().sort_values()
sns.boxplot(x='stint', y='manager', data=df_def, order=means.index, color='lightgreen')
plt.savefig('figures/def_stints.png')






# Load data
matchups = pd.read_csv("data/matchups.csv", index_col=0)
drafts = pd.read_csv("data/drafts.csv", index_col=0)
transactions = pd.read_csv("data/transactions.csv", index_col=0)
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

picks = teams.loc[teams['nickname'].str.contains('Chad'), 'team_key'].tolist()


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

matchups_melted = pd.wide_to_long(matchups, stubnames=['points','team_key'], i=['year','week','matchup'], j='team_id')

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

# Shorten names
nms = hier_df.indices
nms_short = [re.split('[ _]', x)[0].lower() for x in nms]

# BUG fix this- not working now

fig, ax = plt.subplots(tight_layout=True)
ax = sns.heatmap(hier_df.mat, cmap='mako', annot=True, fmt=".0f", xticklabels=nms_short, yticklabels=nms_short, ax=ax)
ax.xaxis.set_ticks_position('top')  # Move x-axis labels to the top
ax.xaxis.set_label_position('top')  # Move x-axis labels to the top
ax.tick_params(axis='x', rotation=90)
ax.set_xlabel('Loser')
ax.set_ylabel('Winner')
plt.savefig("figures/dominance_matrix.png")

print(hier_df.mat)
print(hier_df.indices)

# Davids score is a measure of dominance
hier_df.indices = nms_short
davids_scores = hier_df.davids_score()
print(davids_scores)

# Save plot
plt.barh(davids_scores.keys(), davids_scores.values())

df_davids_scores = pd.DataFrame.from_dict(davids_scores, orient='index')
df_davids_scores = df_davids_scores.reset_index()
df_davids_scores.columns = ['index', 'david']

sns.barplot(df_davids_scores, x='david', y='index')
plt.xlabel("David's Score")
plt.ylabel("")
plt.savefig("figures/davids.png")

%matplotlib inline

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



# Looking at the relationship between point ranks and draft pick

ranks = pd.read_csv("ranks.csv")

ranks[ranks['position']=='RB'].sort_values('points', ascending=False)

ranks['differential'] = ranks['pick'] - ranks['points_rank']

sns.scatterplot(ranks, x='points_rank', y='pick', hue='position')

sns.scatterplot(ranks, x='position', y='differential')

sns.lineplot(ranks, x='pick', y='differential', hue='position')

ranks_rb_top20 = ranks[(ranks["pick"] < 50) & (ranks["position"] == "RB")]

sns.barplot(ranks_rb_top20, x='pick', y='differential')

ranks[ranks['position']=='TE'].sort_values('points', ascending=False)



# NOTE I want to think of some metric that captures how good someone did in a draft.


df = pd.read_csv("wins_moves.csv")

sns.pairplot(df, hue='year', palette='tab10')


df[['roster_moves','adds','games_won','total_points']].corr()



# Draft value
import pandas as pd
import seaborn as sns

df = pd.read_csv("/Users/chad/Downloads/supabase_rpeohwliutyvtvmcwkwh_Roster Data Retrieval.csv")

df_top50 = df[df['overall_pick'] < 50]

g = sns.FacetGrid(data=df_top50, col='year', hue='position', col_wrap=5)
g.map(sns.scatterplot, 'position_pick', 'season_rank')
# add legend
g.add_legend()
# add fit lines
g.map(sns.regplot, 'position_pick', 'season_rank', scatter=False, ci=None)


sns.scatterplot(df, x='position_pick', y='season_rank', hue='position')


# hexbin plot
sns.jointplot(data=df, x='position_pick', y='season_rank', kind='hex')
# add scatter plot on top of jointplot
sns.scatterplot(data=df, x='position_pick', y='season_rank', alpha=0.15, color='red')


print(df.head())



g = sns.FacetGrid(data=df, col='position', hue='year', col_wrap=3, sharex=False)
g.map(sns.scatterplot, 'position_pick', 'season_rank')
# add legend
g.add_legend()
# add fit lines
g.map(sns.regplot, 'position_pick', 'season_rank', scatter=False, ci=None)


# TODO account for keepers
# TODO account for league winners

