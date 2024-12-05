import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from HierarchiaPy import Hierarchia
from datetime import datetime
import re
from sqlalchemy import text
from sqlalchemy import create_engine
import xgboost as xgb
import shap
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# Connect to database
password = st.secrets['supa_password']
db_uri = f"postgresql://postgres.rpeohwliutyvtvmcwkwh:{password}@aws-0-us-west-1.pooler.supabase.com:6543/postgres"
engine = create_engine(db_uri, echo=True)

db = engine.connect()

#--------------- Dropoff Analysis ----------------
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

#--------------- Player Stints Plotting ----------------
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

#--------------- Dominance Analysis ----------------
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

# %matplotlib inline

def calc_david(df):
    hmat = Hierarchia(df, "winner", "loser")
    scores = hmat.davids_score()
    return scores

df.groupby('year').apply(calc_david)

plt.figure(figsize=(12, 8))

heatmap_data = teams.pivot_table(index='nickname', columns='year', values='draft_grade', aggfunc=lambda x: x.mode()[0] if not x.mode().empty else None)

import seaborn as sns

#--------------- Draft Grades Plot ----------------
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


#--------------- Ranks and Total Points Analysis ----------------
# Looking at the relationship between point ranks and draft pick

query = open("ranks.sql").read()
ranks = pd.read_sql_query(text(query), con=db)

ranks[ranks['position']=='RB'].sort_values('points', ascending=False)

ranks['differential'] = ranks['pick'] - ranks['points_rank']

sns.scatterplot(ranks, x='points_rank', y='pick', hue='position')

sns.scatterplot(ranks, x='position', y='differential')

sns.lineplot(ranks, x='pick', y='differential', hue='position')

ranks_rb_top20 = ranks[(ranks["pick"] <= 20) & (ranks["position"] == "RB")]
sns.barplot(ranks_rb_top20, x='pick', y='differential')

ranks_wr_top20 = ranks[(ranks["pick"] <= 20) & (ranks["position"] == "WR")]

sns.barplot(ranks_wr_top20, x='pick', y='differential')

sns.scatterplot(data=ranks_rb_top20, x='pick', y='points', hue='differential', palette='mako')
sns.scatterplot(data=ranks_wr_top20, x='pick', y='points', hue='differential', palette='mako')

ranks_rb_top20[ranks_rb_top20['points']==0]

ranks_te_top10 = ranks[(ranks["pick"] < 20) & (ranks["position"] == "TE")]
sns.barplot(ranks_te_top10, x='pick', y='differential')



# NOTE I want to think of some metric that captures how good someone did in a draft.


df = pd.read_csv("wins_moves.csv")

sns.pairplot(df, hue='year', palette='tab10')


df[['roster_moves','adds','games_won','total_points']].corr()


#--------------- Draft Value Analysis ----------------

query = open("draft_value.sql").read()
df_value = pd.read_sql_query(text(query), con=db)
df_value['type'] = np.where(df_value['draft_value'].isna(), 'FA', 'drafted')

# Look at a given player
idx = (df_value['player']=='James Conner') & (df_value['year']==2018)
df_value.loc[idx]

# Look at the top WR in 2010
df_value[(df_value['position']=='WR') & (df_value['year']==2010)]

sns.scatterplot(data=df_value, x='season_rank', y='season_pts', hue='position', style='type')

sns.boxplot(data=df_value, x='position', y='draft_value', hue='year')

df_value['position'] = pd.Categorical(df_value['position'], categories=['QB', 'RB', 'WR', 'TE', 'K', 'DEF'], ordered=True)

df_value[(df_value['year']==2009) & (df_value['position']=='RB')]['type'].value_counts()

# Overall
sns.stripplot(data=df_value, x='position', y='season_pts', hue='type', alpha=0.5, dodge=True)
sns.pointplot(data=df_value, x='position', y='season_pts', hue='type', dodge=True)

# Get top 3 FA pickups by year
df_value[df_value['type']=='FA'].groupby('position').apply(lambda x: x.nlargest(3, 'season_pts'))



# Plot
g = sns.FacetGrid(data=df_value, col='year', hue='type', col_wrap=5, sharex=False)
g.map(sns.stripplot, 'position', 'season_pts', alpha=0.25)
g.map(sns.pointplot, 'position', 'season_pts')
g.add_legend()
g.set_xlabels('Position')
g.set_ylabels('Season Points')
g.savefig("figures/draft_value_by_pos.png")

# Figure out best FA pickups each year
best_fa_pickups = df[df['type'] == 'FA'].groupby(['year', 'position']).apply(lambda x: x.loc[x['season_rank'].idxmin()])
best_fa_pickups.reset_index(drop=True, inplace=True)
best_fa_pickups.sort_values('season_rank', inplace=True)

# Look at top 5 FA pickups by position
print(best_fa_pickups[best_fa_pickups['position']=='QB'].head())
print(best_fa_pickups[best_fa_pickups['position']=='RB'].head())
print(best_fa_pickups[best_fa_pickups['position']=='WR'].head())

df[df['player']=='Kareem Hunt']

df.sort_values('draft_value', ascending=False)

#--------------- In-season management or a good draft.. what makes a winner? ----------------

# calculate total season points
# fraction gained by drafted players
# fraction gained by free agent pickups

df = pd.read_csv("/Users/chad/Downloads/supabase_rpeohwliutyvtvmcwkwh_Fetch Recent Events.csv")

df = df.groupby(['year','manager','type2'])['points'].sum().unstack()

df = df.reset_index()

df['total_points'] = df['draft'] + df['pickup']

# proportion drafted points
df['draft_prop'] = df['draft'] / df['total_points']

# proportion pickup points
df['pickup_prop'] = df['pickup'] / df['total_points']


g = sns.FacetGrid(data=df, col='manager', col_wrap=5, sharey=True, sharex=True)
g.map(sns.lineplot, 'year', 'draft_prop', marker='o')


df2 = pd.read_csv("/Users/chad/Downloads/supabase_rpeohwliutyvtvmcwkwh_Standings View.csv")
df3 = pd.read_csv("/Users/chad/Downloads/supabase_rpeohwliutyvtvmcwkwh_Standings Table.csv")
df2 = df2.merge(df3, on=['year','manager'])

df = df.merge(df2, on=['year','manager'])

df['moves'] = df['adds'] + df['drops'] + df['rosters']

df




#--------------- Engagement Analysis ----------------
# How do we think about engagement?
# Maybe- number of transactions, number of messages in chat, number of trades, number of draft picks, number of players added, number of players dropped
# engagement = number of moves?


# Plot number of moves vs. end of season rank
g = sns.FacetGrid(data=df, col='manager', col_wrap=5, sharey=False, sharex=False)
g.map(sns.regplot, 'adds', 'rank')
g.set(ylim=(13, 0))
g.savefig("figures/engagement_adds_rank.png")

# Plot number of moves vs. total points scored
g = sns.FacetGrid(data=df, col='manager', col_wrap=5, sharey=False, sharex=False)
g.map(sns.regplot, 'adds', 'points_scored')
g.savefig("figures/engagement_adds_points.png")

# Plot number of roster adjustments vs. end of season rank
g = sns.FacetGrid(data=df, col='manager', col_wrap=5, sharey=True, sharex=False)
g.map(sns.regplot, 'rosters', 'rank')
g.set(ylim=(13, 0))
g.savefig("figures/engagement_rosters_rank.png")

# Plot number of moves vs. total points scored
g = sns.FacetGrid(data=df, col='manager', col_wrap=5, sharey=False, sharex=False)
g.map(sns.regplot, 'rosters', 'points_scored')
g.savefig("figures/engagement_rosters_points.png")

# Plots overall for all managers
sns.regplot(data=df, x='adds', y='points_scored', logx=True)
sns.regplot(data=df, x='rosters', y='points_scored')


# Roster moves by year
g = sns.FacetGrid(data=df[df['year'].isin([2014,2019,2023])], col='year', col_wrap=5, sharey=False, sharex=False)
g.map(sns.regplot, 'rosters', 'rank', logx=True)
g.set(ylim=(13, 0))
# set axis labels
g.set_axis_labels("Roster Moves", "End of Season Rank")
g.savefig("figures/roster_moves_rank.png")


g = sns.FacetGrid(data=df[df['year'].isin([2014,2018,2023])], col='year', col_wrap=5, sharey=False, sharex=False)
g.map(sns.regplot, 'adds', 'rank', logx=True)
g.set(ylim=(13, 0))
g.set_axis_labels("Roster Adds", "End of Season Rank")
g.savefig("figures/roster_adds_rank.png")

sns.lmplot(data=df, x='adds', y='rank', hue='year', ci=False, logx=True)

sns.pairplot(np.log(df[['points_scored','adds','rosters']]))

np.corrcoef(df[['points_scored','adds','rosters']].T)

sns.regplot(data=df, x='adds', y='rank')
sns.regplot(data=df, x='rosters', y='rank')

sns.boxplot(data=df, x='rank', y='adds')
sns.boxplot(data=df, x='rank', y='rosters')

sns.regplot(data=df, x='wins', y='rank') # no team has ever won it all with < 8 wins
df[df['rank']==1]['points_scored'].min() # # you don't need tons of points, 1512 is lowest #1

#--------------- Regression Analysis ----------------
# predictors: wins, add/drops, roster moves, points scored, points allowed, year?
# response: champ or not

# Create a binary champ variable
df['champ'] = np.where(df['rank']==1, 1, 0)


# Trying linear regression and logistic regression

X = df[['adds','rosters','draft','pickup','points_allowed']]
y = df['wins']
# y = df['champ']


# X['manager'] = pd.Categorical(X['manager'])

# test train split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Logistic regression
from sklearn.linear_model import LinearRegression

linr = LinearRegression()

linr.fit(X, y)

# Make predictions

pred = linr.predict(X)

np.corrcoef(y, pred)

sns.histplot(df, x='wins')

import statsmodels.api as sm
import statsmodels.formula.api as smf

# defining the poisson glm 
logr = smf.glm(formula = 'champ ~ wins + draft + pickup + adds + rosters + points_allowed', 
               data = df,
               family = sm.families.Binomial())

# fitting the model 
results = logr.fit()

# Printing the summary of the regression results
print(results.summary())

np.exp(results.params)
# every win increases the odds of winning the championship by 53%
# every point scored by a drafted player increases the odds of winning the championship by 0.9%
# every point scored by a free agent pickup increases the odds of winning the championship by 1.5%
# every point scored by a player added DECREASES the odds of winning the championship by 0.6%
# every point allowed DECREASES the odds of winning the championship by 1%

# interpret coefficients
X.columns
linr.coef_

.009 * 100

sns.regplot(data=df, x='draft', y='wins')
sns.regplot(data=df, x='pickup', y='wins')

# Create a confusion matrix

pd.crosstab(y, pred)

# Summary of logistic regression model

from sklearn.metrics import classification_report, accuracy_score

# Print classification report
print(classification_report(y, pred))

# Print accuracy score
print(f"Accuracy: {accuracy_score(y, pred)}")

X.columns
np.exp(logr.coef_)
# for every 1 unit increase in draft pts, the odds of winning the championship increase by 1.2%
logr.coef_
# Interpret coefficients as log odds
log_odds = np.exp(logr.coef_)
print(f"Log odds: {log_odds}")


model = xgb.XGBClassifier(objective='binary:logistic', learning_rate=0.5, max_depth=5, n_estimators=100)

model.fit(X_train, y_train)

# Plot the decision tree
xgb.plot_tree(model, num_trees=1)

# Make predictions
pred = model.predict(X_test)

y_test.value_counts()

# Create a confusion matrix
pd.crosstab(y_test, pred)

sns.scatterplot(df, x='draft', y='wins', hue='champ')

sns.regplot(df, x='draft', y='champ', logistic=True)


sns.scatterplot(df, x='pickup', y='wins', hue='champ')



# Feature importance
xgb.plot_importance(model)

# Pairplot - save as figure
sns.pairplot(X)
plt.savefig('figures/pairplot.png')


# Shap value analysis
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)

df[df['manager']=='Chad']

# year I won
shap.plots.waterfall(explainer(X)[133])

# year I was 12th
shap.plots.waterfall(explainer(X)[109])

shap.summary_plot(shap_values, X)


#--------------- Best Draft Picks ----------------

# Load data
query = open("draft_picks.sql").read()
df = pd.read_sql_query(text(query), engine.connect())

df.sort_values('wins', ascending=False) # best spot is #2 in terms of wins and points

sns.barplot(data=df, x='overall_pick', y='wins', hue='topdog')

df.plot(x='overall_pick', y='points', kind='line', marker='o')
df.plot(x='overall_pick', y='wins', kind='line', marker='o')
df.plot(x='overall_pick', y='topdog', kind='line', marker='o')



#--------------- Ranking Bump Charts ----------------
# see plotly_ranks.py





events = pd.read_csv("/Users/chad/Downloads/supabase_rpeohwliutyvtvmcwkwh_Event Activity Analysis.csv")

events_sum = events[events['week']!=0].groupby(['year','week','manager']).size()
events_sum = events_sum.reset_index(name='event_count')

g = sns.FacetGrid(data=events_sum, col='manager', col_wrap=5, sharey=False, sharex=False)
g.map(sns.lineplot, 'week', 'event_count', marker='o')

events_sum['year_manager'] = events_sum['year'].astype(str) + '_' + events_sum['manager']

# Time series clustering
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from sklearn.preprocessing import StandardScaler

events_ts = events_sum.pivot(index=['week'], columns='year_manager', values='event_count')
events_ts.fillna(0, inplace=True)


# smooth the time series data
events_ts_smooth = events_ts.apply(lambda x: np.convolve(x, np.ones(5)/5, mode='same'))

events_ts['2007_Andy'].plot()
events_ts_smooth['2007_Andy'].plot()


# Do different managers have different tendencies?
# We'll use sklearn

from sklearn.svm import LinearSVC

# Get the cluster labels
labels = X.index.str.split("_").str[1]

# Fit the model to your data

# Create a linear SVM model
svm = LinearSVC()

# only keep weeks 1-13
X = events_ts_smooth.T

# Fit the model to your data
svm.fit(X, labels)

# Get the cluster labels
pred = svm.predict(X)

# confusion matrix
print(pd.crosstab(labels, pred))
print(classification_report(labels, pred))

# Do a PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_pca = pd.DataFrame(pca.fit_transform(X))
X_pca.columns = ['PC1', 'PC2']

sns.scatterplot(X_pca, x='PC1', y='PC2', hue=labels, palette='Set1')



# Generate plots
p = sns.lmplot(data=events_sum, x='week', y='event_count', hue='year', order=2, ci=False, palette='mako')
p.set_axis_labels("Week", "Manager Activity")
p.fig.set_figwidth(12)
p.fig.set_figheight(8)
p.savefig("figures/events_over_time_lmplot.png", bbox_inches='tight')
# we see an increase in events around week 8 then a decrease from week 8 on
# from polynomial fitting, we see that the curve is shifting to the right over time
# this suggests that more managers are staying active into later weeks of the season 
# this could be due to changes in league rules (we changed playoff structure in year X)

p = sns.lmplot(data=events_sum[events_sum['week'] == 2], x='year', y='event_count')
p.set_axis_labels("Year", "Manager Activity")
p.savefig("figures/events_over_time_week2.png", bbox_inches='tight')

p = sns.lmplot(data=events_sum[events_sum['week'] == 15], x='year', y='event_count')
p.set_axis_labels("Year", "Manager Activity")
p.savefig("figures/events_over_time_week15.png", bbox_inches='tight')

# when did we change league rules?

# when did we add keepers

