# Fantasy

## Introduction

idea is to develop a fantasy football app
AI powered? haha. maybe.
add rankings integration from whatever sources you want (fantasy pros, etc.)
streamlit? not sure where I would host it..
dominance metrics (head-to-head, like they do in college football)
top 6 teams go to playoffs
pull stats from...?

## Cleaning the data


## Building the PostgreSQL database



## Optimizing database structures

The `rosters` table contains information about the roster, or the set of NFL players on a given person's team in a given week. You can imagine that sometimes a manager will leave a player in an active position all year. In that case, we would be recording data for each week (say, 16 weeks) when really all we need is 1 data point. The idea here is to look at transitions between states (for example, between an active roster spot and a bench spot). This is stored in a table called `events`. Doing this results in a __38% savings in storage__, although the SQL queries are more challenging to write given the dynamic nature of the `events` table. This is similar 

## Connecting the database to a LLM

I might want to connect an LLM to SQL database

Sample questions we could ask the LLM:
1. What are some sample trades I could make?
2. Who is the best of all time?
3. Who makes the most trades?
4. Show me a graph of points.
5. How often does the person with the most points win it all?

## Understanding league engagement

Possible metrics that could be useful in understanding manager engagement are roster turnover (the xx) and the overall number of moves (or transactions) made by a player.

## Bringing primate dominance hierarchies to fantasy football

Here's a plot that shows overall "dominance" (a measure of how often someone beats other people, and the strength of the opponent in terms of number of wins they themselves have). Someone who wins a lot of games against winless teams would have a lower David's score than someone who wins a lot of games against teams that beat other teams a lot.

![](figures/davids.png)


## References

What others have done
https://introductory.medium.com/fantasy-football-stats-gpt-fb92c1006f92

Useful for setting up db, llm
https://medium.com/dataherald/how-to-langchain-sqlchain-c7342dd41614

