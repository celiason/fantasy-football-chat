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



## Connecting the database to a LLM

I might want to connect an LLM to SQL database

Sample questions we could ask the LLM:
1. What are some sample trades I could make?
2. Who is the best of all time?
3. Who makes the most trades?
4. Show me a graph of points.
5. How often does the person with the most points win it all?

## Understanding league engagement

Here's a plot that shows overall "dominance" (a measure of how often someone beats other people, and the strength of the opponent in terms of number of wins they themselves have). Someone who wins a lot of games against winless teams would have a lower David's score than someone who wins a lot of games against teams that beat other teams a lot.

![](figures/davids.png)


## References

What others have done
https://introductory.medium.com/fantasy-football-stats-gpt-fb92c1006f92

Useful for setting up db, llm
https://medium.com/dataherald/how-to-langchain-sqlchain-c7342dd41614

