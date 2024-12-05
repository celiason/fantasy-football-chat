# Fantasy football chatbot

<!-- ![](assets/dalle_turtle.jpg) -->
![](assets/dalle_logo2.jpg)
<!-- ![](assets/dalle_logo3.webp) -->

## Introduction

I've been in a fantasy football league for 15 odds years now. I care a lot about the league and the people in it. We've been through a lot together. Something I've noticed is that engagement ebbs and flows. We've had a total of 19 friends in the league, but a few dropped out after only a season or two. I always wondered if having a way to quickly access insights from historical data would be a way to increase engagement. Excited by some of the recent developments in large language models (LLMs), I had the idea of developing a SQL database of our league data and connecting it to a AI chatbot.

Feel free to try the webapp out here- [https://slow-learners-chat.streamlit.app](https://slow-learners-chat.streamlit.app)

## Terminology

There are a bunch of terms in the fantasy football world that need to be defined before we get into the details.

1. league - This is a group of managers
2. manager - This is a person that picks players to be on a team
3. player - An NFL player that gains points for a manager in a given week
4. roster - A set of players on a given week
5. team - A group of rosters throughtout a season associated with a given manager. The team name can vary week-to-week. 
6. statistic - A number that describes an NFL metric (yards gained, touchdown scored, etc.)
7. week - A week in an NFL season
8. season - A set of managers playing together in a given year

## Challenges with sports data

<!-- talk about database normalization maybe (1NF, 2NF, 3NF compliant) -->

In early 2000, Yahoo allowed managers to drop players immediately after they played a game. Since yahoo doesn't provide the timestamp for when a player played on a roster in a given week, I had to use the end of the NFL week as the timestamp. This ensured that the player would not be dropped _before_ playing that week. However, in cases where a manager dropped the player earlier in the week, the events table would say that the player was only active the week before. So, to deal with this challenge, I wrote a function in python that scans through the events table and adjusts the drop time to the first minute of the following week. This way, if a player is active, he will show up that way for the given week and then be dropped for the following week.

## Optimizing database structure

I used a combination of SQLalchemy and PostgreSQL to build the database. We have three tables that have to do with players and managers: `rosters`, `transactions`, and `drafts`. One challenge I faced was how to efficiently store all the data.

The `rosters` table contains information about the roster, or the set of NFL players on a given person's team in a given week. You can imagine that sometimes a manager will leave a player in an active position all year. In that case, we would be recording data for each week (say, 16 weeks) when really all we need is 1 data point.

A better idea is to look at transitions between states (for example, between an active roster spot and a bench spot). This is stored in a table called `events`. Doing this resulted in a __31% reduction in storage__, although the SQL queries are more challenging to write given the dynamic nature of the `events` table. A benefit of this approach is that I can easily calculate time-dependent features like like "roster turnover" as the amount of time a player has been in a position.

## Prompt engineering for the LLM

I wanted to connect my database to a LLM so that we could ask some interesting questions. For example:
1. What are some sample trades I could make?
2. Who is the best of all time?
3. Who makes the most trades?
4. Show me a graph of points.
5. How often does the person with the most points win it all?

I might want to connect an LLM to SQL database. I did this using langchain.

The first version worked ok, but the chatbot often misunderstood the question I was asking and either failed to return anything or hallucinated and gave a ridiculous answer (e.g., shane won 900 games in a season.. which is impossible).

To solve this problem, I decided to create SQL views that have summarized data and more accessible names (with fewer many-to-many relationships). For example, I created a `slots` view that has NFL players on a given manager's roster for each week and year. I also created a `standings` view that has the records for each manager in each year. By only providing these views and the context of the column names, I was able to greatly improve the accuracy of the AI.

Another challenge I ran into was what LLM to use. I wanted to make this free for my friends, so I opted for groq over OpenAI's GPT-3 model. 

## Understanding league engagement

### How much does in-season management matter?

We can also look at in-season number of moves to see if this has an effect on end-of-season rank. If there is a positive relationship, then it would make sense for a player to stay engaged throughout the course of a season.

Interestingly, in 2014 there was a negative effect, while in 2018 in-season adds had a large effect on final rank. Then in 2023, there was no relationship. In these plots, each point is a manager.

![](figures/roster_adds_rank.png)

The take-home is that the effect of in-season management depends on who is managing the team, and the set of players that are available in a given year.

### How much does the draft matter?

This might influence in-season engagement. If it matters a lot, then why do in season management?

Draft pick location...

{% include_relative bumpchart.html %}

### How have league rule changes affected engagement?

To understand whether shifts in league settings have an effect on engagement, I plotted the relationship between manager activity (total number of adds, drops, roster shifts, and trades) and NFL week in a season. The fitted lines are second-order polynomials. You can see below that there is a clear bump in activity around week 8, then a decrease from week 8 on. It also appears that the curves arae shifting to the right over time. This suggests that more managers are staying active into later weeks of the season, which could be due to changes in league rules (e.g., we changed playoff structure in year X).

![](figures/events_over_time_lmplot.png)

To look at this more closely, I plotted manager activity by year for only week 15. There is a clear linear increase in activity from 2007 to 2023.

![](figures/events_over_time_week15.png)

When we look at week 2, this pattern is not present:

![](figures/events_over_time_week2.png)

### Are there temporal changes in manager behavior that can be used to predict when they might leave the league?

Possible metrics that could be useful in understanding manager engagement are roster turnover (the xx) and the overall number of moves (or transactions) made by a player.

Some possibilities- 
number of moves a manager makes
outcome maybe- if they win a week or not.

some players make very few moves. maybe if they knew if the moves mattered they would be more engaged? I think that's the idea here.

Overall number of moves would be the unique events made by a manager.

Let's say a player makes 25 moves in a week before a game and they win. The opposing player made 0 moves and they lost. Then we might suspect that the number of moves mattered. However, in the case of the player with 0 moves it could be that they have a really good team from the draft and they just don't need to make any moves because the team is fine as it is. In this case, we would expect a negative correlation between number of moves and number of wins.

This is analogous to customer purchase behavior. We could have someone that clicks on a lot but never buys (wins), or someone that clicks strategically and makes a purchase (wins a game). But what about the influence that a draft has on future behavior? In that case, we can start to think about 

(NB: I got this from someowhere else-https://www.vitroagency.com/the-parrot/what-marketers-learn-fantasy-football/)

- Opportunity cost (the fantasy football draft or swapping players during season play)
- Comparative advantage and gains (trading players to fill a position need)
- Market behavior including supply and demand shocks (injury, Bye Weeks)
- Consumer surplus (again, the draft)
- Imperfectly competitive markets (fantasy “super team” rosters)
- Game theory (analysis of player value in draft or trade)

One thing that would be cool is to somehow see if there are any trends in behavior. That is, do some managers act the same way year-to-year in terms of pickups, draft selections, etc.

## Bringing primate dominance hierarchies to fantasy football

Below is a matrix showing the overall number of times a team beat another team. For example, if we look at "shane" we see that he beat bo 10 times but only beat chad 7 times. By contrast, chad beat shane 9 times. So his record against shane is 9-7. Pretty cool!

![](figures/dominance_matrix.png)

We can go a step further and borrow from the scientific literature on folks that study primate social structures and dominance. A useful metric is the "David's Score." Someone who wins a lot of games against winless teams would have a lower David's score than someone who wins a lot of games against teams that beat other teams a lot. Here's a plot that shows overall "dominance" (i.e., David's score) for each time across the 17 yearas the league has been in existence.

![](figures/davids.png)


## References

What others have done
https://introductory.medium.com/fantasy-football-stats-gpt-fb92c1006f92

Useful for setting up db, llm
https://medium.com/dataherald/how-to-langchain-sqlchain-c7342dd41614

