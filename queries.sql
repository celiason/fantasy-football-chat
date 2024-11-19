-- I removed a column:
-- ALTER TABLE events
-- DROP COLUMN name;

SELECT name, t.player_id
FROM transactions t
LEFT JOIN players p
    ON t.player_id = p.player_id

-- LEFT JOIN managers
--     ON managers.manager_id
-- WHERE year = 2023
-- LIMIT 10;

-- Look at adds per year
SELECT year, COUNT(*)
FROM transactions
WHERE type = 'add'
GROUP BY year
ORDER BY COUNT(*) DESC
LIMIT 10;


-- What football players was dropped the most
SELECT p.name, COUNT(*) AS adds
FROM transactions t
LEFT JOIN players p
    ON t.player_id = p.player_id
WHERE type = 'add'
GROUP BY p.name
ORDER BY adds DESC
LIMIT 25;

-- Who made the most adds


-- Look at adds per year
SELECT year,
    SUM(CASE WHEN type = 'add' THEN 1 ELSE 0 END) AS adds,
    SUM(CASE WHEN type = 'drop' THEN 1 ELSE 0 END) AS drops,
    SUM(CASE WHEN type = 'trade' THEN 1 ELSE 0 END) AS trades
FROM transactions
-- WHERE trans_type = 'add'
GROUP BY year
ORDER BY year;


-- Who did I (Chad) play the most over the years?
SELECT m.name, g.year, m.manager_id
FROM managers m
LEFT JOIN games g
    ON m.manager_id = g.winning_manager_id OR m.manager_id = g.losing_manager_id
WHERE m.name = 'Chad';

-- What player moved around between teams the most in 2007?
SELECT p.name, COUNT(DISTINCT team_key) AS num_teams
FROM transactions t
LEFT JOIN players p
    ON t.player_id = p.player_id
WHERE year = 2023
GROUP BY p.name
ORDER BY num_teams DESC
LIMIT 10;

-- Who had the best draft pick in 2007?
-- best draft pick - pick that contributed most points per week
-- for that team
-- if a manager dropped a player, the point scoring ends
-- look at events table for that

SELECT m.name
FROM managers m
LEFT JOIN 


-- What was the "best move" of all time?
-- see what player added from FA/waivers (or trade made) that gave the most points to the receiving team


-- What player had the most points in a season overall?
SELECT w.year, p.name, ROUND(SUM(total_points)::numeric, 0) tot_points
FROM statistics s
LEFT JOIN players p
    ON s.player_id = p.player_id
LEFT JOIN weeks w
    ON s.week_id = w.week_id
GROUP BY p.name, w.year
ORDER BY tot_points DESC;


-- How close was the in-season rank to the draft position?
-- only top 12 draft picks
WITH top12 AS (
    SELECT player_id
    FROM drafts
    WHERE year = 2023-- and event = 'draft'
    ORDER BY pick
    LIMIT 12
),
-- running backs
rbs AS (
    SELECT player_id
    FROM players
    WHERE position = 'RB'
)
-- get statistic summary
SELECT p.name, ROUND(SUM(total_points)::numeric, 0)
FROM statistics s
LEFT JOIN players p
    ON s.player_id = p.player_id
WHERE s.player_id IN (SELECT * FROM rbs) AND 
        s.player_id IN (SELECT * from top12) AND
        year = 2023
GROUP BY p.name
;


-- How close was the in-season rank to the draft position?
WITH rbs AS (
    SELECT player_id
    FROM players
    WHERE position = 'RB'
)
SELECT p.name, RANK(SUM(total_points)) AS rank
FROM statistics s
LEFT JOIN players p
    ON s.player_id = p.player_id
WHERE s.player_id IN (SELECT * from rbs) AND year = 2023
GROUP BY p.name
ORDER BY rank DESC
;

-- Positional ranks and draf ranks by season
-- Idea is to see if #1 picks produce like a #1

-- Another query-
-- SELECT *
-- FROM season_pts
-- WHERE position = 'TE'
-- ORDER BY points DESC
-- LIMIT 100
-- ;


-- ALTER TABLE statistics ALTER COLUMN rush_yds TYPE real;

SELECT p.name, p.position_type, p.eligible_positions, ROUND(SUM(total_points)::numeric, 0)
FROM statistics s
LEFT JOIN players p
    ON s.player_id = p.player_id
WHERE s.player_id IN (SELECT * FROM top12) AND year = 2023
GROUP BY p.name;

-- calculate position ranks for that year




-- What was my starting lineup in week 2 of 2008?

select * from events where year = 2008 and week = 2 and event = 'active';

-- there's a trick- if a player was active last week (week 1) they won't have an
-- event active since they're still sitting in the lineup.
-- we'll need to maybe do a lag?


select *, LAG() OVER (PARTITION BY year ORDER BY )




WITH player_status_history AS (
    SELECT
        destination_mid AS manager_id,
        player_id,
        type,
        year,
        week,
        ROW_NUMBER() OVER (PARTITION BY destination_mid, player_id ORDER BY year, week) AS row_num
    FROM
        events
    WHERE
        type = 'active'
    ORDER BY year, week, destination_mid, row_num
),
active_players_per_week AS (
    SELECT
        manager_id,
        player_id,
        year,
        week
    FROM
        player_status_history
    WHERE
        row_num = 1
    UNION ALL
    SELECT
        e.destination_mid,
        e.player_id,
        e.year,
        e.week
    FROM
        events e
    JOIN player_status_history psh
        ON e.destination_mid = psh.manager_id
        AND e.player_id = psh.player_id
    WHERE
        e.type = 'active'
        AND e.year >= psh.year
        AND e.week > psh.week
)
SELECT
    manager_id,
    player_id,
    year,
    week
FROM
    active_players_per_week
ORDER BY
    year, week, manager_id;

