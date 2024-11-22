-- VIEWS!!!

-- Create STANDINGS view (regular season only):
CREATE VIEW standings AS
SELECT 
    seasons.year,
    m.manager,
    COUNT(game_id) AS games_played,
    SUM(CASE WHEN s.manager_id = s.winner_manager_id THEN 1 ELSE 0 END) AS games_won,
    ROUND(SUM(points_scored)::numeric,1) AS total_points,
    ROUND(AVG(points_scored)::numeric,1) AS ppg,
    ROUND(SUM(points_allowed)::numeric,1) AS total_points_allowed
FROM (
    -- Union query to normalize manager roles
    SELECT 
        season_id,
        manager_id1 AS manager_id,
        points1 AS points_scored,
        points2 AS points_allowed,
        winner_manager_id,
        game_id
    FROM games
    WHERE playoffs = 'no'
    UNION ALL
    SELECT 
        season_id,
        manager_id2 AS manager_id,
        points2 AS points_scored,
        points1 AS points_allowed,
        winner_manager_id,
        game_id
    FROM games
    WHERE playoffs = 'no'
) s
LEFT JOIN managers m
    ON s.manager_id = m.manager_id
LEFT JOIN seasons
    ON s.season_id = seasons.season_id
GROUP BY seasons.year, s.manager_id, m.manager
ORDER BY seasons.year, games_won DESC;




-- CREATE VIEW 

CREATE VIEW turnover AS (

with weekly_changes as (
    select manager_id, w.week, s.year, player_id, selected_position AS current_pos,
    lag(selected_position) over(partition by manager_id, player_id order by week) AS last_pos
    from rosters r
    left join weeks w
        on r.week_id = w.week_id
    left join seasons s
        on w.season_id = s.season_id
),
turnover as (
    select manager_id, year, week, sum(case when current_pos = last_pos then 1 else 0 end) as weekly_turnover
    from weekly_changes
    where current_pos != 'BN' and last_pos != 'BN'
    group by manager_id, week, year
)

select manager, week, year, weekly_turnover
    from turnover
    left join managers m
        on turnover.manager_id = m.manager_id
    order by weekly_turnover desc

);

-- Compare roster turnover (bn/active moves) to number of seasons played
with manager_seasons as (
select manager, count(*) as num_seasons from standings group by manager
)
select t.manager, SUM(weekly_turnover) turnover, MAX(num_seasons) seasons
from turnover t
left join manager_seasons m
    on t.manager = m.manager
group by t.manager
order by turnover desc
;


WITH rost AS (
    SELECT manager_id, player_id, selected_position AS position, w.week, s.year
    FROM rosters r
    LEFT JOIN weeks w
        ON r.week_id = w.week_id
    LEFT JOIN seasons s
        ON s.season_id = w.season_id
),
PreviousWeekRoster AS (
    SELECT 
        r1.manager_id,
        r1.week AS current_week,
        r2.week AS previous_week,
        r1.player_id,
        r1.position
    FROM 
        rost r1
    LEFT JOIN 
        rost r2
    ON 
        r1.manager_id = r2.manager_id 
        AND r1.player_id = r2.player_id 
        AND r1.week = r2.week + 1
),
Turnover AS (
    SELECT 
        manager_id,
        current_week,
        COUNT(CASE WHEN previous_week IS NULL THEN 1 END) AS players_added,
        COUNT(CASE WHEN current_week IS NULL THEN 1 END) AS players_removed
    FROM 
        PreviousWeekRoster
    GROUP BY 
        manager_id, current_week
)
SELECT 
    manager_id,
    current_week,
    players_added,
    players_removed,
    (players_added + players_removed) AS total_turnover
FROM 
    Turnover
ORDER BY 
    manager_id, current_week;



-- CREATE VIEW drafts AS
-- ...
-- ;

-- Get database table sizes for comparing dynamic vs static database structure

-- before
SELECT pg_size_pretty( pg_total_relation_size('rosters') ); -- 3240 kB
SELECT pg_size_pretty( pg_total_relation_size('transactions') ); -- 776 kB
SELECT pg_size_pretty( pg_total_relation_size('drafts') ); -- 248 kB

-- after
SELECT pg_size_pretty( pg_total_relation_size('events') ); -- 2960 kB

-- 100 * ((4264 - 2960) / 4264) # 30.6% reduction in size

-- 
SELECT e.player_id
FROM events e
WHERE e.year = 2007 and e.week = 1 and e.player_id = 6770
  AND e.type = 'active' -- Player was active in week 1
  AND NOT EXISTS (         -- Exclude players who became inactive between week 1 and week 4
      SELECT 1
      FROM events e_other
      WHERE e_other.player_id = e.player_id
        AND e_other.week > 1 AND e_other.week <= 1
        -- AND e_other.status != 'active'
  );


-- still trying to get rosters here...
drop view test;

create view test as
    select * from events
    where year = 2007 and (source_mid = 2 or destination_mid = 2) and week > 0;


-- Create ROSTERS view:
WITH status_intervals AS (
    -- Step 1: Calculate intervals of activity based on status changes
    SELECT 
        player_id,
        year,
        week AS start_week,
        source_mid,
        destination_mid,
        selected_position AS pos,
        timestamp,
        CASE 
            WHEN LEAD(week, 1) OVER (PARTITION BY year, player_id ORDER BY timestamp) IS NULL THEN week
            WHEN LEAD(week, 1) OVER (PARTITION BY year, player_id ORDER BY timestamp) > week THEN 
                LEAD(week, 1) OVER (PARTITION BY year, player_id ORDER BY timestamp) - 1
            ELSE week
        END AS end_week,
        -- LEAD(week, 1, NULL) OVER (PARTITION BY year, player_id ORDER BY timestamp) - 1 AS end_week,
        type
    FROM events
    WHERE week > 0 order by timestamp;
    -- uncomment to only show the SELECT in this first CTE
    -- WHERE week > 0 order by player_id, timestamp;
),
active_periods AS (
    -- Step 2: Filter only active intervals
    SELECT 
        player_id,
        year,
        source_mid,
        destination_mid,
        start_week,
        COALESCE(end_week, (SELECT MAX(week) FROM events)) AS end_week,
        type,
        pos
    FROM status_intervals
    WHERE type = 'active'
),
roster AS (
    -- Step 3: Generate all active weeks for each player
    SELECT 
        a.player_id,
        a.source_mid,
        a.destination_mid,
        a.year,
        w.week,
        a.type,
        a.pos
    FROM active_periods a
    JOIN (SELECT DISTINCT week FROM events) w
      ON w.week BETWEEN a.start_week AND a.end_week
)
-- Final Step: Output the player roster by week
SELECT
    year, week, source_mid, COUNT(player_id) AS num_players--, pos
FROM roster
GROUP BY year, source_mid, week
ORDER BY year, week, source_mid;





--  2008 |    7 |          5 |           6
select * from events where year = 2008 and (source_mid = 5 or destination_mid = 5) order by timestamp;

create view t1 AS (select *,
     CASE 
            WHEN LEAD(week, 1) OVER (PARTITION BY year, player_id ORDER BY timestamp) IS NULL THEN week
            WHEN LEAD(week, 1) OVER (PARTITION BY year, player_id ORDER BY timestamp) > week THEN 
                LEAD(week, 1) OVER (PARTITION BY year, player_id ORDER BY timestamp) - 1
            ELSE week
        END AS end_week
FROM events
WHERE week > 0
ORDER BY timestamp
);

create view t2 AS (select *, COALESCE(end_week, (SELECT MAX(week) FROM events)) AS end_week2 from t1 where type = 'active');

create view t3 as (
SELECT 
    t2.year,
    w.week,
    t2.player_id,
    t2.source_mid AS manager,
    t2.selected_position
    FROM t2
    JOIN (SELECT DISTINCT week FROM events) w
      ON w.week BETWEEN t2.week AND t2.end_week2
    -- WHERE t2.player_id = 100030 and year = 2008
    ORDER BY t2.year, w.week
);

-- mgr 6 2007 week 11 is problematic (11 players on active roster)

-- pid = 6781
select * from t2 where player_id = 6781 and year = 2007;
