-- Create a table with total season points by year, position, and player
COPY ( WITH season_pts AS (
    SELECT p.position, p.player_id, p.name, w.year,
            ROUND(SUM(total_points)::numeric, 1) points
    FROM stats s
    LEFT JOIN players p
        ON s.player_id = p.player_id
    LEFT JOIN weeks w
        ON s.week_id = w.week_id
    GROUP BY p.player_id, w.year, p.name, p.position
    -- ORDER BY year ASC
)

-- COPY (select * from draft_ranks
-- limit 250) TO STDOUT;

-- Now we can combine the draft and points ranking tables into a neat summary
-- Show ranks of draft and points by player

    SELECT s.name,
           s.year,
           s.position,
           s.points,
           RANK() OVER (PARTITION BY s.year, s.position ORDER BY s.points DESC) AS points_rank,
    -- SELECT s.name, s.year, s.position, s.points, RANK() OVER (PARTITION BY s.year, position ORDER BY s.points DESC) AS points_rank,
        pick
    FROM season_pts s
    INNER JOIN drafts d
        ON s.player_id = d.player_id AND s.year = d.year
    WHERE s.year = 2016
    ORDER BY pick ASC
) TO '/Users/chad/github/fantasy_app/ranks.csv' WITH CSV DELIMITER ',' HEADER;


COPY (
    SELECT e.year, m.name,
            SUM(CASE WHEN source_mid = destination_mid THEN 1 ELSE 0 END) AS roster_moves,
            SUM(CASE WHEN type = 'add' THEN 1 ELSE 0 END) AS adds,
            r.games_won, r.total_points
    FROM events e
    LEFT JOIN managers m
        ON e.destination_mid = m.manager_id
    LEFT JOIN records r
        ON e.destination_mid = r.manager_id AND e.year = r.year
    WHERE e.week > 0
    GROUP BY e.year, m.name, r.games_won, r.total_points
    ORDER BY roster_moves DESC
) TO '/Users/chad/github/fantasy_app/wins_moves.csv' WITH CSV DELIMITER ',' HEADER;

