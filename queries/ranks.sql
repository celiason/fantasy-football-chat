-- Create a table with total season points by year, position, and player
WITH season_pts AS (
    SELECT
        p.position,
        p.player_id,
        p.player,
        year,
        ROUND(SUM(total_points)::numeric, 1) points
    FROM stats s
    LEFT JOIN players p
        ON s.player_id = p.player_id
    GROUP BY p.player_id, year, p.player, p.position
    -- ORDER BY year ASC
)

-- Now we can combine the draft and points ranking tables into a neat summary
-- Show ranks of draft and points by player
SELECT
    s.player,
    s.year,
    s.position,
    s.points,
    RANK() OVER (PARTITION BY s.year, s.position ORDER BY s.points DESC) AS points_rank,
    pick
FROM season_pts s
INNER JOIN drafts d
    ON s.player_id = d.player_id AND s.year = d.year
ORDER BY pick ASC
;
