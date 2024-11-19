-- VIEWS!!!

CREATE VIEW records AS

SELECT 
    s.year,
    s.manager_id,
    m.name,
    COUNT(game_id) AS games_played,
    SUM(CASE WHEN s.manager_id = s.winner_manager_id THEN 1 ELSE 0 END) AS games_won,
    ROUND(SUM(points_scored)::numeric,1) AS total_points,
    ROUND(SUM(points_allowed)::numeric,1) AS total_points_allowed
FROM (
    -- Union query to normalize manager roles
    SELECT 
        year,
        manager_id1 AS manager_id,
        points1 AS points_scored,
        points2 AS points_allowed,
        winner_manager_id,
        game_id
    FROM games
    UNION ALL
    SELECT 
        year,
        manager_id2 AS manager_id,
        points2 AS points_scored,
        points1 AS points_allowed,
        winner_manager_id,
        game_id
    FROM games
) s
LEFT JOIN managers m
    ON s.manager_id = m.manager_id
GROUP BY s.year, s.manager_id, m.name
ORDER BY s.year, games_won DESC;



CREATE VIEW drafts AS
...
;

