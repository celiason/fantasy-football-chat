-- Create STANDINGS view (regular season only):

CREATE VIEW standings AS
SELECT 
    s.year,
    m.manager,
    -- COUNT(game_id) AS games_played,
    SUM(CASE WHEN s.manager_id = s.winner_manager_id THEN 1 ELSE 0 END) AS wins,
    SUM(CASE WHEN s.manager_id != s.winner_manager_id THEN 1 ELSE 0 END) AS losses,
    ROUND(SUM(points_scored)::numeric,1) AS points_scored,
    ROUND(AVG(points_scored)::numeric,1) AS ppg,
    ROUND(SUM(points_allowed)::numeric,1) AS points_allowed
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
    WHERE playoffs = 'no'
    UNION ALL
    SELECT 
        year,
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
GROUP BY s.year, s.manager_id, m.manager
ORDER BY s.year ASC;

