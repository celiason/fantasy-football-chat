CREATE VIEW slots AS(

WITH max_weeks AS (
       SELECT year, MAX(week) AS max_week
       FROM weeks
       GROUP BY year
       ORDER BY year ASC
),
rosters AS (
SELECT
       e.year,
       week AS first_week,
       player_id,
       source_manager_id AS manager_id,
       selected_position,
       type,
       CASE WHEN LEAD(week, 1) OVER (PARTITION BY e.year, player_id ORDER BY timestamp) - 1 IS NULL
              THEN max_week
            WHEN LEAD(week, 1) OVER (PARTITION BY e.year, player_id ORDER BY timestamp) - 1 > week
              THEN LEAD(week, 1) OVER (PARTITION BY e.year, player_id ORDER BY timestamp) - 1
            ELSE week
       END AS last_week
FROM events e
INNER JOIN max_weeks AS mw
       ON mw.year = e.year
WHERE e.week > 0
ORDER BY e.week
),
slots AS (
    SELECT DISTINCT player_id, r.year, type, manager_id, w.week, selected_position
        FROM rosters r
        JOIN weeks w
            ON w.week >= r.first_week AND w.week <= r.last_week
        WHERE r.type = 'active'
        ORDER BY r.year, r.manager_id, w.week
)
SELECT s.year, s.week, m.manager, p.player, selected_position AS position, stats.total_points AS points
FROM slots s
LEFT JOIN players p
    ON p.player_id = s.player_id
LEFT JOIN managers m
    ON m.manager_id = s.manager_id
LEFT JOIN stats
    ON s.year = stats.year AND s.week = stats.week AND stats.player_id = s.player_id
);
