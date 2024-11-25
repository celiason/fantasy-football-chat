WITH max_stints AS (
    SELECT year, MAX(timestamp) AS max_time
    FROM events
    GROUP BY year
),
cte AS (
    SELECT e.year, week,
    timestamp AS start,
    player_id AS pid,
    source_manager_id AS smid,
    destination_manager_id AS dmid,
    selected_position AS pos,
    LEAD(timestamp, 1, m.max_time) OVER (PARTITION BY e.year, player_id ORDER BY timestamp) AS finish,
    LEAD(type) OVER (PARTITION BY e.year, player_id ORDER BY timestamp) AS next_pos,
    type
    FROM events e
    LEFT JOIN max_stints m
        ON e.year = m.year
),
avg_stints AS (
    SELECT *, finish::timestamp - start::timestamp AS diff
    FROM cte
    WHERE type = 'active'
)
SELECT year, week, m.manager, pos, AVG(diff) stint
FROM avg_stints s
LEFT JOIN managers m
    ON m.manager_id = s.smid
GROUP BY year, week, m.manager, pos
ORDER BY stint DESC
;
