-- looking at how a player stays on a team

with rownums AS (
    select year, week, manager, player,
    ROW_NUMBER() OVER (PARTITION BY manager, player ORDER BY year, week)
from slots
)

-- select * from rownums limit 10;

SELECT year, manager, week, COUNT(player)
FROM rownums
WHERE week = row_number
GROUP BY year, manager, week
ORDER BY year, manager, week;


WITH week1 AS (
    SELECT year, manager_id, player_id
    FROM rosters
    WHERE week = 1 and selected_position not in ('BN','IR')
),
week13 AS (
    SELECT year, manager_id, player_id
    FROM rosters
    WHERE week = 13 and selected_position not in ('BN','IR')
)

SELECT manager, w1.year, COUNT(w1.player_id) AS players_remaining
FROM week1 w1
INNER JOIN week13 w13
    ON w1.player_id = w13.player_id AND
    w1.manager_id = w13.manager_id AND
    w1.year = w13.year
LEFT JOIN managers m
    ON w1.manager_id = m.manager_id
GROUP BY w1.year, m.manager
ORDER BY players_remaining ASC
;
