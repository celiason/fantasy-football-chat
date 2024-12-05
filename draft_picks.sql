-- generate ranks
WITH picks AS (
  SELECT e.year,
    row_number() over (partition by e.year, destination_manager_id order by timestamp ASC) AS pick,
    row_number() over (partition by e.year order by timestamp ASC) AS overall_pick,
    player,
    position,
    m.manager,
    wins,
    rank,
    points_scored
  FROM events e
  LEFT JOIN players p
    ON e.player_id = p.player_id
  LEFT JOIN managers m
    ON e.destination_manager_id = m.manager_id
  LEFT JOIN standings s
    ON m.manager = s.manager and e.year = s.year
  WHERE type = 'draft'
  ORDER BY timestamp ASC
)

-- calculate stats
SELECT
  overall_pick,
  SUM(CASE WHEN rank = 1 THEN 1 ELSE 0 END) AS topdog,
  ROUND(AVG(wins),2) AS wins,
  ROUND(STDDEV(wins),2) AS wins_sd,
  ROUND(AVG(points_scored),2) AS points,
  ROUND(STDDEV(points_scored),2) AS points_sd
FROM picks
WHERE pick = 1
GROUP BY overall_pick
;
