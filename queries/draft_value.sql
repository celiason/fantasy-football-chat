-- rank drafts by when they were picked (earlier = lower numbers)
WITH draft_ranks AS(
  SELECT e.year,
    timestamp,
    p.player,
    p.player_id,
    p.position,
    ROW_NUMBER() OVER (PARTITION BY year, p.position ORDER BY timestamp) AS position_pick,
    ROW_NUMBER() OVER (PARTITION BY year ORDER BY timestamp) AS overall_pick
  FROM events e
  LEFT JOIN players p
    ON e.player_id = p.player_id
  WHERE type = 'draft'
  ORDER BY year, overall_pick, position
),

-- sum up total points by year and player
season_points AS (
  SELECT player, p.player_id, position, year, SUM(total_points) AS season_pts
  FROM players p
  LEFT JOIN stats s
  ON s.player_id = p.player_id
  GROUP BY year, p.player_id, player, position
  ORDER BY season_pts DESC
),

-- create in-season ranks based on total points by position
points_rank AS (
  SELECT year, season_pts, position, player, ROW_NUMBER() OVER (PARTITION BY year, position ORDER BY season_pts DESC) AS season_rank
  FROM season_points
),

-- combine
draft_season_combined AS(
  SELECT p.year, p.player, p.position, p.season_rank, d.position_pick, d.overall_pick, p.season_pts
  FROM points_rank p 
  LEFT JOIN draft_ranks d
    ON p.player = d.player AND p.position = d.position AND p.year = d.year
  -- WHERE p.position = 'RB' and p.year = 2018 limit 10;
)

-- value = in-season rank MINUS position rank

SELECT year, player, position, season_rank, season_pts, position_pick, overall_pick, position_pick - season_rank AS draft_value
FROM draft_season_combined
-- WHERE position = 'WR' AND year = 2010
-- ORDER BY season_pts DESC
-- WHERE year = 2018
-- LIMIT 250;
;
