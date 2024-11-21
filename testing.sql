WITH FreeAgentAdds AS (
    SELECT 
        transaction_id,         -- Unique ID for the addition
        destination AS manager_id,     -- ID of the manager who added the player
        player_id,      -- ID of the player added
        timestamp,  -- Timestamp of the addition
        w.week AS add_week -- see below how we join
    FROM 
        transactions t
    JOIN
        weeks w
    ON
        t.timestamp BETWEEN w.start AND w.end
    WHERE
        type = 'add'
),
RosterStreak AS (
    SELECT 
        r.manager_id,
        r.player_id,
        r.week,
        SUM(CASE WHEN r.player_id IS NOT NULL THEN 1 ELSE 0 END) OVER (
            PARTITION BY r.manager_id, r.player_id
            ORDER BY r.week
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS streak_count
    FROM 
        rosters r
    WHERE 
        r.player_id IN (SELECT player_id FROM FreeAgentAdds)
),
ConsecutiveWeeks AS (
    SELECT 
        fa.manager_id,
        fa.player_id,
        fa.add_week,
        MIN(r.week) AS start_week,
        MAX(r.week) AS end_week
    FROM 
        FreeAgentAdds fa
    JOIN 
        RosterStreak r
    ON 
        fa.manager_id = r.manager_id 
        AND fa.player_id = r.player_id 
        AND r.week >= fa.add_week
    GROUP BY 
        fa.manager_id, fa.player_id, fa.add_week
)
select * from ConsecutiveWeeks limit 10;

PlayerFantasyPoints AS (
    SELECT 
        manager_id,
        player_id,
        SUM(total_points) AS total_points
        w.week
    FROM 
        stats s
    LEFT JOIN
        weeks w
    ON

    JOIN 
        ConsecutiveWeeks cw
    ON 
        s.player_id = cw.player_id 
        AND s.week BETWEEN cw.start_week AND cw.end_week
    GROUP BY 
        manager_id, player_id
)
SELECT 
    pfp.manager_id,
    pfp.player_id,
    pfp.total_points,
    cw.start_week,
    cw.end_week
FROM 
    PlayerFantasyPoints pfp
JOIN 
    ConsecutiveWeeks cw
ON 
    pfp.manager_id = cw.manager_id 
    AND pfp.player_id = cw.player_id
ORDER BY 
    pfp.total_points DESC
LIMIT 10; -- Top 10 pickups based on total fantasy points
