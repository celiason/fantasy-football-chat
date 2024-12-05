

CREATE TABLE matchups (
    matchup_id SERIAL PRIMARY KEY,  -- Unique identifier for each matchup
    team_key1 INT NOT NULL,         -- Foreign key for the first team
    team_key2 INT NOT NULL,         -- Foreign key for the second team
    matchup_date DATE NOT NULL,     -- Date of the matchup
    team1_score INT,                -- Score for team_key1 (optional, if scores are relevant)
    team2_score INT,                -- Score for team_key2 (optional)
    winner_team_key INT,            -- Foreign key for the winning team (if known)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- Record creation timestamp
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP, -- Record update timestamp
    CONSTRAINT chk_unique_teams CHECK (team_key1 <> team_key2), -- Ensures teams are different
    CONSTRAINT fk_team1 FOREIGN KEY (team_key1) REFERENCES teams(team_key), -- Reference to teams table
    CONSTRAINT fk_team2 FOREIGN KEY (team_key2) REFERENCES teams(team_key), -- Reference to teams table
    CONSTRAINT fk_winner FOREIGN KEY (winner_team_key) REFERENCES teams(team_key) -- Reference for winner
);

-- set primary key
ALTER TABLE managers
ADD CONSTRAINT managers_pkey PRIMARY KEY (manager_id);

-- add constraints
ALTER TABLE games
ADD CONSTRAINT fk_manager1 FOREIGN KEY (manager_id1)
    REFERENCES managers (manager_id);
ALTER TABLE games
ADD CONSTRAINT fk_manager2 FOREIGN KEY (manager_id2)
    REFERENCES managers (manager_id),
ADD CONSTRAINT fk_winner FOREIGN KEY (winner_manager_id)
    REFERENCES managers (manager_id);

-- set primary key
ALTER TABLE players
ADD CONSTRAINT players_pkey PRIMARY KEY (player_id);

-- add constraints
ALTER TABLE rosters
ADD CONSTRAINT fk_players FOREIGN KEY (player_id)
    REFERENCES players (player_id),
ADD CONSTRAINT fk_managers FOREIGN KEY (manager_id)
    REFERENCES managers (manager_id);

-- set primary key to weeks table
ALTER TABLE weeks
ADD CONSTRAINT weeks_pkey PRIMARY KEY (week_id);

-- add constraints
ALTER TABLE stats
ADD CONSTRAINT fk_players FOREIGN KEY (player_id)
    REFERENCES players (player_id),
ADD CONSTRAINT fk_weeks FOREIGN KEY (week_id)
    REFERENCES weeks (week_id)
;

-- set primary key to weeks table
ALTER TABLE transactions
ADD CONSTRAINT transactions_pkey PRIMARY KEY (tid);

-- add constraints
ALTER TABLE transactions
ADD CONSTRAINT fk_source FOREIGN KEY (source_mid)
    REFERENCES managers (manager_id),
ADD CONSTRAINT fk_destination FOREIGN KEY (destination_mid)
    REFERENCES managers (manager_id)
;
