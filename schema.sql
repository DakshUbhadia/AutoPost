-- autopost/src/autopost/store/schema.sql   (Revision 2)
-- Single SQLite file (runs/autopost.sqlite), WAL mode. Source of truth.
-- Rev.2: verifier versions on all derived state; raw-completion index;
-- read budgets and human-change budget; segments + gate decisions replace
-- fixed cycles; global (not lineage-scoped) pass-rate/exposure accounting.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ---------- pre-registration ----------
CREATE TABLE prereg (
  version     TEXT PRIMARY KEY,         -- e.g. 'stageB-v1'
  sha256      TEXT NOT NULL,            -- hash of prereg.json
  frozen_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  superseded_by TEXT                    -- a change requires a new version, logged
);

-- ---------- data ----------
CREATE TABLE tasks (
  task_id      TEXT PRIMARY KEY,
  cluster_id   TEXT NOT NULL,            -- dedup cluster, or perturbation program id
  source_id    TEXT NOT NULL,
  source_ver   TEXT NOT NULL,
  skill        TEXT NOT NULL,
  answer_type  TEXT NOT NULL,
  measurable   INTEGER NOT NULL,         -- 1 = integer/rational/terminating decimal
  low_entropy  INTEGER NOT NULL DEFAULT 0,
  split        TEXT NOT NULL CHECK (split IN (
                 'train','dev','pilot','screen','gate_a','gate_a_reservoir','gate_b',
                 'gate_b_reservoir','transfer_guard','transfer_sealed',
                 'hidden_audit','hidden_sealed')),
  strat_base_passrate REAL,              -- from samples never used in comparisons
  problem      TEXT NOT NULL,
  answer       TEXT NOT NULL,
  CHECK (split IN ('train','dev') OR measurable = 1)
);
CREATE INDEX tasks_split ON tasks(split, source_id, skill);
CREATE INDEX tasks_cluster ON tasks(cluster_id);

CREATE TABLE ingest_rejections (key TEXT, stage TEXT, reason TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP);

CREATE TABLE passrate_obs (                -- global accounting in V1
  task_id          TEXT NOT NULL REFERENCES tasks(task_id),
  state_id         TEXT NOT NULL,
  n                INTEGER NOT NULL,
  n_correct        INTEGER NOT NULL,
  top_answer       TEXT,
  top_count        INTEGER,
  origin           TEXT NOT NULL CHECK (origin IN ('estimate','training_rollout')),
  verifier_version TEXT NOT NULL,          -- rows from a buggy version can be excluded
  created_at       TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX passrate_task ON passrate_obs(task_id, verifier_version);

CREATE TABLE exposures (task_id TEXT NOT NULL, cluster_id TEXT NOT NULL,
  n_groups INTEGER NOT NULL, state_id TEXT NOT NULL);
CREATE INDEX exposures_cluster ON exposures(cluster_id);

CREATE TABLE quarantine (                  -- exclusion: consensus against label only
  task_id          TEXT PRIMARY KEY,
  reason           TEXT NOT NULL CHECK (reason = 'consensus_disagrees_label'),
  verifier_version TEXT NOT NULL,
  status           TEXT NOT NULL DEFAULT 'pending'
                   CHECK (status IN ('pending','label_ok','label_fixed','dropped')),
  created_at       TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE audit_flags (                 -- informational; does NOT exclude
  task_id TEXT NOT NULL, flag TEXT NOT NULL,   -- e.g. 'persistent_inband'
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- ---------- raw completions (Parquet files; this table indexes them) ----------
CREATE TABLE completion_files (
  path             TEXT PRIMARY KEY,       -- runs/completions/<kind>/<...>.parquet
  kind             TEXT NOT NULL CHECK (kind IN ('rollout','eval','estimate','offline')),
  state_id         TEXT NOT NULL,
  split            TEXT,                   -- for eval reads
  suite_version    TEXT,
  seeds            TEXT,                   -- JSON
  checker_version  TEXT NOT NULL,          -- verifier (rollouts) or scorer (eval) at write time
  stack_hash       TEXT NOT NULL,
  n_rows           INTEGER NOT NULL,
  created_at       TEXT DEFAULT CURRENT_TIMESTAMP
);

-- ---------- lineage ----------
CREATE TABLE states (
  state_id         TEXT PRIMARY KEY,
  parent_id        TEXT,
  path             TEXT NOT NULL,
  global_step      INTEGER NOT NULL,
  trainer_id       TEXT NOT NULL,
  config_hash      TEXT NOT NULL,
  stack_hash       TEXT NOT NULL,
  verifier_version TEXT NOT NULL,
  status           TEXT NOT NULL CHECK (status IN
                   ('base','checkpoint','champion','promoted','rejected','pruned')),
  created_at       TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE champion_pointer (id INTEGER PRIMARY KEY CHECK (id = 1),
  state_id TEXT NOT NULL REFERENCES states(state_id));

-- ---------- training segments and gate decisions ----------
CREATE TABLE segments (                    -- N steps between checkpoints
  segment_id   TEXT PRIMARY KEY,
  start_state  TEXT NOT NULL,
  end_state    TEXT,
  steps        INTEGER NOT NULL,
  slice_hash   TEXT NOT NULL,
  phase        TEXT NOT NULL CHECK (phase IN ('slice','train','monitor','done','failed')),
  monitor_outcome TEXT,                    -- 'ok' | 'restore'
  gpu_seconds_train REAL, gpu_seconds_monitor REAL,
  started_at TEXT, finished_at TEXT
);

CREATE TABLE gate_decisions (
  decision_id      TEXT PRIMARY KEY,
  candidate_state  TEXT NOT NULL,
  champion_state   TEXT NOT NULL,
  steps_since_promotion INTEGER NOT NULL,
  outcome          TEXT NOT NULL CHECK (outcome IN ('promote','continue','restore','halt')),
  reasons          TEXT NOT NULL,          -- JSON
  evidence         TEXT NOT NULL,          -- JSON (BootResults)
  gpu_seconds_eval REAL,
  created_at       TEXT DEFAULT CURRENT_TIMESTAMP
);

-- ---------- looks, budgets, human changes ----------
CREATE TABLE looks (split TEXT NOT NULL, suite_version TEXT NOT NULL,
  state_ids TEXT NOT NULL, decision_id TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE INDEX looks_split ON looks(split, suite_version);

CREATE TABLE budgets (                     -- pre-registered; decremented by ledger
  name        TEXT NOT NULL,               -- gate_a | gate_b | hidden_audit | hidden_sealed |
                                           -- transfer_sealed | human_changes | step_budget
  version     TEXT NOT NULL,               -- set version or prereg version
  limit_value INTEGER NOT NULL,
  used        INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (name, version)
);

CREATE TABLE human_changes (               -- every config change after a halt
  change_id   TEXT PRIMARY KEY,
  description TEXT NOT NULL,
  informed_by TEXT NOT NULL,               -- JSON: decision ids / metrics consulted
  forced_gate_a_refresh INTEGER NOT NULL DEFAULT 1,
  created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE eval_results (                -- aggregates; per-sample data in Parquet
  state_id TEXT, ref_state_id TEXT, split TEXT, suite_version TEXT, metric TEXT,
  mean REAL, lower REAL, upper REAL, alpha REAL, n_items INTEGER, n_clusters INTEGER,
  k INTEGER, k_ref INTEGER, fresh_reference INTEGER, stack_hash TEXT,
  scorer_version TEXT, decision_id TEXT
);

CREATE TABLE events (kind TEXT NOT NULL, payload TEXT NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP);

CREATE TABLE controller_state (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  status TEXT NOT NULL CHECK (status IN ('RUNNING','HALTED','MILESTONE_REVIEW')),
  reason TEXT,
  steps_since_promotion INTEGER NOT NULL DEFAULT 0,
  consecutive_restores INTEGER NOT NULL DEFAULT 0,
  promotions_since_milestone INTEGER NOT NULL DEFAULT 0
);

-- ---------- Phase I experiment results ----------
CREATE TABLE phase1_results (
  experiment TEXT NOT NULL,                -- E1..E9
  metric     TEXT NOT NULL,
  value      REAL,
  lower      REAL,
  upper      REAL,
  detail     TEXT,                         -- JSON
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
