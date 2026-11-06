-- Additive Revision 3 storage. Legacy scientific tables remain unchanged.
CREATE TABLE ap_runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE ap_jobs (
    job_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES ap_runs(run_id),
    spec BLOB NOT NULL,
    semantic_digest TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending','running','succeeded','failed','cancelled')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE ap_attempts (
    job_id TEXT NOT NULL REFERENCES ap_jobs(job_id),
    attempt INTEGER NOT NULL CHECK(attempt > 0),
    spec BLOB NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending','running','succeeded','failed','cancelled')),
    result BLOB,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(job_id, attempt)
);
CREATE TABLE ap_artifacts (
    artifact_id TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    manifest BLOB NOT NULL
);
CREATE TABLE ap_states (
    state_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES ap_runs(run_id),
    parent_id TEXT REFERENCES ap_states(state_id),
    job_id TEXT NOT NULL,
    global_step INTEGER NOT NULL CHECK(global_step >= 0),
    manifest BLOB NOT NULL
);
CREATE INDEX ap_states_run ON ap_states(run_id);
CREATE TABLE ap_records (
    run_id TEXT NOT NULL REFERENCES ap_runs(run_id),
    namespace TEXT NOT NULL,
    key TEXT NOT NULL,
    record BLOB NOT NULL,
    PRIMARY KEY(run_id, namespace, key)
);
CREATE TABLE ap_pointers (
    run_id TEXT NOT NULL REFERENCES ap_runs(run_id),
    name TEXT NOT NULL,
    state_id TEXT NOT NULL REFERENCES ap_states(state_id),
    PRIMARY KEY(run_id, name)
);
CREATE TABLE ap_budget_limits (
    run_id TEXT NOT NULL REFERENCES ap_runs(run_id),
    name TEXT NOT NULL,
    limit_value INTEGER NOT NULL CHECK(limit_value >= 0),
    used INTEGER NOT NULL DEFAULT 0 CHECK(used >= 0 AND used <= limit_value),
    PRIMARY KEY(run_id, name)
);
CREATE TABLE ap_budget_charges (
    run_id TEXT NOT NULL,
    budget TEXT NOT NULL,
    item_id TEXT NOT NULL,
    job_id TEXT NOT NULL REFERENCES ap_jobs(job_id),
    units INTEGER NOT NULL CHECK(units > 0),
    PRIMARY KEY(run_id, budget, item_id),
    FOREIGN KEY(run_id, budget) REFERENCES ap_budget_limits(run_id, name)
);
CREATE TABLE ap_acceptances (
    job_id TEXT PRIMARY KEY REFERENCES ap_jobs(job_id),
    result_digest TEXT NOT NULL,
    effects_digest TEXT NOT NULL,
    result BLOB NOT NULL,
    effects BLOB NOT NULL,
    accepted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE ap_quarantine (
    quarantine_id INTEGER PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES ap_jobs(job_id),
    attempt INTEGER NOT NULL,
    reason TEXT NOT NULL,
    result BLOB NOT NULL,
    effects BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE ap_events (
    event_id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES ap_runs(run_id),
    job_id TEXT,
    kind TEXT NOT NULL,
    payload BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX ap_events_run ON ap_events(run_id, event_id);
-- Immutable history is append-only even if a backend accidentally issues UPDATE.
CREATE TRIGGER ap_artifacts_immutable_update BEFORE UPDATE ON ap_artifacts
BEGIN SELECT RAISE(ABORT, 'immutable artifact'); END;
CREATE TRIGGER ap_artifacts_immutable_delete BEFORE DELETE ON ap_artifacts
BEGIN SELECT RAISE(ABORT, 'immutable artifact'); END;
CREATE TRIGGER ap_states_immutable_update BEFORE UPDATE ON ap_states
BEGIN SELECT RAISE(ABORT, 'immutable state'); END;
CREATE TRIGGER ap_states_immutable_delete BEFORE DELETE ON ap_states
BEGIN SELECT RAISE(ABORT, 'immutable state'); END;
CREATE TRIGGER ap_acceptances_immutable_update BEFORE UPDATE ON ap_acceptances
BEGIN SELECT RAISE(ABORT, 'immutable acceptance'); END;
CREATE TRIGGER ap_acceptances_immutable_delete BEFORE DELETE ON ap_acceptances
BEGIN SELECT RAISE(ABORT, 'immutable acceptance'); END;
CREATE TRIGGER ap_records_immutable_update BEFORE UPDATE ON ap_records
BEGIN SELECT RAISE(ABORT, 'immutable record'); END;
CREATE TRIGGER ap_records_immutable_delete BEFORE DELETE ON ap_records
BEGIN SELECT RAISE(ABORT, 'immutable record'); END;
