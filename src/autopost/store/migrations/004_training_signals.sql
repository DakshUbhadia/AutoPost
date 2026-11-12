-- Training signals are separate from legacy science and job-result quarantine.
CREATE TABLE ap_passrate_groups (
  group_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  task_digest TEXT NOT NULL,
  checker_id TEXT NOT NULL,
  record BLOB NOT NULL
);
CREATE INDEX ap_passrate_lookup ON ap_passrate_groups(task_id,checker_id);
CREATE TABLE ap_training_samples (
  completion_id TEXT PRIMARY KEY,
  group_id TEXT NOT NULL REFERENCES ap_passrate_groups(group_id)
);
CREATE TABLE ap_label_reviews (
  sequence INTEGER PRIMARY KEY,
  review_id TEXT NOT NULL UNIQUE,
  task_id TEXT NOT NULL,
  task_digest TEXT NOT NULL,
  record BLOB NOT NULL
);
CREATE TABLE ap_curriculum_slices (slice_id TEXT PRIMARY KEY, record BLOB NOT NULL);
CREATE TABLE ap_cluster_exposures (
  slice_id TEXT NOT NULL REFERENCES ap_curriculum_slices(slice_id),
  task_id TEXT NOT NULL,
  cluster_id TEXT NOT NULL,
  PRIMARY KEY(slice_id,task_id)
);
CREATE INDEX ap_global_cluster_exposure ON ap_cluster_exposures(cluster_id);
CREATE TABLE ap_training_audit (
  sequence INTEGER PRIMARY KEY,
  event_id TEXT NOT NULL UNIQUE,
  task_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  record BLOB NOT NULL
);
CREATE TRIGGER ap_passrate_immutable_update BEFORE UPDATE ON ap_passrate_groups BEGIN SELECT RAISE(ABORT,'immutable pass-rate observation'); END;
CREATE TRIGGER ap_passrate_immutable_delete BEFORE DELETE ON ap_passrate_groups BEGIN SELECT RAISE(ABORT,'immutable pass-rate observation'); END;
CREATE TRIGGER ap_samples_immutable_update BEFORE UPDATE ON ap_training_samples BEGIN SELECT RAISE(ABORT,'immutable training sample'); END;
CREATE TRIGGER ap_samples_immutable_delete BEFORE DELETE ON ap_training_samples BEGIN SELECT RAISE(ABORT,'immutable training sample'); END;
CREATE TRIGGER ap_reviews_immutable_update BEFORE UPDATE ON ap_label_reviews BEGIN SELECT RAISE(ABORT,'immutable label review'); END;
CREATE TRIGGER ap_reviews_immutable_delete BEFORE DELETE ON ap_label_reviews BEGIN SELECT RAISE(ABORT,'immutable label review'); END;
CREATE TRIGGER ap_slices_immutable_update BEFORE UPDATE ON ap_curriculum_slices BEGIN SELECT RAISE(ABORT,'immutable slice'); END;
CREATE TRIGGER ap_slices_immutable_delete BEFORE DELETE ON ap_curriculum_slices BEGIN SELECT RAISE(ABORT,'immutable slice'); END;
CREATE TRIGGER ap_exposures_immutable_update BEFORE UPDATE ON ap_cluster_exposures BEGIN SELECT RAISE(ABORT,'immutable exposure'); END;
CREATE TRIGGER ap_exposures_immutable_delete BEFORE DELETE ON ap_cluster_exposures BEGIN SELECT RAISE(ABORT,'immutable exposure'); END;
CREATE TRIGGER ap_training_audit_immutable_update BEFORE UPDATE ON ap_training_audit BEGIN SELECT RAISE(ABORT,'immutable training audit'); END;
CREATE TRIGGER ap_training_audit_immutable_delete BEFORE DELETE ON ap_training_audit BEGIN SELECT RAISE(ABORT,'immutable training audit'); END;
