-- Additive Task/data migration. Legacy tasks and scientific rows remain intact.
ALTER TABLE ap_runs ADD COLUMN fixture_only INTEGER NOT NULL DEFAULT 0 CHECK(fixture_only IN (0,1));
CREATE TABLE ap_datasets (
  dataset_id TEXT PRIMARY KEY,
  sha256 TEXT NOT NULL,
  split TEXT NOT NULL CHECK(split IN ('train','dev','pilot','screen','gate_a','gate_a_reservoir','gate_b','gate_b_reservoir','transfer_guard','hidden_dev')),
  fixture_only INTEGER NOT NULL CHECK(fixture_only IN (0,1)),
  manifest BLOB NOT NULL,
  artifact_ref BLOB NOT NULL,
  audit_ref BLOB NOT NULL,
  UNIQUE(dataset_id,split)
);
CREATE TABLE ap_dataset_tasks (
  dataset_id TEXT NOT NULL,
  task_id TEXT NOT NULL,
  cluster_id TEXT NOT NULL,
  source_id TEXT NOT NULL,
  source_ver TEXT NOT NULL,
  source_item_id TEXT NOT NULL,
  source_role TEXT NOT NULL CHECK(source_role IN ('train','transfer','benchmark','hidden')),
  answer_type TEXT NOT NULL CHECK(answer_type IN ('int','rational','decimal','expr','interval','tuple')),
  measurable INTEGER NOT NULL CHECK(measurable IN (0,1)),
  split TEXT NOT NULL,
  task_schema_version INTEGER NOT NULL CHECK(task_schema_version=1),
  record BLOB NOT NULL,
  PRIMARY KEY(dataset_id,task_id),
  FOREIGN KEY(dataset_id,split) REFERENCES ap_datasets(dataset_id,split),
  CHECK(measurable = (answer_type IN ('int','rational','decimal'))),
  CHECK(split IN ('train','dev') OR measurable=1),
  CHECK(split NOT IN ('train','dev') OR source_role='train')
);
CREATE INDEX ap_data_cluster ON ap_dataset_tasks(cluster_id,split);
CREATE TABLE ap_task_keys (
  dataset_id TEXT NOT NULL,
  task_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  value TEXT NOT NULL,
  PRIMARY KEY(dataset_id,task_id,kind,value),
  FOREIGN KEY(dataset_id,task_id) REFERENCES ap_dataset_tasks(dataset_id,task_id)
);
CREATE TABLE ap_measurement_blocks (
  kind TEXT NOT NULL,
  value TEXT NOT NULL,
  PRIMARY KEY(kind,value)
);
CREATE TABLE ap_hidden_reviews (
  program_digest TEXT NOT NULL,
  report_digest TEXT NOT NULL,
  record BLOB NOT NULL,
  PRIMARY KEY(program_digest,report_digest)
);
CREATE TABLE ap_data_catalog_checks (build_id TEXT PRIMARY KEY, report BLOB NOT NULL);
CREATE TRIGGER ap_catalog_checks_immutable_update BEFORE UPDATE ON ap_data_catalog_checks BEGIN SELECT RAISE(ABORT,'immutable catalog check'); END;
CREATE TRIGGER ap_catalog_checks_immutable_delete BEFORE DELETE ON ap_data_catalog_checks BEGIN SELECT RAISE(ABORT,'immutable catalog check'); END;
CREATE TRIGGER ap_datasets_immutable_update BEFORE UPDATE ON ap_datasets BEGIN SELECT RAISE(ABORT,'immutable dataset'); END;
CREATE TRIGGER ap_datasets_immutable_delete BEFORE DELETE ON ap_datasets BEGIN SELECT RAISE(ABORT,'immutable dataset'); END;
CREATE TRIGGER ap_dataset_tasks_immutable_update BEFORE UPDATE ON ap_dataset_tasks BEGIN SELECT RAISE(ABORT,'immutable task'); END;
CREATE TRIGGER ap_dataset_tasks_immutable_delete BEFORE DELETE ON ap_dataset_tasks BEGIN SELECT RAISE(ABORT,'immutable task'); END;
CREATE TRIGGER ap_task_keys_immutable_update BEFORE UPDATE ON ap_task_keys BEGIN SELECT RAISE(ABORT,'immutable task provenance'); END;
CREATE TRIGGER ap_task_keys_immutable_delete BEFORE DELETE ON ap_task_keys BEGIN SELECT RAISE(ABORT,'immutable task provenance'); END;
CREATE TRIGGER ap_measurement_blocks_immutable_update BEFORE UPDATE ON ap_measurement_blocks BEGIN SELECT RAISE(ABORT,'immutable measurement provenance'); END;
CREATE TRIGGER ap_measurement_blocks_immutable_delete BEFORE DELETE ON ap_measurement_blocks BEGIN SELECT RAISE(ABORT,'immutable measurement provenance'); END;
CREATE TRIGGER ap_hidden_reviews_immutable_update BEFORE UPDATE ON ap_hidden_reviews BEGIN SELECT RAISE(ABORT,'immutable review'); END;
CREATE TRIGGER ap_hidden_reviews_immutable_delete BEFORE DELETE ON ap_hidden_reviews BEGIN SELECT RAISE(ABORT,'immutable review'); END;
