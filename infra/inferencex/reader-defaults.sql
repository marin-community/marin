-- Run as inferencex_writer before migrations. No administrative membership is needed.
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT SELECT ON TABLES TO inferencex_reader;
