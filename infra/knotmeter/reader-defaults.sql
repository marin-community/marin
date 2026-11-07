-- Run as knotmeter_writer before migrations. No administrative membership is needed.
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT SELECT ON TABLES TO knotmeter_reader;
