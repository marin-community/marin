-- Run as a database administrator after creating the native LOGIN roles out of band.
-- No passwords belong in this file or in Pulumi state.
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
GRANT CONNECT ON DATABASE inferencex TO inferencex_writer, inferencex_reader;
GRANT USAGE, CREATE ON SCHEMA public TO inferencex_writer;
GRANT USAGE ON SCHEMA public TO inferencex_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO inferencex_reader;
ALTER DEFAULT PRIVILEGES FOR ROLE inferencex_writer IN SCHEMA public
  GRANT SELECT ON TABLES TO inferencex_reader;
