-- Run as a database administrator after creating the native LOGIN roles out of band.
-- No passwords belong in this file or in Pulumi state.
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
GRANT CONNECT ON DATABASE knotmeter TO knotmeter_writer, knotmeter_reader;
GRANT USAGE, CREATE ON SCHEMA public TO knotmeter_writer;
GRANT USAGE ON SCHEMA public TO knotmeter_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO knotmeter_reader;
