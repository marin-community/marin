-- Periodic profiling storage: profiles table with ring buffer retention.

CREATE TABLE IF NOT EXISTS profiles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    target_id       TEXT    NOT NULL,
    profile_type    TEXT    NOT NULL,
    data            BLOB    NOT NULL,
    captured_at_ms  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_profiles_target_type_time
    ON profiles(target_id, profile_type, captured_at_ms DESC);

-- Ring buffer: keep last 10 per (target_id, profile_type)
CREATE TRIGGER IF NOT EXISTS trg_profiles_retention
AFTER INSERT ON profiles
BEGIN
    DELETE FROM profiles
    WHERE target_id = NEW.target_id
      AND profile_type = NEW.profile_type
      AND id NOT IN (
        SELECT id FROM profiles
        WHERE target_id = NEW.target_id
          AND profile_type = NEW.profile_type
        ORDER BY id DESC
        LIMIT 10
      );
END;
