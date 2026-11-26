# Rollout checkpoint compatibility fixture

Generated with Marin commit `f2c0a3f847727d54e7805b238c9c6524999a2daf`
using three synthetic task entries in one family. The original selector drew
`task-0` and `task-1` in order and queued `task-1` for regeneration. The
original buffer admitted `task-0` and saved its in-memory group in the
serialized rollout state. The same group was written with the original
object-store payload writer.

The restored buffer can train on `task-0`. The selector offers the saved
retry `task-1` next, followed by `task-2` from the saved family cursor.
