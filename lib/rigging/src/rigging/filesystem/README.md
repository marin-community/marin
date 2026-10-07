# Filesystem components

[storage_path.py](storage_path.py) provides parsed storage locations and bounded
I/O operations. [factory.py](factory.py) resolves guarded filesystem access;
the other modules provide atomic writes, locks, listings, hashing, transfer and
mirror support. Use the [fsutil reference](../../../../../docs/references/fsutil.md)
for object-storage operations from a checkout.

[path_validation.py](path_validation.py) validates relative POSIX file paths and
mount collections without storage access. It preserves legal Linux names,
including colons, backslashes, trailing spaces and case distinctions. It rejects
absolute paths, empty or traversal segments, NUL bytes, duplicate files and
file/directory ancestor collisions. Task resources use this shared validator;
dataset conversion and grading policy belong to their callers.
