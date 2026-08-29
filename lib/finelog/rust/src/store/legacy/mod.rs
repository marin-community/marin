//! Mechanics that exist only for tables predating object-native storage.
//!
//! A legacy table keeps its rows in local files under its own directory, copies
//! compacted segments to a flat remote key, and evicts local copies once the
//! archive holds them. Placement and encoding are properties of the directory,
//! so both need background convergence when the writer policy changes.
//!
//! An object-backed table has none of this: its segments are immutable objects
//! named by content, its cache is the object store's concern, and its placement
//! is not a path. Every module here retires when the last version-0 table has
//! been imported.

pub mod archive;
pub mod layout;
