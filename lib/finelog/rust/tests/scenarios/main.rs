// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! Scenario tests, one failure case per module.
//!
//! Each module drives a complete [`finelog::store::Store`] — real table
//! manager, controllers, compaction, and migration — over a local object
//! directory through the shared [`support`] fixture, then interrupts it the
//! way a process failure, a replacement writer, or an operator's rollout
//! would. A crash is dropping the store without shutting it down; a restart is
//! a new store over the same object directory; a replacement writer is a
//! second store over that directory from its own data directory. Faults come
//! from `FaultInjectingObjectStore`, armed as an explicit queue, so every
//! interleaving reproduces exactly.
//!
//! Protocol scenarios (`migration_backfill_crash`, `fence_steal`,
//! `cold_restart`) place faults at chosen commit points and re-check
//! [`support::Invariants`] after every step: the durable revision never
//! decreases, HEAD names a complete state whose objects all exist, and no
//! acknowledged sequence number is lost or duplicated. Incident journeys
//! (`legacy_import_rehearsal`, `registration_race`, `missing_remote_object`)
//! reproduce failure points the rollout rehearsals hit, validating the
//! operator-visible outcome — content digests, exactly-once sequences, clean
//! failure on missing durable objects — in seconds where a rehearsal needed a
//! 5 GB production copy.

mod support;

mod cold_restart;
mod fence_steal;
mod legacy_import_rehearsal;
mod migration_backfill_crash;
mod missing_remote_object;
mod registration_race;
