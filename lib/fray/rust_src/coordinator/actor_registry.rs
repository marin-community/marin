use crate::types::{ActorId, WorkerId};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;

/// Actor lifetime specification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActorLifetime {
    /// Actor lives only for the duration of the task
    Task,
    /// Actor persists across tasks
    Persistent,
    /// Actor is a singleton (one per system)
    Singleton,
}

/// Metadata about an actor
#[derive(Debug, Clone)]
pub struct ActorMetadata {
    pub id: ActorId,
    pub name: Option<String>,
    pub worker_id: WorkerId,
    pub lifetime: ActorLifetime,
    pub created_at: SystemTime,
}

/// Thread-safe actor registry for named discovery
#[derive(Clone)]
pub struct ActorRegistry {
    inner: Arc<RwLock<ActorRegistryInner>>,
}

struct ActorRegistryInner {
    /// Actor metadata by ID
    actors: HashMap<ActorId, ActorMetadata>,
    /// Named actor lookup
    named_actors: HashMap<String, ActorId>,
}

impl ActorRegistry {
    /// Create a new actor registry
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(ActorRegistryInner {
                actors: HashMap::new(),
                named_actors: HashMap::new(),
            })),
        }
    }

    /// Register an actor with its metadata
    /// If the actor has a name, it will be registered in the named lookup as well
    pub fn register(&self, id: ActorId, metadata: ActorMetadata) {
        let mut inner = self.inner.write();
        if let Some(ref name) = metadata.name {
            inner.named_actors.insert(name.clone(), id);
        }
        inner.actors.insert(id, metadata);
    }

    /// Get actor metadata by ID
    pub fn get(&self, id: &ActorId) -> Option<ActorMetadata> {
        let inner = self.inner.read();
        inner.actors.get(id).cloned()
    }

    /// Get actor ID by name
    pub fn get_by_name(&self, name: &str) -> Option<ActorId> {
        let inner = self.inner.read();
        inner.named_actors.get(name).copied()
    }

    /// Get actor metadata by name
    pub fn get_metadata_by_name(&self, name: &str) -> Option<ActorMetadata> {
        let inner = self.inner.read();
        inner
            .named_actors
            .get(name)
            .and_then(|id| inner.actors.get(id).cloned())
    }

    /// Remove an actor from the registry
    /// Returns the metadata if the actor was found
    pub fn remove(&self, id: &ActorId) -> Option<ActorMetadata> {
        let mut inner = self.inner.write();
        if let Some(metadata) = inner.actors.remove(id) {
            if let Some(ref name) = metadata.name {
                inner.named_actors.remove(name);
            }
            Some(metadata)
        } else {
            None
        }
    }

    /// Get all actors on a specific worker
    pub fn get_by_worker(&self, worker_id: &WorkerId) -> Vec<ActorMetadata> {
        let inner = self.inner.read();
        inner
            .actors
            .values()
            .filter(|metadata| &metadata.worker_id == worker_id)
            .cloned()
            .collect()
    }

    /// Get all actors with a specific lifetime
    pub fn get_by_lifetime(&self, lifetime: ActorLifetime) -> Vec<ActorMetadata> {
        let inner = self.inner.read();
        inner
            .actors
            .values()
            .filter(|metadata| metadata.lifetime == lifetime)
            .cloned()
            .collect()
    }

    /// Get the total number of registered actors
    pub fn count(&self) -> usize {
        let inner = self.inner.read();
        inner.actors.len()
    }
}

impl Default for ActorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

