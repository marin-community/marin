use crate::types::ObjectId;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Thread-safe in-memory object storage with reference counting
#[derive(Clone)]
pub struct ObjectStore {
    inner: Arc<RwLock<ObjectStoreInner>>,
}

struct ObjectStoreInner {
    /// Storage for object data
    objects: HashMap<ObjectId, Vec<u8>>,
    /// Reference counts for objects
    ref_counts: HashMap<ObjectId, usize>,
}

impl ObjectStore {
    /// Create a new empty object store
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(ObjectStoreInner {
                objects: HashMap::new(),
                ref_counts: HashMap::new(),
            })),
        }
    }

    /// Insert an object into the store with an initial reference count of 1
    pub fn insert(&self, id: ObjectId, data: Vec<u8>) {
        let mut inner = self.inner.write();
        inner.objects.insert(id, data);
        inner.ref_counts.insert(id, 1);
    }

    /// Get a reference to an object's data
    pub fn get(&self, id: &ObjectId) -> Option<Vec<u8>> {
        let inner = self.inner.read();
        inner.objects.get(id).cloned()
    }

    /// Remove an object from the store
    pub fn remove(&self, id: &ObjectId) -> Option<Vec<u8>> {
        let mut inner = self.inner.write();
        inner.ref_counts.remove(id);
        inner.objects.remove(id)
    }

    /// Increment the reference count for an object
    /// Returns the new reference count, or None if the object doesn't exist
    pub fn incr_ref(&self, id: &ObjectId) -> Option<usize> {
        let mut inner = self.inner.write();
        inner.ref_counts.get_mut(id).map(|count| {
            *count += 1;
            *count
        })
    }

    /// Decrement the reference count for an object
    /// Returns the new reference count, or None if the object doesn't exist
    /// If the reference count reaches 0, the object is NOT automatically removed
    pub fn decr_ref(&self, id: &ObjectId) -> Option<usize> {
        let mut inner = self.inner.write();
        inner.ref_counts.get_mut(id).map(|count| {
            if *count > 0 {
                *count -= 1;
            }
            *count
        })
    }

    /// Get the current reference count for an object
    pub fn ref_count(&self, id: &ObjectId) -> Option<usize> {
        let inner = self.inner.read();
        inner.ref_counts.get(id).copied()
    }
}

impl Default for ObjectStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_get() {
        let store = ObjectStore::new();
        let id = ObjectId(1);
        let data = vec![1, 2, 3, 4];

        store.insert(id, data.clone());
        assert_eq!(store.get(&id), Some(data));
    }

    #[test]
    fn test_ref_counting() {
        let store = ObjectStore::new();
        let id = ObjectId(1);
        let data = vec![1, 2, 3];

        store.insert(id, data);
        assert_eq!(store.ref_count(&id), Some(1));

        assert_eq!(store.incr_ref(&id), Some(2));
        assert_eq!(store.ref_count(&id), Some(2));

        assert_eq!(store.decr_ref(&id), Some(1));
        assert_eq!(store.ref_count(&id), Some(1));
    }

    #[test]
    fn test_remove() {
        let store = ObjectStore::new();
        let id = ObjectId(1);
        let data = vec![1, 2, 3];

        store.insert(id, data.clone());
        assert_eq!(store.remove(&id), Some(data));
        assert_eq!(store.get(&id), None);
        assert_eq!(store.ref_count(&id), None);
    }
}
