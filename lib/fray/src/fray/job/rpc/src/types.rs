use std::fmt;
use std::hash::Hash;
use uuid::Uuid;
use bytes::Bytes;

/// ObjectId is a wrapper around UUID for type-safe object references.
/// Objects can represent tasks, actors, workers, or other entities in the system.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ObjectId(Uuid);

impl ObjectId {
    /// Generate a new ObjectId with a random v4 UUID.
    pub fn new() -> Self {
        ObjectId(Uuid::new_v4())
    }

    /// Create an ObjectId from raw bytes.
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        ObjectId(Uuid::from_bytes(bytes))
    }

    /// Get the raw bytes of the UUID.
    pub fn as_bytes(&self) -> &[u8; 16] {
        self.0.as_bytes()
    }

    /// Parse an ObjectId from a string representation.
    pub fn from_str(s: &str) -> Result<Self, uuid::Error> {
        Ok(ObjectId(Uuid::parse_str(s)?))
    }

    /// Convert to bytes for protobuf serialization
    pub fn to_proto_bytes(&self) -> Bytes {
        Bytes::copy_from_slice(self.as_bytes())
    }

    /// Create from protobuf bytes
    pub fn from_proto_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != 16 {
            return Err(format!("Expected 16 bytes, got {}", bytes.len()));
        }
        let mut arr = [0u8; 16];
        arr.copy_from_slice(bytes);
        Ok(Self::from_bytes(arr))
    }
}

impl fmt::Display for ObjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for ObjectId {
    fn default() -> Self {
        Self::new()
    }
}

/// TaskId is a wrapper around UUID for type-safe task references.
/// TaskIds uniquely identify tasks within the Fray coordinator system.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TaskId(Uuid);

impl TaskId {
    /// Generate a new TaskId with a random v4 UUID.
    pub fn new() -> Self {
        TaskId(Uuid::new_v4())
    }

    /// Create a TaskId from raw bytes.
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        TaskId(Uuid::from_bytes(bytes))
    }

    /// Get the raw bytes of the UUID.
    pub fn as_bytes(&self) -> &[u8; 16] {
        self.0.as_bytes()
    }

    /// Parse a TaskId from a string representation.
    pub fn from_str(s: &str) -> Result<Self, uuid::Error> {
        Ok(TaskId(Uuid::parse_str(s)?))
    }

    /// Convert to bytes for protobuf serialization
    pub fn to_proto_bytes(&self) -> Bytes {
        Bytes::copy_from_slice(self.as_bytes())
    }

    /// Create from protobuf bytes
    pub fn from_proto_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != 16 {
            return Err(format!("Expected 16 bytes, got {}", bytes.len()));
        }
        let mut arr = [0u8; 16];
        arr.copy_from_slice(bytes);
        Ok(Self::from_bytes(arr))
    }
}

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for TaskId {
    fn default() -> Self {
        Self::new()
    }
}

/// ActorId is a wrapper around UUID for type-safe actor references.
/// ActorIds uniquely identify actors (execution entities) in the Fray system.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ActorId(Uuid);

impl ActorId {
    /// Generate a new ActorId with a random v4 UUID.
    pub fn new() -> Self {
        ActorId(Uuid::new_v4())
    }

    /// Create an ActorId from raw bytes.
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        ActorId(Uuid::from_bytes(bytes))
    }

    /// Get the raw bytes of the UUID.
    pub fn as_bytes(&self) -> &[u8; 16] {
        self.0.as_bytes()
    }

    /// Parse an ActorId from a string representation.
    pub fn from_str(s: &str) -> Result<Self, uuid::Error> {
        Ok(ActorId(Uuid::parse_str(s)?))
    }

    /// Convert to bytes for protobuf serialization
    pub fn to_proto_bytes(&self) -> Bytes {
        Bytes::copy_from_slice(self.as_bytes())
    }

    /// Create from protobuf bytes
    pub fn from_proto_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != 16 {
            return Err(format!("Expected 16 bytes, got {}", bytes.len()));
        }
        let mut arr = [0u8; 16];
        arr.copy_from_slice(bytes);
        Ok(Self::from_bytes(arr))
    }
}

impl fmt::Display for ActorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for ActorId {
    fn default() -> Self {
        Self::new()
    }
}

/// WorkerId is a wrapper around UUID for type-safe worker identification.
/// WorkerIds uniquely identify worker processes that execute tasks.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct WorkerId(Uuid);

impl WorkerId {
    /// Generate a new WorkerId with a random v4 UUID.
    pub fn new() -> Self {
        WorkerId(Uuid::new_v4())
    }

    /// Create a WorkerId from raw bytes.
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        WorkerId(Uuid::from_bytes(bytes))
    }

    /// Get the raw bytes of the UUID.
    pub fn as_bytes(&self) -> &[u8; 16] {
        self.0.as_bytes()
    }

    /// Parse a WorkerId from a string representation.
    pub fn from_str(s: &str) -> Result<Self, uuid::Error> {
        Ok(WorkerId(Uuid::parse_str(s)?))
    }

    /// Convert to bytes for protobuf serialization
    pub fn to_proto_bytes(&self) -> Bytes {
        Bytes::copy_from_slice(self.as_bytes())
    }

    /// Create from protobuf bytes
    pub fn from_proto_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != 16 {
            return Err(format!("Expected 16 bytes, got {}", bytes.len()));
        }
        let mut arr = [0u8; 16];
        arr.copy_from_slice(bytes);
        Ok(Self::from_bytes(arr))
    }
}

impl fmt::Display for WorkerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for WorkerId {
    fn default() -> Self {
        Self::new()
    }
}

/// Task definition
#[derive(Debug, Clone)]
pub struct Task {
    pub id: TaskId,
    pub payload: Vec<u8>,
}

/// Task result
#[derive(Debug, Clone)]
pub struct TaskResult {
    pub task_id: TaskId,
    pub result: Vec<u8>,
}

