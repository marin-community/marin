use crate::types::{ActorId, ObjectId, TaskId, WorkerId};
use std::net::SocketAddr;
use tonic::transport::Channel;

// Import the generated protobuf code
pub mod proto {
    tonic::include_proto!("fray");
}

use proto::coordinator_client::CoordinatorClient as GrpcCoordinatorClient;
use proto::*;

#[derive(Clone)]
pub struct CoordinatorClient {
    client: GrpcCoordinatorClient<Channel>,
}

impl CoordinatorClient {
    pub async fn connect(addr: SocketAddr) -> Result<Self, Box<dyn std::error::Error>> {
        let endpoint = format!("http://{}", addr);
        let client = GrpcCoordinatorClient::connect(endpoint).await?;
        Ok(Self { client })
    }

    pub async fn put(&mut self, data: Vec<u8>) -> Result<ObjectId, Box<dyn std::error::Error>> {
        let response = self.client.put(PyObject { data }).await?;
        let obj_ref = response.into_inner();
        Ok(ObjectId::from_proto_bytes(&obj_ref.id)?)
    }

    pub async fn get(&mut self, object_id: ObjectId) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let response = self
            .client
            .get(ObjectRef {
                id: object_id.to_proto_bytes().to_vec(),
            })
            .await?;
        let py_obj = response.into_inner();
        Ok(py_obj.data)
    }

    pub async fn submit_task(
        &mut self,
        task_id: TaskId,
        payload: Vec<u8>,
    ) -> Result<TaskId, Box<dyn std::error::Error>> {
        let response = self
            .client
            .submit_task(Task {
                id: task_id.to_proto_bytes().to_vec(),
                payload,
            })
            .await?;
        let task_ref = response.into_inner();
        Ok(TaskId::from_proto_bytes(&task_ref.id)?)
    }

    pub async fn get_task_result(
        &mut self,
        task_id: TaskId,
    ) -> Result<Option<Vec<u8>>, Box<dyn std::error::Error>> {
        let response = self
            .client
            .get_task_result(TaskRef {
                id: task_id.to_proto_bytes().to_vec(),
            })
            .await?;
        let result = response.into_inner();

        match result.result {
            Some(task_result::Result::Success(data)) => Ok(Some(data)),
            Some(task_result::Result::Error(_)) | None => Ok(None),
        }
    }

    pub async fn wait_tasks(
        &mut self,
        task_ids: Vec<TaskId>,
        num_returns: u32,
    ) -> Result<Vec<TaskId>, Box<dyn std::error::Error>> {
        let task_refs: Vec<TaskRef> = task_ids
            .iter()
            .map(|id| TaskRef {
                id: id.to_proto_bytes().to_vec(),
            })
            .collect();

        let response = self
            .client
            .wait_tasks(WaitTasksRequest {
                task_refs,
                num_returns,
                timeout_ms: 5000, // 5 second timeout
            })
            .await?;

        let ready = response.into_inner().ready;
        let ready_ids: Result<Vec<TaskId>, String> = ready
            .iter()
            .map(|task_ref| TaskId::from_proto_bytes(&task_ref.id))
            .collect();

        Ok(ready_ids?)
    }

    pub async fn create_actor(
        &mut self,
        class_name: String,
        args: Vec<Vec<u8>>,
        name: String,
    ) -> Result<ActorId, Box<dyn std::error::Error>> {
        let response = self
            .client
            .create_actor(ActorSpec {
                class_name,
                args,
                kwargs: vec![],
                name,
                lifetime: actor_spec::Lifetime::Detached as i32,
            })
            .await?;

        let actor_ref = response.into_inner();
        Ok(ActorId::from_proto_bytes(&actor_ref.id)?)
    }

    pub async fn get_actor_by_name(
        &mut self,
        name: &str,
    ) -> Result<Option<ActorId>, Box<dyn std::error::Error>> {
        let response = self
            .client
            .get_actor_by_name(GetActorByNameRequest {
                name: name.to_string(),
            })
            .await;

        match response {
            Ok(resp) => {
                let actor_ref = resp.into_inner();
                Ok(Some(ActorId::from_proto_bytes(&actor_ref.id)?))
            }
            Err(status) if status.code() == tonic::Code::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub async fn register_worker(
        &mut self,
        worker_id: WorkerId,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.client
            .register_worker(RegisterWorkerRequest {
                worker_id: worker_id.to_proto_bytes().to_vec(),
                capabilities: "default".to_string(),
            })
            .await?;

        Ok(())
    }

    pub async fn heartbeat(&mut self, worker_id: WorkerId) -> Result<(), Box<dyn std::error::Error>> {
        self.client
            .heartbeat(HeartbeatRequest {
                worker_id: worker_id.to_proto_bytes().to_vec(),
            })
            .await?;

        Ok(())
    }
}
