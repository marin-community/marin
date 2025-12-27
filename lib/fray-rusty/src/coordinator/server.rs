use crate::coordinator::Coordinator;
use crate::types::{ActorId, ObjectId, TaskId, WorkerId};
use std::net::SocketAddr;
use std::sync::Arc;
use tonic::{transport::Server, Request, Response, Status};

// Import the generated protobuf code
pub mod proto {
    tonic::include_proto!("fray");
}

use proto::coordinator_server::{Coordinator as CoordinatorTrait, CoordinatorServer};
use proto::*;

pub struct CoordinatorImpl {
    coordinator: Arc<Coordinator>,
}

impl CoordinatorImpl {
    pub fn new(coordinator: Arc<Coordinator>) -> Self {
        Self { coordinator }
    }
}

#[tonic::async_trait]
impl CoordinatorTrait for CoordinatorImpl {
    async fn put(&self, request: Request<PyObject>) -> Result<Response<ObjectRef>, Status> {
        let py_obj = request.into_inner();
        let data = py_obj.data;

        let object_id = ObjectId::new();
        self.coordinator.object_store.insert(object_id, data);

        Ok(Response::new(ObjectRef {
            id: object_id.to_proto_bytes().to_vec(),
        }))
    }

    async fn get(&self, request: Request<ObjectRef>) -> Result<Response<PyObject>, Status> {
        let obj_ref = request.into_inner();
        let object_id = ObjectId::from_proto_bytes(&obj_ref.id)
            .map_err(|e| Status::invalid_argument(e))?;

        let data = self
            .coordinator
            .object_store
            .get(&object_id)
            .ok_or_else(|| Status::not_found("Object not found"))?;

        Ok(Response::new(PyObject { data }))
    }

    async fn submit_task(&self, request: Request<Task>) -> Result<Response<TaskRef>, Status> {
        let task = request.into_inner();
        let task_id = TaskId::from_proto_bytes(&task.id)
            .map_err(|e| Status::invalid_argument(e))?;

        let task_obj = crate::types::Task {
            id: task_id,
            payload: task.payload,
        };

        self.coordinator.task_scheduler.enqueue(task_obj);

        Ok(Response::new(TaskRef {
            id: task_id.to_proto_bytes().to_vec(),
        }))
    }

    async fn get_task_result(
        &self,
        request: Request<TaskRef>,
    ) -> Result<Response<TaskResult>, Status> {
        let task_ref = request.into_inner();
        let task_id = TaskId::from_proto_bytes(&task_ref.id)
            .map_err(|e| Status::invalid_argument(e))?;

        let task_result = self
            .coordinator
            .task_scheduler
            .get_result(&task_id)
            .ok_or_else(|| Status::not_found("Task not found or still pending"))?;

        Ok(Response::new(TaskResult {
            task_id: task_id.to_proto_bytes().to_vec(),
            result: Some(task_result::Result::Success(task_result.result)),
        }))
    }

    async fn wait_tasks(
        &self,
        request: Request<WaitTasksRequest>,
    ) -> Result<Response<WaitTasksResponse>, Status> {
        let req = request.into_inner();
        let num_returns = req.num_returns as usize;

        let mut ready_tasks = Vec::new();

        for task_ref in req.task_refs {
            if ready_tasks.len() >= num_returns {
                break;
            }

            let task_id = TaskId::from_proto_bytes(&task_ref.id)
                .map_err(|e| Status::invalid_argument(e))?;

            if self.coordinator.task_scheduler.get_result(&task_id).is_some() {
                ready_tasks.push(TaskRef {
                    id: task_id.to_proto_bytes().to_vec(),
                });
            }
        }

        Ok(Response::new(WaitTasksResponse { ready: ready_tasks }))
    }

    async fn create_actor(
        &self,
        request: Request<ActorSpec>,
    ) -> Result<Response<ActorRef>, Status> {
        let _spec = request.into_inner();

        let actor_id = ActorId::new();

        // For now, just generate an actor ID without actually creating it on a worker
        // Full implementation would select a worker and create the actor

        Ok(Response::new(ActorRef {
            id: actor_id.to_proto_bytes().to_vec(),
        }))
    }

    async fn get_actor_by_name(
        &self,
        request: Request<GetActorByNameRequest>,
    ) -> Result<Response<ActorRef>, Status> {
        let req = request.into_inner();

        let actor_id = self
            .coordinator
            .actor_registry
            .get_by_name(&req.name)
            .ok_or_else(|| Status::not_found("Actor not found"))?;

        Ok(Response::new(ActorRef {
            id: actor_id.to_proto_bytes().to_vec(),
        }))
    }

    async fn call_actor_method(
        &self,
        request: Request<ActorMethodCall>,
    ) -> Result<Response<TaskRef>, Status> {
        let _call = request.into_inner();

        // Create a task ref for the method call result
        let task_id = TaskId::new();

        Ok(Response::new(TaskRef {
            id: task_id.to_proto_bytes().to_vec(),
        }))
    }

    async fn register_worker(
        &self,
        request: Request<RegisterWorkerRequest>,
    ) -> Result<Response<RegisterWorkerResponse>, Status> {
        let req = request.into_inner();
        let worker_id = WorkerId::from_proto_bytes(&req.worker_id)
            .map_err(|e| Status::invalid_argument(e))?;

        self.coordinator.worker_pool.register_worker(worker_id, 10);

        Ok(Response::new(RegisterWorkerResponse {}))
    }

    async fn heartbeat(
        &self,
        request: Request<HeartbeatRequest>,
    ) -> Result<Response<HeartbeatResponse>, Status> {
        let req = request.into_inner();
        let worker_id = WorkerId::from_proto_bytes(&req.worker_id)
            .map_err(|e| Status::invalid_argument(e))?;

        self.coordinator.worker_pool.update_heartbeat(&worker_id);

        Ok(Response::new(HeartbeatResponse {}))
    }
}

pub async fn run_coordinator_server(addr: SocketAddr) -> Result<(), Box<dyn std::error::Error>> {
    let coordinator = Arc::new(Coordinator::new());
    let coordinator_impl = CoordinatorImpl::new(coordinator);

    println!("Coordinator listening on {}", addr);

    Server::builder()
        .add_service(CoordinatorServer::new(coordinator_impl))
        .serve(addr)
        .await?;

    Ok(())
}
