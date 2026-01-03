use crate::coordinator::Coordinator;
use crate::types::{ActorId, ObjectId, TaskId, WorkerId};
use log::{debug, error, info, warn};
use std::net::SocketAddr;
use std::sync::Arc;
use tonic::{transport::Server, Request, Response, Status};
use pyo3::types::{PyAnyMethods, PyBytesMethods, PyListMethods, PyTuple};

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
        debug!(target: "fray::rpc::coordinator::objects",
               "Storing object {}, size: {} bytes", object_id, data.len());

        self.coordinator.object_store.insert(object_id, data);

        Ok(Response::new(ObjectRef {
            id: object_id.to_proto_bytes().to_vec(),
        }))
    }

    async fn get(&self, request: Request<ObjectRef>) -> Result<Response<PyObject>, Status> {
        let obj_ref = request.into_inner();
        let object_id = ObjectId::from_proto_bytes(&obj_ref.id)
            .map_err(|e| Status::invalid_argument(e))?;

        debug!(target: "fray::rpc::coordinator::objects",
               "Retrieving object {}", object_id);

        let data = self
            .coordinator
            .object_store
            .get(&object_id)
            .ok_or_else(|| {
                warn!(target: "fray::rpc::coordinator::objects",
                      "Object {} not found", object_id);
                Status::not_found("Object not found")
            })?;

        debug!(target: "fray::rpc::coordinator::objects",
               "Retrieved object {}, size: {} bytes", object_id, data.len());

        Ok(Response::new(PyObject { data }))
    }

    async fn submit_task(&self, request: Request<Task>) -> Result<Response<TaskRef>, Status> {
        let task = request.into_inner();
        let task_id = TaskId::from_proto_bytes(&task.id)
            .map_err(|e| {
                error!(target: "fray::rpc::coordinator::tasks",
                       "Invalid task ID: {}", e);
                Status::invalid_argument(e)
            })?;

        info!(target: "fray::rpc::coordinator::tasks",
              "Submitting task {}, payload size: {} bytes", task_id, task.payload.len());

        // The payload contains a pickled list: [func, arg1, arg2, ...]
        // We need to deserialize it to get the function and arguments
        use pyo3::Python;
        use pyo3::types::{PyBytes, PyList, PyModule};

        let result = Python::attach(|py| {
            // Import cloudpickle
            let cloudpickle = PyModule::import(py, "cloudpickle")
                .map_err(|e| {
                    let err_msg = format!("Failed to import cloudpickle: {}", e);
                    error!(target: "fray::rpc::coordinator::tasks",
                           "Task {}: {}", task_id, err_msg);
                    err_msg
                })?;
            let loads = cloudpickle.getattr("loads")
                .map_err(|e| format!("Failed to get cloudpickle.loads: {}", e))?;

            // Unpickle the payload (should be a list)
            let payload_bytes = PyBytes::new(py, &task.payload);
            let payload_list = loads.call1((payload_bytes,))
                .map_err(|e| {
                    let err_msg = format!("Failed to unpickle payload: {}", e);
                    error!(target: "fray::rpc::coordinator::tasks",
                           "Task {}: {}", task_id, err_msg);
                    err_msg
                })?;

            let list = payload_list.cast::<PyList>()
                .map_err(|e| format!("Payload is not a list: {}", e))?;

            if list.is_empty() {
                return Err("Empty payload list".to_string());
            }

            // First item is the function
            let func = list.get_item(0)
                .map_err(|e| format!("Failed to get function from payload: {}", e))?;

            // Remaining items are arguments
            let mut args_vec = Vec::new();
            for i in 1..list.len() {
                let arg = list.get_item(i)
                    .map_err(|e| format!("Failed to get arg {}: {}", i, e))?;
                args_vec.push(arg);
            }

            // Call the function
            let args_tuple = PyTuple::new(py, &args_vec)
                .map_err(|e| format!("Failed to create args tuple: {}", e))?;
            let result = func.call1(&args_tuple)
                .map_err(|e| {
                    let err_msg = format!("Function call failed: {}", e);
                    error!(target: "fray::rpc::coordinator::tasks",
                           "Task {}: {}", task_id, err_msg);
                    err_msg
                })?;

            // Pickle the result
            let dumps = cloudpickle.getattr("dumps")
                .map_err(|e| format!("Failed to get cloudpickle.dumps: {}", e))?;
            let pickled_result = dumps.call1((result,))
                .map_err(|e| format!("Failed to pickle result: {}", e))?;
            let result_bytes = pickled_result.cast::<PyBytes>()
                .map_err(|e| format!("Failed to convert pickled result to bytes: {}", e))?;

            Ok(result_bytes.as_bytes().to_vec())
        });

        let result_data = result.map_err(|e| {
            error!(target: "fray::rpc::coordinator::tasks",
                   "Task {} failed: {}", task_id, e);
            Status::internal(format!("Task execution failed: {}", e))
        })?;

        info!(target: "fray::rpc::coordinator::tasks",
              "Task {} completed, result size: {} bytes", task_id, result_data.len());

        // Store result in task scheduler
        let task_result = crate::types::TaskResult {
            task_id,
            result: result_data,
        };

        self.coordinator
            .task_scheduler
            .complete_task(task_id, task_result);

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
        let spec = request.into_inner();

        // Generate new actor ID
        let actor_id = ActorId::new();

        info!(target: "fray::rpc::coordinator::actors",
              "Creating actor {}, name: {:?}, class_def size: {} bytes",
              actor_id,
              if spec.name.is_empty() { None } else { Some(&spec.name) },
              spec.class_def.len());

        // Convert protobuf spec to worker ActorSpec
        let worker_spec = crate::worker::actor_host::ActorSpec {
            id: actor_id,
            class_def: crate::worker::executor::PickledData::new(spec.class_def),
            init_args: spec
                .args
                .into_iter()
                .map(crate::worker::executor::PickledData::new)
                .collect(),
        };

        // Create actor in in-process actor host
        self.coordinator
            .actor_host
            .create_actor(worker_spec)
            .map_err(|e| {
                error!(target: "fray::rpc::coordinator::actors",
                       "Failed to create actor {}: {}", actor_id, e);
                Status::internal(format!("Failed to create actor: {}", e))
            })?;

        // Register in actor registry
        use crate::coordinator::{ActorLifetime, ActorMetadata};
        let metadata = ActorMetadata {
            id: actor_id,
            name: if spec.name.is_empty() {
                None
            } else {
                Some(spec.name)
            },
            worker_id: WorkerId::default(), // In-process, no real worker
            lifetime: match spec.lifetime {
                1 => ActorLifetime::Task,       // JOB_SCOPED
                _ => ActorLifetime::Persistent, // DETACHED (0) or default
            },
            created_at: std::time::SystemTime::now(),
        };

        self.coordinator
            .actor_registry
            .register(actor_id, metadata);

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
        let call = request.into_inner();

        // Parse actor ID
        let actor_id = ActorId::from_proto_bytes(&call.actor_id)
            .map_err(|e| {
                error!(target: "fray::rpc::coordinator::actors",
                       "Invalid actor ID: {}", e);
                Status::invalid_argument(format!("Invalid actor ID: {}", e))
            })?;

        // Save method name for logging before moving call
        let method_name = call.method_name.clone();

        debug!(target: "fray::rpc::coordinator::actors",
               "Calling method {} on actor {}", method_name, actor_id);

        // Verify actor exists
        self.coordinator
            .actor_registry
            .get(&actor_id)
            .ok_or_else(|| {
                warn!(target: "fray::rpc::coordinator::actors",
                      "Actor {} not found", actor_id);
                Status::not_found(format!("Actor {} not found", actor_id))
            })?;

        // Create task ID for the method call result
        let task_id = TaskId::new();

        // Convert to ActorMethodCall for worker
        let method_call = crate::worker::actor_host::ActorMethodCall {
            actor_id,
            method_name: call.method_name,
            args: call
                .args
                .into_iter()
                .map(crate::worker::executor::PickledData::new)
                .collect(),
        };

        // Execute method call synchronously (in-process)
        let result = self.coordinator.actor_host.call_method(method_call);

        // Convert worker TaskResult to coordinator TaskResult
        let task_result = match result {
            crate::worker::executor::TaskResult::Success { data } => {
                debug!(target: "fray::rpc::coordinator::actors",
                       "Actor {} method {} succeeded, result size: {} bytes",
                       actor_id, method_name, data.len());
                crate::types::TaskResult {
                    task_id,
                    result: data,
                }
            }
            crate::worker::executor::TaskResult::Error { traceback } => {
                error!(target: "fray::rpc::coordinator::actors",
                       "Actor {} method {} failed: {}",
                       actor_id, method_name, traceback);
                return Err(Status::internal(format!(
                    "Actor method failed: {}",
                    traceback
                )));
            }
        };

        // Store result in task scheduler
        self.coordinator
            .task_scheduler
            .complete_task(task_id, task_result);

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

        info!(target: "fray::rpc::coordinator",
              "Registering worker {}", worker_id);

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

        debug!(target: "fray::rpc::coordinator",
               "Heartbeat from worker {}", worker_id);

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
