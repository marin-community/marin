# Fray RPC Migration Plan: Connect RPC Implementation

## Overview

**Goal**: Migrate the existing Fray RPC implementation from standard gRPC to Connect RPC (https://connectrpc.com/)

**Current State**:
- ✅ Full gRPC implementation with controller, worker, and context
- ✅ Proto definitions with all messages and services
- ✅ Controller unit tests
- ❌ Not integrated into `create_job_ctx()` factory
- ❌ No end-to-end integration tests
- ❌ No worker tests

**Why Connect RPC?**
- HTTP/1.1, HTTP/2, HTTP/3 support (not just HTTP/2)
- JSON + binary protobuf support
- Standard ASGI/WSGI server deployment
- Type-safe clients with cleaner API
- Better tooling via Buf

---

## Migration Stages

### Stage 1: Dependencies & Build System

**Objective**: Replace gRPC dependencies with Connect RPC and set up Buf for proto generation.

**Tasks**:
1. Update `lib/fray/pyproject.toml`:
   - Remove: `grpcio`, `grpcio-tools`
   - Add: `connect-python`, `uvicorn`, `buf-python` (or buf CLI)
2. Create `buf.yaml` and `buf.gen.yaml` in `lib/fray/` for Buf configuration
3. Update `lib/fray/scripts/generate_proto.sh` to use Buf instead of protoc
4. Test proto generation produces correct output

**Success Criteria**:
- Dependencies install cleanly via `uv sync`
- Proto generation script produces Connect stubs
- Generated files include service base classes and clients

---

### Stage 2: Protocol Updates (if needed)

**Objective**: Review and update proto definitions for Connect RPC best practices.

**Tasks**:
1. Review `lib/fray/src/fray/job/rpc/proto/fray.proto` for compatibility
2. Ensure method names follow Connect conventions
3. Verify message definitions work with both JSON and binary serialization
4. Regenerate protos with Buf

**Success Criteria**:
- Proto compiles with Buf
- Generated Python code includes both ASGI and client classes

---

### Stage 3: Controller Migration

**Objective**: Migrate `FrayControllerServicer` from gRPC to Connect ASGI service.

**Tasks**:
1. Update `lib/fray/src/fray/job/rpc/controller.py`:
   - Replace `FrayControllerServicer` with Connect service class (inherits from generated base)
   - Convert RPC methods to async ASGI handlers
   - Keep existing task queue logic (thread-safe with locks)
   - Replace `FrayControllerServer` to use uvicorn instead of grpc.server
2. Update method signatures to match Connect patterns (request, context)
3. Handle async context properly
4. Add graceful shutdown support

**API Changes**:
```python
# Before (gRPC):
class FrayControllerServicer(fray_pb2_grpc.FrayControllerServicer):
    def SubmitTask(self, request, context):
        ...

# After (Connect):
from fray.job.rpc.proto.fray_connect import FrayControllerService

class FrayControllerServicer(FrayControllerService):
    async def submit_task(self, request: TaskSpec, context) -> TaskHandle:
        ...
```

**Success Criteria**:
- Controller starts as ASGI application
- All RPC methods implemented as async handlers
- Existing controller unit tests pass (after test updates)

---

### Stage 4: Context & Client Migration

**Objective**: Migrate `FrayContext` to use Connect RPC client.

**Tasks**:
1. Update `lib/fray/src/fray/job/rpc/context.py`:
   - Replace gRPC channel/stub with Connect client
   - Update `_FrayFuture` to use Connect client methods
   - Convert polling logic to async/await or keep sync with blocking calls
   - Update error handling for Connect exceptions
2. Decide on sync vs async API (current implementation is sync)
3. Handle connection management (channel creation, cleanup)

**API Changes**:
```python
# Before (gRPC):
from grpc import insecure_channel
channel = insecure_channel(controller_address)
stub = fray_pb2_grpc.FrayControllerStub(channel)

# After (Connect):
from fray.job.rpc.proto.fray_connect import FrayControllerClient
client = FrayControllerClient("http://localhost:50051")
```

**Success Criteria**:
- FrayContext creates Connect client successfully
- Can submit tasks and retrieve results
- Polling/waiting logic works correctly

---

### Stage 5: Worker Migration

**Objective**: Migrate `FrayWorker` to use Connect RPC client for controller communication.

**Tasks**:
1. Update `lib/fray/src/fray/job/rpc/worker.py`:
   - Replace gRPC controller client with Connect client
   - Update `FrayWorkerServicer` to use Connect ASGI service (for health checks)
   - Keep main event loop but use Connect client methods
   - Update error handling for Connect exceptions
2. Handle worker-side ASGI server for health monitoring
3. Test task execution flow

**Success Criteria**:
- Worker connects to controller via Connect RPC
- Worker executes tasks and reports results
- Health check endpoint works via Connect

---

### Stage 6: Testing Infrastructure

**Objective**: Update existing tests and add comprehensive integration tests.

**Tasks**:
1. Update `lib/fray/tests/test_controller.py`:
   - Replace gRPC test fixtures with Connect ASGI test client
   - Update all test methods for Connect API
   - Ensure all 8 existing tests still pass
2. Create `lib/fray/tests/rpc/test_integration.py`:
   - End-to-end test: FrayContext → Controller → Worker → Result
   - Multi-task concurrent execution test
   - Worker failure and recovery test
   - Task timeout and error handling tests
3. Create `lib/fray/tests/rpc/test_worker.py`:
   - Worker registration test
   - Task execution test
   - Health check test
   - Graceful shutdown test

**Success Criteria**:
- All controller unit tests pass
- End-to-end integration test demonstrates full workflow
- Worker tests cover core functionality
- Test coverage >= 80% for RPC code

---

### Stage 7: Integration with Fray Context Factory

**Objective**: Wire FrayContext into the main `create_job_ctx()` factory.

**Tasks**:
1. Update `lib/fray/src/fray/job/context.py`:
   - Add `"fray"` to `context_type` Literal types
   - Add `controller_address` parameter to `create_job_ctx()`
   - Implement `elif context_type == "fray"` branch
   - Update docstrings and type hints
2. Update `ContextConfig` dataclass to include controller address
3. Add validation for required parameters

**Changes**:
```python
def create_job_ctx(
    context_type: Literal["ray", "threadpool", "sync", "fray", "auto"] = "auto",
    max_workers: int = 1,
    controller_address: str | None = None,
    **ray_options,
) -> JobContext:
    ...
    elif context_type == "fray":
        if controller_address is None:
            raise ValueError("controller_address required for fray context")
        from fray.job.rpc.context import FrayContext
        return FrayContext(controller_address)
```

**Success Criteria**:
- `create_job_ctx("fray", controller_address="...")` works
- All existing tests for other context types still pass
- Type checking passes with mypy

---

### Stage 8: Documentation & Examples

**Objective**: Document the new Fray RPC backend for users.

**Tasks**:
1. Update implementation plan to mark Phase 2 as complete
2. Add usage examples to docstrings
3. Create example scripts in `lib/fray/examples/rpc/`:
   - `run_controller.py` - Start controller server
   - `run_worker.py` - Start worker process
   - `submit_task.py` - Submit tasks via FrayContext
4. Document deployment options (uvicorn, gunicorn, etc.)

**Success Criteria**:
- Users can follow examples to run distributed tasks
- Documentation is clear and complete

---

## Task Assignment Strategy

### Sub-Agent Tasks (parallel where possible)

**Task 1: Dependencies & Build** (Stage 1)
- Agent: `python-implementation-expert`
- Update pyproject.toml, create Buf configs, update proto generation script

**Task 2: Controller Migration** (Stage 3)
- Agent: `python-implementation-expert`
- Migrate controller.py to Connect ASGI service
- Keep business logic, change RPC layer

**Task 3: Context Migration** (Stage 4)
- Agent: `python-implementation-expert`
- Migrate context.py to Connect client
- Update _FrayFuture for Connect

**Task 4: Worker Migration** (Stage 5)
- Agent: `python-implementation-expert`
- Migrate worker.py to Connect client + ASGI service

**Task 5: Test Updates** (Stage 6)
- Agent: `python-implementation-expert`
- Update controller tests, create integration tests
- Create worker tests

**Task 6: Integration** (Stage 7)
- Agent: `python-implementation-expert`
- Wire into context.py factory

**Task 7: Code Review** (After all implementations)
- Agent: `senior-code-reviewer`
- Review all changes for quality and consistency

---

## Parallel Execution Plan

### Wave 1 (Sequential - Foundation):
1. Stage 1: Dependencies & Build System

### Wave 2 (Parallel - Core Migration):
2. Stage 3: Controller Migration
3. Stage 4: Context Migration
4. Stage 5: Worker Migration

### Wave 3 (Sequential - Integration):
5. Stage 6: Testing Infrastructure
6. Stage 7: Integration with Factory
7. Stage 8: Documentation

### Wave 4 (Final):
8. Code Review

---

## Risk Mitigation

**Risks**:
1. **Async vs Sync**: Connect encourages async but current implementation is sync
   - Mitigation: Use Connect's sync client/server options initially
2. **Proto compatibility**: Existing proto may need adjustments
   - Mitigation: Review proto early, minimal changes expected
3. **Test breakage**: Controller tests depend on gRPC fixtures
   - Mitigation: Update fixtures first, then tests incrementally
4. **Connection management**: Different connection model than gRPC
   - Mitigation: Follow Connect examples closely

**Rollback Plan**:
- Keep existing gRPC code in git history
- Can revert if Connect RPC shows major issues
- Stage 1 changes are lowest risk (just dependencies)

---

## Success Metrics

**Must Have**:
- ✅ All existing controller tests pass with Connect
- ✅ End-to-end integration test passes
- ✅ FrayContext works via `create_job_ctx("fray", ...)`
- ✅ Worker can execute tasks and report results

**Nice to Have**:
- ✅ Performance benchmarks vs gRPC (similar or better)
- ✅ Example scripts for deployment
- ✅ Documentation for users

---

## Timeline (Not Time-Based)

This is a dependency graph, not a schedule:

```
Dependencies → Build System → Proto Generation
                                    ↓
                        ┌───────────┴────────────┐
                        ↓                        ↓
                   Controller              Context & Worker
                        ↓                        ↓
                        └───────────┬────────────┘
                                    ↓
                               Integration Tests
                                    ↓
                              Factory Integration
                                    ↓
                              Code Review
```

---

## Implementation Notes

- **Preserve business logic**: Task queue, worker registry, execution model unchanged
- **Focus on RPC layer**: Only change how components communicate
- **Type safety**: Leverage Connect's generated clients for better type checking
- **Testing first**: Update tests as we go, not at the end
- **Incremental**: Each stage should be testable independently

---

## Next Steps

1. Execute Stage 1 (Dependencies) - Foundation for everything else
2. Execute Stage 3-5 in parallel (Controller, Context, Worker migrations)
3. Execute Stage 6 (Tests) - Validate everything works
4. Execute Stage 7 (Integration) - Make it available to users
5. Execute Stage 8 (Docs) - Help users adopt it
6. Final code review and cleanup
