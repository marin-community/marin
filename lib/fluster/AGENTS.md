# Agent Tips

* Use the connect/RPC abstractions to implement and perform RPC calls. DO NOT use httpx or raw HTTP.
* Use scripts/generate-protos.py to regenerate files after changing the `.proto` files.
* Prefer shallow, functional interfaces which return control to the user, vs callbacks or inheritance.

e.g.

class Scheduler:
  def add_job()
  def add_worker():
  def compute_schedule() -> ScheduledJobs:

is preferable to:

class Scheduler:
  def __init__(self, job_creator: JobCreator):
    self.job_creator = job_creator
  def run(self):
    ... self.job_creator.create_job()

It's acceptable to have a top-level class which implements the main loop of
course, but prefer to keep other interfaces shallow and functional whenever
possible.

* Tests should evaluate _behavior_, not implementation. Don't test things that are trivially caught by the type checker. Explicitly that means:

- No tests for "constant = constant"
- No tests for "method exists"
- No tests for "create an object(x, y, z) and attributes are x, y, z"

These tests have negative value - they make our code more brittle. Test
_behavior_ instead. You can use mocks as needed to isolate environments (e.g.
mock around a remote API). Prefer "fakes" -- e.g. create a real database but
with fake data -- when reasonable.

## Protocols and Testing

Non-trivial public classes should define a protocol which represents their
_important_ interface characteristics. Use this protocol in type hints for
when the class is used instead of the concrete class.

Test to this protocol, not the concrete class: the protocol should describe the
interesting behavior of the class, but not betray the implementation details.

(You may of course _instantiate_ the concrete class for testing.)
