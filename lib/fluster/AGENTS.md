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
