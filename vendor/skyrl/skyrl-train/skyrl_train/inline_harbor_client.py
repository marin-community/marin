"""Credential-isolated Harbor trials admitted by the trainer's evaluation pool."""

import asyncio
import json
import socket
import sys

from skyrl_train.evaluation_pool import EvaluationPool

CONTROL_MESSAGE_LIMIT = 16 * 1024 * 1024


class HarborTrialClient:
    """Multiplex trial requests over one private socket to one Harbor subprocess."""

    def __init__(self, environment: dict[str, str]):
        self.environment = environment
        self._pending: dict[int, asyncio.Future] = {}
        self._next_id = 0
        self._failure: Exception | None = None

    async def __aenter__(self):
        parent, child = socket.socketpair()
        parent.setblocking(False)
        try:
            self.process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "skyrl_train.inline_harbor",
                "--control-fd",
                str(child.fileno()),
                pass_fds=(child.fileno(),),
                env=self.environment,
            )
        except BaseException:
            parent.close()
            raise
        finally:
            child.close()
        try:
            self.reader, self.writer = await asyncio.open_connection(sock=parent, limit=CONTROL_MESSAGE_LIMIT)
        except BaseException:
            parent.close()
            await self._stop_process()
            raise
        self._receiver = asyncio.create_task(self._receive())
        return self

    async def __aexit__(self, *_args):
        self.writer.close()
        try:
            await self.writer.wait_closed()
        finally:
            try:
                await self._stop_process()
            finally:
                self._receiver.cancel()
                await asyncio.gather(self._receiver, return_exceptions=True)

    async def _stop_process(self):
        if self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 30)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()

    async def run(self, cohort: dict, index: int) -> dict[str, float]:
        if self._failure is not None:
            raise self._failure
        if self.process.returncode is not None:
            raise RuntimeError(f"Harbor evaluation worker exited with status {self.process.returncode}")
        request_id = self._next_id
        self._next_id += 1
        result = asyncio.get_running_loop().create_future()
        self._pending[request_id] = result
        try:
            self.writer.write((json.dumps({"id": request_id, "cohort": cohort, "index": index}) + "\n").encode())
            await self.writer.drain()
            return await result
        finally:
            del self._pending[request_id]

    async def _receive(self):
        try:
            while line := await self.reader.readline():
                message = json.loads(line)
                future = self._pending.get(message["id"])
                if future is None or future.done():
                    continue
                if "error" in message:
                    future.set_exception(RuntimeError(message["error"]))
                else:
                    future.set_result(message["metrics"])
            raise RuntimeError("Harbor evaluation worker closed its control socket")
        except Exception as error:
            self._failure = error
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(error)


async def collect_harbor(cohorts: list[dict], pool: EvaluationPool, environment: dict[str, str]):
    """Admit each trial alongside native requests, preserving cohort quotas and accounting."""
    results: dict[str, list[dict[str, float]]] = {cohort["name"]: [] for cohort in cohorts}
    if len(results) != len(cohorts):
        raise ValueError("duplicate Harbor evaluation cohort")
    if not cohorts:
        return {}
    if any(not cohort["job"]["tasks"] for cohort in cohorts):
        raise ValueError("Harbor evaluation requires explicit nonempty task selections")
    async with HarborTrialClient(environment) as client:

        async def collect(cohort, index):
            metrics = await pool.run(
                f"harbor/{cohort['name']}",
                lambda: client.run(cohort, index),
                limit=int(cohort["job"]["n_concurrent_trials"]),
            )
            results[cohort["name"]].append(metrics)

        async with asyncio.TaskGroup() as tasks:
            for cohort in cohorts:
                for index in range(len(cohort["job"]["tasks"])):
                    tasks.create_task(collect(cohort, index))
    aggregates = {}
    expected = {cohort["name"]: len(cohort["job"]["tasks"]) for cohort in cohorts}
    for name, trials in results.items():
        totals = {key: sum(trial[key] for trial in trials) for key in ("num_attempted", "num_scored", "reward_sum")}
        if len(trials) != expected[name] or totals["num_attempted"] != expected[name]:
            raise RuntimeError(f"Harbor evaluation did not return every selected task for {name}")
        totals["avg_verifier_score"] = totals["reward_sum"] / totals["num_scored"] if totals["num_scored"] else 0.0
        aggregates[name] = totals
    return aggregates
