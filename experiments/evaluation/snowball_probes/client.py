# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HTTP transport authenticated by the applet's run-scoped session key."""

from dataclasses import dataclass, field

from marina.client import marina_request

from infra.marina.applets.snowball_probes.server.sessions import SESSION_HEADER


@dataclass(frozen=True)
class SessionConfig:
    origin: str
    applet: str
    run_id: str
    key: str = field(repr=False)


class WorkerApi:
    def __init__(self, session: SessionConfig):
        self.session = session

    def request(self, method: str, path: str, body: dict | None = None, params: dict | None = None) -> dict:
        result = marina_request(
            self.session.origin,
            method,
            f"/a/{self.session.applet}/api{path}",
            json_body=body,
            params=params,
            headers={SESSION_HEADER: self.session.key},
        )
        if not isinstance(result, dict):
            raise ValueError("Applet returned a non-object response")
        return result

    def run(self, run_id: str) -> dict:
        return self.request("GET", f"/runs/{run_id}")

    def cases(self, run_id: str, state: str = "", limit: int = 200) -> dict:
        return self.request("GET", "/cases", params={"run_id": run_id, "state": state, "limit": limit})

    def case(self, case_id: str, trace_state: str) -> dict:
        return self.request("GET", f"/cases/{case_id}", params={"trace_state": trace_state})

    def save(self, path: str, records: list[dict]) -> dict:
        if path == "/runs":
            if len(records) != 1:
                raise ValueError("Save one run snapshot at a time")
            return self.request("POST", path, records[0])
        if path not in {"/cases", "/traces"}:
            raise ValueError(f"Unknown worker record path: {path}")
        return self.request("POST", path + "/batch", {path[1:]: records})

    def claim(self, run_id: str, case_id: str) -> dict:
        return self.request("POST", f"/runs/{run_id}/ideas/claim", {"case_id": case_id})
