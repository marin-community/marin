# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Actions summaries for asynchronous checkpoint sampling."""

import html

from iris.client.workload import JobStatus

from experiments.grug.moe_hero_ep.ops.vibe_check.completions import SampleRequest
from experiments.grug.moe_hero_ep.ops.vibe_check.jobs import sample_job_names


def render_sampling_summary(
    requests: list[SampleRequest], completed_ids: set[str], jobs: list[JobStatus], dashboard_url: str
) -> str:
    """Show completed results and retained Iris job states without GPU work."""
    requests_by_job = {name: request for request in requests for name in sample_job_names(request)}
    completed_ids = completed_ids.intersection(request.sample_id for request in requests)
    outstanding = len(requests) - len(completed_ids)
    lines = [
        "## Hero checkpoint completions",
        "",
        f"Completed sample sets: {len(completed_ids)}. Requests without results: {outstanding}.",
        "",
        "Submission does not wait for GPU sampling. Job states below come from Iris.",
        "",
        "<table><tr><th>Checkpoint</th><th>Iris job</th><th>State</th><th>Details</th></tr>",
    ]
    for job in jobs:
        if not job.job_id.is_root or job.job_id.name not in requests_by_job:
            continue
        request = requests_by_job[job.job_id.name]
        checkpoint = str(request.checkpoint.step)
        url = html.escape(job.job_id.dashboard_url(dashboard_url), quote=True)
        name = html.escape(job.job_id.name)
        details = html.escape(job.error_message or job.pending_reason or job.status_message)
        lines.append(
            f'<tr><td>{checkpoint}</td><td><a href="{url}">{name}</a></td>'
            f"<td>{job.state.value}</td><td>{details}</td></tr>"
        )
    lines.extend(["</table>", ""])
    return "\n".join(lines)
