import { h } from 'preact';
import htm from 'htm';

const html = htm.bind(h);

export function UsersTab({ users }) {
  if (users.length === 0) {
    return html`<div class="no-jobs">No users found</div>`;
  }

  return html`
    <table>
      <thead>
        <tr>
          <th>User</th>
          <th>Active Jobs</th>
          <th>Running Jobs</th>
          <th>Pending Jobs</th>
          <th>Total Tasks</th>
          <th>Running Tasks</th>
          <th>Succeeded Tasks</th>
        </tr>
      </thead>
      <tbody>
        ${users.map(user => {
          const activeJobs = Object.entries(user.jobStateCounts)
            .filter(([state]) => !['succeeded', 'failed', 'killed', 'worker_failed', 'unschedulable'].includes(state))
            .reduce((total, [, count]) => total + count, 0);
          const totalTasks = Object.values(user.taskStateCounts).reduce((total, count) => total + count, 0);

          return html`
          <tr>
            <td>${user.user}</td>
            <td>${activeJobs}</td>
            <td>${user.jobStateCounts.running}</td>
            <td>${user.jobStateCounts.pending}</td>
            <td>${totalTasks}</td>
            <td>${user.taskStateCounts.running}</td>
            <td>${user.taskStateCounts.succeeded}</td>
          </tr>
          `;
        })}
      </tbody>
    </table>
  `;
}
