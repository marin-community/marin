// Shared GitHub helpers used by both Nightlies (REST, workflow runs) and
// Build (GraphQL, commit status rollup) sources. The constants below belong
// only to Build; Nightlies carries a repository per lane.

export const GH_OWNER = "marin-community";
export const GH_REPO = "marin";

export function githubAuthHeaders(): Record<string, string> {
  const headers: Record<string, string> = {
    accept: "application/vnd.github+json",
    "x-github-api-version": "2022-11-28",
    "user-agent": "marin-infra-dashboard",
  };
  const token = process.env.GITHUB_TOKEN;
  if (token) {
    headers.authorization = `Bearer ${token}`;
  }
  return headers;
}
