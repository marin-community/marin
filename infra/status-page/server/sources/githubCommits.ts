// GitHub CI build status for the Build panel — last N commits on main
// with their aggregate check-run rollup. Mirrors what the GitHub commits
// view renders as the little green check / red X / yellow dot next to
// each commit title.
//
// We use the GraphQL API because it lets us fetch N commits + their
// status rollup in a single request. The REST equivalent would be N+1
// calls (list commits, then per-commit /check-runs), which would blow
// through the 5000/hr rate limit under any reasonable refresh cadence.
//
// GraphQL requires authentication even for public repositories — if
// GITHUB_TOKEN is not set, the call returns 401. Set BUILDS_FIXTURE=1
// to bypass with canned data during local UI dev.

import { GH_OWNER, GH_REPO, githubAuthHeaders } from "./github.js";

const FIXTURE_MODE = process.env.BUILDS_FIXTURE === "1";
const GRAPHQL_URL = "https://api.github.com/graphql";

// StatusCheckRollupState enum from GitHub's GraphQL schema.
// See https://docs.github.com/en/graphql/reference/enums#statuscheckrollupstate
export type CommitState =
  | "SUCCESS"
  | "FAILURE"
  | "ERROR"
  | "PENDING"
  | "EXPECTED"
  | "NONE"; // null rollup — no checks configured for the commit

export interface CommitStatus {
  oid: string;
  shortOid: string;
  headline: string;
  committedAt: string;
  author: string;
  url: string;
  state: CommitState;
}

export interface BuildsResponse {
  commits: CommitStatus[];
  successRate: number | null; // [0, 1] over finalized commits (success + failure + error)
  fetchedAt: string;
  error?: string;
}

// GraphQL response shape for the subset of fields we read. Everything is
// optional because GitHub's schema returns nullable wrappers; narrow on
// the way out.
interface GqlCommitNode {
  oid: string;
  abbreviatedOid: string;
  messageHeadline: string;
  committedDate: string;
  url: string;
  author: {
    user: { login: string } | null;
    name: string | null;
  } | null;
  statusCheckRollup: { state: CommitState } | null;
}

interface GqlResponse {
  data?: {
    repository?: {
      ref?: {
        target?: {
          history?: {
            nodes?: GqlCommitNode[];
          };
        };
      };
    };
  };
  errors?: { message: string }[];
}

const QUERY = `
  query MainCommits($owner: String!, $repo: String!, $count: Int!) {
    repository(owner: $owner, name: $repo) {
      ref(qualifiedName: "refs/heads/main") {
        target {
          ... on Commit {
            history(first: $count) {
              nodes {
                oid
                abbreviatedOid
                messageHeadline
                committedDate
                url
                author {
                  user { login }
                  name
                }
                statusCheckRollup {
                  state
                }
              }
            }
          }
        }
      }
    }
  }
`;

function computeSuccessRate(commits: CommitStatus[]): number | null {
  // Exclude commits that haven't finalized yet (pending / expected / no
  // checks). Success rate over "did CI pass?" only makes sense for runs
  // that have actually finished.
  const finalized = commits.filter(
    (c) => c.state === "SUCCESS" || c.state === "FAILURE" || c.state === "ERROR",
  );
  if (finalized.length === 0) return null;
  const successes = finalized.filter((c) => c.state === "SUCCESS").length;
  return successes / finalized.length;
}

function toCommitStatus(node: GqlCommitNode): CommitStatus {
  return {
    oid: node.oid,
    shortOid: node.abbreviatedOid,
    headline: node.messageHeadline,
    committedAt: node.committedDate,
    author: node.author?.user?.login ?? node.author?.name ?? "unknown",
    url: node.url,
    state: node.statusCheckRollup?.state ?? "NONE",
  };
}

function fixtureSnapshot(count: number): BuildsResponse {
  const states: CommitState[] = ["SUCCESS", "SUCCESS", "SUCCESS", "FAILURE", "SUCCESS", "PENDING"];
  const now = Date.now();
  const commits: CommitStatus[] = Array.from({ length: count }, (_, i) => {
    const state = states[i % states.length];
    const shortOid = (0xa000000 + i).toString(16).slice(0, 7);
    return {
      oid: `${shortOid}${"0".repeat(33)}`,
      shortOid,
      headline: `fixture commit ${i + 1}`,
      committedAt: new Date(now - i * 60 * 60 * 1000).toISOString(),
      author: "fixture",
      url: `https://github.com/${GH_OWNER}/${GH_REPO}/commit/${shortOid}`,
      state,
    };
  });
  return {
    commits,
    successRate: computeSuccessRate(commits),
    fetchedAt: new Date().toISOString(),
  };
}

export async function fetchBuildsOnMain(count: number): Promise<BuildsResponse> {
  const fetchedAt = new Date().toISOString();
  if (FIXTURE_MODE) {
    return fixtureSnapshot(count);
  }

  const res = await fetch(GRAPHQL_URL, {
    method: "POST",
    headers: {
      ...githubAuthHeaders(),
      "content-type": "application/json",
    },
    body: JSON.stringify({
      query: QUERY,
      variables: { owner: GH_OWNER, repo: GH_REPO, count },
    }),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    return {
      commits: [],
      successRate: null,
      fetchedAt,
      error: `GitHub GraphQL ${res.status}: ${body.slice(0, 200)}`,
    };
  }

  const payload = (await res.json()) as GqlResponse;
  if (payload.errors && payload.errors.length > 0) {
    return {
      commits: [],
      successRate: null,
      fetchedAt,
      error: `GitHub GraphQL errors: ${payload.errors.map((e) => e.message).join("; ").slice(0, 200)}`,
    };
  }
  const nodes = payload.data?.repository?.ref?.target?.history?.nodes ?? [];
  const commits = nodes.map(toCommitStatus);
  return {
    commits,
    successRate: computeSuccessRate(commits),
    fetchedAt,
  };
}
