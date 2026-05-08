// Controller VM discovery via GCE instance labels.
//
// TypeScript port of infra/iris-iap-proxy/discovery.py — queries the
// Compute Engine API for VMs matching the iris controller label, returns
// the internal HTTP URL, and caches the result for 60s to avoid per-request
// GCE calls.
//
// For local dev, set CONTROLLER_URL to bypass discovery entirely (e.g. to
// a `gcloud compute start-iap-tunnel` forwarded port).

import { InstancesClient } from "@google-cloud/compute";

const CONTROLLER_URL_OVERRIDE = process.env.CONTROLLER_URL ?? "";
const GCP_PROJECT = process.env.GCP_PROJECT ?? "hai-gcp-models";
const CONTROLLER_ZONE = process.env.CONTROLLER_ZONE ?? "us-central1-a";
const CONTROLLER_LABEL = process.env.CONTROLLER_LABEL ?? "iris-marin-controller";
const CONTROLLER_PORT = Number(process.env.CONTROLLER_PORT ?? "10000");

const CACHE_TTL_MS = 60_000;

let cachedUrl: string | null = null;
let cacheExpiresAt = 0;
let client: InstancesClient | null = null;

function getClient(): InstancesClient {
  if (!client) {
    client = new InstancesClient();
  }
  return client;
}

async function queryControllerIp(): Promise<string> {
  const [instances] = await Promise.race([
    getClient().list({
      project: GCP_PROJECT,
      zone: CONTROLLER_ZONE,
      filter: `labels.${CONTROLLER_LABEL}=true AND status=RUNNING`,
    }),
    new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error("GCE discovery timed out after 15s")), 15_000),
    ),
  ]);
  if (!instances || instances.length === 0) {
    throw new Error(
      `No controller VM found (label=${CONTROLLER_LABEL}=true, ` +
        `project=${GCP_PROJECT}, zone=${CONTROLLER_ZONE})`,
    );
  }
  const instance = instances[0];
  for (const iface of instance.networkInterfaces ?? []) {
    if (iface.networkIP) {
      return iface.networkIP;
    }
  }
  throw new Error(`Controller VM ${instance.name ?? "<unknown>"} has no internal IP`);
}

export async function getControllerUrl(): Promise<string> {
  if (CONTROLLER_URL_OVERRIDE) {
    return CONTROLLER_URL_OVERRIDE;
  }

  const now = Date.now();
  if (cachedUrl && now < cacheExpiresAt) {
    return cachedUrl;
  }

  const ip = await queryControllerIp();
  const url = `http://${ip}:${CONTROLLER_PORT}`;
  cachedUrl = url;
  cacheExpiresAt = now + CACHE_TTL_MS;
  return url;
}
