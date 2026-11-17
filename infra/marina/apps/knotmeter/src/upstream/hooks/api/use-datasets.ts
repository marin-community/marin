// SPDX-License-Identifier: GPL-3.0-only
// InferenceX 315128ce corpus structure contract.
export const DEFAULT_BLOCK_SIZE = 64;

export interface RawWekaRequest {
  t?: number;
  type?: string;
  model?: string;
  in?: number;
  out?: number;
  hash_ids?: number[];
  api_time?: number;
}

export interface RawWekaSubagent {
  t?: number;
  type: "subagent";
  agent_id?: string;
  subagent_type?: string;
  duration_ms?: number;
  requests?: RawWekaRequest[];
  models?: string[];
}

export type RawWekaEntry = RawWekaRequest | RawWekaSubagent;

export interface RawWekaConversation {
  id: string;
  models?: string[];
  block_size?: number;
  hash_id_scope?: string;
  requests?: RawWekaEntry[];
}

export interface TurnNode {
  kind: "turn";
  turnIndex: number;
  rawIndex?: number;
  innerIndex?: number;
  startS?: number;
  endS?: number;
  model?: string;
  in: number;
  out: number;
  cached: number;
  uncached: number;
}

export interface SubagentNode {
  kind: "subagent";
  label: string;
  agentId?: string;
  rawIndex?: number;
  startS?: number;
  endS?: number;
  durationMs?: number;
  in: number;
  out: number;
  cached: number;
  uncached: number;
  children: TurnNode[];
}

export type StructureNode = TurnNode | SubagentNode;

export interface ConversationStructure {
  blockSize: number;
  nodes: StructureNode[];
  totals: {
    in: number;
    out: number;
    cached: number;
    uncached: number;
    numTurns: number;
    numSubagentGroups: number;
  };
}
