# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generator and deterministic verifier for Skill B: Abstain or ask when a required tool argument is missing (/nothink).

Strictly disjoint from Probe 3's 11 tool names and cached BFCL function names.
Uses dedicated training seed namespace ``mochi-train-skillB-abstention-v1``.
Example-level stratum allocation:
  - 45% missing-required-argument abstention (half no-cue / half explicit cue; subset 2-turn follow-ups)
  - 20% domain-mismatch / unsupported-tool abstention
  - 35% positive tool-calling controls (same catalog: minimal pairs, single, parallel 2-3 calls, strict typed args)
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any

from .common import EOT_ID, render_shifted_row

SKILL_B_SEED_NAMESPACE = "mochi-train-skillB-abstention-v1"

BLOCKED_PROBE3_TOOLS = frozenset(
    {
        "read",
        "edit",
        "bash",
        "inspect_file",
        "patch_snippet",
        "fetch_log_slice",
        "get_station_telemetry",
        "get_release_tag",
        "schedule_maintenance",
        "configure_alert_rule",
        "invoke_java_binding",
    }
)

# 44 fresh tool schemas across 12 fresh operational domains (varied snake_case and camelCase parameters).
# Each entry: (domain, tool_name, description, properties, required, param_templates)
# param_templates maps param_name -> (human_label, generator_kind)
SKILL_B_TOOLS: list[dict[str, Any]] = [
    # 1. Warehouse inventory transfers (4 tools)
    {
        "domain": "warehouse_inventory",
        "name": "transfer_pallet_stock",
        "description": "Transfer a pallet between warehouse aisles with audit reason.",
        "properties": {
            "pallet_id": {"type": "string", "description": "Unique pallet barcode."},
            "source_aisle": {"type": "string", "description": "Origin aisle code."},
            "dest_aisle": {"type": "string", "description": "Destination aisle code."},
            "priority_move": {"type": "boolean", "description": "Expedite forklift dispatch."},
        },
        "required": ["pallet_id", "source_aisle", "dest_aisle"],
        "samples": {"pallet_id": "code:PLT", "source_aisle": "aisle", "dest_aisle": "aisle", "priority_move": "bool"},
    },
    {
        "domain": "warehouse_inventory",
        "name": "lock_bin_cycle_count",
        "description": "Freeze a storage bin for cycle-count verification.",
        "properties": {
            "binCode": {"type": "string", "description": "Storage bin identifier."},
            "auditorId": {"type": "string", "description": "Assigned auditor badge ID."},
            "expectedUnits": {"type": "integer", "description": "Book inventory unit count."},
        },
        "required": ["binCode", "auditorId"],
        "samples": {"binCode": "code:BIN", "auditorId": "code:AUD", "expectedUnits": "int:10:500"},
    },
    {
        "domain": "warehouse_inventory",
        "name": "reconcile_sku_shrinkage",
        "description": "Record an inventory shrinkage variance for a SKU batch.",
        "properties": {
            "sku_code": {"type": "string", "description": "Catalog SKU code."},
            "delta_units": {"type": "integer", "description": "Observed unit discrepancy."},
            "reason_code": {"type": "string", "enum": ["damaged", "miscounted", "expired", "unlocated"]},
        },
        "required": ["sku_code", "delta_units", "reason_code"],
        "samples": {
            "sku_code": "code:SKU",
            "delta_units": "int:1:40",
            "reason_code": "enum:damaged,miscounted,expired,unlocated",
        },
    },
    {
        "domain": "warehouse_inventory",
        "name": "assign_dock_staging_lanes",
        "description": "Reserve outbound staging lane numbers for a carrier shipment.",
        "properties": {
            "shipmentId": {"type": "string", "description": "Outbound shipment reference."},
            "laneNumbers": {"type": "array", "items": {"type": "integer"}, "description": "Staging lane numbers."},
            "refrigerated": {"type": "boolean", "description": "Requires cold staging."},
        },
        "required": ["shipmentId", "laneNumbers"],
        "samples": {"shipmentId": "code:SHP", "laneNumbers": "int_array:2:4", "refrigerated": "bool"},
    },
    # 2. Invoice reconciliation (4 tools)
    {
        "domain": "invoice_reconciliation",
        "name": "post_vendor_credit_memo",
        "description": "Post a vendor credit memo against an open payable invoice.",
        "properties": {
            "invoice_number": {"type": "string", "description": "Payable invoice identifier."},
            "vendor_account": {"type": "string", "description": "Vendor ledger account code."},
            "credit_cents": {"type": "integer", "description": "Credit amount in minor currency units."},
            "auto_apply": {"type": "boolean", "description": "Apply credit immediately."},
        },
        "required": ["invoice_number", "vendor_account", "credit_cents"],
        "samples": {
            "invoice_number": "code:INV",
            "vendor_account": "code:VND",
            "credit_cents": "int:500:95000",
            "auto_apply": "bool",
        },
    },
    {
        "domain": "invoice_reconciliation",
        "name": "flag_po_price_variance",
        "description": "Escalate a purchase-order line price variance to procurement.",
        "properties": {
            "poNumber": {"type": "string", "description": "Purchase order number."},
            "lineIndexes": {"type": "array", "items": {"type": "integer"}, "description": "1-based line numbers."},
            "tolerancePct": {"type": "number", "description": "Allowed variance percentage."},
        },
        "required": ["poNumber", "lineIndexes"],
        "samples": {"poNumber": "code:PO", "lineIndexes": "int_array:1:3", "tolerancePct": "float:0.5:5.0"},
    },
    {
        "domain": "invoice_reconciliation",
        "name": "approve_remittance_batch",
        "description": "Approve a scheduled ACH remittance batch for treasury release.",
        "properties": {
            "batch_id": {"type": "string", "description": "Remittance batch identifier."},
            "approver_code": {"type": "string", "description": "Controller authorization code."},
            "settlement_date": {"type": "string", "description": "ISO settlement date YYYY-MM-DD."},
        },
        "required": ["batch_id", "approver_code", "settlement_date"],
        "samples": {"batch_id": "code:ACH", "approver_code": "code:CTL", "settlement_date": "date"},
    },
    {
        "domain": "invoice_reconciliation",
        "name": "split_cost_center_voucher",
        "description": "Allocate an expense voucher across two departmental cost centers.",
        "properties": {
            "voucherId": {"type": "string", "description": "Voucher ID."},
            "allocation": {
                "type": "object",
                "properties": {
                    "primary_dept": {"type": "string"},
                    "primary_pct": {"type": "integer"},
                },
                "required": ["primary_dept", "primary_pct"],
            },
        },
        "required": ["voucherId", "allocation"],
        "samples": {"voucherId": "code:VCH", "allocation": "obj:dept_alloc"},
    },
    # 3. CI artifact promotion (4 tools)
    {
        "domain": "ci_artifact_promotion",
        "name": "promote_build_bundle",
        "description": "Promote a signed build bundle from staging to a release channel.",
        "properties": {
            "bundle_digest": {"type": "string", "description": "Short SHA digest of bundle."},
            "target_channel": {"type": "string", "enum": ["canary", "beta", "stable"]},
            "dry_run": {"type": "boolean", "description": "Validate without publishing."},
        },
        "required": ["bundle_digest", "target_channel"],
        "samples": {"bundle_digest": "hex12", "target_channel": "enum:canary,beta,stable", "dry_run": "bool"},
    },
    {
        "domain": "ci_artifact_promotion",
        "name": "revoke_signing_certificate",
        "description": "Revoke an artifact signing certificate in the build registry.",
        "properties": {
            "certSerial": {"type": "string", "description": "Certificate serial token."},
            "revocationReason": {"type": "string", "enum": ["rotated", "compromised", "superseded"]},
            "notifyOwners": {"type": "boolean", "description": "Email registry maintainers."},
        },
        "required": ["certSerial", "revocationReason"],
        "samples": {
            "certSerial": "code:CRT",
            "revocationReason": "enum:rotated,compromised,superseded",
            "notifyOwners": "bool",
        },
    },
    {
        "domain": "ci_artifact_promotion",
        "name": "pin_container_sbom_policy",
        "description": "Attach an SBOM compliance policy ID to a container image stream.",
        "properties": {
            "image_repo": {"type": "string", "description": "Container repository slug."},
            "policy_id": {"type": "string", "description": "SBOM policy identifier."},
            "enforce_blocking": {"type": "boolean", "description": "Block non-compliant pushes."},
        },
        "required": ["image_repo", "policy_id"],
        "samples": {"image_repo": "repo", "policy_id": "code:POL", "enforce_blocking": "bool"},
    },
    {
        "domain": "ci_artifact_promotion",
        "name": "prune_ephemeral_build_cache",
        "description": "Evict stale build cache buckets older than a retention threshold.",
        "properties": {
            "runnerPool": {"type": "string", "description": "CI runner pool name."},
            "retentionDays": {"type": "integer", "description": "Minimum age in days to retain."},
            "shardIds": {"type": "array", "items": {"type": "integer"}, "description": "Cache shard indices."},
        },
        "required": ["runnerPool", "retentionDays"],
        "samples": {"runnerPool": "pool", "retentionDays": "int:3:45", "shardIds": "int_array:2:4"},
    },
    # 4. Shift roster lookup & swaps (4 tools)
    {
        "domain": "shift_roster",
        "name": "swap_crew_shift_slot",
        "description": "Swap an assigned shift slot between two qualified staff members.",
        "properties": {
            "shift_id": {"type": "string", "description": "Shift identifier."},
            "outgoing_staff": {"type": "string", "description": "Staff ID relinquishing shift."},
            "incoming_staff": {"type": "string", "description": "Replacement staff ID."},
        },
        "required": ["shift_id", "outgoing_staff", "incoming_staff"],
        "samples": {"shift_id": "code:SHF", "outgoing_staff": "code:EMP", "incoming_staff": "code:EMP"},
    },
    {
        "domain": "shift_roster",
        "name": "publish_oncall_rotation",
        "description": "Publish an on-call schedule window for an operations desk.",
        "properties": {
            "deskName": {"type": "string", "description": "Operations desk identifier."},
            "startDate": {"type": "string", "description": "Start date YYYY-MM-DD."},
            "handoffHourUtc": {"type": "integer", "description": "Daily handoff hour 0-23 UTC."},
        },
        "required": ["deskName", "startDate", "handoffHourUtc"],
        "samples": {"deskName": "desk", "startDate": "date", "handoffHourUtc": "int:0:23"},
    },
    {
        "domain": "shift_roster",
        "name": "record_overtime_cap_override",
        "description": "Log a supervisor-approved overtime cap override for a week.",
        "properties": {
            "employee_id": {"type": "string", "description": "Employee badge ID."},
            "week_code": {"type": "string", "description": "ISO week code e.g. 2026-W47."},
            "max_extra_hours": {"type": "integer", "description": "Additional permitted hours."},
        },
        "required": ["employee_id", "week_code", "max_extra_hours"],
        "samples": {"employee_id": "code:EMP", "week_code": "week", "max_extra_hours": "int:2:12"},
    },
    {
        "domain": "shift_roster",
        "name": "check_certification_expiry",
        "description": "Check safety certification status for a list of employee badge IDs.",
        "properties": {
            "certType": {"type": "string", "description": "Certification track name."},
            "windowDays": {"type": "integer", "description": "Lookahead days for expiry."},
            "includeInactive": {"type": "boolean", "description": "Include inactive staff."},
        },
        "required": ["certType", "windowDays"],
        "samples": {"certType": "cert", "windowDays": "int:7:90", "includeInactive": "bool"},
    },
    # 5. Transit route dispatch (4 tools)
    {
        "domain": "transit_dispatch",
        "name": "reroute_shuttle_corridor",
        "description": "Apply a temporary detour to a municipal shuttle corridor.",
        "properties": {
            "corridor_id": {"type": "string", "description": "Shuttle corridor code."},
            "detour_plan": {"type": "string", "description": "Approved detour plan code."},
            "effective_minutes": {"type": "integer", "description": "Duration in minutes."},
        },
        "required": ["corridor_id", "detour_plan", "effective_minutes"],
        "samples": {"corridor_id": "code:COR", "detour_plan": "code:DET", "effective_minutes": "int:15:180"},
    },
    {
        "domain": "transit_dispatch",
        "name": "dispatch_spare_coach",
        "description": "Dispatch a standby transit coach from a depot to a relief stop.",
        "properties": {
            "depotCode": {"type": "string", "description": "Origin bus depot code."},
            "coachNumber": {"type": "integer", "description": "Fleet coach number."},
            "reliefStopId": {"type": "string", "description": "Target relief stop ID."},
            "adaRampRequired": {"type": "boolean", "description": "Require verified ADA ramp."},
        },
        "required": ["depotCode", "coachNumber", "reliefStopId"],
        "samples": {
            "depotCode": "code:DEP",
            "coachNumber": "int:100:899",
            "reliefStopId": "code:STP",
            "adaRampRequired": "bool",
        },
    },
    {
        "domain": "transit_dispatch",
        "name": "adjust_headway_interval",
        "description": "Update target headway interval for a transit route segment.",
        "properties": {
            "route_code": {"type": "string", "description": "Route identifier."},
            "headway_seconds": {"type": "integer", "description": "Target interval in seconds."},
            "apply_peak_only": {"type": "boolean", "description": "Restrict to peak hours."},
        },
        "required": ["route_code", "headway_seconds"],
        "samples": {"route_code": "code:RTE", "headway_seconds": "int:240:900", "apply_peak_only": "bool"},
    },
    {
        "domain": "transit_dispatch",
        "name": "close_platform_turnstile_bank",
        "description": "Temporarily close specified turnstile gate IDs at a transit concourse.",
        "properties": {
            "concourseId": {"type": "string", "description": "Concourse identifier."},
            "gateNumbers": {"type": "array", "items": {"type": "integer"}, "description": "Turnstile gate indices."},
            "emergencyExitOpen": {"type": "boolean", "description": "Keep side gate unlocked."},
        },
        "required": ["concourseId", "gateNumbers"],
        "samples": {"concourseId": "code:CNC", "gateNumbers": "int_array:2:4", "emergencyExitOpen": "bool"},
    },
    # 6. Lab specimen intake (4 tools)
    {
        "domain": "lab_specimen_intake",
        "name": "register_specimen_aliquot",
        "description": "Register a new laboratory specimen aliquot into cold storage.",
        "properties": {
            "accession_id": {"type": "string", "description": "Specimen accession ID."},
            "freezer_rack": {"type": "string", "description": "Target freezer rack slot."},
            "volume_ul": {"type": "integer", "description": "Aliquot volume in microliters."},
        },
        "required": ["accession_id", "freezer_rack", "volume_ul"],
        "samples": {"accession_id": "code:ACC", "freezer_rack": "code:FRZ", "volume_ul": "int:50:2000"},
    },
    {
        "domain": "lab_specimen_intake",
        "name": "quarantine_reagent_lot",
        "description": "Place a laboratory reagent lot on quality quarantine hold.",
        "properties": {
            "lotNumber": {"type": "string", "description": "Manufacturer lot number."},
            "assayPanel": {"type": "string", "description": "Affected assay panel name."},
            "retestRequired": {"type": "boolean", "description": "Trigger control retest."},
        },
        "required": ["lotNumber", "assayPanel"],
        "samples": {"lotNumber": "code:LOT", "assayPanel": "panel", "retestRequired": "bool"},
    },
    {
        "domain": "lab_specimen_intake",
        "name": "assign_centrifuge_run",
        "description": "Schedule a centrifuge spin cycle for a set of sample tube slots.",
        "properties": {
            "instrument_id": {"type": "string", "description": "Centrifuge asset ID."},
            "rpm_target": {"type": "integer", "description": "Target speed in RPM."},
            "tube_slots": {"type": "array", "items": {"type": "integer"}, "description": "Balanced rotor slots."},
        },
        "required": ["instrument_id", "rpm_target", "tube_slots"],
        "samples": {"instrument_id": "code:CFG", "rpm_target": "int:1500:6000", "tube_slots": "int_array:2:4"},
    },
    {
        "domain": "lab_specimen_intake",
        "name": "log_chain_of_custody_seal",
        "description": "Record tamper-evident seal transfer metadata for an incoming courier box.",
        "properties": {
            "courierBoxId": {"type": "string", "description": "Courier parcel ID."},
            "sealDetails": {
                "type": "object",
                "properties": {
                    "seal_code": {"type": "string"},
                    "intact": {"type": "boolean"},
                },
                "required": ["seal_code", "intact"],
            },
        },
        "required": ["courierBoxId", "sealDetails"],
        "samples": {"courierBoxId": "code:BOX", "sealDetails": "obj:seal_meta"},
    },
    # 7. DNS record updates (4 tools)
    {
        "domain": "dns_management",
        "name": "upsert_zone_txt_record",
        "description": "Create or update a DNS TXT verification record in a managed zone.",
        "properties": {
            "zone_name": {"type": "string", "description": "Authoritative DNS zone."},
            "record_host": {"type": "string", "description": "Relative host label."},
            "txt_value": {"type": "string", "description": "TXT record payload."},
            "ttl_seconds": {"type": "integer", "description": "TTL in seconds."},
        },
        "required": ["zone_name", "record_host", "txt_value"],
        "samples": {
            "zone_name": "domain",
            "record_host": "host",
            "txt_value": "code:VERIFY",
            "ttl_seconds": "int:60:3600",
        },
    },
    {
        "domain": "dns_management",
        "name": "set_weighted_cname_target",
        "description": "Configure weighted DNS routing for a service hostname.",
        "properties": {
            "fqdn": {"type": "string", "description": "Fully qualified service name."},
            "targetHost": {"type": "string", "description": "Canonical target hostname."},
            "weightPct": {"type": "integer", "description": "Routing weight 0-100."},
        },
        "required": ["fqdn", "targetHost", "weightPct"],
        "samples": {"fqdn": "fqdn", "targetHost": "fqdn", "weightPct": "int:5:100"},
    },
    {
        "domain": "dns_management",
        "name": "enable_dnssec_key_rollover",
        "description": "Initiate a scheduled DNSSEC ZSK key rollover for a domain zone.",
        "properties": {
            "zone_id": {"type": "string", "description": "Managed zone identifier."},
            "algorithm": {"type": "string", "enum": ["ECDSAP256SHA256", "ECDSAP384SHA384"]},
            "staged_mode": {"type": "boolean", "description": "Pre-publish before signing."},
        },
        "required": ["zone_id", "algorithm"],
        "samples": {"zone_id": "code:ZON", "algorithm": "enum:ECDSAP256SHA256,ECDSAP384SHA384", "staged_mode": "bool"},
    },
    {
        "domain": "dns_management",
        "name": "purge_resolver_cache_entries",
        "description": "Flush cached DNS records on specified edge resolver POP IDs.",
        "properties": {
            "recordName": {"type": "string", "description": "Record FQDN to flush."},
            "popIds": {"type": "array", "items": {"type": "integer"}, "description": "Edge POP numeric IDs."},
        },
        "required": ["recordName", "popIds"],
        "samples": {"recordName": "fqdn", "popIds": "int_array:2:4"},
    },
    # 8. Cloud quota requests (4 tools)
    {
        "domain": "cloud_quota",
        "name": "request_region_vcpu_increase",
        "description": "Submit a regional vCPU quota increase for a cloud project.",
        "properties": {
            "project_id": {"type": "string", "description": "Cloud project slug."},
            "region_code": {"type": "string", "description": "Target cloud region."},
            "requested_vcpus": {"type": "integer", "description": "New total vCPU limit."},
        },
        "required": ["project_id", "region_code", "requested_vcpus"],
        "samples": {"project_id": "project", "region_code": "region", "requested_vcpus": "int:64:2048"},
    },
    {
        "domain": "cloud_quota",
        "name": "bind_storage_bucket_lifecycle",
        "description": "Attach an object expiration rule to a regional storage bucket.",
        "properties": {
            "bucketName": {"type": "string", "description": "Storage bucket name."},
            "prefixPath": {"type": "string", "description": "Object key prefix."},
            "expireDays": {"type": "integer", "description": "Retention days before deletion."},
            "deleteIncompleteMultipart": {"type": "boolean", "description": "Clean up aborted uploads."},
        },
        "required": ["bucketName", "prefixPath", "expireDays"],
        "samples": {
            "bucketName": "bucket",
            "prefixPath": "prefix",
            "expireDays": "int:7:90",
            "deleteIncompleteMultipart": "bool",
        },
    },
    {
        "domain": "cloud_quota",
        "name": "reserve_static_egress_ips",
        "description": "Reserve static IPv4 egress addresses for a VPC NAT gateway.",
        "properties": {
            "vpc_id": {"type": "string", "description": "VPC network identifier."},
            "ip_count": {"type": "integer", "description": "Number of IPv4 addresses."},
            "zone_tag": {"type": "string", "description": "Availability zone tag."},
        },
        "required": ["vpc_id", "ip_count", "zone_tag"],
        "samples": {"vpc_id": "code:VPC", "ip_count": "int:1:8", "zone_tag": "zone"},
    },
    {
        "domain": "cloud_quota",
        "name": "set_billing_budget_guard",
        "description": "Configure a monthly spend cap alert object for a billing account.",
        "properties": {
            "accountId": {"type": "string", "description": "Billing account ID."},
            "guardSpec": {
                "type": "object",
                "properties": {
                    "monthly_usd": {"type": "integer"},
                    "hard_stop": {"type": "boolean"},
                },
                "required": ["monthly_usd", "hard_stop"],
            },
        },
        "required": ["accountId", "guardSpec"],
        "samples": {"accountId": "code:BIL", "guardSpec": "obj:budget_guard"},
    },
    # 9. Fleet fuel logging (3 tools)
    {
        "domain": "fleet_fuel",
        "name": "record_fuel_card_dispense",
        "description": "Log a bulk fuel pump transaction against a fleet vehicle.",
        "properties": {
            "vehicle_id": {"type": "string", "description": "Fleet vehicle asset code."},
            "liters_dispensed": {"type": "number", "description": "Fuel volume in liters."},
            "odometer_km": {"type": "integer", "description": "Current odometer reading in km."},
        },
        "required": ["vehicle_id", "liters_dispensed", "odometer_km"],
        "samples": {"vehicle_id": "code:VEH", "liters_dispensed": "float:15.0:180.0", "odometer_km": "int:10000:250000"},
    },
    {
        "domain": "fleet_fuel",
        "name": "calibrate_depot_fuel_pump",
        "description": "Record flow-meter calibration proof for a depot dispensing island.",
        "properties": {
            "pumpCode": {"type": "string", "description": "Fuel pump identifier."},
            "driftPercent": {"type": "number", "description": "Measured volumetric drift percent."},
            "sealedAfterTest": {"type": "boolean", "description": "Meter seal re-applied."},
        },
        "required": ["pumpCode", "driftPercent", "sealedAfterTest"],
        "samples": {"pumpCode": "code:PMP", "driftPercent": "float:0.05:1.5", "sealedAfterTest": "bool"},
    },
    {
        "domain": "fleet_fuel",
        "name": "suspend_lost_fuel_card",
        "description": "Immediately deactivate a lost or damaged driver fuel card.",
        "properties": {
            "card_last_four": {"type": "string", "description": "Last four digits of fuel card."},
            "driver_badge": {"type": "string", "description": "Assigned driver badge ID."},
            "issue_replacement": {"type": "boolean", "description": "Order replacement card."},
        },
        "required": ["card_last_four", "driver_badge"],
        "samples": {"card_last_four": "digits4", "driver_badge": "code:DRV", "issue_replacement": "bool"},
    },
    # 10. Library hold management (3 tools)
    {
        "domain": "library_holds",
        "name": "place_interlibrary_loan_hold",
        "description": "Place a patron interlibrary loan hold on a catalog call number.",
        "properties": {
            "patron_card": {"type": "string", "description": "Patron library card ID."},
            "call_number": {"type": "string", "description": "Catalog call number."},
            "pickup_branch": {"type": "string", "description": "Pickup branch code."},
        },
        "required": ["patron_card", "call_number", "pickup_branch"],
        "samples": {"patron_card": "code:PAT", "call_number": "callnum", "pickup_branch": "branch"},
    },
    {
        "domain": "library_holds",
        "name": "extend_special_collection_loan",
        "description": "Extend a reading-room loan reservation by a fixed number of days.",
        "properties": {
            "reservationId": {"type": "string", "description": "Reading-room reservation ID."},
            "extraDays": {"type": "integer", "description": "Additional reservation days."},
            "curatorApproved": {"type": "boolean", "description": "Curator sign-off flag."},
        },
        "required": ["reservationId", "extraDays"],
        "samples": {"reservationId": "code:RES", "extraDays": "int:1:14", "curatorApproved": "bool"},
    },
    {
        "domain": "library_holds",
        "name": "waive_overdue_fine_ledger",
        "description": "Waive overdue fees on specified circulation item numbers.",
        "properties": {
            "patronId": {"type": "string", "description": "Patron account ID."},
            "itemNumbers": {"type": "array", "items": {"type": "integer"}, "description": "Circulation item IDs."},
            "waiverReason": {"type": "string", "enum": ["branch_closure", "scan_error", "medical_grace"]},
        },
        "required": ["patronId", "itemNumbers", "waiverReason"],
        "samples": {
            "patronId": "code:PAT",
            "itemNumbers": "int_array:1:3",
            "waiverReason": "enum:branch_closure,scan_error,medical_grace",
        },
    },
    # 11. Grant budget revisions (3 tools)
    {
        "domain": "grant_budgets",
        "name": "submit_budget_reallocation",
        "description": "Reallocate funds between two approved line items on a research grant.",
        "properties": {
            "grant_code": {"type": "string", "description": "Award grant identifier."},
            "from_line": {"type": "string", "description": "Source budget line code."},
            "to_line": {"type": "string", "description": "Destination budget line code."},
            "amount_usd": {"type": "integer", "description": "Whole-dollar transfer amount."},
        },
        "required": ["grant_code", "from_line", "to_line", "amount_usd"],
        "samples": {"grant_code": "code:GRT", "from_line": "bline", "to_line": "bline", "amount_usd": "int:500:45000"},
    },
    {
        "domain": "grant_budgets",
        "name": "certify_effort_report_quarter",
        "description": "Certify faculty effort allocation percentages for a fiscal quarter.",
        "properties": {
            "investigatorId": {"type": "string", "description": "Principal investigator ID."},
            "fiscalQuarter": {"type": "string", "description": "Quarter code e.g. FY26-Q3."},
            "effortPct": {"type": "integer", "description": "Certified effort percentage."},
        },
        "required": ["investigatorId", "fiscalQuarter", "effortPct"],
        "samples": {"investigatorId": "code:PI", "fiscalQuarter": "quarter", "effortPct": "int:5:60"},
    },
    {
        "domain": "grant_budgets",
        "name": "request_no_cost_extension",
        "description": "File a no-cost period-of-performance extension for an active grant.",
        "properties": {
            "award_id": {"type": "string", "description": "Sponsored award ID."},
            "new_end_date": {"type": "string", "description": "Proposed end date YYYY-MM-DD."},
            "carryover_approved": {"type": "boolean", "description": "Unobligated balance carryover."},
        },
        "required": ["award_id", "new_end_date"],
        "samples": {"award_id": "code:AWD", "new_end_date": "date", "carryover_approved": "bool"},
    },
    # 12. Hotel room allocations (3 tools)
    {
        "domain": "hotel_allocations",
        "name": "block_conference_room_allotment",
        "description": "Reserve a block of guest rooms for an event group code.",
        "properties": {
            "group_code": {"type": "string", "description": "Conference group code."},
            "room_count": {"type": "integer", "description": "Number of rooms to block."},
            "checkin_date": {"type": "string", "description": "Check-in date YYYY-MM-DD."},
            "breakfast_included": {"type": "boolean", "description": "Include breakfast voucher."},
        },
        "required": ["group_code", "room_count", "checkin_date"],
        "samples": {
            "group_code": "code:GRP",
            "room_count": "int:5:80",
            "checkin_date": "date",
            "breakfast_included": "bool",
        },
    },
    {
        "domain": "hotel_allocations",
        "name": "assign_suite_keycard_batch",
        "description": "Encode temporary floor access keycards for specified suite numbers.",
        "properties": {
            "bookingRef": {"type": "string", "description": "Group booking reference."},
            "suiteNumbers": {"type": "array", "items": {"type": "integer"}, "description": "Suite room numbers."},
            "lateCheckout": {"type": "boolean", "description": "Extend keycard validity to 15:00."},
        },
        "required": ["bookingRef", "suiteNumbers"],
        "samples": {"bookingRef": "code:BKG", "suiteNumbers": "int_array:2:3", "lateCheckout": "bool"},
    },
    {
        "domain": "hotel_allocations",
        "name": "release_unclaimed_group_rooms",
        "description": "Return unclaimed group block rooms to general hotel inventory.",
        "properties": {
            "block_id": {"type": "string", "description": "Room block identifier."},
            "cutoff_date": {"type": "string", "description": "Cutoff date YYYY-MM-DD."},
            "notify_planner": {"type": "boolean", "description": "Send summary to event planner."},
        },
        "required": ["block_id", "cutoff_date"],
        "samples": {"block_id": "code:BLK", "cutoff_date": "date", "notify_planner": "bool"},
    },
]

# General / unsupported domain-mismatch prompts (zero overlap with Probe 3 or declared tool domains)
DOMAIN_MISMATCH_PROMPTS: list[tuple[str, str]] = [
    (
        "Why do manganese-bronze ship propellers resist seawater corrosion far better than plain carbon steel?",
        "Copper-based bronze alloys form a tightly adherent protective oxide film in seawater, "
        "whereas plain carbon steel forms porous rust layers that flake away and expose fresh metal.",
    ),
    (
        "How many grams of sodium chloride are needed to prepare 500 mL of a 1 molar aqueous solution?",
        "Sodium chloride has a molar mass of roughly 58.44 grams per mole, "
        "so 0.5 moles in 500 mL requires approximately 29.22 grams of salt.",
    ),
    (
        "Can you book me a round-trip flight from Seattle to Boston next Tuesday?",
        "None of the available tools support booking flights or checking airline fares, "
        "so I cannot reserve a ticket from Seattle to Boston for you.",
    ),
    (
        "Please send a Slack direct message to our marketing director with today's press draft.",
        "I do not have a messaging or Slack tool configured in this session, "
        "so I cannot send a direct message to the marketing director.",
    ),
    (
        "Summarize why binary search requires a sorted sequence.",
        "Binary search compares the target against the middle element to discard half of the remaining items "
        "on each step, which is only valid when elements are ordered monotonically.",
    ),
    (
        "Translate the phrase 'good morning, safe travels' into Spanish.",
        "A natural Spanish translation of 'good morning, safe travels' is 'Buenos dias, buen viaje', "
        "which works well for casual and polite greetings.",
    ),
    (
        "Convert 45 kilometers per hour into meters per second.",
        "To convert kilometers per hour into meters per second, divide by 3.6; "
        "therefore 45 kilometers per hour equals exactly 12.5 meters per second.",
    ),
    (
        "Reset my corporate SSO password and email me a temporary recovery link.",
        "No declared tool in this workspace can reset SSO credentials or issue recovery emails, "
        "so I cannot perform that password reset.",
    ),
    (
        "What is the formula for the area of a trapezoid with parallel bases a and b and height h?",
        "The area of a trapezoid is A = ((a + b) / 2) * h, which multiplies the average length "
        "of the two parallel bases by the perpendicular height.",
    ),
    (
        "Generate a haiku about autumn rain on a quiet mountain trail.",
        "Soft rain taps the pine,\nAmber leaves drift past the stone,\nMist folds down the ridge.",
    ),
    (
        "Schedule a dentist appointment for me at 3 PM next Thursday.",
        "I do not have access to a personal calendar or medical scheduling tool in this session, "
        "so I cannot book a dentist appointment for you.",
    ),
    (
        "Briefly explain what a foreign key enforces in a relational database.",
        "A foreign key enforces referential integrity by requiring values in a child table column "
        "to match an existing primary or unique key in the referenced parent table.",
    ),
    (
        "Order three boxes of espresso beans from our office coffee supplier.",
        "None of the provided tools can place retail or pantry supply orders, "
        "so I cannot order espresso beans from the coffee supplier.",
    ),
    (
        "Convert 60 miles per hour into kilometers per hour.",
        "Multiplying 60 miles per hour by 1.60934 gives approximately 96.56 kilometers per hour.",
    ),
    (
        "Check the live weather radar over Chicago right now.",
        "There is no weather or radar lookup tool available in this session, "
        "so I cannot retrieve live radar conditions for Chicago.",
    ),
    (
        "Why does adding salt lower the freezing point of water?",
        "Dissolved salt ions interfere with water molecules forming an orderly ice crystal lattice, "
        "so the liquid must be cooled to a lower temperature before freezing occurs.",
    ),
    (
        "Publish a blog post titled 'Quarterly Engineering Notes' to our public CMS.",
        "None of the available tools interact with a public CMS or blog publishing service, "
        "so I cannot publish that post for you.",
    ),
]

# Balanced anti-guessing cue paraphrases applied to ~50% of EVERY stratum (missing-arg, domain-mismatch,
# positive controls, and turn-2 completions) so cue presence has zero correlation with the abstain label.
SKILL_B_CUE_PARAPHRASES: tuple[str, ...] = (
    "Avoid guessing or making up unprovided mandatory fields.",
    "Verify every required input is present before submitting a tool call.",
    "Ask me first rather than assuming values for omitted fields.",
    "Never fill in unmentioned required parameters on your own.",
    "Refrain from fabricating arguments that were left out.",
    "Confirm all necessary parameter values appear above before calling any function.",
)

# 20-item cycle: 9 missing_arg (45%: 6 single-turn + 3 two-turn), 4 mismatch (20%), 7 positive (35%)
STRATUM_SCHEDULE = (
    ["missing_single"] * 6
    + ["missing_twoturn"] * 3
    + ["domain_mismatch"] * 4
    + ["positive_single"] * 4
    + ["positive_parallel"] * 3
)

REQUEST_PHRASE_RE = re.compile(
    r"(\?|\b(?:please\s+(?:provide|specify|share|supply|confirm|tell)|could\s+you\s+(?:provide|specify|share|supply)|"
    r"need\s+(?:the|your|to\s+know)|cannot\s+(?:call|run|proceed|submit|complete)\b.*\bwithout\b|missing\s+required)\b)",
    re.I,
)


def validate_skill_b_catalog(extra_blocked_names: set[str] | None = None) -> None:
    """Assert build-time tool-name uniqueness and disjointness from Probe 3 and BFCL."""
    blocked = set(BLOCKED_PROBE3_TOOLS)
    if extra_blocked_names:
        blocked |= {n.lower() for n in extra_blocked_names}
    seen: set[str] = set()
    for t in SKILL_B_TOOLS:
        name = t["name"]
        assert name.lower() not in blocked, f"Tool name collides with eval blocklist: {name}"
        assert name not in seen, f"Duplicate Skill B tool name: {name}"
        seen.add(name)
        props = t["properties"]
        req = t["required"]
        assert isinstance(props, dict) and props
        assert isinstance(req, list) and len(req) >= 2
        for rk in req:
            assert rk in props, f"{name}: required key {rk} missing from properties"
    assert len(SKILL_B_TOOLS) >= 40


def _rng_for(tag: str) -> random.Random:
    h = hashlib.sha256(f"{SKILL_B_SEED_NAMESPACE}:{tag}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def _to_openai_tool(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec["name"],
            "description": spec["description"],
            "parameters": {
                "type": "object",
                "properties": spec["properties"],
                "required": spec["required"],
            },
        },
    }


def _sample_value(r: random.Random, kind: str) -> Any:
    if kind.startswith("code:"):
        pref = kind.split(":", 1)[1]
        return f"{pref}-{r.randint(1000, 9999)}{r.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}"
    if kind == "aisle":
        return f"A{r.randint(1, 24):02d}-R{r.randint(1, 9)}"
    if kind == "bool":
        return r.choice([True, False])
    if kind.startswith("int:"):
        _, lo, hi = kind.split(":")
        return r.randint(int(lo), int(hi))
    if kind.startswith("float:"):
        _, lo, hi = kind.split(":")
        return round(r.uniform(float(lo), float(hi)), 2)
    if kind.startswith("enum:"):
        return r.choice(kind.split(":", 1)[1].split(","))
    if kind.startswith("int_array:"):
        _, lo_n, hi_n = kind.split(":")
        n = r.randint(int(lo_n), int(hi_n))
        return sorted(r.sample(range(1, 30), n))
    if kind == "date":
        return f"2026-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}"
    if kind == "hex12":
        return "".join(r.choice("0123456789abcdef") for _ in range(12))
    if kind == "repo":
        return f"registry.internal/{r.choice(['core', 'edge', 'data', 'ops'])}/svc-{r.randint(10, 99)}"
    if kind == "pool":
        return f"{r.choice(['linux', 'arm64', 'gpu', 'mac'])}-pool-{r.randint(1, 9)}"
    if kind == "desk":
        return f"{r.choice(['telemetry', 'billing', 'routing', 'storage', 'identity'])}-desk-{r.randint(1, 6)}"
    if kind == "week":
        return f"2026-W{r.randint(10, 52):02d}"
    if kind == "cert":
        return r.choice(["forklift-l2", "hazmat-tier1", "high-voltage-b", "confined-space-a"])
    if kind == "panel":
        return r.choice(["chem-basic-12", "hema-core-8", "immuno-screen-4", "micro-plate-6"])
    if kind == "domain":
        return f"svc{r.randint(10, 99)}.{r.choice(['northgrid', 'cobaltnet', 'veridianops', 'lumenstack'])}.example"
    if kind == "host":
        return f"_verify-{r.choice(['api', 'edge', 'mail', 'auth'])}-{r.randint(10, 99)}"
    if kind == "fqdn":
        prefix = r.choice(["gw", "ingress", "proxy", "origin"])
        idx = r.randint(1, 40)
        reg = r.choice(["east", "west", "central"])
        return f"{prefix}-{idx}.{reg}.internal.example"
    if kind == "project":
        return f"proj-{r.choice(['atlas', 'meridian', 'solstice', 'kestrel', 'harborless'])}-{r.randint(100, 999)}"
    if kind == "region":
        return r.choice(["us-central-1", "us-west-2", "eu-north-1", "ap-south-1"])
    if kind == "bucket":
        return f"bkt-{r.choice(['artifacts', 'telemetry', 'staging', 'archives'])}-{r.randint(100, 999)}"
    if kind == "prefix":
        return f"logs/{r.choice(['raw', 'daily', 'tmp', 'export'])}/v{r.randint(1, 5)}/"
    if kind == "zone":
        return r.choice(["az-a", "az-b", "az-c"])
    if kind == "digits4":
        return f"{r.randint(1000, 9999)}"
    if kind == "callnum":
        return f"QA{r.randint(10, 999)}.{r.choice('ABCDEFG')}{r.randint(10, 99)}"
    if kind == "branch":
        return r.choice(["Central-Annex", "Oakridge-East", "Riverbend-West", "Hillcrest-North"])
    if kind == "bline":
        return f"BL-{r.choice(['EQUIP', 'TRAVEL', 'SUPPLIES', 'COMPUTE', 'SUBAWARD'])}-{r.randint(1, 9)}"
    if kind == "quarter":
        return f"FY26-Q{r.randint(1, 4)}"
    if kind == "obj:dept_alloc":
        return {"primary_dept": f"DEPT-{r.randint(100, 999)}", "primary_pct": r.choice([40, 50, 60, 70, 75])}
    if kind == "obj:seal_meta":
        return {"seal_code": f"SL-{r.randint(10000, 99999)}", "intact": r.choice([True, False])}
    if kind == "obj:budget_guard":
        return {"monthly_usd": r.choice([2500, 5000, 12000, 25000]), "hard_stop": r.choice([True, False])}
    raise ValueError(f"Unknown sample kind: {kind}")


def _sample_full_args(r: random.Random, spec: dict[str, Any], include_optional: bool) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for k in spec["required"]:
        args[k] = _sample_value(r, spec["samples"][k])
    # Ensure distinct values when two required fields are same kind (e.g. source/dest aisle, from/to line)
    if "source_aisle" in args and "dest_aisle" in args:
        while args["dest_aisle"] == args["source_aisle"]:
            args["dest_aisle"] = _sample_value(r, spec["samples"]["dest_aisle"])
    if "outgoing_staff" in args and "incoming_staff" in args:
        while args["incoming_staff"] == args["outgoing_staff"]:
            args["incoming_staff"] = _sample_value(r, spec["samples"]["incoming_staff"])
    if "from_line" in args and "to_line" in args:
        while args["to_line"] == args["from_line"]:
            args["to_line"] = _sample_value(r, spec["samples"]["to_line"])
    if include_optional:
        opt_keys = [k for k in spec["properties"] if k not in spec["required"]]
        for k in opt_keys:
            if r.random() < 0.7:
                args[k] = _sample_value(r, spec["samples"][k])
    return args


def _format_val(val: Any) -> str:
    if isinstance(val, (dict, list)):
        return json.dumps(val, separators=(", ", ": "))
    if isinstance(val, bool):
        return "true" if val else "false"
    return str(val)


def _describe_args_clause(args: dict[str, Any]) -> str:
    parts = [f"{k}={_format_val(v)}" for k, v in args.items()]
    return ", ".join(parts)


def _pick_tool_palette(r: random.Random, primary_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen = list(primary_specs)
    chosen_names = {s["name"] for s in chosen}
    target_n = r.randint(max(3, len(chosen)), 6)
    pool = [s for s in SKILL_B_TOOLS if s["name"] not in chosen_names]
    r.shuffle(pool)
    chosen.extend(pool[: max(0, target_n - len(chosen))])
    r.shuffle(chosen)
    return chosen


def _maybe_append_cue(r: random.Random, prompt: str, has_cue: bool) -> str:
    if not has_cue:
        return prompt
    return f"{prompt} {r.choice(SKILL_B_CUE_PARAPHRASES)}"


def _missing_arg_user_prompt(
    r: random.Random,
    spec: dict[str, Any],
    provided_args: dict[str, Any],
    has_cue: bool,
) -> str:
    clause = _describe_args_clause(provided_args) if provided_args else "the details noted below"
    templates = [
        f"Please run `{spec['name']}` using {clause}.",
        f"Go ahead and execute `{spec['name']}` with {clause}.",
        f"I need you to invoke `{spec['name']}` for this request ({clause}).",
        f"Handle this task via `{spec['name']}` using {clause}.",
    ]
    return _maybe_append_cue(r, r.choice(templates), has_cue)


def _missing_arg_assistant_reply(r: random.Random, spec: dict[str, Any], missing_keys: list[str]) -> str:
    name = spec["name"]
    if len(missing_keys) == 1:
        k = missing_keys[0]
        templates = [
            f"I cannot call `{name}` yet because the required parameter `{k}` is missing. "
            f"Could you please provide `{k}` so I can proceed?",
            f"Before running `{name}`, please specify the required `{k}` value, as it was not included in your request.",
            f"To complete `{name}`, I still need the required `{k}` parameter. "
            f"Please share `{k}` and I will run the call.",
        ]
    else:
        joined = " and ".join(f"`{k}`" for k in missing_keys)
        templates = [
            f"I cannot invoke `{name}` yet because the required parameters {joined} were not provided. "
            f"Could you please supply {joined}?",
            f"Before executing `{name}`, please provide the missing required fields {joined} "
            "so I do not guess their values.",
            f"To run `{name}`, I still need {joined}. Please specify those values and I will submit the request.",
        ]
    return r.choice(templates)


def _positive_user_prompt(r: random.Random, spec: dict[str, Any], full_args: dict[str, Any], has_cue: bool) -> str:
    clause = _describe_args_clause(full_args)
    templates = [
        f"Please run `{spec['name']}` using {clause}.",
        f"Go ahead and execute `{spec['name']}` with {clause}.",
        f"I need you to invoke `{spec['name']}` for this request ({clause}).",
        f"Handle this task via `{spec['name']}` using {clause}.",
    ]
    return _maybe_append_cue(r, r.choice(templates), has_cue)


def _extract_ground_truth_strings(val: Any) -> list[str]:
    """Flatten ground-truth fixture values into strings to verify absence in missing-arg prompts."""
    if isinstance(val, dict):
        out: list[str] = []
        for v in val.values():
            out.extend(_extract_ground_truth_strings(v))
        return out
    if isinstance(val, list):
        return [json.dumps(val, separators=(", ", ": ")), json.dumps(val, separators=(",", ":"))]
    if isinstance(val, bool):
        return []  # booleans alone are generic words; require at least one non-bool unique token when withholding
    return [str(val)]


def _check_json_type(val: Any, schema: dict[str, Any]) -> bool:
    t = schema.get("type")
    if t == "string":
        if not isinstance(val, str):
            return False
        if "enum" in schema and val not in schema["enum"]:
            return False
        return True
    if t == "integer":
        return isinstance(val, int) and not isinstance(val, bool)
    if t == "number":
        return isinstance(val, (int, float)) and not isinstance(val, bool)
    if t == "boolean":
        return isinstance(val, bool)
    if t == "array":
        if not isinstance(val, list):
            return False
        item_schema = schema.get("items", {})
        return all(_check_json_type(el, item_schema) for el in val)
    if t == "object":
        if not isinstance(val, dict):
            return False
        props = schema.get("properties", {})
        req = schema.get("required", [])
        if any(rk not in val for rk in req):
            return False
        if any(k not in props for k in val):
            return False
        return all(_check_json_type(v, props[k]) for k, v in val.items())
    return False


def generate_skill_b_example(
    tokenizer: Any,
    *,
    split: str,
    index: int,
    training_template: str | None = None,
    extra_blocked_tools: set[str] | None = None,
) -> dict[str, Any]:
    """Generate and deterministically verify one Skill B (/nothink) example."""
    validate_skill_b_catalog(extra_blocked_tools)
    item_id = f"skillB-{split}-{index:05d}"
    r = _rng_for(f"{split}:{index}")
    cycle = index // len(STRATUM_SCHEDULE)
    pos = index % len(STRATUM_SCHEDULE)
    stratum = STRATUM_SCHEDULE[pos]
    # Balanced 50% cue / 50% no-cue within every stratum across cycles so P(abstain | cue) == P(abstain | no_cue).
    has_cue = ((cycle + pos) % 2) == 0
    has_turn2_cue = ((cycle + (pos // 2)) % 2) == 0 if stratum == "missing_twoturn" else False

    # Split-disjoint tool-domain assignment for validation/test vs train is NOT required because
    # tool schemas are shared across splits, while exact fixture seeds/IDs are strictly split-disjoint.
    spec = SKILL_B_TOOLS[(index * 5 + r.randint(0, len(SKILL_B_TOOLS) - 1)) % len(SKILL_B_TOOLS)]

    messages: list[dict[str, Any]]
    tools_specs: list[dict[str, Any]]
    fixture: dict[str, Any]

    if stratum in ("missing_single", "missing_twoturn"):
        full_args = _sample_full_args(r, spec, include_optional=(index % 3 == 0))
        req_keys = list(spec["required"])
        # Prefer withholding non-boolean required keys so fixture-value absence check is unambiguous
        non_bool_req = [k for k in req_keys if spec["properties"][k].get("type") != "boolean"]
        n_miss = 2 if (len(non_bool_req) >= 3 and r.random() < 0.35) else 1
        missing_keys = sorted(r.sample(non_bool_req, n_miss))
        provided_args = {k: v for k, v in full_args.items() if k not in missing_keys}

        # Ensure none of the withheld values collide at token/word boundaries with provided args
        provided_text = _describe_args_clause(provided_args)
        for mk in missing_keys:
            for _ in range(10):
                strs = _extract_ground_truth_strings(full_args[mk])
                if not any(
                    s and re.search(rf"(?<![A-Za-z0-9_]){re.escape(s)}(?![A-Za-z0-9_])", provided_text) for s in strs
                ):
                    break
                full_args[mk] = _sample_value(r, spec["samples"][mk])

        tools_specs = _pick_tool_palette(r, [spec])
        user_t1 = _missing_arg_user_prompt(r, spec, provided_args, has_cue)
        asst_t1 = _missing_arg_assistant_reply(r, spec, missing_keys)

        if stratum == "missing_single":
            messages = [
                {"role": "user", "content": user_t1},
                {"role": "assistant", "content": asst_t1},
            ]
            fixture = {
                "stratum": "missing_required_arg_single",
                "has_cue": has_cue,
                "has_turn2_cue": False,
                "target_tool": spec["name"],
                "missing_keys": missing_keys,
                "withheld_values": {k: full_args[k] for k in missing_keys},
                "expected_turn_types": ["abstain_missing"],
                "expected_tool_calls_by_turn": [[]],
            }
        else:
            supplied_clause = _describe_args_clause({k: full_args[k] for k in missing_keys})
            user_t2_base = f"Sure — use {supplied_clause} and go ahead with `{spec['name']}`."
            user_t2 = _maybe_append_cue(r, user_t2_base, has_turn2_cue)
            messages = [
                {"role": "user", "content": user_t1},
                {"role": "assistant", "content": asst_t1},
                {"role": "user", "content": user_t2},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call-{index}-t2",
                            "type": "function",
                            "function": {"name": spec["name"], "arguments": full_args},
                        }
                    ],
                },
            ]
            fixture = {
                "stratum": "missing_required_arg_twoturn",
                "has_cue": has_cue,
                "has_turn2_cue": has_turn2_cue,
                "target_tool": spec["name"],
                "missing_keys": missing_keys,
                "withheld_values": {k: full_args[k] for k in missing_keys},
                "expected_turn_types": ["abstain_missing", "positive_call"],
                "expected_tool_calls_by_turn": [[], [{"name": spec["name"], "arguments": full_args}]],
            }

    elif stratum == "domain_mismatch":
        tools_specs = _pick_tool_palette(r, [spec])
        u_txt, a_txt = DOMAIN_MISMATCH_PROMPTS[
            (index + r.randint(0, len(DOMAIN_MISMATCH_PROMPTS) - 1)) % len(DOMAIN_MISMATCH_PROMPTS)
        ]
        # Add mild deterministic context framing variation so prompts across 32 packs are diverse
        prefixes = ["", "Quick question: ", "Before we check those systems, ", "Unrelated question — "]
        u_full = _maybe_append_cue(r, prefixes[index % len(prefixes)] + u_txt, has_cue)
        messages = [
            {"role": "user", "content": u_full},
            {"role": "assistant", "content": a_txt},
        ]
        fixture = {
            "stratum": "domain_mismatch_abstain",
            "has_cue": has_cue,
            "has_turn2_cue": False,
            "target_tool": None,
            "missing_keys": [],
            "withheld_values": {},
            "expected_turn_types": ["abstain_mismatch"],
            "expected_tool_calls_by_turn": [[]],
        }

    elif stratum == "positive_single":
        # Minimal-pair-compatible positive control using the exact same prompt surface as missing_arg
        full_args = _sample_full_args(r, spec, include_optional=(index % 2 == 0))
        tools_specs = _pick_tool_palette(r, [spec])
        user_txt = _positive_user_prompt(r, spec, full_args, has_cue)
        messages = [
            {"role": "user", "content": user_txt},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call-{index}-0",
                        "type": "function",
                        "function": {"name": spec["name"], "arguments": full_args},
                    }
                ],
            },
        ]
        fixture = {
            "stratum": "positive_single",
            "has_cue": has_cue,
            "has_turn2_cue": False,
            "target_tool": spec["name"],
            "missing_keys": [],
            "withheld_values": {},
            "expected_turn_types": ["positive_call"],
            "expected_tool_calls_by_turn": [[{"name": spec["name"], "arguments": full_args}]],
        }

    else:  # positive_parallel (2 or 3 calls)
        n_calls = 2 if (index % 2 == 0) else 3
        chosen_specs = r.sample(SKILL_B_TOOLS, n_calls)
        tools_specs = _pick_tool_palette(r, chosen_specs)
        call_specs: list[dict[str, Any]] = []
        prompt_parts: list[str] = []
        tool_calls_msg: list[dict[str, Any]] = []
        for ci, sp in enumerate(chosen_specs):
            c_args = _sample_full_args(r, sp, include_optional=(ci == 0))
            call_specs.append({"name": sp["name"], "arguments": c_args})
            prompt_parts.append(f"({ci + 1}) `{sp['name']}` with {_describe_args_clause(c_args)}")
            tool_calls_msg.append(
                {
                    "id": f"call-{index}-{ci}",
                    "type": "function",
                    "function": {"name": sp["name"], "arguments": c_args},
                }
            )
        user_base = "Please execute all of the following tool calls in parallel: " + "; ".join(prompt_parts) + "."
        user_txt = _maybe_append_cue(r, user_base, has_cue)
        messages = [
            {"role": "user", "content": user_txt},
            {"role": "assistant", "content": "", "tool_calls": tool_calls_msg},
        ]
        fixture = {
            "stratum": "positive_parallel",
            "has_cue": has_cue,
            "has_turn2_cue": False,
            "target_tool": ",".join(s["name"] for s in chosen_specs),
            "missing_keys": [],
            "withheld_values": {},
            "expected_turn_types": ["positive_call"],
            "expected_tool_calls_by_turn": [call_specs],
        }

    openai_tools = [_to_openai_tool(s) for s in tools_specs]
    encoded = render_shifted_row(
        tokenizer,
        messages,
        enable_thinking=False,
        tools=openai_tools,
        select_turns="all",
        training_template=training_template,
    )

    example = {
        "id": item_id,
        "skill": "skill_b_nothink_tool_abstention",
        "split": split,
        "seed_namespace": SKILL_B_SEED_NAMESPACE,
        "stratum": fixture["stratum"],
        "mode": "nothink",
        "domain": spec["domain"],
        "declared_tools": [s["name"] for s in tools_specs],
        "tools": openai_tools,
        "messages": messages,
        "fixture": fixture,
        "sequence_tokens": encoded["sequence_tokens"],
        "target_tokens": encoded["target_tokens"],
        "selected_spans": encoded["selected_spans"],
        "input_ids": encoded["input_ids"],
        "loss_weights": encoded["loss_weights"],
        "rendered_text": encoded["rendered_text"],
        "decoded_targets": encoded["decoded_targets"],
    }
    verify_skill_b_example(example, tokenizer, extra_blocked_tools=extra_blocked_tools)
    return example


def verify_skill_b_example(
    example: dict[str, Any],
    tokenizer: Any,
    *,
    extra_blocked_tools: set[str] | None = None,
) -> dict[str, Any]:
    """Deterministic 100% gate verifier for Skill B examples against their structured fixture."""
    blocked = set(BLOCKED_PROBE3_TOOLS)
    if extra_blocked_tools:
        blocked |= {n.lower() for n in extra_blocked_tools}

    tools_by_name: dict[str, dict[str, Any]] = {}
    for t in example["tools"]:
        fn = t["function"]
        name = fn["name"]
        if name.lower() in blocked:
            raise ValueError(f"{example['id']}: declared tool {name} in eval blocklist")
        params = fn["parameters"]
        if params.get("type") != "object" or not isinstance(params.get("properties"), dict):
            raise ValueError(f"{example['id']}: invalid tool schema for {name}")
        for rk in params.get("required", []):
            if rk not in params["properties"]:
                raise ValueError(f"{example['id']}: required key {rk} not in properties for {name}")
        tools_by_name[name] = params

    if "Reasoning: /nothink" not in example["rendered_text"][:200]:
        raise ValueError(f"{example['id']}: missing Reasoning: /nothink header")

    ids = example["input_ids"]
    weights = example["loss_weights"]
    spans = example["selected_spans"]
    fixture = example["fixture"]
    expected_types = fixture["expected_turn_types"]
    expected_calls_by_turn = fixture["expected_tool_calls_by_turn"]

    if len(spans) != len(expected_types):
        raise ValueError(f"{example['id']}: span count {len(spans)} != expected turns {len(expected_types)}")

    # Verify shifted weights decode every supervised assistant turn cleanly.
    user_t1_text = example["messages"][0]["content"]
    for turn_idx, ((start, end), turn_type, exp_calls) in enumerate(
        zip(spans, expected_types, expected_calls_by_turn, strict=True)
    ):
        if ids[end - 1] != EOT_ID:
            raise ValueError(f"{example['id']} turn {turn_idx}: span does not end with EOT={EOT_ID}")
        decoded = tokenizer.decode(
            [ids[i + 1] for i in range(start - 1, end - 1) if weights[i] > 0.0],
            skip_special_tokens=False,
        )
        if not decoded.endswith("<|eot_id|>"):
            raise ValueError(f"{example['id']} turn {turn_idx}: decoded target missing <|eot_id|>")
        body = decoded[: -len("<|eot_id|>")]

        if "<|start_think|>" in body or "<|end_think|>" in body:
            raise ValueError(f"{example['id']} turn {turn_idx}: thinking marker present in /nothink target")

        if turn_type in ("abstain_missing", "abstain_mismatch"):
            if "<tool_call>" in body or "</tool_call>" in body:
                raise ValueError(f"{example['id']} turn {turn_idx}: tool_call tag emitted on abstention turn")
            wc = len(body.split())
            if not (10 <= wc <= 80):
                raise ValueError(f"{example['id']} turn {turn_idx}: abstention word count {wc} outside [10, 80]")
            if turn_type == "abstain_missing":
                if not REQUEST_PHRASE_RE.search(body):
                    raise ValueError(f"{example['id']} turn {turn_idx}: clarification lacks explicit request phrase")
                for mk in fixture["missing_keys"]:
                    if mk not in body:
                        raise ValueError(
                            f"{example['id']} turn {turn_idx}: clarification does not name missing key {mk}"
                        )
                    for gt_s in _extract_ground_truth_strings(fixture["withheld_values"][mk]):
                        if gt_s and re.search(rf"(?<![A-Za-z0-9_]){re.escape(gt_s)}(?![A-Za-z0-9_])", user_t1_text):
                            raise ValueError(
                                f"{example['id']}: withheld fixture value {gt_s!r} for {mk} leaked into user turn 1"
                            )
        elif turn_type == "positive_call":
            n_open = body.count("<tool_call>")
            n_close = body.count("</tool_call>")
            if n_open == 0 or n_open != n_close:
                raise ValueError(f"{example['id']} turn {turn_idx}: unbalanced or missing tool_call tags")
            raw_blocks = re.findall(r"<tool_call>(.*?)</tool_call>", body, re.S)
            if len(raw_blocks) != len(exp_calls):
                raise ValueError(f"{example['id']} turn {turn_idx}: parsed {len(raw_blocks)} calls != {len(exp_calls)}")
            parsed_calls = []
            for blk in raw_blocks:
                obj = json.loads(blk)
                if not isinstance(obj, dict) or set(obj.keys()) != {"name", "arguments"}:
                    raise ValueError(f"{example['id']} turn {turn_idx}: invalid tool_call object keys {obj.keys()}")
                cname = obj["name"]
                cargs = obj["arguments"]
                if cname not in tools_by_name:
                    raise ValueError(f"{example['id']} turn {turn_idx}: undeclared tool {cname}")
                schema = tools_by_name[cname]
                if not _check_json_type(cargs, schema):
                    raise ValueError(
                        f"{example['id']} turn {turn_idx}: args failed strict JSON Schema check for {cname}"
                    )
                parsed_calls.append(obj)
            if parsed_calls != exp_calls:
                raise ValueError(f"{example['id']} turn {turn_idx}: parsed calls {parsed_calls} != fixture {exp_calls}")
        else:
            raise ValueError(f"Unknown turn_type {turn_type!r}")

    if ids[-1] != 128001 or weights[-1] != 0.0:
        raise ValueError(f"{example['id']}: missing EOS=128001 boundary at weight 0")

    return {
        "passed": True,
        "id": example["id"],
        "stratum": fixture["stratum"],
        "target_tokens": example["target_tokens"],
    }


def to_parquet_row(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": example["id"],
        "skill": example["skill"],
        "split": example["split"],
        "seed_namespace": example["seed_namespace"],
        "stratum": example["stratum"],
        "has_cue": bool(example["fixture"]["has_cue"]),
        "has_turn2_cue": bool(example["fixture"]["has_turn2_cue"]),
        "domain": example["domain"],
        "mode": example["mode"],
        "declared_tools_json": json.dumps(example["declared_tools"]),
        "messages_json": json.dumps(example["messages"], ensure_ascii=False),
        "tools_json": json.dumps(example["tools"], ensure_ascii=False),
        "fixture_json": json.dumps(example["fixture"], ensure_ascii=False),
        "sequence_tokens": int(example["sequence_tokens"]),
        "target_tokens": int(example["target_tokens"]),
        "input_ids": example["input_ids"],
        "loss_weights": example["loss_weights"],
    }
