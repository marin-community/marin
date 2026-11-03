"""
Data models and metadata cleaning utilities for SkyRL database registration.

Copied from LLaMA-Factory's unified_db module.
"""

from typing import Any, Dict


def clean_dataset_metadata(dataset_data: Dict[str, Any]) -> Dict[str, Any]:
    """Clean dataset metadata for API responses."""
    if not dataset_data:
        return {}

    cleaned = {
        "id": str(dataset_data.get("id")) if dataset_data.get("id") else None,
        "name": dataset_data.get("name"),
        "created_by": dataset_data.get("created_by"),
        "creation_location": dataset_data.get("creation_location"),
        "creation_time": dataset_data.get("creation_time"),
        "generation_start": dataset_data.get("generation_start"),
        "generation_end": dataset_data.get("generation_end"),
        "data_location": dataset_data.get("data_location"),
        "generation_parameters": dataset_data.get("generation_parameters", {}),
        "generation_status": dataset_data.get("generation_status"),
        "dataset_type": dataset_data.get("dataset_type"),
        "data_generation_hash": dataset_data.get("data_generation_hash"),
        "hf_fingerprint": dataset_data.get("hf_fingerprint"),
        "hf_commit_hash": dataset_data.get("hf_commit_hash"),
        "num_tasks": dataset_data.get("num_tasks"),
        "last_modified": dataset_data.get("last_modified"),
        "updated_at": dataset_data.get("updated_at"),
    }

    return {k: v for k, v in cleaned.items() if v is not None}


def clean_model_metadata(model_data: Dict[str, Any]) -> Dict[str, Any]:
    """Clean model metadata for API responses."""
    if not model_data:
        return {}

    cleaned = {
        "id": str(model_data.get("id")) if model_data.get("id") else None,
        "name": model_data.get("name"),
        "base_model_id": str(model_data.get("base_model_id")) if model_data.get("base_model_id") else None,
        "created_by": model_data.get("created_by"),
        "creation_location": model_data.get("creation_location"),
        "creation_time": model_data.get("creation_time"),
        "dataset_id": str(model_data.get("dataset_id")) if model_data.get("dataset_id") else None,
        "is_external": model_data.get("is_external"),
        "weights_location": model_data.get("weights_location"),
        "wandb_link": model_data.get("wandb_link"),
        "updated_at": model_data.get("updated_at"),
        "training_start": model_data.get("training_start"),
        "training_end": model_data.get("training_end"),
        "training_parameters": model_data.get("training_parameters", {}),
        "training_status": model_data.get("training_status"),
        "agent_id": str(model_data.get("agent_id")) if model_data.get("agent_id") else None,
        "training_type": model_data.get("training_type"),
        "traces_location_s3": model_data.get("traces_location_s3"),
        "description": model_data.get("description"),
        "dataset_names": model_data.get("dataset_names"),
    }

    return {k: v for k, v in cleaned.items() if v is not None}


def clean_agent_metadata(agent_data: Dict[str, Any]) -> Dict[str, Any]:
    """Clean agent metadata for API responses."""
    if not agent_data:
        return {}

    cleaned = {
        "id": str(agent_data.get("id")) if agent_data.get("id") else None,
        "name": agent_data.get("name"),
        "agent_version_hash": agent_data.get("agent_version_hash"),
        "description": agent_data.get("description"),
        "updated_at": agent_data.get("updated_at"),
    }

    return {k: v for k, v in cleaned.items() if v is not None}
