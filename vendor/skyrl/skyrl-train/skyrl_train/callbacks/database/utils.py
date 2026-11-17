"""
Database registration utilities for SkyRL.

This module provides functions for registering trained RL models, agents,
and datasets to Supabase. Copied from LLaMA-Factory's unified_db module.
"""

import json
import logging
import os
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .config import get_default_client, get_admin_client
from .models import clean_dataset_metadata, clean_model_metadata, clean_agent_metadata

logger = logging.getLogger(__name__)


def load_supabase_keys() -> bool:
    """Load Supabase credentials from KEYS env var if available."""
    keys_env = os.environ.get("KEYS")
    if not keys_env:
        warnings.warn(
            "Supabase credentials not loaded: set KEYS env variable to a secrets file to enable database registration."
        )
        return False

    keys_path = os.path.expandvars(keys_env)
    if not os.path.isfile(keys_path):
        warnings.warn(
            f"Supabase credentials file not found at '{keys_path}'. "
            "Model uploads will not be registered in the database until KEYS points to a valid file."
        )
        return False

    try:
        with open(keys_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].strip()
                key, value = line.split("=", 1)
                key = key.strip()
                if not key:
                    continue
                value = value.strip().strip('"').strip("'")
                os.environ[key] = os.path.expandvars(value)
    except Exception as exc:
        warnings.warn(
            f"Failed to load Supabase credentials from '{keys_path}': {exc!r}. Database registration will be skipped."
        )
        return False

    required = ["SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_SERVICE_ROLE_KEY"]
    missing = [var for var in required if not os.environ.get(var)]
    if missing:
        warnings.warn(
            "Missing Supabase settings "
            f"{', '.join(missing)} after loading KEYS file. "
            "Model uploads will not be registered; ensure the KEYS file exports these values."
        )
        return False

    return True


def get_supabase_client(use_admin: bool = False):
    """Get Supabase client for database operations."""
    if use_admin:
        return get_admin_client()
    return get_default_client()


# ==================== DATASET UTILITIES ====================


def get_dataset_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Retrieve a dataset from the database by name."""
    try:
        client = get_supabase_client()
        response = client.table("datasets").select("*").eq("name", name).execute()

        if not response.data:
            return None

        return clean_dataset_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error retrieving dataset by name {name}: {e}")
        return None


def create_dataset(dataset_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new dataset in the database."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table("datasets").insert(dataset_data).execute()

        if not response.data:
            raise ValueError("Failed to create dataset")

        return clean_dataset_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        raise


def update_dataset(dataset_id: str, dataset_data: Dict[str, Any]) -> Dict[str, Any]:
    """Update an existing dataset in the database."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table("datasets").update(dataset_data).eq("id", dataset_id).execute()

        if not response.data:
            raise ValueError(f"Failed to update dataset with ID {dataset_id}")

        return clean_dataset_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error updating dataset {dataset_id}: {e}")
        raise


def register_hf_dataset(
    repo_name: str,
    dataset_type: str,
    name: Optional[str] = None,
    created_by: Optional[str] = None,
    forced_update: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Register a HuggingFace dataset (simplified version for SkyRL)."""
    try:
        dataset_name = name or repo_name
        existing = get_dataset_by_name(dataset_name)
        if existing and not forced_update:
            logger.info(f"Dataset {dataset_name} already exists")
            return {"success": True, "dataset": existing, "exists": True}

        if not created_by:
            if "/" in repo_name:
                created_by = repo_name.split("/")[0]
            else:
                created_by = "hf-uploader"

        now = datetime.now(timezone.utc)
        dataset_data = {
            "creation_time": now.isoformat(),
            "updated_at": now.isoformat(),
            "name": dataset_name,
            "created_by": created_by,
            "data_location": f"https://huggingface.co/datasets/{repo_name}",
            "creation_location": "HuggingFace",
            "generation_status": "completed",
            "generation_parameters": {
                "hf_repo": repo_name,
                "source": "huggingface_hub",
                "registered_at": now.isoformat(),
            },
            "dataset_type": dataset_type,
        }

        if kwargs:
            dataset_data.update(kwargs)

        if existing:
            updated = update_dataset(existing["id"], dataset_data)
            return {"success": True, "dataset": updated, "updated": True}
        else:
            created = create_dataset(dataset_data)
            return {"success": True, "dataset": created}

    except Exception as e:
        logger.error(f"Failed to register HuggingFace dataset {repo_name}: {e}")
        return {"success": False, "error": str(e)}


# ==================== MODEL UTILITIES ====================


def get_model_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Retrieve a model from the database by name."""
    try:
        client = get_supabase_client()
        response = client.table("models").select("*").eq("name", name).execute()

        if not response.data:
            return None

        return clean_model_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error retrieving model by name {name}: {e}")
        return None


def create_model(model_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new model in the database."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table("models").insert(model_data).execute()

        if not response.data:
            raise ValueError("Failed to create model")

        return clean_model_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        raise


def update_model(model_id: str, model_data: Dict[str, Any]) -> Dict[str, Any]:
    """Update an existing model in the database."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table("models").update(model_data).eq("id", model_id).execute()

        if not response.data:
            raise ValueError(f"Failed to update model with ID {model_id}")

        return clean_model_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error updating model {model_id}: {e}")
        raise


# ==================== AGENT UTILITIES ====================


def get_agent_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Retrieve an agent from the database by name."""
    try:
        client = get_supabase_client()
        response = client.table("agents").select("*").eq("name", name).execute()

        if not response.data:
            return None

        return clean_agent_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error retrieving agent by name {name}: {e}")
        return None


def create_agent(agent_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new agent in the database."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table("agents").insert(agent_data).execute()

        if not response.data:
            raise ValueError("Failed to create agent")

        return clean_agent_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        raise


def update_agent(agent_id: str, agent_data: Dict[str, Any]) -> Dict[str, Any]:
    """Update an existing agent in the database."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table("agents").update(agent_data).eq("id", agent_id).execute()

        if not response.data:
            raise ValueError(f"Failed to update agent with ID {agent_id}")

        return clean_agent_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error updating agent {agent_id}: {e}")
        raise


def register_agent(
    name: str, agent_version_hash: Optional[str] = None, description: Optional[str] = None
) -> Dict[str, Any]:
    """Register or update an agent in the database."""
    try:
        existing = get_agent_by_name(name)
        now = datetime.now(timezone.utc).isoformat()
        agent_data = {
            "name": name,
            "agent_version_hash": agent_version_hash,
            "description": description,
            "updated_at": now,
        }

        if existing:
            updated = update_agent(existing["id"], agent_data)
            return {"success": True, "agent": updated, "updated": True}
        else:
            created = create_agent(agent_data)
            return {"success": True, "agent": created}

    except Exception as e:
        logger.error(f"Failed to register agent {name}: {e}")
        return {"success": False, "error": str(e)}


# ==================== TRAINED MODEL REGISTRATION ====================


def register_trained_model(training_record: Dict[str, Any], forced_update: bool = False) -> Dict[str, Any]:
    """
    Register a newly trained model (SFT/RL) to the database.

    Args:
        training_record: Dictionary containing:
            - agent_name: Name of the training agent
            - training_start: ISO datetime string when training started
            - training_end: ISO datetime string when training ended
            - created_by: Username/creator identifier
            - base_model_name: Base model path (e.g., "Qwen/Qwen3-8B")
            - dataset_name or dataset_names: Training dataset(s)
            - training_type: "SFT" or "RL"
            - training_parameters: Dict of training config
            - wandb_link: Optional W&B run URL
            - traces_location_s3: Optional S3 path for traces
            - model_name: Optional explicit model name (HF repo ID)
        forced_update: If True, update existing model records

    Returns:
        Dict with success status and model data or error message
    """
    try:

        def _unwrap(value):
            """Unwrap single-element lists/tuples to their value."""
            if isinstance(value, (list, tuple, set)):
                if not value:
                    return None
                try:
                    first = value[0]
                except TypeError:
                    first = next(iter(value))
                return _unwrap(first)
            return value

        agent_name = _unwrap(training_record.get("agent_name"))
        base_model_name = _unwrap(training_record.get("base_model_name"))
        training_type = _unwrap(training_record.get("training_type"))

        if not agent_name:
            return {"success": False, "error": "agent_name is required"}
        if not base_model_name:
            return {"success": False, "error": "base_model_name is required"}
        if training_type not in ("SFT", "RL"):
            return {"success": False, "error": "training_type must be 'SFT' or 'RL'"}

        def _normalize_dataset_list(raw: Any) -> List[str]:
            """Normalize dataset names to a list of strings."""
            if raw is None:
                return []
            if isinstance(raw, str):
                parts = raw.split(",")
            elif isinstance(raw, (list, tuple, set)):
                parts = list(raw)
            else:
                parts = [raw]
            normalized: List[str] = []
            for item in parts:
                name = str(item).strip()
                if name and name not in normalized:
                    normalized.append(name)
            return normalized

        dataset_list = _normalize_dataset_list(training_record.get("dataset_names"))
        if not dataset_list:
            dataset_list = _normalize_dataset_list(training_record.get("dataset_name"))
        if not dataset_list:
            return {"success": False, "error": "dataset_name is required"}

        def _parse_ts(val):
            """Parse timestamp from string or datetime."""
            if val is None:
                return None
            if isinstance(val, datetime):
                return val
            if isinstance(val, str):
                if val.endswith("Z"):
                    return datetime.fromisoformat(val.replace("Z", "+00:00"))
                return datetime.fromisoformat(val)
            raise ValueError("timestamp must be datetime or ISO string")

        raw_start = training_record.get("training_start")
        if not raw_start:
            return {"success": False, "error": "training_start is required"}
        training_start_dt = _parse_ts(raw_start)
        training_end_dt = _parse_ts(training_record.get("training_end"))

        # Clean training parameters
        training_params = training_record.get("training_parameters")
        if training_params is None:
            training_params = {}
        elif isinstance(training_params, str):
            try:
                training_params = json.loads(training_params)
            except Exception:
                training_params = {"raw": training_params}
        elif isinstance(training_params, dict):
            cleaned: Dict[str, Any] = {}
            for key, value in training_params.items():
                try:
                    json.dumps(value)
                    cleaned[key] = value
                except TypeError:
                    cleaned[key] = str(value)
            training_params = cleaned
        else:
            try:
                json.dumps(training_params)
                training_params = {"value": training_params}
            except TypeError:
                training_params = {"raw": str(training_params)}

        created_by = training_record.get("created_by") or ""
        wandb_link = training_record.get("wandb_link") or ""
        traces_location_s3 = training_record.get("traces_location_s3") or ""
        explicit_name = training_record.get("model_name") or ""

        # Register agent
        agent_res = register_agent(name=agent_name)
        if not agent_res.get("success"):
            return agent_res
        agent = agent_res["agent"]
        agent_id = agent["id"]

        # Register datasets
        dataset_id: Optional[str] = None
        dataset_names_csv: Optional[str] = ",".join(dataset_list)
        if len(dataset_list) == 1:
            dataset_name_single = dataset_list[0]
            ds = get_dataset_by_name(dataset_name_single)
            if not ds:
                ds_res = register_hf_dataset(
                    repo_name=dataset_name_single,
                    dataset_type=training_type,
                    name=dataset_name_single,
                    created_by=created_by,
                )
                if not ds_res.get("success"):
                    return {"success": False, "error": ds_res.get("error", "Dataset registration failed")}
                ds = ds_res["dataset"]
            dataset_id = ds["id"]
        else:
            for name in dataset_list:
                ds = get_dataset_by_name(name)
                if not ds:
                    ds_res = register_hf_dataset(
                        repo_name=name,
                        dataset_type=training_type,
                        name=name,
                        created_by=created_by,
                    )
                    if not ds_res.get("success"):
                        return {"success": False, "error": ds_res.get("error", "Dataset registration failed")}

        # Register or find base model
        base_m = get_model_by_name(base_model_name)
        if not base_m:
            now_dt = datetime.now(timezone.utc)
            now_ts = now_dt.isoformat()
            base_training_start = training_start_dt or now_dt
            base_training_end = training_end_dt or base_training_start
            base_payload = {
                "name": base_model_name,
                "created_by": (
                    created_by or (base_model_name.split("/")[0] if "/" in base_model_name else "hf-uploader")
                ),
                "creation_location": "HuggingFace",
                "creation_time": now_ts,
                "updated_at": now_ts,
                "is_external": True,
                "weights_location": f"https://huggingface.co/{base_model_name}",
                "training_status": "completed",
                "training_parameters": {
                    "source": "huggingface_hub",
                    "registered_at": now_ts,
                },
                "agent_id": agent_id,
                "training_type": training_type,
                "training_start": base_training_start.isoformat(),
                "training_end": base_training_end.isoformat(),
            }
            base_m = create_model(base_payload)
        base_model_id = base_m["id"]

        # Determine model name
        if explicit_name:
            model_name = explicit_name
        else:
            dataset_name_for_default = dataset_list[0]
            date_str = (training_end_dt or training_start_dt).strftime("%Y%m%d")
            model_name = f"{dataset_name_for_default}_{date_str}"

        existing = get_model_by_name(model_name)
        now_ts = datetime.now(timezone.utc).isoformat()
        weights_location = f"https://huggingface.co/{model_name}"
        training_status = "completed" if training_end_dt else "in_progress"

        model_data = {
            "name": model_name,
            "created_by": created_by,
            "creation_location": "HuggingFace",
            "creation_time": now_ts,
            "updated_at": now_ts,
            "is_external": True,
            "weights_location": weights_location,
            "training_status": training_status,
            "training_parameters": training_params,
            "description": None,
            "agent_id": agent_id,
            "base_model_id": base_model_id,
            "dataset_id": dataset_id,
            "dataset_names": dataset_names_csv,
            "training_type": training_type,
            "training_start": training_start_dt.isoformat(),
            "training_end": training_end_dt.isoformat() if training_end_dt else None,
            "wandb_link": wandb_link,
            "traces_location_s3": traces_location_s3,
        }

        if existing and not forced_update:
            return {"success": True, "model": existing, "exists": True}
        if existing and forced_update:
            updated = update_model(existing["id"], model_data)
            return {"success": True, "model": updated, "updated": True}
        created = create_model(model_data)
        return {"success": True, "model": created}

    except Exception as e:
        logger.error(f"Failed to register trained model: {e}")
        return {"success": False, "error": str(e)}
