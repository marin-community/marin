import contextlib
import json
import logging
import os
import shutil
import tempfile
import time
import urllib.parse
from typing import Optional

import fsspec
import jax
import requests  # type: ignore
from fsspec import AbstractFileSystem
from jax.experimental.multihost_utils import sync_global_devices


logger = logging.getLogger(__name__)

SENTINEL_FILE = "/tmp/tpu_shutdown_sentinel"


def _checked_request(url):
    try:
        response = requests.get(url, headers={"Metadata-Flavor": "Google"})
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException:
        logger.exception(f"Could not get {url} from metadata server. Is this a TPU VM?", exc_info=True)
        raise


def _checked_delete(url):
    # first get the token
    token = _checked_request(
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
    )
    token = json.loads(token)["access_token"]
    headers = {"Authorization": f"Bearer {token}", "Metadata-Flavor": "Google"}
    try:
        response = requests.delete(url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException:
        logger.exception(f"Could not delete {url} from metadata server. Is this a TPU VM?", exc_info=True)
        raise


def _shutdown_tpu_with_queued_resource():
    queued_resource = _checked_request(
        "http://metadata.google.internal/computeMetadata/v1/instance/attributes/queued-resource-name"
    )
    # queued resource looks like:
    # projects/999999/locations/us-central2-b/queuedResources/NAME
    # to delete we need to use delete against
    # https://tpu.googleapis.com/v2/projects/9999/locations/us-central2-b/queuedResources/NAME?force=true
    if queued_resource:
        queued_resource_name = queued_resource.split("/")[-1]
        # quiet really works like -y
        if jax.process_index() == 0:
            logger.critical(f"Found queued resource {queued_resource_name}. Attempting to delete it.")
            # We need to use curl
            # curl -X DELETE -H "Authorization: Bearer $(gcloud auth print-access-token)" \
            # -H "Content-Type: application/json" \
            # https://tpu.googleapis.com/v2/projects/my-project/locations/us-central2-b/queuedResources/my-queued-resource?force=true
            # os.system(f"gcloud compute tpus queued-resources delete {queued_resource} --zone {zone} --force --quiet")
            url = f"https://tpu.googleapis.com/v2/{queued_resource}?force=true"
            _checked_delete(url)
        return True
    else:
        logger.info("No queued resource found.")
        return False


def shutdown_tpu_vm(sleep_seconds=60 * 5):
    """You should probably call this from atexit or something like that."""
    # fork a process to do the delete so the main process can exit before the delete is done
    logger.info("Forking a process to delete...")
    logger.critical(f"Create a file {SENTINEL_FILE} to cancel the shutdown")
    logger.critical(f"$ touch {SENTINEL_FILE}")

    # fork works better for our use case
    pid = os.fork()
    if pid == 0:
        _do_shutdown_tpu_vm(sleep_seconds)
        os._exit(0)
    else:
        logger.info(f"Forked process {pid} to delete TPU VM")


def _do_shutdown_tpu_vm(sleep_seconds):
    # the gcloud command we would run is something like:
    # gcloud compute tpus tpu-vm delete tpu-vm-1 --zone us-central1-a --quiet
    try:
        zone = _checked_request("http://metadata.google.internal/computeMetadata/v1/instance/zone")
        zone = zone.split("/")[-1]
        name = _checked_request("http://metadata.google.internal/computeMetadata/v1/instance/attributes/instance-id")
        project = _checked_request("http://metadata.google.internal/computeMetadata/v1/project/project-id")
    except requests.exceptions.RequestException:
        logger.warning("Could not get zone or instance-id from metadata server. Is this a TPU VM? Not shutting down.")
        return

    logger.critical(f"Shutting down TPU VM {name} in zone {zone} in {sleep_seconds} seconds")
    time.sleep(sleep_seconds)
    if os.path.exists(SENTINEL_FILE):
        logger.critical(f"Found sentinel file {SENTINEL_FILE}, not shutting down TPU VM")
        return
    logger.critical(f"Shutting down TPU VM {name} in zone {zone}")

    try:
        success = _shutdown_tpu_with_queued_resource()
        if success:
            return
    except requests.exceptions.RequestException:
        logger.info("This is not a queued resource, deleting the old fashioned way.")

    logger.critical(f"Shutting down TPU VM {name} in zone {zone}")
    if jax.process_index() != 0:
        logger.info(f"Letting process 0 handle the shutdown. We are process {jax.process_index()}")
        return

    # os.system(f"gcloud compute tpus tpu-vm delete {name} --zone {zone} --quiet")
    # https://tpu.googleapis.com/v2/projects/PROJECT/locations/us-central2-b/nodes/NAME
    url = f"http://tpu.googleapis.com/v2/projects/{project}/locations/{zone}/nodes/{name}"
    _checked_delete(url)


_sync_count = 0


@contextlib.contextmanager
def temp_dir_before_upload(path, process_should_upload: Optional[bool] = None):
    """
    Creates a temp_dir, yields it, then uploads it to the given path or url on exit using fsspec.
    If it's a local path instead of a url, it just yields the path.

    :param process_should_upload: If None, only upload if jax.process_index() == 0. Otherwise, upload if this is True.
    """

    if process_should_upload is None:
        process_should_upload = jax.process_index() == 0

    def _is_url_like(path):
        return urllib.parse.urlparse(path).scheme != ""

    try:
        tmpdir: Optional[str] = None
        if _is_url_like(path):
            tmpdir = tempfile.mkdtemp()
            local_path = tmpdir
        else:
            local_path = path

        yield local_path

        if tmpdir is not None:  # we're uploading to GCS or similar
            if process_should_upload:
                logger.info(f"Copying from temp dir to {path}")
                fs: AbstractFileSystem = fsspec.core.get_fs_token_paths(path, mode="wb")[0]
                fs.put(os.path.join(local_path, "*"), path, recursive=True)
                logger.info(f"Finished copying to {path}")
            else:
                logger.info(f"Waiting for process 0 to finish saving checkpoint to {path}")

            global _sync_count
            sync_global_devices(f"upload? {path}{_sync_count}")
            _sync_count += 1
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)
