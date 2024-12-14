# runs the build_rpv2_bloom_filter as a gcloud task
import json
import os

import fsspec
import google
from google.auth import default
from google.auth.transport.requests import Request
import google.oauth2.id_token
from google.cloud import tasks_v2

from build_rpv2_bloom_filter import _fsspec_exists, all_paths_for_crawl
from marin.web.rpv2 import RPV2_CRAWLS


def create_process_slice_task(out_path, lang, crawl, part, chunk_range, credentials, id_token):
    # Package relevant data needed for the task into a dictionary
    task_data = {
        "out_path": out_path,
        "lang": lang,
        "crawl": crawl,
        "part": part,
        "chunk_range": list(chunk_range),  # Make sure to convert the range into a list
    }

    # Specify your project ID and Cloud Task queue information
    project_id = "hai-gcp-models"
    queue_name = "rpv1-crawl-bloom-filters"
    location = "us-central1"

    # Specify the URL of the Cloud Function that will process the task
    url = "https://us-central1-hai-gcp-models.cloudfunctions.net/build_rpv2_bloom_filter"
    # url = "https://us-central1-hai-gcp-models.cloudfunctions.net/hello_world_function"

    # Create a Cloud Task
    client = tasks_v2.CloudTasksClient(credentials=credentials)

    queue_path = client.queue_path(project_id, location, queue_name)

    # Create the task object
    # task = {
    #     "http_request": {
    #             "http_method": tasks_v2.HttpMethod.POST,
    #             "url": url,
    #             "body": json.dumps(task_data).encode(),
    #             "headers": {"Authorization": f"Bearer {id_token}",
    #                         "Content-Type": "application/json"}
    #     },
    #     # "dispatch_deadline": 30 * 60,  # 30 minutes
    # }

    task = tasks_v2.Task(
        http_request=tasks_v2.HttpRequest(
            http_method=tasks_v2.HttpMethod.POST,
            url=url,
            headers={
                "Content-type": "application/json",
                "Authorization": f"Bearer {id_token}",
            },
            body=json.dumps(task_data).encode(),
        ),
        name=None,
    )
    task.dispatch_deadline = {"seconds": 30 * 60}

    # response = client.create_task(request={"parent": queue_path, "task": task})
    task_resp = client.create_task(
        tasks_v2.CreateTaskRequest(
            # The queue to add the task to
            parent=queue_path,
            # The task itself
            task=task,
        )
    )
    print(task_resp)
    return task_resp


def create_union_blooms_task(out_path, paths, credentials, id_token):
    # Package relevant data needed for the task into a dictionary
    task_data = {
        "out_path": out_path,
        "paths": paths,
    }

    # Specify your project ID and Cloud Task queue information
    project_id = "hai-gcp-models"
    queue_name = "rpv1-crawl-bloom-filters"
    location = "us-central1"

    # Specify the URL of the Cloud Function that will process the task
    url = "https://us-central1-hai-gcp-models.cloudfunctions.net/union_blooms"

    # Create a Cloud Task
    client = tasks_v2.CloudTasksClient(credentials=credentials)

    queue_path = client.queue_path(project_id, location, queue_name)

    # Create the task object
    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": url,
            "body": json.dumps(task_data).encode(),
            "headers": {"Authorization": f"Bearer {id_token}", "Content-Type": "application/json"},
        },
    }

    response = client.create_task(request={"parent": queue_path, "task": task})
    return response


def purge_merged_blooms(out_path, paths):
    if not _fsspec_exists(out_path):
        return "Merged bloom does not exist, not purging", 412

    for path in paths:
        fs = fsspec.core.url_to_fs(path)[0]
        fs.rm(path)


if __name__ == "__main__":
    lang = "en"
    # need to be able to invoke a gen2 cloud function
    credentials, project = google.auth.default(
        scopes=[
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/cloud-tasks",
            # also need to read/write gcs
            "https://www.googleapis.com/auth/devstorage.read_write",
        ]
    )

    PATH = "gs://levanter-data/marin/v0/url_blooms/"

    skipped = 0
    total = 0

    auth_req = Request()
    id_token = google.oauth2.id_token.fetch_id_token(
        auth_req, audience="https://us-central1-hai-gcp-models.cloudfunctions.net/build_rpv2_bloom_filter"
    )

    for crawl in RPV2_CRAWLS[0:10]:
        for part, slice, path in all_paths_for_crawl(PATH, crawl, lang):
            if _fsspec_exists(path):
                skipped += 1
                continue
            create_process_slice_task(path, lang, crawl, part, slice, credentials, id_token)
            total += 1

    print(f"Skipped {skipped} tasks, created {total} tasks")

    # auth_req = Request()
    # id_token = google.oauth2.id_token.fetch_id_token(auth_req, audience="https://us-central1-hai-gcp-models.cloudfunctions.net/union_blooms")
    #
    # for crawl in RPV2_CRAWLS[:1]:
    #     paths = []
    #     for part, slice, path in all_paths_for_crawl(PATH, crawl, lang):
    #         paths.append(path)
    #
    #     out_path = f"gs://levanter-data/marin/v0/url_blooms/{lang}_{crawl}.bloom"
    #     create_union_blooms_task(out_path, paths, credentials, id_token)
