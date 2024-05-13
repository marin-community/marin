from google.cloud import storage

storage_client = storage.Client()
bucket = storage_client.get_bucket("levanter-data")
filenames = [
                "wiki-0000.json.gz",
                "wiki-0001.json.gz",
            ]

for filename in filenames:
    blob = bucket.blob(f"{filename}")
    contents = blob.download_to_filename(f"/nlp/scr/cychou/{filename}")
    

