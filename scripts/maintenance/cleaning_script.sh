#!/bin/bash
# set cron job to run the following script daily

# Print current date and time
echo "Script started at: $(date)"

# clean up cache directory
rm -rf /home/shared/.cache

# clean up gcloud log
rm -rf /home/shared/.config/gcloud/logs

# clean up old docker image
bash /home/shared/maintenance/delete_old_docker_images.sh

# Print the status of the disk
df

# clean up inactive TPU queued resources
bash /home/shared/maintenance/delete_inactive_tpu_queued_resources.sh us-central2-b
bash /home/shared/maintenance/delete_inactive_tpu_queued_resources.sh us-west4-a
bash /home/shared/maintenance/delete_inactive_tpu_queued_resources.sh europe-west4-b
