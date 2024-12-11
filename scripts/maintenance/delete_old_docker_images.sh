#!/bin/bash
# Delete all docker images except the most recent 2 ones by each docker image name (created by users)

# The number of the most recent images to keep
keep_count=2

# For each image names, sort by their creation time, delete all except the most recent ones
docker images --format "{{.Repository}}" | sort -u | while read repo; do
    docker images --format "{{.ID}}" "$repo" |
    sort -r |
    tail -n +$((keep_count + 1)) |
    xargs -r docker rmi --force
done
