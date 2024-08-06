#!/bin/bash
# Delete all docker images except the most recent 2 ones by each docker image name (created by users)

# Default number of images to keep
keep_count=2

# Get unique image names
image_names=$(docker images --format "{{.Repository}}" | sort -u)

for name in $image_names
do
  # Get all image IDs for this name, sorted by creation date (newest first)
  image_ids=$(docker images --format "{{.Repository}}|{{.ID}}|{{.CreatedAt}}" "$name" | sort -t '|' -k3 -r | awk -F '|' '{print $2}')
  
  # Convert to array
  IFS=$'\n' read -d '' -r -a id_array <<< "$image_ids"
  
  # If there are more than $keep_count images, remove the older ones
  if [ ${#id_array[@]} -gt $keep_count ]; then
    echo "Processing repository: $name"
    for i in "${!id_array[@]}"; do
      if [ $i -ge $keep_count ]; then
        echo "Attempting to remove image: ${id_array[$i]}"
        if docker rmi --force "${id_array[$i]}" 2>/dev/null; then
          echo "Successfully removed image: ${id_array[$i]}"
        else
          echo "Failed to remove image: ${id_array[$i]}. It may be used by other repositories."
        fi
      fi
    done
  else
    echo "Repository $name has $keep_count or fewer images. No action needed."
  fi
done
