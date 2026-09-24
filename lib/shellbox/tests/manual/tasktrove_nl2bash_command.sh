# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

bash /setup_files/setup_seeds.sh
cd /workspace
find . -path ./ignored_directory -prune -o -type f -iname 'fileName*.txt' -print0 2>&1 | sort -z > /output/command_capture.txt 2>&1
test -f /output/command_capture.txt
