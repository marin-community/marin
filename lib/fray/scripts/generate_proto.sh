#!/bin/bash
# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FRAY_DIR="$(dirname "$SCRIPT_DIR")"

if ! command -v buf &> /dev/null; then
    echo "Error: buf is not installed. Please install buf from https://buf.build/docs/installation"
    exit 1
fi

cd "$FRAY_DIR"

echo "Generating protobuf and Connect RPC code with buf..."
buf generate

echo "Fixing imports in generated files..."
PROTO_OUT="$FRAY_DIR/src/fray/job/rpc/proto"

# Fix imports in fray_connect.py and fray_pb2_grpc.py to use relative imports
for file in "$PROTO_OUT/fray_connect.py" "$PROTO_OUT/fray_pb2_grpc.py"; do
    if [ -f "$file" ]; then
        sed -i '' 's/^import fray_pb2 as fray__pb2$/from . import fray_pb2 as fray__pb2/g' "$file"
    fi
done

echo "Cleaning up __pycache__ directories..."
find "$PROTO_OUT" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

echo "Generated files:"
ls -1 "$PROTO_OUT"/*.py 2>/dev/null || echo "No Python files generated"

echo "Generation complete."
