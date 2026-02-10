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

from connectrpc.code import Code

from iris.rpc.errors import connect_error_with_traceback, extract_error_details


def test_connect_error_with_traceback_populates_timestamp() -> None:
    try:
        raise ValueError("boom")
    except ValueError as exc:
        err = connect_error_with_traceback(Code.INTERNAL, "Error launching job", exc=exc)

    details = extract_error_details(err)
    assert details is not None
    assert details.exception_type.endswith("ValueError")
    assert details.timestamp.epoch_ms > 0
