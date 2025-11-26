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

import timeit

import time

NUMBER = 1000000


def format_time(time_ns: float) -> str:
    return f"{time_ns / 1000:.04} us"


def main():
    res = timeit.timeit(
        setup=f"from rbloom import Bloom; b = Bloom({NUMBER}, 0.01)",
        stmt="b.add(object())",
        timer=time.perf_counter_ns,
        number=NUMBER,
    )
    print("Time to insert an element:")
    print(format_time(res / NUMBER))

    results = timeit.repeat(
        setup=f"from rbloom import Bloom; b = Bloom({NUMBER}, 0.01); objects = [object() for _ in range({NUMBER})]",
        stmt="b.update(objects)",
        timer=time.perf_counter_ns,
        number=1,
        repeat=20,
    )
    res = min(results)
    print("Time to insert each element in a batch:")
    print(format_time(res / NUMBER))

    results = timeit.repeat(
        setup=f"from rbloom import Bloom; b = Bloom({NUMBER}, 0.01); objects = (object() for _ in range({NUMBER}))",
        stmt="b.update(objects)",
        timer=time.perf_counter_ns,
        number=1,
        repeat=20,
    )
    res = min(results)
    print("Time to insert each element in a batch via an iterable:")
    print(format_time(res / NUMBER))

    setup_stmt = (
        f"from rbloom import Bloom; b = Bloom({NUMBER}, 0.01); stored_obj = object(); "
        f"b.add(stored_obj); b.update(object() for _ in range({NUMBER}))"
    )
    res = timeit.timeit(
        setup=setup_stmt,
        stmt="stored_obj in b",
        timer=time.perf_counter_ns,
        number=NUMBER,
    )
    print("Time to check if an object is present:")
    print(format_time(res / NUMBER))


if __name__ == "__main__":
    main()
