#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "protobuf>=4.0",
# ]
# ///
"""Analyze XPlane profiler trace to identify distributed training bottlenecks.

Usage:
    uv run analyze_profiler.py t1v-n-4806fb8d-w-20.xplane.pb

This script parses the XPlane protobuf format directly without needing
TensorFlow's generated protobuf code.
"""

import sys
from collections import defaultdict
from google.protobuf import descriptor_pb2
from google.protobuf.internal.decoder import _DecodeVarint
from google.protobuf.internal import wire_format


def parse_varint(data, pos):
    """Parse a varint from data at position pos."""
    result = 0
    shift = 0
    while True:
        if pos >= len(data):
            return None, pos
        byte = data[pos]
        result |= (byte & 0x7F) << shift
        pos += 1
        if not (byte & 0x80):
            break
        shift += 7
    return result, pos


def parse_length_delimited(data, pos):
    """Parse a length-delimited field."""
    length, pos = parse_varint(data, pos)
    if length is None:
        return None, pos
    return data[pos:pos + length], pos + length


def parse_fixed64(data, pos):
    """Parse a fixed64 field."""
    return int.from_bytes(data[pos:pos + 8], 'little'), pos + 8


def parse_fixed32(data, pos):
    """Parse a fixed32 field."""
    return int.from_bytes(data[pos:pos + 4], 'little'), pos + 4


def parse_message(data):
    """Parse a protobuf message into a dict of field_number -> values."""
    fields = defaultdict(list)
    pos = 0
    while pos < len(data):
        tag, new_pos = parse_varint(data, pos)
        if tag is None:
            break
        pos = new_pos

        field_number = tag >> 3
        wire_type = tag & 0x7

        if wire_type == 0:  # Varint
            value, pos = parse_varint(data, pos)
        elif wire_type == 1:  # Fixed64
            value, pos = parse_fixed64(data, pos)
        elif wire_type == 2:  # Length-delimited
            value, pos = parse_length_delimited(data, pos)
        elif wire_type == 5:  # Fixed32
            value, pos = parse_fixed32(data, pos)
        else:
            break

        if value is not None:
            fields[field_number].append((wire_type, value))

    return fields


def decode_string(data):
    """Try to decode bytes as UTF-8 string."""
    try:
        return data.decode('utf-8')
    except:
        return None


def analyze_xplane_raw(filepath: str):
    """Analyze XPlane file using raw protobuf parsing."""
    print(f"Reading file: {filepath}")
    with open(filepath, 'rb') as f:
        data = f.read()

    print(f"File size: {len(data) / 1024 / 1024:.1f} MB")

    # XSpace structure (from tensorflow/core/profiler/protobuf/xplane.proto):
    # message XSpace {
    #   repeated XPlane planes = 1;
    #   repeated string errors = 2;
    #   repeated string warnings = 3;
    #   string hostnames = 4;
    # }
    # message XPlane {
    #   int64 id = 1;
    #   string name = 2;
    #   repeated XLine lines = 3;
    #   map<int64, XEventMetadata> event_metadata = 4;
    #   map<int64, XStatMetadata> stat_metadata = 5;
    #   repeated XStat stats = 6;
    # }

    results = {
        'planes': [],
        'total_time_ps': 0,
        'event_counts': defaultdict(int),
        'event_durations': defaultdict(int),
        'op_categories': defaultdict(int),
        # Separate TPU-only stats
        'tpu_total_time_ps': 0,
        'tpu_event_counts': defaultdict(int),
        'tpu_event_durations': defaultdict(int),
        'tpu_op_categories': defaultdict(int),
    }

    xspace = parse_message(data)

    # Field 1 = planes (repeated XPlane)
    planes = xspace.get(1, [])
    print(f"Found {len(planes)} planes")

    for plane_idx, (wire_type, plane_data) in enumerate(planes):
        if wire_type != 2:  # Must be length-delimited
            continue

        plane = parse_message(plane_data)

        # Field 2 = name (string)
        plane_name = ""
        if 2 in plane:
            _, name_bytes = plane[2][0]
            plane_name = decode_string(name_bytes) or f"plane_{plane_idx}"

        # Field 4 = event_metadata (map<int64, XEventMetadata>)
        event_metadata = {}
        for _, meta_entry in plane.get(4, []):
            meta_fields = parse_message(meta_entry)
            # Map entry: field 1 = key (int64), field 2 = value (XEventMetadata)
            if 1 in meta_fields and 2 in meta_fields:
                key = meta_fields[1][0][1]
                value_data = meta_fields[2][0][1]
                value_fields = parse_message(value_data)
                # XEventMetadata: field 1 = id, field 2 = name
                meta_name = ""
                if 2 in value_fields:
                    _, name_bytes = value_fields[2][0]
                    meta_name = decode_string(name_bytes) or ""
                event_metadata[key] = meta_name

        # Field 5 = stat_metadata
        stat_metadata = {}
        for _, meta_entry in plane.get(5, []):
            meta_fields = parse_message(meta_entry)
            if 1 in meta_fields and 2 in meta_fields:
                key = meta_fields[1][0][1]
                value_data = meta_fields[2][0][1]
                value_fields = parse_message(value_data)
                meta_name = ""
                if 2 in value_fields:
                    _, name_bytes = value_fields[2][0]
                    meta_name = decode_string(name_bytes) or ""
                stat_metadata[key] = meta_name

        # Field 3 = lines (repeated XLine)
        lines = plane.get(3, [])
        line_count = len(lines)
        event_count = 0

        # Check if this is a TPU plane
        is_tpu_plane = '/device:TPU' in plane_name

        for wire_type_line, line_data in lines:
            if wire_type_line != 2:  # Must be length-delimited
                continue
            line = parse_message(line_data)
            # XLine: field 4 = events (repeated XEvent), not field 3!
            events = line.get(4, [])

            for wire_type_evt, event_data in events:
                if wire_type_evt != 2:  # Must be length-delimited
                    continue
                event = parse_message(event_data)
                event_count += 1

                # XEvent: field 1 = metadata_id, field 2 = offset_ps, field 3 = duration_ps
                metadata_id = event.get(1, [(0, 0)])[0][1]
                duration_ps = event.get(3, [(0, 0)])[0][1]

                results['total_time_ps'] += duration_ps

                event_name = event_metadata.get(metadata_id, f"unknown_{metadata_id}")
                results['event_counts'][event_name] += 1
                results['event_durations'][event_name] += duration_ps

                # Categorize by name
                name_lower = event_name.lower()
                category = 'other'
                if any(x in name_lower for x in ['all-reduce', 'allreduce', 'all-gather',
                       'allgather', 'reduce-scatter', 'collective', 'send-done', 'recv-done']):
                    category = 'communication'
                elif any(x in name_lower for x in ['dot', 'conv', 'fusion', 'custom-call',
                         'splash_mha', 'flash', 'attention', 'softmax', 'matmul', 'gemm']):
                    category = 'compute'
                elif any(x in name_lower for x in ['copy', 'slice', 'reshape', 'transpose',
                         'broadcast', 'pad', 'gather', 'scatter', 'concatenate']):
                    category = 'memory'
                elif 'infeed' in name_lower:
                    category = 'data_loading'
                elif any(x in name_lower for x in ['while', 'conditional', 'call', 'tuple']):
                    category = 'control_flow'
                elif 'jit_' in name_lower or 'xla' in name_lower:
                    category = 'jit_overhead'

                results['op_categories'][category] += duration_ps

                # Also track TPU-specific stats
                if is_tpu_plane:
                    results['tpu_total_time_ps'] += duration_ps
                    results['tpu_event_counts'][event_name] += 1
                    results['tpu_event_durations'][event_name] += duration_ps
                    results['tpu_op_categories'][category] += duration_ps

        results['planes'].append({
            'name': plane_name,
            'lines': line_count,
            'events': event_count,
            'event_metadata_count': len(event_metadata),
        })

        print(f"  Plane: {plane_name} - {line_count} lines, {event_count} events")

    return results


def format_time(ps):
    """Format picoseconds to human readable."""
    ns = ps / 1000
    if ns >= 1e9:
        return f"{ns/1e9:.2f}s"
    elif ns >= 1e6:
        return f"{ns/1e6:.2f}ms"
    elif ns >= 1e3:
        return f"{ns/1e3:.2f}us"
    return f"{ns:.0f}ns"


def print_analysis(results):
    """Print analysis results."""
    print("\n" + "=" * 80)
    print("XPlane Profiler Analysis Report")
    print("=" * 80)

    print(f"\nPlane Summary:")
    for p in results['planes']:
        print(f"  {p['name']}: {p['lines']} lines, {p['events']} events, {p['event_metadata_count']} metadata entries")

    # Show TPU-specific breakdown (most important for optimization)
    tpu_total_ps = results['tpu_total_time_ps'] or 1
    print(f"\n{'='*40}")
    print("TPU-ONLY Time Breakdown (most important)")
    print(f"{'='*40}")
    print(f"Total TPU time: {format_time(tpu_total_ps)}")

    tpu_categories = results['tpu_op_categories']
    for cat in ['compute', 'communication', 'memory', 'control_flow', 'jit_overhead', 'data_loading', 'other']:
        time_ps = tpu_categories.get(cat, 0)
        pct = 100 * time_ps / tpu_total_ps
        print(f"  {cat.capitalize():15} {format_time(time_ps):>12} ({pct:5.1f}%)")

    print(f"\nTop 25 TPU Operations by Total Time:")
    sorted_ops = sorted(results['tpu_event_durations'].items(), key=lambda x: x[1], reverse=True)
    for name, duration_ps in sorted_ops[:25]:
        count = results['tpu_event_counts'][name]
        pct = 100 * duration_ps / tpu_total_ps
        avg_us = (duration_ps / 1000 / count) if count else 0
        short_name = name[:55] if len(name) <= 55 else name[:52] + "..."
        print(f"  {short_name:55} cnt={count:<6} tot={format_time(duration_ps):>10} avg={avg_us:.1f}us ({pct:.2f}%)")

    # Also show overall breakdown for reference
    print(f"\n{'='*40}")
    print("Overall (including Host CPU) - for reference")
    print(f"{'='*40}")
    total_ps = results['total_time_ps'] or 1
    print(f"Total time: {format_time(total_ps)}")

    categories = results['op_categories']
    for cat in ['compute', 'communication', 'memory', 'control_flow', 'jit_overhead', 'data_loading', 'other']:
        time_ps = categories.get(cat, 0)
        pct = 100 * time_ps / total_ps
        print(f"  {cat.capitalize():15} {format_time(time_ps):>12} ({pct:5.1f}%)")

    return results


def print_recommendations(results):
    """Print optimization recommendations based on analysis."""
    print("\n" + "=" * 80)
    print("Optimization Recommendations (based on TPU stats)")
    print("=" * 80)

    # Use TPU-specific stats for recommendations
    total_ps = results['tpu_total_time_ps'] or 1
    categories = results['tpu_op_categories']

    comm_pct = 100 * categories.get('communication', 0) / total_ps
    compute_pct = 100 * categories.get('compute', 0) / total_ps
    memory_pct = 100 * categories.get('memory', 0) / total_ps
    control_flow_pct = 100 * categories.get('control_flow', 0) / total_ps
    jit_pct = 100 * categories.get('jit_overhead', 0) / total_ps
    data_pct = 100 * categories.get('data_loading', 0) / total_ps
    other_pct = 100 * categories.get('other', 0) / total_ps

    print(f"\nTPU Profile Summary:")
    print(f"  Compute:       {compute_pct:.1f}%")
    print(f"  Communication: {comm_pct:.1f}%")
    print(f"  Memory:        {memory_pct:.1f}%")
    print(f"  Control Flow:  {control_flow_pct:.1f}%  <-- while loops, gradient accumulation")
    print(f"  JIT Overhead:  {jit_pct:.1f}%  <-- jit compilation wrappers")
    print(f"  Data Loading:  {data_pct:.1f}%")
    print(f"  Other:         {other_pct:.1f}%")

    if comm_pct > 30:
        print(f"""
[HIGH PRIORITY] Communication overhead is {comm_pct:.1f}%

This is significant. Recommendations:
1. INCREASE GRADIENT ACCUMULATION
   - Current: 4 steps
   - Try: 8 or 16 steps to amortize collective ops
   - In demo_vlm_train.py: GRADIENT_ACCUMULATION_STEPS = 8

2. INCREASE BATCH SIZE (if memory allows)
   - Current: per_device_parallelism = 1
   - Try: per_device_parallelism = 2

3. CHECK SHARDING STRATEGY
   - For small models like Qwen3-1.7B, FSDP may have more overhead
   - Consider if tensor parallelism is beneficial at this scale
""")
    elif comm_pct > 15:
        print(f"""
[MODERATE] Communication overhead is {comm_pct:.1f}%

Acceptable but could be improved:
1. Increase gradient accumulation to 8 steps
2. Profile with larger batch sizes
""")
    else:
        print(f"""
[GOOD] Communication overhead is {comm_pct:.1f}% - well controlled
""")

    if compute_pct < 40:
        print(f"""
[LOW] Compute utilization is only {compute_pct:.1f}%

Recommendations:
1. VERIFY FLASH ATTENTION is being used
   - Check logs for "Using flash attention" message
   - block_size=1024 is configured

2. INCREASE WORKLOAD
   - Larger batch size = better compute utilization
   - Longer sequences if applicable

3. CHECK FOR JIT RECOMPILATION
   - Look for repeated "Compiling" messages in logs
""")

    if data_pct > 10:
        print(f"""
[ATTENTION] Data loading overhead is {data_pct:.1f}%

The data pipeline may be a bottleneck:
1. Increase streaming_prefetch_size from 8 to 16
2. Increase streaming_max_buffered_batches from 16 to 32
3. Verify data is on fast regional GCS storage
4. Run: python experiments/VLM/profile_data_pipeline.py
""")

    if control_flow_pct > 30:
        print(f"""
[HIGH PRIORITY] Control flow overhead is {control_flow_pct:.1f}%

This is mainly from gradient accumulation while loops!

Recommendations:
1. REDUCE GRADIENT ACCUMULATION STEPS
   - Current: 4 steps with while loops
   - Each step has XLA control flow overhead

2. INCREASE PER-DEVICE BATCH SIZE INSTEAD
   - Try: per_device_parallelism = 2 or 4
   - Reduce gradient_accumulation_steps to 2 or 1
   - Keep effective batch size the same

Example change in demo_vlm_train.py:
   # Current: 64 chips * 1 * 4 = 256 batch
   PER_DEVICE_PARALLELISM = 1
   GRADIENT_ACCUMULATION_STEPS = 4

   # Better: 64 chips * 2 * 2 = 256 batch (less while loop overhead)
   PER_DEVICE_PARALLELISM = 2
   GRADIENT_ACCUMULATION_STEPS = 2

   # Best (if memory allows): 64 chips * 4 * 1 = 256 batch (no while loops!)
   PER_DEVICE_PARALLELISM = 4
   GRADIENT_ACCUMULATION_STEPS = 1
""")

    if jit_pct > 15:
        print(f"""
[ATTENTION] JIT overhead is {jit_pct:.1f}%

This could indicate:
1. Repeated recompilation - check logs for "Compiling" messages
2. First few steps always have JIT overhead - ensure profiler starts after warmup
""")

    if other_pct > 10:
        print(f"""
[INFO] "Other" category is {other_pct:.1f}%
""")
        # Analyze what's in "other"
        other_ops = []
        for name, duration_ps in results['tpu_event_durations'].items():
            name_lower = name.lower()
            # Check if it would be categorized as "other"
            is_comm = any(x in name_lower for x in ['all-reduce', 'allreduce', 'all-gather',
                       'allgather', 'reduce-scatter', 'collective', 'send-done', 'recv-done'])
            is_compute = any(x in name_lower for x in ['dot', 'conv', 'fusion', 'custom-call',
                         'splash_mha', 'flash', 'attention', 'softmax', 'matmul', 'gemm'])
            is_memory = any(x in name_lower for x in ['copy', 'slice', 'reshape', 'transpose',
                         'broadcast', 'pad', 'gather', 'scatter', 'concatenate'])
            is_data = 'infeed' in name_lower
            is_control = any(x in name_lower for x in ['while', 'conditional', 'call', 'tuple'])
            is_jit = 'jit_' in name_lower or 'xla' in name_lower

            if not any([is_comm, is_compute, is_memory, is_data, is_control, is_jit]):
                other_ops.append((name, duration_ps, results['tpu_event_counts'][name]))

        other_ops.sort(key=lambda x: x[1], reverse=True)
        print("Top 'Other' operations:")
        for name, dur, cnt in other_ops[:10]:
            pct = 100 * dur / total_ps
            avg_s = dur / 1e12 / cnt if cnt else 0
            # Check if it's a numeric-only name (likely internal XLA op)
            is_numeric = name.isdigit() or (name.startswith('unknown_') and name[8:].isdigit())
            marker = " [internal XLA op]" if is_numeric else ""
            print(f"  {name[:40]:40} cnt={cnt:<5} tot={format_time(dur):>10} avg={avg_s:.2f}s ({pct:.2f}%){marker}")

    # Look at specific expensive TPU ops
    sorted_ops = sorted(results['tpu_event_durations'].items(), key=lambda x: x[1], reverse=True)
    print("\nKey Observations from Top TPU Operations:")

    for name, duration_ps in sorted_ops[:5]:
        pct = 100 * duration_ps / total_ps
        name_lower = name.lower()

        if pct > 10:
            if 'reduce' in name_lower or 'gather' in name_lower:
                print(f"  - '{name[:40]}' ({pct:.1f}%): Collective op - increase gradient accumulation")
            elif 'infeed' in name_lower:
                print(f"  - '{name[:40]}' ({pct:.1f}%): Data loading - optimize prefetch settings")
            elif 'dot' in name_lower or 'fusion' in name_lower:
                print(f"  - '{name[:40]}' ({pct:.1f}%): Compute op - this is expected/good")


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run analyze_profiler.py <xplane.pb file>")
        print("Example: uv run analyze_profiler.py t1v-n-4806fb8d-w-20.xplane.pb")
        sys.exit(1)

    filepath = sys.argv[1]
    results = analyze_xplane_raw(filepath)
    print_analysis(results)
    print_recommendations(results)


if __name__ == "__main__":
    main()
