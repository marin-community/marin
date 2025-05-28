import json
import logging

from tqdm import tqdm

logger = logging.getLogger("ray")


def decompress_zstd_stream(raw_stream, content_length: int, chunk_size: int):
    """
    Decompresses a zstd stream and parses JSON lines.
    Args:
        raw_stream: Raw binary stream from requests
        content_length: Content length from response headers
    Returns:
        list: List of parsed JSON objects
    """
    import zstandard as zstd

    contents = []
    dctx = zstd.ZstdDecompressor()

    # Read chunk by chunk so we can split on newlines ourselves
    with dctx.stream_reader(raw_stream) as reader:
        buffer = b""

        # Get total size for progress bar
        total_size = int(content_length)
        progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True, desc="Downloading")
        bytes_read = 0

        while True:
            chunk = reader.read(chunk_size)
            if not chunk:
                break
            buffer += chunk
            bytes_read += len(chunk)
            progress_bar.update(len(chunk))

            while True:
                newline_pos = buffer.find(b"\n")
                if newline_pos < 0:
                    break
                line_bytes = buffer[:newline_pos]
                buffer = buffer[newline_pos + 1 :]

                if not line_bytes.strip():
                    continue

                try:
                    content = json.loads(line_bytes.decode("utf-8"))
                    contents.append(content)
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse JSON line: {line_bytes[:80]}...")
                    continue

        progress_bar.close()

    return contents
