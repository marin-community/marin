import json
import os
import random

import fsspec

from marin.utils import rebase_file_path


def check_and_sample(domain, version):
    fs = fsspec.filesystem("gcs")
    base_path = f"gcs://marin-data/processed/{domain}/{version}/"
    examples_path = f"gcs://marin-data/examples/{domain}/{version}/"

    # Get list of formats (directories) that start with XML, HTML, md, or txt
    formats = fs.ls(base_path, use_listings_cache=False)
    valid_formats = [fmt for fmt in formats if
                     os.path.basename(fmt).lower().startswith(('xml', 'html', 'md', 'txt', "text"))]

    # Collect md5Hash for all files in each valid format
    all_md5_hashes = {}
    for format_type in valid_formats:
        files = fs.glob(os.path.join(format_type, "**/*.jsonl.gz"), detail=True, use_listings_cache=False)
        all_md5_hashes[format_type] = {file: info['md5Hash'] for file, info in files.items()}

    # Load previous md5 hashes if available
    md5_hashes_path = os.path.join(examples_path, 'md5_hashes.json')
    if fs.exists(md5_hashes_path, use_listings_cache=False):
        with fsspec.open(md5_hashes_path, 'r', use_listings_cache=False) as f:
            previous_md5_hashes = json.load(f)
    else:
        previous_md5_hashes = {}

    # Check if any hash has changed
    reprocess = False
    for format_type, hashes in all_md5_hashes.items():
        if format_type not in previous_md5_hashes or previous_md5_hashes[format_type] != hashes:
            reprocess = True
            break

    if not reprocess:
        print("No changes detected, skipping reprocessing.")
        return "No changes detected, skipping reprocessing."

    # Perform reservoir sampling for all formats at once
    sample_size = 1000
    sample_lines = {fmt: [''] * sample_size for fmt in valid_formats}

    # Get file paths for the first format
    primary_format = valid_formats[0]
    primary_files = fs.glob(os.path.join(primary_format, "**/*.jsonl.gz"), use_listings_cache=False)

    # Randomly shuffle the files
    random.shuffle(primary_files)

    idx = 0
    print("Processing files...")
    # Iterate over the primary files and corresponding files in other formats
    for num_file, primary_file in enumerate(primary_files):
        if num_file > sample_size * 20:
            break

        files_in_diff_fmts = []
        for fmt in valid_formats:
            corresponding_file = rebase_file_path(primary_format, primary_file, fmt)
            files_in_diff_fmts.append(corresponding_file)

        file_pointers = [fs.open(file, 'rt', compression="infer", use_listings_cache=False) for file in
                         files_in_diff_fmts]

        for i, lines in enumerate(zip(*file_pointers)):
            # Do reservoir sampling
            if idx < sample_size:
                for j, line in enumerate(lines):
                    sample_lines[valid_formats[j]][idx] = line
            else:
                j = random.randint(0, idx)
                if j < sample_size:
                    for k, line in enumerate(lines):
                        sample_lines[valid_formats[k]][j] = line

            idx += 1
            if i > sample_size * 20:
                break
            if idx % 10000 == 0:
                print(f"Processed {idx} lines")

    # Save sampled lines to examples directory
    # first get the path from format type and then save
    return_text = ""
    if idx == 0:
        print("No samples were found or processed. Exiting.")
        return "No samples were processed. Exiting."
    elif idx < sample_size:
        print(f"Only {idx} samples were processed.")
        return_text = f"Only {idx} samples were processed."

    for format_type, lines in sample_lines.items():
        example_file_path = os.path.join(examples_path, os.path.basename(format_type), "samples.jsonl")
        with fsspec.open(example_file_path, 'wt', use_listings_cache=False) as example_file:
            for line in lines:
                example_file.write(line)

    # Save the current md5 hashes
    with fsspec.open(md5_hashes_path, 'w', use_listings_cache=False) as f:
        json.dump(all_md5_hashes, f)

    print("New samples have been processed and saved.")
    return f"New samples have been processed and saved, {return_text}"
