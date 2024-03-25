from build_rpv2_bloom_filter import mk_bloom_for_shard_slice, union_blooms as union_blooms_rpv2, _fsspec_exists


def build_rpv2_bloom_filter(request):
    request_json = request.get_json(silent=True)

    if request_json:
        # Extract the necessary information from the request
        out_path = request_json.get('out_path')
        lang = request_json.get('lang')
        crawl = request_json.get('crawl')
        part = request_json.get('part')
        chunk_range = request_json.get('chunk_range')

        if _fsspec_exists(out_path):
            return "Bloom filter already exists", 200

        # Call your existing logic with the extracted information
        mk_bloom_for_shard_slice(out_path, crawl, part, lang, chunk_range)

        return "Slice processed successfully", 200
    else:
        return "Invalid task data", 400

def hello_world_function(request):
    print("Hello World! Request received.")
    return "Success", 200


def union_blooms(request):
    request_json = request.get_json(silent=True)

    if request_json:
        # Extract the necessary information from the request
        out_path = request_json.get('out_path')
        paths = request_json.get('paths')

        print(f"Unioning blooms to {out_path} with first path {paths[0]}")

        if _fsspec_exists(out_path):
            return "Bloom filter already exists", 200

        # Call your existing logic with the extracted information
        union_blooms_rpv2(out_path, paths)

        return "Blooms unioned successfully", 200
    else:
        return "Invalid task data", 400