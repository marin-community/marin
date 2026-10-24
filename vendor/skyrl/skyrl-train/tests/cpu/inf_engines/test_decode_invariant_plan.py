from skyrl_train.inference_engines.vllm.decode_invariant.fa3 import fa3_window_requests


def test_fa3_window_requests_run_rows_past_the_window_alone():
    # (query rows, keys) per request: a prefill from position 5 across position 8, a decode row past it, a prefill
    # from position 0, a decode row, a prefill chunk from position 9, a padded request without rows, and prefills
    # from cached prefixes, the last one across position 8.
    requests = [(6, 11), (1, 20), (5, 5), (1, 3), (3, 12), (0, 7), (3, 7), (6, 10)]
    query_start = [0]
    for rows, _ in requests:
        query_start.append(query_start[-1] + rows)

    split = fa3_window_requests(query_start, [keys for _, keys in requests], window=8)

    starts, keys = split.query_start.tolist(), split.key_lengths.tolist()
    sizes = [end - start for start, end in zip(starts, starts[1:])]
    # (first position, rows) of each new request: its rows are the last ones of its keys.
    assert [(key - size, size) for key, size in zip(keys, sizes)] == [
        (5, 3),
        (8, 1),
        (9, 1),
        (10, 1),
        (19, 1),
        (0, 5),
        (2, 1),
        (9, 1),
        (10, 1),
        (11, 1),
        (4, 3),
        (4, 4),
        (8, 1),
        (9, 1),
    ]
    assert starts[0] == 0 and starts[-1] == query_start[-1]
    assert split.owner.tolist() == [0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 6, 7, 7, 7]
