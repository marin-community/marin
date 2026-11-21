from skyrl_train.inference_engines.request_timing import RequestTimings


def test_request_evidence_distinguishes_queued_and_decoding_when_response_is_lost():
    records = []
    timings = RequestTimings(records.append)
    timings.start("decoding", "eval-None-0--266_0")
    timings.start("queued", "eval-None-0--267_0")
    timings.observe("chatcmpl-decoding", [("QUEUED", 10), ("SCHEDULED", 15)], 1, 20)
    timings.observe("chatcmpl-decoding", [], 600, 80)
    timings.finish("decoding")
    timings.finish("queued")
    first = next(row for row in records if row["event"] == "first_token")
    assert (first["session_id"], first["queue_wait"], first["first_token_at"]) == ("eval-None-0--266_0", 5, 20)
    decoded, queued = [row for row in records if row["event"] == "ended"]
    assert (decoded["generated_tokens"], decoded["last_token_at"]) == (601, 80)
    assert (queued["generated_tokens"], queued["first_token_at"]) == (0, None)
