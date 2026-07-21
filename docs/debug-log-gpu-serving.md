# Debugging log for GPU serving

Make `marin-serve iris Qwen/Qwen3-0.6B --cluster marin --gpu h100` start a usable
vLLM endpoint on CoreWeave.

## Initial status

The serve task installs stock CUDA vLLM 0.25.1, then fails while its Run:ai model
streamer lists the cached model in CoreWeave object storage. CoreWeave rejects
the streamer's path-style `ListObjectsV2` request with
`PathStyleRequestNotAllowed`.

## Hypothesis 1

The default 14-day serve cache changes the Hugging Face model ID into a regional
`s3://` cache path. Both stock and Marin-fork CUDA launchers include the Run:ai
S3 loader, but `IsolatedCudaVllm.env()` only writes and exports boto3's
virtual-hosted S3 configuration for the Marin fork. The stock launcher therefore
inherits the CoreWeave endpoint and credentials without the addressing style
that endpoint requires.

## Changes to make

Update `IsolatedCudaVllm.env()` so every CUDA vLLM variant receives the shared
virtual-hosted S3 config. Strengthen the existing launcher test to read the
exported config file and add stock-launcher coverage for the same behavior.

## Results

The regression test failed before the fix because the stock launcher's
environment lacked `AWS_CONFIG_FILE`. After moving the virtual-hosted config to
the shared CUDA launcher environment, the focused launcher tests pass for both
stock and Marin-fork vLLM.

A live rerun on 2026-07-21 submitted `/power/serve-qwen3-0-6b-bf954a` with the
reported Qwen model on one H100. The Run:ai pull completed, vLLM initialized,
`POST /v1/chat/completions` returned 200 during the readiness probe, and Iris
registered `/serve/serve-qwen3-0-6b-bf954a`. The temporary job was then stopped.

## Future work

None.
