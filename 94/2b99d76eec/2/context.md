# Session Context

## User Prompts

### Prompt 1

f33cf904b78eee8eb85afa18fa544699276bddb1 commit removed window_size_bytes, that's good, I want an optional argument to allow to override the number of shards for tokenization

### Prompt 2

add comment along the lines: `This could be useful if you want more shards the max workers for example to mitigate the cost of retry on a single shard.`

### Prompt 3

do we need the config to exists in both TokenizeConfig and HfTokenizeConfig?

### Prompt 4

should the other flags also be moved to the base class?

### Prompt 5

let's move things like `worker_resources`, `writer_batch_size` and `max_workers`

