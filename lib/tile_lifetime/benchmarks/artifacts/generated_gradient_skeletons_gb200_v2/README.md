# Generated gradient skeletons on GB200, second checkpoint

This checkpoint records three bounded experiments on one NVIDIA GB200. The
device reported driver 595.71.05, a 1200 W power limit, 1950 MHz SM clock, and
3996 MHz memory clock. The source archive SHA-256 was
`b33f3f83c5c2c6daee84efb058e5bf3f9f50bc9162ae59638fc0904c99cd12cb`.
The complete local artifact bundle had SHA-256
`911bfbf3414fb7a6857970cf299f95eaa32447627e54abbd86d720cc3959a3b0`.

## JAX-owned RMS reverse through the generated Torch scaffold

At Shuttle revision `a017469b27`, ordinary JAX owned AD and exported a
44-operation StableHLO reverse graph containing five generic reductions.
Shuttle recovered and generated the Fold bodies. Torch 2.10.0+cu130 was used
only as the transient compilation/timing scaffold.

- Generated median: 0.0396336004 ms.
- Matched compiled algebra median: 0.0527904004 ms.
- Ratio: 0.7507728701.
- Maximum/mean dX error: 1.90735e-6 / 1.02739e-8.
- Maximum/mean feature-Fold error: 7.62939e-5 / 1.09826e-5.
- Generated dX hash: `af0ff0513196b72de07c39f7aa8314102dc60e2205ed6442c4e96ab72778a32a`.
- Generated feature-Fold hash: `497f17524ff862c17baf46204208852c386048e83cea0daf50ac81dced7445b5`.

Raw generated samples in milliseconds:

```text
0.0404192001, 0.0399488002, 0.0390112013, 0.0399232000, 0.0391808003,
0.0398400009, 0.0391871989, 0.0400927991, 0.0391712010, 0.0399616003,
0.0391167998, 0.0399776012, 0.0391680002, 0.0399455994, 0.0389759988,
0.0396768004, 0.0391295999, 0.0399744004, 0.0390464008, 0.0400000006,
0.0390527993, 0.0399776012, 0.0390623987, 0.0398880005, 0.0388864011,
0.0397855997, 0.0386815995, 0.0395904005, 0.0389919996, 0.0400799990
```

Raw matched-compiled-algebra samples in milliseconds:

```text
0.0708800018, 0.0591232002, 0.0630783975, 0.0570623994, 0.0634015977,
0.0559455991, 0.0557792008, 0.0522176027, 0.0576864004, 0.0542208016,
0.0576704025, 0.0480832011, 0.0541728020, 0.0491840005, 0.0513535976,
0.0498207986, 0.0533631980, 0.0482208014, 0.0497087985, 0.0482560009,
0.0511168003, 0.0539903998, 0.0534943998, 0.0510528028, 0.0504000008,
0.0460543990, 0.0513920009, 0.0469119996, 0.0548128009, 0.0514496028
```

## Torch-free JAX typed-FFI Fold execution

The performance comparison in this subsection is withdrawn. The matched XLA
function closed over benchmark arrays, so XLA constant-folded the computation.
The raw samples remain preserved, and the generated correctness, handler-call,
and determinism evidence remains valid. The corrected H100 runtime-input replay
is under `../jax_row_normalization_backward_h100_components_corrected_v1`; GB200
must be replayed with the same correction before making a GB200 performance
claim.

At Shuttle revision `1e0512923d`, JAX 0.11.0 registered and called the
generated CUDA Fold family through typed FFI. The runtime path has no Torch
dependency. NVCC was CUDA 13.0.88 and the generated source SHA-256 was
`a4a08bb7f932c2e584ddb1a5e401e6a8dcd35e0eb335042ace9a211d0ca7a3ae`.

- Generated typed-FFI median: 0.0610992545 ms.
- Matched XLA algebra median: 0.0669200905 ms.
- Withdrawn ratio: 0.9130181101; invalid constant-folded XLA baseline.
- Handler executions: 312.
- Maximum matched-FP32 dX/feature-Fold errors: 9.53674e-7 / 2.28882e-5.
- Output hashes: `db038d17c1d830c0bd24007e2ada6f785b7421e0dcbec75e75cd5a0a45c0de51`,
  `fda783d534c9d729e4bf5de05819b5edd0f3dc8a6e6b1d154e0664dd3186f2d4`.

Raw generated typed-FFI samples in milliseconds:

```text
0.0648321118, 0.0610208139, 0.0637121033, 0.0663361046, 0.0637504738,
0.0589441042, 0.0624992885, 0.0631040893, 0.0608833041, 0.0603265129,
0.0625441317, 0.0582079869, 0.0621953048, 0.0674113166, 0.0629568938,
0.0611776952, 0.0608607661, 0.0603297260, 0.0618529040, 0.0578592997,
0.0588640105, 0.0612000935, 0.0655297190, 0.0586817041, 0.0592319760,
0.0596417114, 0.0676289201, 0.0601248816, 0.0598816201, 0.0593089033
```

Raw matched-XLA samples in milliseconds:

```text
0.0693088863, 0.0669665169, 0.0692961272, 0.0653504860, 0.0684736762,
0.0673696864, 0.0689535867, 0.0659713056, 0.0673569273, 0.0683168881,
0.0677792821, 0.0691873021, 0.0678016804, 0.0656289048, 0.0663072802,
0.0673185103, 0.0686081126, 0.0633057207, 0.0667233020, 0.0658080913,
0.0686720945, 0.0637600664, 0.0657184981, 0.0656769145, 0.0668736640,
0.0643360894, 0.0664097257, 0.0638656784, 0.0678016804, 0.0658497214
```

This result uses a deterministic reduction tree. It is not source-order
equivalent to XLA's selected tree. After casting to the natural BF16 VJP
outputs, maximum differences are 0.0625 for dX and 1.0 for the feature Fold.

## Rejected partitioned key/value reverse Fold

At Shuttle revision `ace2636514`, the S=2048 causal GQA backward used BF16,
32 query heads, eight KV heads, head dimension 128, 32x32 tiles, eight warps,
and three pipeline stages. Both schedules were deterministic and used no
atomic accumulation.

One partition:

- Generated/oracle medians: 0.864582396 / 0.148534399 ms.
- Ratio: 5.820755325.
- Raw generated samples: `0.862297630, 0.866099166, 0.856768036,
  0.867705631, 0.865190411, 0.865068817, 0.865433598, 0.860991955,
  0.860780811, 0.864095973`.
- Raw oracle samples: `0.157535994, 0.146156800, 0.150726402,
  0.146239996, 0.150860798, 0.145644796, 0.153548801, 0.146464002,
  0.148787200, 0.148281598`.

Four partitions:

- Generated/oracle medians: 0.854435205 / 0.154431999 ms.
- Ratio: 5.532760131.
- Raw generated samples: `0.854540825, 0.854329586, 0.849055958,
  0.855532837, 0.852601624, 0.858316803, 0.855443192, 0.853548813,
  0.849491215, 0.859955215`.
- Raw oracle samples: `0.158687997, 0.159206402, 0.153343999,
  0.149388802, 0.159840000, 0.149337602, 0.155519998, 0.147577596,
  0.158854401, 0.153318405`.

The four-partition schedule improves generated latency by only 1.17% while
adding 64 MiB of FP32 partial buffers, increasing key/value tasks from 512 to
2048, and adding 512 finalizers. It was rejected and removed in `0f28376aea`.
The remaining gap is attributed to the reverse QK/PV physical pipeline and
data reuse rather than Fold task parallelism.
