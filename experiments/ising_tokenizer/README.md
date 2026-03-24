# Ising Tokenizer Experiments

This experiment tree is a controlled physics-side analogue of the JPEG
tokenizer work:

> can a tokenized autoregressive model recover useful latent structure from the
> natural event language of a dynamical system, without being handed the physics
> directly?

The first target is the 2D Ising model with continuous-time single-spin-flip
dynamics. The initial state is tokenized as explicit `[pos][spin]` pairs and
the trajectory is tokenized as explicit `[pos][dt_bin]` pairs. Temperature is a
continuous conditioning signal, not a token.

The canonical template lives in `/experiments/ising_tokenizer/base/`.
Future variants should start as explicit copies of `base/` and keep local
changes obvious.

## V0 Decisions

- Dynamics: naive rejection-free BKL-style event sampling with Glauber flip
  rates and full-lattice rate recomputation.
- Temperature conditioning: scalar residual modulation added to token
  embeddings.
- Data: synthetic local splits only.
- Evaluation: teacher-forced likelihood first, rollout evaluation later.

## Layout

- `data.py`: synthetic trajectory generation and tokenization.
- `model.py`: grug-style conditioned transformer.
- `train.py`: local smoke trainer.
- `launch.py`: small runnable configuration for the first local Ising smoke.
