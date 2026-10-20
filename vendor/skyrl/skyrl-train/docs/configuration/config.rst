Configuration Overview
======================

Data Configuration
------------------

.. code-block:: yaml

    data:
      train_data: ["${oc.env:HOME}/data/gsm8k/train.parquet"]
      val_data: ["${oc.env:HOME}/data/gsm8k/validation.parquet"]

- ``data.train_data``: A list of files for the training dataset.
- ``data.val_data``: A list of files for the evaluation dataset.

A dataset file can be a path to a parquet or json file, or the name of a Hugging Face dataset.

.. note::
    Currently, all datasets are loaded into memory, so the dataset size is limited by available CPU memory on a worker node.


Model Placement Configuration
-----------------------------

.. code-block:: yaml

  placement:
    colocate_all: true
    colocate_policy_ref: true
    colocate_critic_reward: false
    policy_num_nodes: 1
    policy_num_gpus_per_node: 4
    critic_num_nodes: 1
    critic_num_gpus_per_node: 4
    ref_num_nodes: 1
    ref_num_gpus_per_node: 4
    reward_num_nodes: 1
    reward_num_gpus_per_node: 4
    policy_strict_spread_pg: false
    overlap_init: false

For an in-depth guide on model placement and colocation, please refer to the :doc:`model placement and colocation guide <placement>`.

``trainer.placement.overlap_init`` starts inference engines while policy workers build their models.
It defaults to false and requires dedicated policy placement (``policy_strict_spread_pg: true``),
disaggregated local engines, an objective without a reference model, and disabled distillation.
It controls startup concurrency separately from placement reservation so recipes can opt in individually.

General Training Configuration
------------------------------

.. code-block:: yaml

    epochs: 1  # Number of passes over the full dataset
    update_epochs_per_batch: 1
    train_batch_size: 1024
    policy_mini_batch_size: 256
    critic_mini_batch_size: 256
    micro_train_batch_size_per_gpu: 1
    micro_forward_batch_size_per_gpu: 1
    update_ref_every_epoch: false
    use_sample_packing: true
    max_prompt_length: 512
    gradient_checkpointing: true
    seed: 42


- ``epochs``: Number of epochs/ passes over the full dataset (similar to SFT)
- ``update_epochs_per_batch``: Number of gradient update passes over each training batch. This is equivalent to the concept of "PPO epochs" where you iterate over the same experience multiple times.
- ``train_batch_size``: Number of prompt groups in each training step's batch.
- ``policy_mini_batch_size``: Mini batch size used during RL training step. Each mini batch corresponds to one optimizer step. For example, if the ``train_batch_size`` is 4 and ``policy_mini_batch_size`` is 2, then there will be 2 optimizer steps (i.e., model updates) for a given training batch. Note that is this the global mini batch size. The actual size of the mini batch per worker would be ``policy_mini_batch_size/ number of DP ranks``
- ``critic_mini_batch_size``: Similar to ``policy_mini_batch_size`` but for the critic model (if applicable). Note that in general, the critic model can tolerate off-policy updates more than the policy. Thus, you would want to set ``critic_mini_batch_size`` to be lower compared ``policy_mini_batch_size`` (i.e., more critic updates).
- ``micro_train_batch_size_per_gpu``: Micro batch size during training step. This is common for both policy and critic models. Each mini batch is split into micro batches of this size, gradients are computed and accumulated over these micro batches.
- ``micro_forward_batch_size_per_gpu``: Micro batch size during forward pass (i.e., for log probability or value computation). This is common for both policy and critic models. Each mini batch is split into micro batches of this size, model forward pass is performed over these micro batches.
- ``update_ref_every_epoch``: Whether to update the reference model every epoch.
- ``use_sample_packing``: Whether to use sample packing during model forward pass (common for all models).
- ``max_prompt_length``: Maximum prompt length during training. Longer prompts will be truncated.
- ``gradient_checkpointing``: Whether to use gradient checkpointing.
- ``seed``: Random seed for training.


.. tip::
  If you're facing issues with tuning the right values for ``micro_train_batch_size_per_gpu``, ``policy_mini_batch_size`` and ``micro_forward_batch_size_per_gpu``, see ``utils/utils.py::validate_batch_sizes`` for details on constraints.

Evaluation Configuration
------------------------------
.. code-block:: yaml

    eval_batch_size: 1024
    eval_before_train: true
    eval_interval: 5 # Set to -1 to disable evaluation.

- ``eval_batch_size``: Batch size for evaluation.
- ``eval_before_train``: Whether to evaluate the model before training.
- ``eval_interval``: The frequency of evaluating the model with the validation dataset (in terms of number of steps). If set to ``-1``, evaluation will not be performed.

.. note::
  If multiple validation datasets are provided (e.g. ``data.val_data="['$DATA_DIR/validation1.parquet', '$DATA_DIR/validation2.parquet']" \``),
  then the evaluation will be performed on all of them. The metrics for each dataset, and the aggregated metrics, will
  all be logged in WandB. If ``dump_eval_results`` is set to ``true``, the per-dataset and aggregated results will be
  dumped.

Checkpoint Configuration
---------------------------------------

.. code-block:: yaml

    resume_mode: latest # null/"none", "latest", "from_path"
    resume_path: null
    ckpt_path: "${oc.env:HOME}/ckpts/" # Local directory path or cloud storage path (S3, GCP) for resumable training checkpoints (model state, optimizer state, etc.)
    max_ckpts_to_keep: -1 # -1 to keep all checkpoints, N to keep the last N checkpoints
    ckpt_interval: 10  # Save full training checkpoint every `ckpt_interval` steps.
    hf_save_interval: -1  # Save HF format model(s)every `hf_save_interval` steps.
    export_path: "${oc.env:HOME}/exports" # Path for exported artifacts (HF models, debug dumps, etc.)
    project_name: "skyrl"
    run_name: "test_run"
    logger: "wandb"

For an in-depth guide on checkpointing and resumption, please refer to the :doc:`checkpointing guide <../checkpointing-logging/checkpointing>`.

Rollout Buffer Configuration
----------------------------

.. code-block:: yaml

    rollout_buffer:
      max_staleness_steps: 0
      batch_policy: full_batch
      max_in_flight: null
      object_store_root: null
    teacher_scoring:
      max_queued_per_teacher: 8
      workers_per_teacher: 1

Rollout workers generate prompt groups under leases from a rollout buffer, and each training step trains on
``train_batch_size`` groups. See :doc:`../tutorials/fully_async` for the full design.

- ``rollout_buffer.max_staleness_steps``: How many policy steps may separate the step at which a group was leased
  from the step that trains on it. ``0`` is synchronous on-policy training and is required when ``placement.colocate_all=true``.
  A positive value lets generation run ahead of training.
- ``rollout_buffer.batch_policy``: How committed groups form batches when ``max_staleness_steps`` is positive.
  ``full_batch`` trains each step on exactly the groups leased for it and waits for the slowest; ``rolling`` fills
  each batch in commit order, so a slow group never blocks a step but quick groups train sooner and more often.
- ``rollout_buffer.max_in_flight``: Maximum number of prompt groups generating at once. ``null`` bounds generation
  only by staleness. A value below ``train_batch_size`` generates each batch in several waves.
- ``rollout_buffer.object_store_root``: Directory, usually an S3 prefix, that holds each trainable group as its own
  object; a checkpoint then records the objects' URIs instead of copying the groups. Nothing deletes the objects,
  so use an expiring prefix. ``null`` keeps groups only in Ray's object store.
- ``teacher_scoring.max_queued_per_teacher``: Maximum number of score requests queued for each distillation teacher.
  A full queue holds back admission of further groups.
- ``teacher_scoring.workers_per_teacher``: Number of score requests each teacher runs concurrently.

Logging and Debugging Configuration
-----------------------------------

.. code-block:: yaml

    logger: "wandb"
    project_name: "skyrl"
    run_name: "test_run"
    dump_data_batch: false
    dump_eval_results: true

- ``logger``: Logger to use. Currently, we support ``wandb``, ``mlflow``, and ``console``. ``console`` will simply log metrics to the console.
- ``project_name``: Name of the project in WandB and MLFlow.
- ``run_name``: Name of the run in WandB and MLFlow.
- ``dump_data_batch``: Whether to dump the data batch to a file. This is useful for debugging. When ``true``, the data batch will be dumped to a file in the ``export_path`` directory. The training batch at global step ``N`` is saved to ``self.cfg.trainer.export_path / "dumped_data" / global_step_N_training_input``
- ``dump_eval_results``: Whether to dump the evaluation results to a file. When ``true``, the full evaluation results will be dumped to a file in the ``export_path`` directory. The evaluation results at global step ``N`` is saved to ``self.cfg.trainer.export_path / "dumped_eval" / global_step_N_eval_results``

Training Backend
----------------

Megatron is the supported training backend. Set ``trainer.strategy=megatron`` and configure
policy and reference parallelism under ``trainer.<role>.megatron_config``.

.. _megatron-configurations:

Megatron Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    megatron_config:
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
      context_parallel_size: 1
      expert_model_parallel_size: 1
      expert_tensor_parallel_size: null
      optimizer_checkpoint_sharding_type: fully_reshardable

      ddp_config: # pass-through config to Megatron's `DistributedDataParallelConfig` object
        # https://github.com/NVIDIA/Megatron-LM/blob/core_r0.13.0/megatron/core/distributed/distributed_data_parallel_config.py#L8
        ...
      optimizer_config_kwargs: # pass-through kwargs to Megatron's `OptimizerConfig` object
        # any overlapping arguments with those we attempt to resolve in trainer.policy.optimizer_config will be overridden by the values here
        # https://github.com/NVIDIA/Megatron-LM/blob/core_r0.13.0/megatron/core/optimizer/optimizer_config.py#L12
        ...
      model_config_kwargs: # pass-through kwargs to the HuggingFace model config (i.e. for overriding vocab size, etc)
        ...
      transformer_config_kwargs: # pass-through kwargs to the Megatron's `TransformerConfig` object
        # https://github.com/NVIDIA/Megatron-LM/blob/core_r0.13.0/megatron/core/transformer/transformer_config.py#L33
        ...
      # flag to manually empty torch's cuda cache between the forward/backward pass and the optimizer step
      # this will free reserved but unallocated memory, and can help avoid OoMs in the optimizer
      empty_cuda_cache: true

``optimizer_checkpoint_sharding_type`` chooses the Megatron distributed-optimizer checkpoint layout.
``fully_reshardable`` is the default and permits changes to model-parallel geometry, but gathers optimizer state
onto the CPU of data-parallel rank zero during save. ``dp_reshardable`` saves DP-local shards without that gather;
use it only when subsequent training will keep the same tensor, pipeline, context, and expert geometry. The loader
reads each checkpoint's recorded format, so a run can resume an older ``fully_reshardable`` checkpoint and write
new ``dp_reshardable`` checkpoints.


- ``megatron_config.tensor_model_parallel_size``: Tensor model parallel size for reducing memory across model parameters and activations. Megatron sequence parallelism is also enabled by default if tensor parallel size is greater than 1.
- ``megatron_config.pipeline_model_parallel_size``: Pipeline model parallel size for sharding model layers across multiple GPUs.
- ``megatron_config.context_parallel_size``: Context parallel size for reducing activation memory across the sequence length dimension.
- ``megatron_config.expert_model_parallel_size``: The expert parallel size for sharding expert modules across multiple GPUs.
- ``megatron_config.expert_tensor_parallel_size``: The tensor parallel size for each expert module. If set to ``null``, then the value will be resolved to ``tensor_model_parallel_size`` by Megatron. It is recommended to set this to ``1`` when enabling ``expert_model_parallel_size > 1`` for the best performance.

Some rules for configuring these parameters:

- ``model_size = pp_size * tp_size * cp_size``
- ``dp_size = world_size / model_size``
- ``world_size % (pp_size * ep_size * etp_size) == 0``
    - This means that ``ep_size * etp_size`` can scale independently of ``tp_size * cp_size``, and can go across data parallel ranks.

Optimizer Configuration
-----------------------
For both the critic and policy model, we provide a common optimizer configuration

.. code-block:: yaml

    optimizer_config:
       lr: 1.0e-6
       adam_betas: [0.9, 0.999]
       weight_decay: 1e-2
       max_grad_norm: 1.0
       offload_after_step: true
       num_warmup_steps: 0
       scheduler: "constant_with_warmup"

- ``optimizer_config.lr``: Learning rate for the optimizer
- ``optimizer_config.adam_betas``: Betas for AdamW optimizer.
- ``optimizer_config.weight_decay``: L2 regularization strength for AdamW.
- ``optimizer_config.max_grad_norm``: Gradient clipping parameter. The total L2 norm of the model gradients will be scaled to this value during training.
- ``optimizer_config.offload_after_step``: Whether to offload optimizer state to CPU after step if colocated. When generation and training workers are colocated, we recommend using the default setting of ``true``. In some cases with non-colocation, it can be desirable to leave optimizer state on GPU memory to avoid offloading costs as well as additional CPU memory usage.
- ``optimizer_config.num_warmup_steps``: Number of mini-batch steps to warmup the optimizer for.
- ``optimizer_config.scheduler``: Which learning rate scheduler to use. Intended to align with ``transformers.SchedulerType`` from `Huggingface <https://huggingface.co/docs/transformers/main/en/main_classes/optimizer_schedules#transformers.SchedulerType>`_.

Policy and Reference Configuration
----------------------------------

Set ``trainer.policy.model.path`` to the Hugging Face model identifier or local model directory.
The policy optimizer uses ``trainer.policy.optimizer_config``. Policy and reference model parallelism,
router replay, and Megatron runtime settings live under ``trainer.policy.megatron_config`` and
``trainer.ref.megatron_config`` respectively. See :ref:`megatron-configurations`.

Megatron training currently requires ``trainer.critic.model.path=null`` and does not support LoRA.

Algorithm Configuration
-----------------------

.. code-block:: yaml

    algorithm:
      advantage_estimator: "grpo"  # "grpo", "gae", or customizable with AdvantageEstimatorRegistry

      # KL Penalty Parameters
      kl_ctrl: # only used if use_kl_in_reward is true (not applied in the case of use_kl_loss=true) - uses kl_loss_coef as the initial KL coefficient
        type: "fixed" # "fixed" or "adaptive"
        kl_target: 0.1 # target KL divergence for adaptive KL controller
        horizon: 10000 # controls the update rate of the adaptive KL controller

      kl_estimator_type: "k3_unbiased_gradient"

      # note: use_kl_in_reward and use_kl_loss should be mutually exclusive
      use_kl_in_reward: false # apply kl loss to rewards
      use_kl_loss: true # used in policy model
      kl_loss_coef: 0.001
      advantage_batch_normalize: false
      value_head_prefix: "value_head"
      policy_loss_type: "regular" # "regular", "dual_clip", "gspo", "clip_cov", "kl_cov" or customizable with PolicyLossRegistry
      loss_reduction: "token_mean" # "token_mean", "sequence_mean", "seq_mean_token_sum_norm", "seq_mean_token_sum_norm_global"
      grpo_norm_by_std: true # set to false to disable normalization by std in GRPO (used in Dr. GRPO)

      # GAE parameters
      lambd: 1.0
      gamma: 1.0

      # PPO parameters
      eps_clip_low: 0.2
      eps_clip_high: 0.2
      # dual clip parameters
      clip_ratio_c: 3.0

      # clip-cov parameters (only used when policy_loss_type: "clip_cov")
      clip_cov:
        clip_ratio: 0.0002 # fraction of tokens to clip based on covariance
        clip_cov_lb: 1.0 # lower bound for covariance clipping
        clip_cov_ub: 5.0 # upper bound for covariance clipping

      # kl-cov parameters (only used when policy_loss_type: "kl_cov")
      kl_cov:
        kl_cov_frac: 0.2 # percentage of tokens to apply KL regularization to (20%)
        ppo_kl_coef: 1.0 # coefficient for KL regularization term

      # cispo parameters (only used when policy_loss_type: "cispo")
      cispo:
        cispo_eps_clip_low: 1.0  # offset for lower bound of importance sampling ratio clipping (as opposed to PPO token update clipping)
        cispo_eps_clip_high: 5 # offset for upper bound of importance sampling ratio clipping (as opposed to PPO token update clipping)

      # value loss parameters
      value_clip: 0.2

      # dynamic sampling parameters
      dynamic_sampling:
        type: null # filter (DAPO) or null
        max_sample_batches: 30 # inspect at most this many batches of candidate groups per step, -1 for no limit

      # Detached rollout correction for OLD-anchored policy rows
      off_policy_correction: null
      off_policy_correction_rules: null

      # SAPO parameters (only used when policy_loss_type: "sapo") (https://arxiv.org/pdf/2511.20347)
      sapo:
        tau_pos: 1.0
        tau_neg: 1.05 # default values used in the paper with Qwen3-30B-A3B-Base

- ``algorithm.advantage_estimator``: Advantage estimator to use. We currently implement ``grpo``, ``gae``, ``rloo``, ``reinforce++``, ``reward``, and custom advantage estimators can be registered with the ``AdvantageEstimatorRegistry``.
- ``algorithm.kl_ctrl`` Configuration for the KL controller - only used if ``use_kl_in_reward`` is ``true`` (not applied in the case of ``use_kl_loss`` is ``true``). ``kl_loss_coef`` is used as the initial KL coefficient for both ``fixed`` and ``adaptive`` KL controllers.

 - ``type``: Type of KL controller to use. Options include: ``fixed`` or ``adaptive``.
 - ``kl_target``: Target KL divergence for adaptive KL controller.
 - ``horizon``: Controls the update rate of the adaptive KL controller.

- ``algorithm.kl_estimator_type``: ``k1``, ``abs``, ``k2``, ``k3``, or the default ``k3_unbiased_gradient``.
  The default has k3's reported value and k2's gradient through the clamped log probability ratio.
  Under on-policy sampling, away from the log-ratio clamp, its expected gradient is that of
  reverse KL, ``KL(policy || reference)``. The k3 value clamp does not limit that gradient.
  Plain ``k3`` has the forward-KL gradient under on-policy sampling when neither clamp is active.
  Metrics and the KL-in-reward penalty use values only. verl calls this value/gradient
  combination ``k3+``; see `Approximating KL Divergence <http://joschu.net/blog/kl-approx.html>`_
  for the k1, k2 and k3 value estimators.
- ``algorithm.use_kl_in_reward``: Whether to apply KL divergence penalty to rewards. The new rewards will be computed as ``rewards - kl * kl_loss_coef``.
- ``algorithm.use_kl_loss``: Whether to add a KL divergence loss to the policy model. The policy loss will be computed as ``policy_loss + kl * kl_loss_coef``.
- ``algorithm.kl_loss_coef``: Coefficient for the KL divergence loss.
- ``algorithm.advantage_batch_normalize``: Whether to normalize advantages by the (global) batch mean and standard deviation.
- ``algorithm.value_head_prefix``: The name used to identify the value head in the critic model.
- ``algorithm.policy_loss_type``: Type of policy loss to use. Options include:

  - ``regular``: Vanilla PPO loss with token-level importance sampling
  - ``importance_sampling``: Unclipped advantage-weighted loss with the current-to-old policy ratio; see the :doc:`objective usage guide </algorithms/objective_guide>`.
  - ``dual_clip``: Dual clip PPO loss proposed in `this paper <https://arxiv.org/pdf/1912.09729>`_
  - ``gspo``: `Group Sequence Policy Optimization <https://arxiv.org/abs/2507.18071>`_ with sequence-level importance sampling for improved training stability. Implements the "GSPO-token" variant from the paper and requires ``algorithm.loss_reduction=sequence_mean``.
  - ``clip_cov``: Clip-Cov combines standard PPO clipping with covariance-based correction masking for improved stability. Based on `this paper <https://arxiv.org/abs/2505.22617>`_.
  - ``kl_cov``: KL-Cov applies KL regularization to tokens selected based on covariance values. Based on `this paper <https://arxiv.org/abs/2505.22617>`_.
  - ``cispo``: Clipped Importance Sampling Weight Policy Optimization (CISPO) proposed in `MiniMax-M1 <https://arxiv.org/abs/2506.13585>`_.
  - ``sapo``: Smooth sigmoid-gated policy loss with separate positive- and negative-advantage temperatures; see the :doc:`objective usage guide </algorithms/objective_guide>`.
  - ``behavior_clip``: PPO clipping against the sampling policy, with a dual bound for negative advantages; see the :doc:`objective usage guide </algorithms/objective_guide>`.
  - ``sft``: Negative log likelihood on eligible response tokens, independent of advantages; see the :doc:`objective usage guide </algorithms/objective_guide>`.
  - ``dpo``: Direct Preference Optimization over dataset-supplied chosen/rejected pairs against the frozen reference; requires ``algorithm.loss_reduction=pair_mean`` and ``environment.env_class=preference_pair``. See the :doc:`objective usage guide </algorithms/objective_guide>`.
  - Custom policy losses can be registered with the ``PolicyLossRegistry``


- ``algorithm.loss_reduction``: Type of loss reduction to use. Options include:

  - ``token_mean``: computes average loss over all valid tokens in the batch. Used in `DAPO <https://dapo-sia.github.io/>`_.
  - ``sequence_mean``: computes per-sequence avg token loss, then averages over the batch.
  - ``pair_mean``: sums the per-token surrogate values and divides by the global pair count; used by ``dpo`` whose adjacent chosen/rejected rows form one training unit.
  - ``seq_mean_token_sum_norm``: computes the sum of token losses for each sequence, normalizes by the max sequence length (computed as ``cfg.generator.max_input_length + cfg.generator.sampling_params.max_generate_length``), and then averages over the batch. This is used in `Dr. GRPO <https://arxiv.org/abs/2503.20783>`_.
  - ``seq_mean_token_sum_norm_global``: GLOBAL length-unbiased variant of Dr. GRPO. Sums the masked per-token loss over the whole DP batch and divides by a single global denominator ``Z = global_num_seqs * max_seq_len`` (computed once on the driver via a single all-reduce), instead of dividing each micro-batch by ``accumulation_steps``. This sidesteps the mean-of-per-microbatch-means size bias under gradient accumulation + async rollouts.

- ``algorithm.grpo_norm_by_std``: Whether to normalize advantages by the standard deviation in GRPO. This is set to ``false`` in `Dr. GRPO <https://arxiv.org/abs/2503.20783>`_.
- ``algorithm.lambd``: Lambda parameter for GAE.
- ``algorithm.gamma``: Gamma parameter for GAE.
- ``algorithm.eps_clip_low``: Lower bound for PPO clipping.
- ``algorithm.eps_clip_high``: Upper bound for PPO clipping.
- ``algorithm.clip_ratio_c``: Clip ratio for dual clip PPO loss.
- ``algorithm.value_clip``: Clip value for value loss.
- ``algorithm.dynamic_sampling``: Dynamic sampling configuration.
  - ``algorithm.dynamic_sampling.type``: ``filter`` (`DAPO <https://dapo-sia.github.io/>`_) or ``null`` for no dynamic sampling. The filter judges each group as it arrives at the rollout buffer, discards groups without enough reward spread, and keeps drawing prompts until the batch is full.
  - ``algorithm.dynamic_sampling.max_sample_batches``: Per-step limit on candidate groups, in units of ``train_batch_size``: a step that inspects ``max_sample_batches * train_batch_size`` candidates without filling its batch fails. Set to ``-1`` for no limit. The training batch is never shortened.
- ``algorithm.off_policy_correction``: Policy numerator correction: ``tis``, ``icepop``, ``seq_mask_tis``, ``outlier_mask``, ``none`` or ``custom``. Asynchronous OLD-anchored losses require an explicit choice.
- ``algorithm.off_policy_correction_rules``: Token or sequence mask/truncate rules for ``custom`` corrections.
- ``algorithm.dynamic_sampling.max_mean_reward``: Optional exclusive upper bound on the mean final outcome reward of a group. Groups at or above the bound are discarded, including groups with a single final outcome. The selected ``informative_on`` reward source and minimum-spread requirement also apply.
- ``algorithm.advantage_estimator=reward``: Sum each response's eligible rewards and broadcast the sum to its eligible tokens, without group centering or standardization.

- ``algorithm.clip_cov``: Clip-Cov parameters (only used when ``policy_loss_type`` is ``clip_cov``):

  - ``clip_ratio``: Fraction of tokens to clip based on covariance values.
  - ``clip_cov_lb``: Lower bound for covariance clipping.
  - ``clip_cov_ub``: Upper bound for covariance clipping.

- ``algorithm.kl_cov``: KL-Cov parameters (only used when ``policy_loss_type`` is ``kl_cov``):

  - ``kl_cov_frac``: Percentage of tokens to apply KL regularization to.
  - ``ppo_kl_coef``: Coefficient for KL regularization term.

- ``algorithm.cispo``: CISPO parameters (only used when ``policy_loss_type`` is ``cispo``):

  - ``cispo_eps_clip_low``: Defaults to 1.0, giving a zero lower ratio bound. Offset for lower bound of importance sampling ratio clipping. Tokens with importance sampling ratio less than ``1 - cispo_eps_clip_low`` will have their ratio clipped, but can still be updated in the policy gradient update.
  - ``cispo_eps_clip_high``: Offset for upper bound of importance sampling ratio clipping. Tokens with importance sampling ratio greater than ``1 + cispo_eps_clip_high`` will have their ratio clipped, but can still be updated in the policy gradient update.

- ``algorithm.sapo``: SAPO (as proposed in `this paper <https://arxiv.org/pdf/2511.20347>`) parameters (only used when ``policy_loss_type`` is ``sapo``):

  - ``tau_pos``: Temperature for gating function for tokens with positive advantages.
  - ``tau_neg``: Temperature for gating function for tokens with negative (or zero) advantages.

Correction weights multiply the policy numerator and leave the reduction counts, KL, entropy and teacher rows unchanged.
``tis`` caps the old-policy/behavior ratio at 2. ``icepop`` keeps that ratio within [0.5, 5] and gives zero weight outside.
``seq_mask_tis`` combines a sequence geometric-ratio mask in [0.99, 1.01] with token TIS; ``outlier_mask`` discards
sequences with any eligible token ratio outside [1e-4, 100]. A configured correction requires behavior logprobs.
Its metrics are ``policy/correction/weight_mean``, ``policy/correction/truncated_fraction`` and
``policy/correction/masked_fraction``; ratio drift is reported under ``policy/mismatch/pooled/*``.

Launch documents select an objective recipe through ``skyrl.config_groups.algorithm_recipe``. Available recipes are
``grpo``, ``dapo``, ``dr_grpo``, ``gspo``, ``cispo``, ``opd`` and ``mopd``. They set algorithm fields; explicit fields
in the experiment override the recipe. Each recipe cites its paper. They configure the objective, not a full
paper reproduction: model, data, resource layout and generation settings remain experiment choices.
``opd`` and ``mopd`` require the experiment's teacher definitions, routing plan and distillation coefficient.
Asynchronous OLD-anchored recipes also require an explicit ``off_policy_correction`` choice, including ``none``.

Teacher-support objectives include ``sparse_forward_kl``, ``sparse_reverse_kl`` and ``sparse_jsd``. Reverse KL and JSD
use the selected support plus a single remaining-mass bin. JSD requires ``distillation.jsd_beta`` in (0, 1) and uses
``beta * teacher + (1 - beta) * student`` for its mixture. ``distillation.entry_clip`` is an optional upper bound
on each sparse-forward-KL entry contribution. Sparse forward KL conditions the teacher on its support; reverse KL
and JSD use its full-vocabulary-normalized probabilities. The student probabilities use the sampling temperature,
while teacher logprobs are untempered.

Policy Loss Formulation
~~~~~~~~~~~~~~~~~~~~~~~

Policy losses return masked per-token values and clipping diagnostics. The objective
assembler applies data weights and reduces each row using counts for the complete
optimizer step, as described in :doc:`../algorithms/objective`.

.. code-block:: python

  def ppo_policy_loss(inputs: PolicyLossInputs, config: DictConfig) -> TokenLoss:
      ratio = safe_exp_delta(inputs.log_probs - inputs.old_log_probs)
      unclipped = -ratio * inputs.advantages
      clipped = -ratio.clamp(1 - config.eps_clip_low, 1 + config.eps_clip_high) * inputs.advantages
      values = torch.maximum(unclipped, clipped)
      metrics = clipping_metrics(
          ratio,
          clipped > unclipped,
          inputs.loss_mask,
          eps_clip_low=config.eps_clip_low,
          eps_clip_high=config.eps_clip_high,
      )
      if config.use_tis:
          weights = safe_exp_delta(inputs.old_log_probs - inputs.rollout_log_probs)
          values = values * weights.clamp(max=config.tis_imp_ratio_cap)
      return TokenLoss(torch.where(inputs.loss_mask > 0, values, 0), metrics)

Workers retain ``policy/ppo_clip_ratio`` as the pooled clipping fraction and also emit
``policy/ppo_clip_ratio_low`` and ``policy/ppo_clip_ratio_high`` for the bound that changed the objective.
``policy/ppo_clip_pressure_low`` and ``policy/ppo_clip_pressure_high`` report the fraction of ratios outside each
bound before the loss decides whether clipping binds for that token.


Generator Configuration
-----------------------

.. code-block:: yaml

  generator:
    model_dtype: "bfloat16" # should match dtype for inference engine
    run_engines_locally: true
    num_inference_engines: 1
    backend: "vllm"
    weight_sync_backend: "nccl"
    weight_sync_pause:
      mode: keep
      clear_cache: true
    inference_engine_tensor_parallel_size: 4
    inference_engine_pipeline_parallel_size: 1
    inference_engine_expert_parallel_size: 1
    inference_engine_data_parallel_size: 1
    n_samples_per_prompt: 5
    max_input_length: ${trainer.max_prompt_length} # max generator input length used for multi-turn conversations - for single turn set equal to max_prompt_length
    enable_prefix_caching: true
    enable_chunked_prefill: true
    max_num_batched_tokens: 8192
    enforce_eager: false
    gpu_memory_utilization: 0.8
    max_num_seqs: 1024
    remote_inference_engine_urls: ["127.0.0.1:8001"]
    max_turns: 1

    # Custom chat template configuration if needed
    chat_template:
      source: "name"  # "name" or "file"
      name_or_path: null  # e.g., "qwen3_with_thinking" or "/path/to/template.j2"

    # Chat templating kwargs to pass to `tokenizer.apply_chat_template`
    chat_template_kwargs: {}

    engine_init_kwargs: {}

    override_existing_update_group: "auto" # "auto", "enable", "disable"
    # sampling params for generation phase
    sampling_params:
      max_generate_length: 1024
      temperature: 1.0
      top_p: 1.0
      min_p: 0.0
      top_k: -1


    # sampling params for evaluation
    eval_sampling_params:
      max_generate_length: ${generator.sampling_params.max_generate_length}
      temperature: 1.0
      top_p: 1.0
      min_p: 0.0
      top_k: -1

    eval_n_samples_per_prompt: 1

    trajectory_reward_shaping:
      schema_version: 2
      enabled: false
      loop:
        max_period_tokens: 64
        tail_tokens: 256
        minimum_occurrences: 4
        advantage_penalty_per_token: 0.0
        max_advantage_penalty: 0.2
      non_termination:
        penalty: 0.0
        accepted_stop_reasons: [stop, complete, eos, end_turn]
      overlong:
        l_max: ${generator.sampling_params.max_generate_length}
        l_cache: 0
      successful_length:
        free_tokens: 0
        penalty_per_token: 0.0
        max_penalty: 0.2

    trajectory_retention:
      enabled: true
      output_path: ${trainer.export_path}/training_trajectories
      run_id: ${trainer.run_name}
      phases: [train]
      sample_count_per_step: 1
      sample_fraction: 0.0
      always_retain_failures: true
      always_retain_non_terminating: true
      always_retain_loops: true
      accepted_stop_reasons: ${generator.trajectory_reward_shaping.non_termination.accepted_stop_reasons}
      reward_below: null
      reward_above: null
      max_bytes_per_step: 8388608
      max_bytes_per_run: 268435456
      required: false
      redact_fields: []
      model_path: ${trainer.policy.model.path}
      model_source_identity: ${trainer.policy.model.source_identity}
      resume_path: ${trainer.resume_path}
      inference_backend: ${generator.backend}

    apply_overlong_filtering: false

``trajectory_retention`` runs after every generator has produced the common normalized output. It writes gzip-compressed,
content-addressed JSON records for selected training or evaluation trajectories. Count and fraction sampling are deterministic;
failure, non-termination, loop, and reward-threshold selectors retain diagnostic cases independently. The persistent ledger makes
resume idempotent and enforces compressed-byte limits before each write. Set ``required: true`` when a retention write failure must
stop training; best-effort mode instead reports ``generate/trajectory_retention/write_errors``. Iris derives a durable path under
the job's ``trace_jobs`` directory unless the launch configuration supplies an explicit path.


Inference Engine Placement Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``generator.run_engines_locally``: Whether to use local inference engines. If ``true``, the inference engine will be initialized during the training run in the current Ray cluster. We use one Ray actor per inference replica and communication will happen via Ray object store.  If set to ``false``, then the generator expects a list of remote urls and communication will happen over HTTP.
- ``generator.num_inference_engines``: Number of inference engines to use. If ``run_engines_locally`` is ``false``, then this number should match the number of remote urls.
- ``generator.remote_inference_engine_urls``: List of remote urls to use. Applicable only when ``run_engines_locally`` is ``false``.
- ``generator.enable_http_endpoint``: When ``true``, launch an OpenAI-compatible HTTP endpoint for the inference engine client so that generators can send requests to this server instead of using ``.generate()`` Python calls.
- ``generator.http_endpoint_host``: Host for the inference HTTP endpoint.
- ``generator.http_endpoint_port``: Port for the inference HTTP endpoint.

For more details on how different placement options work, please refer to the :doc:`placement guide <placement>`.

Weight Transfer Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``generator.weight_sync_backend``: Backend to use for weight synchronization. Currently, we support ``nccl`` and ``gloo``.
- ``generator.weight_sync_transport``: How weights reach the engines. ``broadcast`` (default) sends every tensor from trainer rank 0 to every engine. ``expert_block`` broadcasts each MoE expert matrix from a Megatron rank that holds it to the vLLM workers that serve that expert, writing into their live parameters; dense weights go to one worker per replica, which broadcasts them within the replica. It requires inference engines that are not colocated with training (``trainer.placement.colocate_all: false``), a Grug MoE trained with the ``megatron`` strategy at ``tensor_model_parallel_size: 1`` and ``expert_tensor_parallel_size: 1``, local vLLM engines at TP=1 with EP equal to DP and DP>=1, ``weight_sync_backend: nccl`` and vLLM's TRITON MoE backend. Worker placements are checked at startup, including a single serving worker. TP=PP=1 replicas with EP=DP>1 can span nodes automatically; other pipeline stages remain node-local. Do not set the removed ``generator.inference_engine_allow_cross_node_ep`` key. Trainer and engine EP sizes may differ, and engines may be pipeline-parallel. Anything else is refused at startup. Each sync logs ``timing/expert_block_sync/{install,policy,receiver,expert,dense}_seconds``.
- ``generator.expert_block_sync.timeout_seconds``: Timeout for creating the sync groups at startup and for the broadcasts of each sync.
- ``generator.expert_block_sync.verify``: If set, replay synchronization and verify it against the trainer values.
- ``generator.override_existing_update_group``: Whether to override the existing update group for the inference engine. This is applicable only for remote inference engines. During training, `skyrl-train` forms a custom process group ("update group") with the rank 0 training worker and all the inference engine ranks.  If ``override_existing_update_group=enable``, then during initialization, a previous weight update group will be overriden in the inference engine. For example, if you have a remote server setup and you run training for the same model multiple times, it is helpful to override the previous update group. We recommend leaving this to ``auto`` - since it will automatically determine if the previous update group should be overridden based on ``run_engines_locally``.

``generator.weight_sync_pause`` sets the local vLLM pause policy during weight sync when
``trainer.rollout_buffer.max_staleness_steps`` is positive:

- ``mode: abort`` ends in-flight requests. Non-streaming single-prompt requests can continue or retry;
  streaming chat completions end with ``finish_reason=abort``.
- ``mode: wait`` lets in-flight requests finish before the sync. It requires
  ``generator.vllm_v1_disable_multiproc=false`` and can delay a step behind long requests.
- ``mode: keep`` (default) freezes in-flight requests and resumes them after the sync, including streams and batches.

``clear_cache: true`` (default) clears KV and prefix caches during the pause. With ``keep``, running requests
re-prefill their prompt and generated tokens under the new weights. ``keep`` with ``clear_cache: false`` retains
KV from the old weights across the sync, which is faster but can mix weight policies in later generation. Only
``keep`` permits ``clear_cache: false``. Non-default pause settings require local vLLM engines; SGLang and remote
engines do not support pausing.

Inference Engine Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``generator.backend``: Backend to use for the inference engine. We support ``vllm`` and ``sglang``. ``sglang`` is supported only for remote inference engines at the moment.
- ``generator.model_dtype``: Dtype used for the inference engine. This is also used during weight transfer - the policy model weights are casted to this dtype before being sent to the inference engine during weight transfer.
- ``generator.inference_engine_tensor_parallel_size``: Tensor parallel size for the inference engine.
- ``generator.inference_engine_pipeline_parallel_size``: Pipeline parallel size for the inference engine. Currently, PP is only supported for vLLM backend.
- ``generator.inference_engine_expert_parallel_size``: Expert parallel size for the inference engine. Currently, EP is only supported for vLLM backend and ep_size must equal dp_size * tp_size.
- ``generator.inference_engine_data_parallel_size``: Data parallel size for the inference engine. Currently, DP is only supported for vLLM backend.
- ``generator.gpu_memory_utilization``: GPU memory utilization for the inference engine. Applicable only for ``run_engines_locally=true``.
- ``generator.vllm_v1_disable_multiproc``: If ``true``, this will set ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` in the environment, which makes the scheduling deterministic. This is useful for reproducibility.
- ``generator.enable_prefix_caching``: Whether to enable prefix caching for the inference engine. Applicable only when ``backend="vllm"``. This can be left to the default ``true`` in most cases. Note that in the case of remote inference engines, you would need to match the setting used when you initialized the remote servers.
- ``generator.enable_chunked_prefill``: Whether to enable chunked prefill for the inference engine. Applicable only when ``backend="vllm"``. With vLLM, this can be left to the default ``true`` in most cases.
- ``generator.max_num_seqs``: Continous batching parameter for vLLM. Maximum number of sequences to pack into a batch.
- ``generator.max_num_batched_tokens``: Continous batching parameter for vLLM. Maximum number of tokens to pack into a batch.

Generation Parameters
~~~~~~~~~~~~~~~~~~~~~

Multi-turn rollouts use structured conversation turns and retain the exact sampled token IDs.

- ``generator.n_samples_per_prompt``: Number of samples to generate per prompt. Note that the total size of the training batch will be ``trainer.train_batch_size * generator.n_samples_per_prompt``.
- ``generator.max_input_length``: Maximum input length for the inference engine. For single turn generation, this can be same as ``trainer.max_prompt_length`` (i.e., the initial prompt length). For multi-turn generation, this is the maximum input length used for multi-turn conversations at each turn.
- ``generator.sampling_params``: Sampling parameters for the inference engine during trajectory generation phase.

    - ``generator.sampling_params.max_generate_length``: Maximum length of the generated response.
    - ``generator.sampling_params.temperature``: Temperature for the inference engine.
    - ``generator.sampling_params.top_p``: Top-p sampling parameter for the inference engine.
    - ``generator.sampling_params.min_p``: Min-p sampling parameter for the inference engine, as proposed in `this paper <https://arxiv.org/pdf/2407.01082>`_.
    - ``generator.sampling_params.top_k``: Top-k sampling parameter for the inference engine.
- ``generator.eval_sampling_params``: Sampling parameters for evaluation.
- ``generator.eval_n_samples_per_prompt``: Number of samples to generate per prompt for evaluation.
- ``generator.max_turns``: Maximum number of turns for generation with multi-turn RL.
- ``generator.engine_init_kwargs``: Inference engine arguments passed directly to the vLLM or SGLang engine. To specify an engine arg in the CLI override, use the format: +generator.engine_init_kwargs.[arg_name]=value. If duplicate kwargs are passed or kwargs clash with existing generator arguments (e.g., ``tensor_parallel_size``), an error is raised.
- ``generator.chat_template``: Custom chat template configuration if needed.
    - ``generator.chat_template.source``: Source of the chat template. Can be either ``name`` or ``file``.
    - ``generator.chat_template.name_or_path``: Name or path of the chat template. If the source is ``name``, then it should be one of the supported templates in :code_link:`skyrl_train/trajectory_runners/trajectory_processing.py`. If the source is ``file``, then this field should be a path to a Jinja2 template file.
- ``generator.chat_template_kwargs``: Chat templating kwargs to pass to ``tokenizer.apply_chat_template``.

Misc Configuration
~~~~~~~~~~~~~~~~~~

- ``generator.trajectory_reward_shaping``: Generator-independent optimization shaping applied after trajectory normalization. ``non_termination`` penalizes stop reasons outside its accepted set. ``overlong`` applies DAPO's outcome-independent linear penalty to the full trajectory between ``l_max - l_cache`` and ``l_max``; the penalty is stored on the final row, and ``l_cache=0`` disables it. The default ``l_max`` follows the generation limit, so multi-turn runners should set it to their intended full-trajectory token budget. ``successful_length`` penalizes trainable response tokens beyond ``free_tokens`` only when the raw task outcome is positive. ``loop`` searches the final trainable segment's tail for the smallest repeating period, then emits capped negative per-token advantage credit for the excess repetitions. This loop credit is applied after advantage normalization and never enters the outcome reward or its group statistics. The raw outcome remains in ``unshaped_rewards`` for pass-rate and verifier-accuracy metrics. ``schema_version`` is stored with the run configuration and emitted on each shaped trajectory.
- ``generator.trajectory_retention``: Generator-independent bounded capture of normalized training trajectories. It samples deterministically per step, always retains configured anomalies, and writes content-addressed compressed records plus a resume-safe ledger. ``required=false`` reports storage failures without stopping training; ``required=true`` fails the run.
- ``generator.apply_overlong_filtering``: Whether to apply DAPO Overlong Filtering to the loss masks. For each trajectory that exceeds the max length (i.e., truncated and does not end with an EOS token), this masks out every token in the loss mask.
- ``trainer.step_wise_training``: Whether to use step-wise training. If ``true``, then the generator will return multi-turn generations with each turn being a separate trajectory. Advantages are computed based on the last step of each trajectory and propagated to the previous steps.
