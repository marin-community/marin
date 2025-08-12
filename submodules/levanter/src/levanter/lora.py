"""
Implements LoRA https://arxiv.org/abs/2106.09685 transforms on Levanter models.

LoRA is a parameter-efficient fine-tuning method that uses a low-rank factorization to reduce the number of parameters
in a model. We support LoRA using a similar approach to the one in [PEFT](https://github.com/huggingface/peft), and implement
routines to export LoRA models in a format compatible with PEFT.

To use LoRA, you need either to specify a rank, a scale (which is typically the same as rank), and a spec for the
target modules. The spec can be specified as:

- None (default), which means all linear modules
- a regex, e.g. `r".*(q_proj|k_proj|v_proj|o_proj).*"`
- a list of strings, e.g. `["q_proj", "k_proj", "v_proj", "o_proj"]`

We recommend using None, which was found to be better than the other options: https://twitter.com/Tim_Dettmers/status/1689375417189412864,
https://arxiv.org/pdf/2305.14314.pdf Section 4.

LoRA is implemented by doing "tree surgery" on the model, replacing [haliax.nn.Linear][] layers with a
[levanter.lora.LoraLinear][] layer that wraps the original linear layer.

Consider a simple model with two parameters, attention and mlp. That might look like:
   ```python
   model = Model(attention=Attention(proj_qkv=Linear), mlp=Mlp(...))
   ```
 With LoRA, our model might look more like:
   ```python
   model = Model(attention=Attention(proj_qkv=LoraLinear(wrapped=orig_proj_qkv, lora=LoraLinear(...)), mlp=Mlp(...))
   ```

 During training, you'll want to keep the base and adapter parameters separate, so that it looks like this:
   ```python
   base_model, adapter_model = partition_lora_params(model)
   base_model = Model(attention=Attention(proj_qkv=LoraLinear(wrapped=orig_proj_qkv, lora=None), mlp=Mlp(...))
   adapter_model = Model(attention=Attention(proj_qkv=LoraLinear(wrapped=None, lora=LoraLinear(...)), mlp=(...))
   ```
 and then we combine them at runtime:
   ```python
   model = combine_lora_params(base_model, lora_params=adapter_model)
   ```
 which just grounds out into a call to [equinox.combine][]
"""
import dataclasses
import functools
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TypeVar, Union

import equinox as eqx
import jax
from jaxtyping import PyTree

import haliax
import haliax as hax
import haliax.nn as hnn
from haliax import Axis
from haliax.jax_utils import shaped_rng_split
from haliax.state_dict import (
    ModuleWithStateDictSerialization,
    StateDict,
    save_state_dict,
    to_torch_compatible_state_dict,
)

from levanter.callbacks import StepInfo
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef, upload_to_hub
from levanter.utils.cloud_utils import temp_dir_before_upload
from levanter.utils.jax_utils import join_key, key_iterator, leaf_key_paths
from levanter.utils.logging import silence_transformer_nag


silence_transformer_nag()
from transformers import PreTrainedTokenizerBase  # noqa: E402


logger = logging.getLogger(__name__)


M = TypeVar("M", bound=PyTree)

# Remaining Tasks
# - bias


LORA_R = "LORA_R"


@dataclass(frozen=True)
class LoraConfig:
    target_modules: Optional[Union[List[str], str]] = None
    """modules to loraize. can either be a regex or a list of strings of module names, or None, meaning all linear modules"""
    r: int = 8  # rank of LoRA transform
    alpha: float = 8.0  # scaling factor for LoRA transform
    dropout: float = 0.0  # dropout probability for LoRA layers
    # TODO: bias

    def matches_target(self, key_path):
        if isinstance(self.target_modules, str):
            compiled = re.compile(self.target_modules)
            return compiled.match(key_path) is not None
        elif self.target_modules is None:
            return True
        else:
            return any(key_path.endswith(target) for target in self.target_modules)


def loraize(model: M, config: LoraConfig, key: jax.random.PRNGKey) -> M:
    """
    Applies LoRA transform to the given model by replacing Linear layers that match the given pattern with LoraLinear layers.
    """
    return _loraize(model, config, key, "", batch_dims=())


class LowRankLinear(eqx.Module):
    """
    A linear layer with a low-rank factorization. Used by LoRA.

    A LowRankLinear is basically 2 matrices that have a common dimension
    """

    lora_A: hnn.Linear
    lora_B: hnn.Linear
    dropout: hnn.Dropout
    scale: float = eqx.field(static=True)

    def __call__(self, x, key=None):
        if key is None and self.dropout.is_active:
            raise RuntimeError(
                "Cannot call LoraLinear with dropout and without a key if dropout is enabled."
                " The base model needs to be retrofitted to pass keys to the Linear layers."
            )
        x = self.dropout(x, key=key)
        z = self.lora_A(x)
        z = self.lora_B(z)
        return z * self.scale

    @staticmethod
    def init(In: hax.Axis, Out: Axis, r: int, alpha: float, dropout_prob: float, *, key):
        """
        Initializes a LoraLinear module.
        """
        _R = hax.Axis(LORA_R, r)
        key_A, key_B = jax.random.split(key)
        # Peft always uses out_first=True (i.e. normal Torch convention) for linear, even for gpt2-style Conv1d
        lora_A = hnn.Linear.init(In, _R, key=key_A, use_bias=False, out_first=True)
        lora_B = hnn.Linear.init(_R, Out, key=key_B, use_bias=False, out_first=True)
        dropout = hnn.Dropout(dropout_prob)

        return LowRankLinear(lora_A, lora_B, dropout, alpha / r)

    def merge(self) -> hax.NamedArray:
        return hax.dot(self.lora_A.weight, self.lora_B.weight, axis=LORA_R) * self.scale


class LoraLinear(ModuleWithStateDictSerialization):
    """
    Linear layer with LoRA transform.
    """

    wrapped: hnn.Linear
    lora: LowRankLinear

    def __call__(self, x, key=None):
        if key is not None:
            k1, k2 = jax.random.split(key)
            return self.lora(x, key=k2) + self.wrapped(x, key=k1)
        else:

            return self.lora(x) + self.wrapped(x)

    def merge(self):
        weight = self.lora.merge() + self.wrapped.weight
        return dataclasses.replace(self.wrapped, weight=weight)

    @staticmethod
    def init(wrapped: hnn.Linear, r: int, alpha: float, dropout: float = 0.0, *, key):
        """
        Initializes a LoraLinear module.
        """
        lora = LowRankLinear.init(wrapped.In, wrapped.Out, r, alpha, dropout, key=key)
        return LoraLinear(wrapped, lora)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"wrapped": None, "lora": None}


def _is_lora_compatible_module(module):
    # TODO: more modules
    return isinstance(module, hnn.Linear)


def filter_lora_params(params: M) -> M:
    """
    Filters LoRA parameters from the given parameter tree.
    """

    return eqx.filter(params, is_lora_param, is_leaf=lambda x: is_lora_param(x) or isinstance(x, haliax.NamedArray))


def partition_lora_params(params: M) -> Tuple[M, M]:
    """
    Partitions the given parameter tree into base/non-LoRA parameters and non-LoRA parameters.

    Returns:
        (base_params, lora_params)
    """
    lora_params, base_params = eqx.partition(params, is_lora_param, is_leaf=is_lora_param)
    return base_params, lora_params


def combine_lora_params(params: M, lora_params: M) -> M:
    """
    Combines the given LoRA parameters with the given parameter tree.
    """
    return eqx.combine(params, lora_params, is_leaf=is_lora_param)


def is_lora_param(node):
    return isinstance(node, LowRankLinear)


def lora_trainable_params_filter(model: M) -> M:
    """
    Creates a filter tree suitable for passing to Trainer.is_trainable marking which parameters are trainable and which
    are not.

    Returns:
       (PyTree) A filter tree marking which parameters are trainable and which are not. This filter is the same as the model,
       except every LoRA param is replaced with True and every other leaf (really, every array) is replaced with False.
    """

    # We only want to train on the lora params. The way to do this in Equinox is generally with
    # a filter tree (cf https://docs.kidger.site/equinox/examples/frozen_layer/),
    # which is a tree with the same structure (or a "tree prefix" thereof) as the model, but with
    # bools or Callable[..., bool] at the leaves. We can then pass this tree to the trainer and it
    # will only train the parameters that are True in the tree.
    # Levanter defines `is_lora_param` for this purpose, but we need to be careful about how we use it.
    # Equinox's primitives don't really have a "match all tree nodes matching a predicate" function (just
    # a "match all tree leaves matching a predicate" function), so we need to be just a bit careful.
    # Basically, we want to halt recursion in the tree whenever we hit a node that is a lora param.
    return jax.tree_util.tree_map(is_lora_param, model, is_leaf=is_lora_param)


def _loraize(model: M, config: LoraConfig, key: jax.random.PRNGKey, prefix: str, batch_dims: Tuple[Axis, ...]) -> M:
    """
    This implementation is mostly straightforward, with one major wrinkle: scan layers like Stacked, which
    add an extra batch dimension and thus require vmap, and thus require a vmap'ed LoRA transform.

    As an example, the GPT-2 Model has a Stacked[Gpt2Block] member, which means that all members of the Gpt2Block
    have an extra Layers dimension. E.g. the c_attn will have shape weight=NamedArray(['layers', 'embed', 'qkv', 'heads', 'head_size']
    even though it's defined as just Linear(In=Embed, Out=(Qkv, Heads, HeadSize)). The Stacked adds the initial Layers dimension.

    There are two ways we can approach scan layers: one is to ask implementors of lora layers to handle
    this themselves, and the other is to handle it here. The former is more flexible, but the latter is
    more convenient, even if it runs the risk of being a leaky abstraction. We choose the latter.
    """
    key_iter = key_iterator(key)

    def _is_special_module(module):
        return _is_lora_compatible_module(module) or isinstance(module, hnn.Stacked)

    def _batchify_ctor(ctor):
        # this is gross but it basically just vmaps the ctor over each batch dimension
        return functools.reduce(lambda ctor, batch_axis: hax.vmap(ctor, batch_axis), reversed(batch_dims), ctor)

    def _loraize_module(module, key_path):
        # TODO: turn into a registry
        if isinstance(module, hnn.Stacked):
            new_inner = _loraize(
                module.stacked,
                config,
                next(key_iter),
                prefix=join_key(key_path, "stacked"),
                batch_dims=batch_dims + (module.Block,),
            )
            return dataclasses.replace(module, stacked=new_inner)
        elif config.matches_target(key_path) and _is_lora_compatible_module(module):
            my_key = next(key_iter)
            batched_key = shaped_rng_split(my_key, [axis.size for axis in batch_dims])
            return _batchify_ctor(LoraLinear.init)(module, config.r, config.alpha, config.dropout, key=batched_key)
        else:
            return module

    return jax.tree_util.tree_map(
        _loraize_module,
        model,
        leaf_key_paths(model, is_leaf=_is_special_module, prefix=prefix),
        is_leaf=_is_special_module,
    )


@hax.named_jit  # needs to be inside (named) jit s.t. it works with sharded parameters
def merge_lora_modules(module: M) -> M:
    """
    Merges LoRA modules into their wrapped modules. That is, it adds the LoRA parameters to the wrapped weights,
    producing a modified base model with no LoRA parameters.
    """

    def _merge_lora_modules(module):
        if isinstance(module, LoraLinear):
            return module.merge()
        else:
            return module

    return jax.tree_util.tree_map(_merge_lora_modules, module, is_leaf=lambda node: isinstance(node, LoraLinear))


SAFETENSORS_WEIGHTS_NAME = "adapter_model.safetensors"
CONFIG_NAME = "adapter_config.json"
DEFAULT_DICT_PREFIX = "base_model.model.transformer"


def save_peft_pretrained(
    lora_model: M,
    config: LoraConfig,
    base_model_name_or_path,
    path: str,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    *,
    prefix: Optional[str] = DEFAULT_DICT_PREFIX,
    upload_to: Optional[Union[bool, str, RepoRef]] = None,
    **upload_kwargs,
):
    """
    Saves a LoRA model as a HuggingFace checkpoint, compatible with Peft.

    Args:
        lora_model: the LoRA model to save
        path: the path to save the model to. May be a url, in which case we will use fsspec to save to that url.
        tokenizer: if provided, will save the tokenizer to the checkpoint
        prefix: the prefix to use for the LoRA parameters. Defaults to "base_model.model.transformer", which is what
            Peft seems to expect.
        upload_to: if provided, will upload the saved model to the given hf hub repo. If a string, will be interpreted
            as a repo name + branch
        upload_kwargs: kwargs to pass to the upload function
    """
    os.makedirs(path, exist_ok=True)
    hf_config = to_hf_config(config, base_model_name_or_path=base_model_name_or_path)
    state_dict = lora_state_dict(lora_model, prefix=prefix)

    with temp_dir_before_upload(path) as local_path:
        save_state_dict(state_dict, f"{local_path}/{SAFETENSORS_WEIGHTS_NAME}")
        with open(f"{local_path}/{CONFIG_NAME}", "w") as f:
            json.dump(hf_config, f)

        if tokenizer is not None:
            tokenizer.save_pretrained(local_path)

        if upload_to is True:
            upload_to = RepoRef.from_string(base_model_name_or_path)

        if upload_to:
            upload_to = RepoRef.from_string(upload_to) if isinstance(upload_to, str) else upload_to
            upload_to_hub(local_path, repo_ref=upload_to, **upload_kwargs)


def save_peft_checkpoint_callback(
    base_path,
    config: LoraConfig,
    base_model_name_or_path,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    upload_to_hf: Optional[Union[bool, str, RepoRef]] = False,
    **hf_upload_kwargs,
):
    """
    If hf_repo is provided, this will upload the checkpoint to the huggingface hub, passing
    any additional kwargs to the huggingface_hub.upload_folder function.

    Args:
        base_path: the base path to save the checkpoint to. `/step-<step>` will be appended to this. base_path
                   may be a GCS bucket path, in which case the checkpoint will be uploaded to GCS after being written to a tmp
        config: the LoRA config to use
        base_model_name_or_path: the name or path of the base model
        tokenizer: If provided, will save the tokenizer to the checkpoint
        upload_to_hf: the repo to upload to. If a string, will be interpreted as a repo name + branch
        hf_upload_kwargs: kwargs to pass to the upload function
    """

    def cb(step: StepInfo):
        nonlocal hf_upload_kwargs
        if step.step == 0:
            return
        if upload_to_hf is not None and "commit_message" not in hf_upload_kwargs:
            my_upload_kwargs = hf_upload_kwargs.copy()
            my_upload_kwargs["commit_message"] = f"Upload for step {step.step} from Levanter"
        else:
            my_upload_kwargs = hf_upload_kwargs

        logger.info(f"Saving PEFT checkpoint for step {step.step} to {base_path}")

        save_peft_pretrained(
            step.eval_model,
            config,
            base_model_name_or_path,
            os.path.join(base_path, f"step-{step.step}"),
            tokenizer,
            upload_to=upload_to_hf,
            **my_upload_kwargs,
        )

        logger.info("Saved checkpoint.")

    return cb


def save_merged_hf_checkpoint_callback(
    base_path,
    converter: HFCheckpointConverter,
    upload_to_hf: Optional[Union[str, RepoRef]] = None,
    **hf_upload_kwargs,
):
    """
    Saves a merged HF checkpoint for the given model. This method essentially combines the base model with the LoRA
    model using [levanter.lora.combine_lora_params][], and then saves the combined model as a HuggingFace checkpoint
    using the given converter.

    If hf_repo is provided, this will upload the checkpoint to the huggingface hub, passing any additional kwargs to the
    huggingface_hub.upload_folder function.

    Args:
        base_path: the base path to save the checkpoint to. `/step-<step>` will be appended to this. base_path
            may be a GCS bucket path, in which case the checkpoint will be uploaded to GCS after being written to a tmp
        converter: the converter to use to convert the base model to HF
        base_model: the base model to save
        upload_to_hf: the repo to upload to. If a string, will be interpreted as a repo name + branch
        **hf_upload_kwargs:
    Returns:
        callback
    """

    def save_merged_hf_model_cb(step: StepInfo):
        if step.step == 0:
            return
        if upload_to_hf is not None and "commit_message" not in hf_upload_kwargs:
            my_upload_kwargs = hf_upload_kwargs.copy()
            my_upload_kwargs["commit_message"] = f"Upload for step {step.step} from Levanter"
        else:
            my_upload_kwargs = hf_upload_kwargs

        logger.info(f"Saving merged HF model for step {step.step} to {base_path}")
        path = os.path.join(base_path, f"step-{step.step}")

        model = step.eval_model

        save_merged_hf_model(model, converter, path, upload_to_hf=upload_to_hf, **my_upload_kwargs)

        logger.info("Saved merged checkpoint.")

    return save_merged_hf_model_cb


def save_merged_hf_model(
    lora_model: M,
    converter: HFCheckpointConverter,
    path: str,
    upload_to_hf: Optional[Union[str, RepoRef]] = None,
    **upload_kwargs,
):
    """
    Saves a merged HF checkpoint for the given model. This method essentially combines the base model with the LoRA
    model using [levanter.lora.merge_lora_modules][], and then saves the combined model as a HuggingFace checkpoint
    """
    merged_model = merge_lora_modules(lora_model)
    if upload_to_hf is None:
        upload_to_hf = False  # type: ignore
    converter.save_pretrained(
        merged_model,
        path,
        upload_to_hf=upload_to_hf,  # type: ignore
        **upload_kwargs,
    )


def to_hf_config(config: LoraConfig, base_model_name_or_path: Optional[str] = None, **kwargs) -> dict:
    """
    Converts a LoraConfig to a HuggingFace config.
    """
    # Example:
    # {
    #   "base_model_name_or_path": "decapoda-research/llama-7b-hf",
    #   "bias": "none",
    #   "enable_lora": null,
    #   "fan_in_fan_out": false,
    #   "inference_mode": true,
    #   "lora_alpha": 16,
    #   "lora_dropout": 0.05,
    #   "merge_weights": false,
    #   "modules_to_save": null,
    #   "peft_type": "LORA",
    #   "r": 16,
    #   "target_modules": [
    #     "q_proj",
    #     "k_proj",
    #     "v_proj",
    #     "o_proj"
    #   ],
    #   "task_type": "CAUSAL_LM"
    # }
    return {
        "base_model_name_or_path": base_model_name_or_path,
        "bias": "none",  # TODO: support bias
        "fan_in_fan_out": False,  # TODO: support fan_in_fan_out
        "inference_mode": True,  # TODO: support inference_mode
        "lora_alpha": config.alpha,
        "lora_dropout": 0.00,  # TODO: support dropout
        "modules_to_save": None,  # TODO: support modules_to_save?
        "peft_type": "LORA",
        "r": config.r,
        "target_modules": config.target_modules,
        "task_type": "CAUSAL_LM",  # TODO: support task_type
        **kwargs,
    }


def lora_state_dict(model: M, prefix: Optional[str] = DEFAULT_DICT_PREFIX) -> StateDict:
    """
    Returns a state dict of the LoRA parameters of the given model without other parameters.
    This method attempts to return a state dict compatible with PEFT's import method.
    """
    state_dict = to_torch_compatible_state_dict(filter_lora_params(model), prefix=prefix)
    return {k: v for k, v in state_dict.items() if v is not None}
