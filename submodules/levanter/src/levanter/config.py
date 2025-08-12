import atexit
import functools
import inspect
import os
import sys
import tempfile
import urllib.parse
from datetime import timedelta
from functools import wraps
from typing import List, Optional, Union

import draccus
import fsspec
import jmp
from draccus import parse
from fsspec import AbstractFileSystem

from levanter.utils.datetime_utils import encode_timedelta, parse_timedelta


JsonAtom = Union[str, int, float, bool, None]


def register_codecs():
    # draccus.encode.register(jnp.dtype, lambda dtype: dtype.name)
    # draccus.encode.register(type(jnp.float32), lambda meta: meta.dtype.name)
    # draccus.decode.register(jnp.dtype, lambda dtype_name: jnp.dtype(dtype_name))

    def policy_encode(policy: jmp.Policy):
        def name(dtype):
            if hasattr(dtype, "name"):
                return dtype.name
            elif hasattr(dtype, "dtype"):
                return name(dtype.dtype)

        out = f"compute={name(policy.compute_dtype)},params={name(policy.param_dtype)},output={name(policy.output_dtype)}"
        assert jmp.get_policy(out) == policy
        return out

    draccus.decode.register(jmp.Policy, lambda policy_str: jmp.get_policy(policy_str))
    draccus.encode.register(jmp.Policy, policy_encode)

    draccus.decode.register(timedelta, parse_timedelta)
    draccus.encode.register(timedelta, encode_timedelta)

    from haliax import ScanCheckpointPolicy

    def scan_checkpoint_policy_encode(policy: ScanCheckpointPolicy):
        return policy

    def scan_checkpoint_policy_decode(policy: str | dict | bool):
        if not isinstance(policy, dict):
            return ScanCheckpointPolicy.from_bool_or_str(policy)

        from draccus.parsers.decoding import decode_dataclass

        return decode_dataclass(ScanCheckpointPolicy, policy)

    draccus.decode.register(ScanCheckpointPolicy, scan_checkpoint_policy_decode)
    draccus.encode.register(ScanCheckpointPolicy, scan_checkpoint_policy_encode)


register_codecs()


DEFAULT_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")


def main(fn=None, *, args: Optional[List[str]] = None, config_dir: Optional[str] = DEFAULT_CONFIG_DIR):
    """
    Like draccus.wrap but can handle config paths that are urls loadable by fsspec.
    This isn't documented in levanter.config.main_decorator, but only the first arg can be config-ified.

    :param args: the args to parse. If None, will use sys.argv[1:]
    :param config_dir: the directory to look for configs in (if the path does not exist already). If None, will only use the current working directory
    """

    if fn is None:
        return functools.partial(main, args=args, config_dir=config_dir)

    _cmdline_args = args
    if args is None:
        _cmdline_args = sys.argv[1:]

    @wraps(fn)
    def wrapper_inner(*args, **kwargs):
        config_path, cmdline_args = _maybe_get_config_path_and_cmdline_args(_cmdline_args)
        paths_to_check = [config_path, f"{config_path}.yaml", f"{config_path}.yml"]
        if config_path is not None and config_dir is not None:
            paths_to_check.extend([os.path.join(config_dir, p) for p in paths_to_check])

        for path in paths_to_check:
            if path is not None and os.path.exists(path):
                config_path = path
                break

        argspec = inspect.getfullargspec(fn)
        argtype = argspec.annotations[argspec.args[0]]
        cfg = parse(config_class=argtype, config_path=config_path, args=cmdline_args)
        response = fn(cfg, *args, **kwargs)
        return response

    return wrapper_inner


def _maybe_get_config_path_and_cmdline_args(args: List[str]):
    """
    We want to accept ... --config_path <config> ... where config could be a path or url.
    If URL, we need to download it and save it to a temp file. We then want to remove --config_path
    from the cmdline args so that draccus doesn't try to load it as a config path and return it separately here
    along with the modified cmdline args.
    We also accept ... --configs <config1> <config2> ... and concatenate them into a single config.
    """
    if "--config_path" not in args and "--config" not in args and "--configs" not in args:
        return None, args
    else:
        config_args = ["--config_path", "--config", "--configs"]
        found_indices = [args.index(arg) for arg in config_args if arg in args]
        if len(found_indices) > 1:
            raise ValueError(f"Multiple config args found in {args}")
        config_path_index = found_indices[0] + 1

        config_paths: List[str] = []
        args = args.copy()
        del args[config_path_index - 1]
        config_path_index -= 1

        while config_path_index < len(args) and not args[config_path_index].startswith("-"):

            config_path = args[config_path_index]

            if urllib.parse.urlparse(config_path).scheme:
                fs: AbstractFileSystem
                fs, fs_path = fsspec.core.url_to_fs(config_path)
                temp_file = tempfile.NamedTemporaryFile(prefix="config", suffix=".yaml", delete=False)
                atexit.register(lambda: os.unlink(temp_file.name))
                fs.get(fs_path, temp_file.name)
                config_path = temp_file.name

            config_paths.append(config_path)
            del args[config_path_index]

        merged_config_path = None

        if len(config_paths) == 1:
            merged_config_path = config_paths[0]
        elif len(config_paths) > 1:
            # merge the configs by concatenating them
            temp_merged_config_path = tempfile.NamedTemporaryFile(prefix="config_merged", suffix=".yaml", delete=False)
            atexit.register(lambda: os.unlink(temp_merged_config_path.name))
            with open(temp_merged_config_path.name, "w") as f:
                for config_path in config_paths:
                    with open(config_path) as config_file:
                        f.write(config_file.read())
            merged_config_path = temp_merged_config_path.name
        else:
            raise ValueError("No config path found in args")

        return merged_config_path, args
