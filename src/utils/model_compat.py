"""Compatibility helpers for loading older serialized models."""
import sys
import functools
import numpy as np
import numpy.random._pickle as np_random_pickle
import gymnasium as gym


_original_torch_load = None


def patch_torch_load_for_legacy():
    global _original_torch_load
    import torch
    if _original_torch_load is None:
        _original_torch_load = torch.load
        @functools.wraps(_original_torch_load)
        def _patched_load(*args, **kwargs):
            kwargs.pop("weights_only", None)
            return _original_torch_load(*args, **kwargs)
        torch.load = _patched_load


def unpatch_torch_load():
    global _original_torch_load
    import torch
    if _original_torch_load is not None:
        torch.load = _original_torch_load
        _original_torch_load = None


def ensure_numpy_pickle_compat():
    """Register legacy numpy module aliases used by older cloudpickle payloads."""
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)

    if getattr(np_random_pickle.__bit_generator_ctor, "__name__", "") != "_compat_bit_generator_ctor":
        original_ctor = np_random_pickle.__bit_generator_ctor

        def _compat_bit_generator_ctor(bit_generator_name="MT19937"):
            if not isinstance(bit_generator_name, str):
                bit_generator_name = getattr(bit_generator_name, "__name__", str(bit_generator_name))
            return original_ctor(bit_generator_name)

        np_random_pickle.__bit_generator_ctor = _compat_bit_generator_ctor


def sb3_custom_objects(state_dim, num_actions=7):
    """Custom objects for loading older SB3 models without pickled spaces."""
    return {
        "observation_space": gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        ),
        "action_space": gym.spaces.Discrete(num_actions),
    }
