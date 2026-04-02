import torch
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig

from typing import Any
from collections.abc import Sequence, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager, ExitStack
from types import MethodType


@dataclass(slots=True)
class ForwardState:
    storage: dict[Any, Any] = field(default_factory=dict)
    layer: int = 0


ForwardMethods = Sequence[tuple[str, Callable]]


class EarlyStopException(Exception):
    pass


@contextmanager
def register(model, hooks: ForwardMethods):
    """Context manager overwrites forward methods on a PyTorch model and removes them automatically."""
    state = ForwardState()
    with ExitStack() as stack:
        for hook_point, hook in hooks:
            module = find_module_by_path(model, hook_point)
            orig_forward = module.forward

            def new_forward(self, *args, _hook=hook, _fwd=orig_forward, **kwargs):
                return _hook(self, *args, **kwargs, state=state, fwd=_fwd)

            module.forward = MethodType(new_forward, module)

            stack.callback(
                lambda mod=module, f=orig_forward: setattr(mod, "forward", f)
            )

        yield state


def find_module_by_path(model, path: str):
    parts = path.split(".")
    current = getattr(model, "model", model)
    current = getattr(model, "language_model", current)
    try:
        for p in parts:
            if p.isdigit():
                current = current[int(p)]
            else:
                current = getattr(current, p)
        return current
    except (AttributeError, IndexError):
        pass  # fallback to recursive search

    matches = []
    for name, module in model.named_modules():
        if name.split(".")[-1] == path:
            matches.append(module)

    if len(matches) == 0:
        raise ValueError(f"Module '{path}' not found in model")
    elif len(matches) > 1:
        raise ValueError(f"Module '{path}' is ambiguous, multiple matches found")
    else:
        return matches[0]


def load_model(
    model_name: str, dtype=torch.bfloat16, device_map="balanced", quant=False
):
    quantization_config = BitsAndBytesConfig(load_in_8bit=True) if quant else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
        experts_implementation="eager",
        quantization_config=quantization_config,
    )
    model.eval()
    moe_keys = (
        "num_experts",
        "moe_num_experts",
        "num_local_experts",
        "n_routed_experts",
    )
    num_experts: int = _first_attr(model.config, moe_keys, 1)

    return model, num_experts > 1, num_experts


def _first_attr(obj, attrs, default=None):
    for a in attrs:
        value = getattr(obj, a, None)
        if value is not None:
            return value
    return default
