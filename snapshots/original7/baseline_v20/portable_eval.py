"""Hardware-aware evaluation helpers; model weights and metric rules are unchanged."""
from contextlib import nullcontext
import os
import torch


def precision(device):
    name = os.environ.get("XC_EVAL_PRECISION", "auto")
    native_bf16 = device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported()
    if name == "auto":
        name = ("bf16" if native_bf16 else "fp16") if device.type == "cuda" else "fp32"
    if name not in ("bf16", "fp16", "fp32"):
        raise ValueError(f"Invalid evaluation precision {name}")
    if name == "bf16" and device.type == "cuda" and not native_bf16:
        raise RuntimeError("This GPU does not support BF16; select fp16 or fp32.")
    if name == "fp16" and device.type != "cuda":
        raise ValueError("FP16 evaluation requires CUDA; use fp32 on CPU.")
    return name


def autocast(device):
    name = precision(device)
    if name == "fp32":
        return nullcontext()
    return torch.autocast(device.type, dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[name])


def batch_size(default=8):
    value = int(os.environ.get("XC_EVAL_BATCH", str(default)))
    if value < 1:
        raise ValueError("Evaluation batch size must be positive")
    return value


def load_adapter(model, path):
    """Frozen backbone keys are absent by design; all trained keys must be present."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    state = ck["ema"]
    expected = set(model.trainable_parameter_names())
    missing = expected - state.keys()
    if missing:
        raise RuntimeError(f"Incomplete adapter checkpoint {path}: {sorted(missing)}")
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected adapter keys: {incompatible.unexpected_keys}")
    return ck
