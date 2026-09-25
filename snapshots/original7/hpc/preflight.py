#!/usr/bin/env python
"""Verify the installed environment and a real GPU operation inside a Slurm job."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("This GPU check must run in a Slurm job, not on the login node.")
    root = Path(__file__).resolve().parents[1]
    backbone = root / "weights/hub/models--timm--vit_large_patch16_dinov3.lvd1689m"
    if not backbone.is_dir() or not any(backbone.rglob("*.safetensors")):
        raise SystemExit(f"Missing offline DINOv3 cache/weights: {backbone}")
    versions = {}
    for name in ("torch", "torchvision", "timm", "numpy", "pandas", "PIL",
                 "huggingface_hub", "safetensors", "scipy", "peft", "transformers", "tqdm"):
        module = importlib.import_module(name)
        versions[name] = str(getattr(module, "__version__", "unknown"))
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU visible. Check the GPU resource request and PyTorch CUDA build.")
    properties = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    try:
        driver_query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"],
            capture_output=True, text=True, check=True, timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        driver_query = f"nvidia-smi query unavailable: {exc}"
    # This checks actual kernel execution plus fp16 backward/scaling, not only
    # CUDA visibility. It also catches incompatible GPU architecture builds.
    model = torch.nn.Linear(64, 4).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        loss = model(torch.randn(16, 64, device="cuda")).square().mean()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    if not bool(torch.isfinite(loss)):
        raise SystemExit("Non-finite result in GPU fp16 preflight.")
    report = {
        "status": "gpu_preflight_passed",
        "note": "Environment check only; this is not a reproduction accuracy result.",
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "versions": versions,
        "torch_cuda": torch.version.cuda,
        "torch_cuda_arch_list": torch.cuda.get_arch_list(),
        "gpu_name": properties.name,
        "gpu_capability": list(capability),
        "gpu_memory_gib": round(properties.total_memory / 1024**3, 2),
        "bf16_native_supported": capability[0] >= 8 and torch.cuda.is_bf16_supported(),
        "nvidia_smi_driver_query": driver_query,
        "offline_backbone": str(backbone),
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "gpu_preflight.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
