"""Three carrier directions using the original A_V13_GRL model and loss loop.

Training fixes final-epoch EMA before seeing the target. The target manifest and
images are loaded only after training has returned and the checkpoint is saved.
Run --help for the CLI; --dry-run validates sources without importing PyTorch.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLASSES = ["Away", "Bend", "Kneel", "Pick", "SStep", "Sit", "Towards"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for command in ("train", "eval"):
        p = sub.add_parser(command)
        p.add_argument("--target", type=int, choices=[10, 24, 77], required=True)
        p.add_argument("--protocol", choices=["grouped", "legacy"], default="grouped")
        p.add_argument("--variant", choices=["proposed", "no_das_acr"], default="proposed",
                       help="no_das_acr disables DAS and both carrier losses; same backbone and generic augmentations")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--run-dir", type=Path, required=True)
        p.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
        p.add_argument("--batch-size", type=int, default=16)
        p.add_argument("--eval-batch-size", type=int, default=16)
        p.add_argument("--num-workers", type=int, default=0)
        p.add_argument("--task-root", type=Path, help="Directory containing manifest/train,val,test.csv")
        p.add_argument("--dataset-root", type=Path, default=ROOT)
        if command == "train":
            p.add_argument("--epochs", type=int, default=100)
            p.add_argument("--max-steps", type=int, help="SMOKE ONLY: cap steps per epoch; starts EMA at epoch 1")
            p.add_argument("--fast-gpu", action="store_true", help="Optional GPU augmentation; changes interpolation/RNG")
            p.add_argument("--dry-run", action="store_true", help="Validate source manifests; no model or target data loaded")
            p.add_argument("--skip-final-eval", action="store_true", help="Save final EMA and evaluate later with eval")
        else:
            p.add_argument("--checkpoint", type=Path, help="Defaults to RUN_DIR/checkpoints/final_ema.pt")
    args = ap.parse_args(argv)
    if args.protocol == "legacy" and args.target != 77:
        ap.error("legacy supports target77 only; use grouped for the two additional directions")
    if args.batch_size < 2 or args.eval_batch_size < 1 or args.num_workers < 0:
        ap.error("batch-size must be >=2, eval-batch-size >=1, num-workers >=0")
    if args.command == "train":
        if args.epochs < 1 or (args.max_steps is not None and args.max_steps < 1):
            ap.error("epochs and max-steps must be positive")
        if args.max_steps is None and args.epochs < 5:
            ap.error("normal runs need epochs>=5 (original EMA starts at 5); use --max-steps for smoke")
    args.sources = [f for f in (10, 24, 77) if f != args.target]
    tree = "cross_frequency_grouped" if args.protocol == "grouped" else "cross_frequency"
    args.task_root = (args.task_root or ROOT / "tasks" / tree / f"target{args.target}").resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.run_dir = args.run_dir.resolve()
    return args


def source_preflight(args):
    """Read source train/val only, not the target manifest contents."""
    counts, paths, recording_ids, hashes = {}, {}, {}, {}
    for split in ("train", "val"):
        manifest = args.task_root / "manifest" / f"{split}.csv"
        with manifest.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"Empty source manifest: {manifest}")
        freqs = {r["frequency"] for r in rows}
        expected = {f"{f}GHz" for f in args.sources}
        if freqs != expected:
            raise ValueError(f"{manifest}: frequencies {freqs}, expected {expected}")
        if {r["class"] for r in rows} != set(CLASSES):
            raise ValueError(f"{manifest}: expected all seven classes")
        for frequency in expected:
            if {r["class"] for r in rows if r["frequency"] == frequency} != set(CLASSES):
                raise ValueError(f"{manifest}: {frequency} does not contain all seven classes")
        if args.protocol == "grouped" and any(not r.get("recording_id") or not r.get("sha256") for r in rows):
            raise ValueError(f"{manifest}: grouped protocol requires recording_id and sha256 provenance")
        for r in rows:
            image_path = (args.dataset_root / r["path"].replace("\\", "/")).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            if r.get("sha256") and hashlib.sha256(image_path.read_bytes()).hexdigest() != r["sha256"]:
                raise ValueError(f"Image content differs from prepared manifest: {image_path}")
        paths[split] = {r["path"].replace("\\", "/") for r in rows}
        recording_ids[split] = {r.get("recording_id") for r in rows}
        hashes[split] = {r.get("sha256") for r in rows}
        counts[split] = {
            "rows": len(rows), "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "frequency_counts": {f: sum(r["frequency"] == f for r in rows) for f in sorted(freqs)},
        }
    if paths["train"] & paths["val"]:
        raise ValueError("Source train/val image paths overlap")
    if args.protocol == "grouped":
        if recording_ids["train"] & recording_ids["val"]:
            raise ValueError("Grouped protocol forbids source train/val recording overlap")
        if hashes["train"] & hashes["val"]:
            raise ValueError("Grouped protocol forbids source train/val identical image content")
    if not (args.task_root / "manifest" / "test.csv").is_file():
        raise FileNotFoundError(args.task_root / "manifest" / "test.csv")
    if counts["train"]["rows"] < args.batch_size:
        raise ValueError("Source training count is smaller than batch size")
    return counts


def resolve_precision(requested, device_type, bf16_supported=False):
    if requested == "auto":
        return ("bf16" if bf16_supported else "fp16") if device_type == "cuda" else "fp32"
    if device_type == "cpu" and requested != "fp32":
        raise ValueError("CPU execution uses fp32; choose --precision auto or fp32")
    if device_type == "cuda" and requested == "bf16" and not bf16_supported:
        raise ValueError("This GPU does not support bf16 (e.g. V100); use auto or fp16")
    return requested


def configure(args):
    if "config" in sys.modules:
        raise RuntimeError("Training config is already imported; launch each direction/run in a fresh Python process")
    # Pin the intended method explicitly. Upstream defaults are V15+V13 and do
    # NOT represent the A_V13_GRL headline row. Set environment before importing
    # config so spawned DataLoader workers inherit the same data/augmentation.
    recipe = {
        "V921_TASK_ROOT": str(args.task_root), "V921_DATASET_ROOT": str(args.dataset_root),
        "V921_TRAIN_FREQS": ",".join(f"{f}GHz" for f in args.sources),
        "V921_TEST_FREQS": f"{args.target}GHz",
        "V921_SOURCE_BATCH_SIZE": str(args.batch_size), "V921_NUM_WORKERS": str(args.num_workers),
        "V921_EVAL_BATCH": str(args.eval_batch_size),
        "V921_EXPERIMENT_NAME": f"A_V13_GRL_{args.protocol}_target{args.target}",
        "V921_USE_DAS": "1", "V921_DAS_MODE": "curriculum",
        "V921_USE_DANN": "0", "V921_USE_HFT": "1", "V921_USE_SPEC_AUGMENT": "1",
        "V921_SEEDED_HFT": "1",
        "V921_FAST_GPU": "1" if getattr(args, "fast_gpu", False) else "0",
        "V921_SKIP_ORACLE": "1", "V921_SKIP_LAST_CKPT": "1",
        "V921_CARRIER_NORM": "off", "V921_LOWBAND_GHZ": "10.0",
        "V921_ARC_MARGIN": "0.25", "V921_SUPCON_WEIGHT": "0.25", "V921_MIRO_WEIGHT": "0.1",
        "V13_GRL_WEIGHT": "0.3", "V13_GRL_TARGET": "shown", "V13_GRL_DISCRETE": "0",
        "V13_FREQ_WEIGHT": "0.05", "V13_DECORR_WEIGHT": "0",
        "V15_FALSIFY_WEIGHT": "0", "V15_KIN_SOURCE_WEIGHT": "0",
        "V15_SENSOR_UNIFORM_WEIGHT": "0", "V15_CONSIST_WEIGHT": "0",
        "V15R_REALISM_WEIGHT": "0", "V15R_WORSTCASE": "0", "V15R_FALSIFY_WEIGHT": "0",
        "V15R_MARGIN_WEIGHT": "0", "V15R_SINGLE_HEAD": "0",
    }
    if args.variant == "no_das_acr":
        recipe.update({"V921_USE_DAS": "0", "V13_GRL_WEIGHT": "0", "V13_FREQ_WEIGHT": "0"})
    recipe["V921_EXPERIMENT_NAME"] = f"{args.variant}_{args.protocol}_target{args.target}"
    os.environ.update(recipe)
    sys.path.insert(0, str(ROOT / "baseline_v20"))
    sys.path.insert(1, str(ROOT / "EXPERIMENTSRESULT"))
    import torch
    import config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    native_bf16 = (device.type == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8
                   and torch.cuda.is_bf16_supported())
    precision = resolve_precision(args.precision, device.type, native_bf16)
    if os.name == "nt" and args.num_workers > 0:
        raise ValueError("The upstream dynamic-library loader requires --num-workers 0 on Windows; Linux supports workers")
    if getattr(args, "fast_gpu", False) and device.type != "cuda":
        raise ValueError("--fast-gpu requires CUDA")
    config.AMP_DTYPE = precision
    os.environ["V921_AMP_DTYPE"] = precision
    if getattr(args, "max_steps", None) is not None:
        config.EMA_START_EPOCH = 1
    args.backbone = config.DEFAULT_BACKBONE
    args.adapter_mode = config.BACKBONE_TUNE_MODE
    args.strict_final_ema = True
    args.resume = None
    return torch, config, device, recipe


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def evaluate_final(args, model, config, device, checkpoint):
    """One pass over the entire target; kinematic logits match ensemble_rules."""
    import numpy as np
    import torch
    import v9_2_1lib as lib
    df = lib.load_manifest("test", keep_7c=True)
    if len(df) == 0 or set(df["frequency"]) != {f"{args.target}GHz"}:
        raise ValueError("Target manifest must contain only the requested target carrier")
    if set(df["class"]) != set(CLASSES):
        raise ValueError("Target manifest must contain all seven classes")
    snapshot_dir = args.run_dir / "manifests"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(snapshot_dir / "test.csv", index=False)
    loader = lib.make_eval_loader(df, batch_size=args.eval_batch_size)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[config.AMP_DTYPE]
    model.eval()
    probabilities, labels, raw_logits = [], [], []
    with torch.inference_mode():
        for x, y, _ in loader:
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
                z = model.encode_adapted(x.to(device, non_blocking=True))
                logits = model.logits_from_neck(z, margin=False)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite target logits")
            probabilities.append(logits.float().softmax(1).cpu().numpy())
            raw_logits.append(logits.float().cpu().numpy())
            labels.append(y.numpy())
    prob, y = np.concatenate(probabilities), np.concatenate(labels)
    pred = prob.argmax(1)
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    np.add.at(cm, (y, pred), 1)
    per_class = {}
    for i, name in enumerate(CLASSES):
        tp, support, predicted = int(cm[i, i]), int(cm[i].sum()), int(cm[:, i].sum())
        per_class[name] = {"precision": tp / max(1, predicted), "recall": tp / max(1, support),
                           "f1": 2 * tp / max(1, support + predicted), "support": support}
    summary = {
        "target_ghz": args.target, "source_ghz": args.sources, "protocol": args.protocol,
        "variant": checkpoint["variant"],
        "seed": int(checkpoint["seed"]), "epoch": int(checkpoint["epoch"]),
        "selection": "fixed_final_epoch_ema", "inference_head": "kinematic_only",
        "target_used_for_training_or_selection": False,
        "smoke_only": bool(checkpoint.get("smoke_only", False)),
        "count": len(y), "acc": float(np.mean(y == pred)),
        "macro_f1": float(np.mean([r["f1"] for r in per_class.values()])),
        "per_class": per_class, "class_order": CLASSES, "confusion_matrix": cm.tolist(),
        "precision": config.AMP_DTYPE,
        "target_manifest_sha256": hashlib.sha256((args.task_root / "manifest" / "test.csv").read_bytes()).hexdigest(),
        "scientific_config_sha256": checkpoint.get("scientific_config_sha256"),
    }
    reports = args.run_dir / "reports"
    save_json(reports / "final_target_metrics.json", summary)
    np.savez_compressed(reports / "final_target_predictions.npz", probabilities=prob, logits=np.concatenate(raw_logits), labels=y,
                        predictions=pred, paths=np.asarray(df["path"].tolist()), classes=np.asarray(CLASSES))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv=None):
    args = parse_args(argv)
    source_counts = source_preflight(args)
    args.source_manifest_hashes = {k: v["sha256"] for k, v in source_counts.items()}
    if args.command == "train" and args.dry_run:
        print(json.dumps({"validated": True, "target_ghz": args.target, "source_ghz": args.sources,
                          "protocol": args.protocol, "variant": args.variant, "task_root": str(args.task_root),
                          "sources": source_counts, "target_data_loaded": False}, indent=2))
        return
    if args.command == "train" and args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty run directory: {args.run_dir}; choose a new run-dir")
    torch, config, device, recipe = configure(args)
    import v15_v18_common_train as trainer
    import v9_2_1lib as lib
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "reports").mkdir(exist_ok=True)
    (args.run_dir / "checkpoints").mkdir(exist_ok=True)
    if args.command == "train":
        effective_config = {k: v for k, v in vars(config).items() if k.isupper()
                            and isinstance(v, (str, int, float, bool, list, dict, type(None)))}
        scientific_config = {
            "target_ghz": args.target, "protocol": args.protocol, "variant": args.variant,
            "epochs": args.epochs, "batch_size": args.batch_size, "num_workers": args.num_workers,
            "precision": config.AMP_DTYPE, "source_manifests": source_counts,
            "effective_config": {k: v for k, v in effective_config.items()
                                 if k not in ("SEED", "EXPERIMENT_NAME", "TERMINAL_PROGRESS")},
            "smoke_only": args.max_steps is not None, "max_steps_per_epoch": args.max_steps,
        }
        args.scientific_config_sha256 = hashlib.sha256(
            json.dumps(scientific_config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        save_json(args.run_dir / "run_config.json", {
            "scientific_config": scientific_config, "scientific_config_sha256": args.scientific_config_sha256,
            "command": [str(x) for x in sys.argv], "source_ghz": args.sources,
            "target_ghz": args.target, "protocol": args.protocol, "variant": args.variant, "seed": args.seed,
            "epochs": args.epochs, "batch_size": args.batch_size, "num_workers": args.num_workers,
            "precision_requested": args.precision, "precision_resolved": config.AMP_DTYPE,
            "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "python": platform.python_version(), "torch": str(torch.__version__),
            "sources": source_counts, "task_root": str(args.task_root), "dataset_root": str(args.dataset_root),
            "environment_recipe": recipe, "effective_config": effective_config,
            "smoke_only": args.max_steps is not None, "max_steps_per_epoch": args.max_steps,
            "selection": "fixed_final_epoch_ema", "resume_supported": False,
            "notes": ["Original DAS schedule kept for all directions; it is not direction-tuned.",
                      "PyTorch/Python/NumPy/sampler seeds are set; GPU kernels are not forced deterministic.",
                      "HFT noise uses seeded NumPy RNG; fixes upstream unseeded default_rng() and changes its retraining RNG stream.",
                      "Default PIL and optional GPU grid_sample augmentation are not numerically equivalent.",
                      "Training CE keeps the original kin+sensor sum; inference uses kin-only logits."],
        })
        log, log_fp = trainer.log_factory(args.run_dir)
        try:
            log(f"run_dir={args.run_dir}")
            log(f"device={device} precision={config.AMP_DTYPE}")
            result = trainer.train(args, args.run_dir, device, log)
            if args.skip_final_eval:
                return
            model = result["model"]
            result["ema"].copy_to(model)
            checkpoint = torch.load(result["final_ema"], map_location="cpu", weights_only=True)
        finally:
            log_fp.close()
    else:
        path = args.checkpoint or args.run_dir / "checkpoints" / "final_ema.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != "cross_frequency_final_ema_v1":
            raise ValueError("Use reproduce.py for shipped checkpoints; eval accepts only this runner's final EMA")
        if checkpoint["target_frequencies"] != config.TEST_FREQS or checkpoint["source_frequencies"] != config.TRAIN_FREQS:
            raise ValueError("Checkpoint carrier direction differs from requested direction")
        if checkpoint.get("protocol") != args.protocol:
            raise ValueError("Checkpoint split protocol differs from requested protocol")
        if checkpoint.get("variant") != args.variant:
            raise ValueError("Checkpoint variant differs; pass the matching --variant")
        if checkpoint.get("source_manifest_hashes") != args.source_manifest_hashes:
            raise ValueError("Source manifest hashes differ from training; use the original --task-root")
        if checkpoint["classes"] != config.CLASSES:
            raise ValueError("Checkpoint class order differs")
        lib.set_seed(int(checkpoint["seed"]))
        model = lib.TimmBackboneV921(checkpoint["backbone"], config.NUM_CLASSES,
                                   adapter_mode=checkpoint["adapter_mode"]).to(device)
        expected = set(model.trainable_parameter_names())
        if set(checkpoint["ema"]) != expected:
            raise ValueError("EMA state keys differ from trainable model parameters")
        model.load_state_dict(checkpoint["ema"], strict=False)
    evaluate_final(args, model, config, device, checkpoint)


if __name__ == "__main__":
    main()
