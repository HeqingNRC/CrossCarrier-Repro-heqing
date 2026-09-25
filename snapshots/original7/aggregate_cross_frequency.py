"""Aggregate completed final-EMA runs, refusing mixed recipes or partial seed sets.

python aggregate_cross_frequency.py --runs-root /path/to/train --output /path/to/summary
Scans recursively, so multiple Slurm array job directories may share a runs root.
Smoke tests are always excluded. No target results are used to choose a run.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def scores(pred, y, nc):
    cm = np.zeros((nc, nc), dtype=int)
    np.add.at(cm, (y, pred), 1)
    den = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * cm.diagonal(), den, out=np.zeros(nc), where=den != 0)
    return {"acc": float((pred == y).mean()), "macro_f1": float(f1.mean()), "per_class_f1": f1.tolist()}


def recipe_signature(config):
    excluded = {"SEED", "EXPERIMENT_NAME", "TRAIN_FREQS", "TEST_FREQS", "FREQ_TO_IDX"}
    effective = {k: v for k, v in config["effective_config"].items() if k not in excluded}
    # Absolute paths/seed/hardware identifiers are provenance, not recipe inputs.
    return {"epochs": config["epochs"], "batch_size": config["batch_size"], "num_workers": config["num_workers"],
            "precision": config["precision_resolved"], "effective_config": effective,
            "sources": config["sources"]}


def aggregate(paths, seeds):
    groups, ignored = {}, []
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("smoke_only"):
            ignored.append(str(path))
            continue
        key = (row["protocol"], row["variant"], row["target_ghz"])
        bucket = groups.setdefault(key, {})
        seed = row["seed"]
        if seed not in seeds:
            raise ValueError(f"Unexpected seed {seed} at {path}; choose an explicit runs-root")
        if seed in bucket:
            raise ValueError(f"Duplicate runs for {key}, seed {seed}; do not select by target score")
        cfg = json.loads((path.parents[1] / "run_config.json").read_text(encoding="utf-8"))
        if cfg["smoke_only"] or cfg["epochs"] != row["epoch"]:
            raise ValueError(f"Incomplete/invalid final epoch: {path}")
        if row["selection"] != "fixed_final_epoch_ema":
            raise ValueError(f"Unexpected model-selection rule: {path}")
        with np.load(path.with_name("final_target_predictions.npz"), allow_pickle=False) as data:
            arrays = {name: data[name].copy() for name in data.files}
        bucket[seed] = (row, cfg, arrays, str(path))
    results = []
    for key, bucket in sorted(groups.items()):
        if set(bucket) != set(seeds):
            raise ValueError(f"Incomplete seeds for {key}: found {sorted(bucket)}, need {seeds}")
        ordered = [bucket[s] for s in seeds]
        first, first_cfg, reference, _ = ordered[0]
        signature = recipe_signature(first_cfg)
        for row, cfg, data, path in ordered:
            if cfg["scientific_config_sha256"] != first_cfg["scientific_config_sha256"] or row["scientific_config_sha256"] != cfg["scientific_config_sha256"]:
                raise ValueError(f"Scientific configuration hash mismatch: {path}")
            if recipe_signature(cfg) != signature:
                raise ValueError(f"Mixed training recipes/source manifests: {path}")
            if row["target_manifest_sha256"] != first["target_manifest_sha256"]:
                raise ValueError(f"Mixed target manifests: {path}")
            for field in ("labels", "paths", "classes"):
                if not np.array_equal(data[field], reference[field]):
                    raise ValueError(f"Prediction alignment mismatch for {field}: {path}")
            if not np.isfinite(data["probabilities"]).all() or not np.isfinite(data["logits"]).all():
                raise ValueError(f"Nonfinite predictions: {path}")
            actual = scores(data["logits"].argmax(1), data["labels"], len(data["classes"]))
            if abs(actual["macro_f1"] - row["macro_f1"]) > 1e-10 or abs(actual["acc"] - row["acc"]) > 1e-10:
                raise ValueError(f"Saved metric/prediction inconsistency: {path}")
        logits = np.stack([x[2]["logits"] for x in ordered])
        probability = np.stack([x[2]["probabilities"] for x in ordered])
        y = reference["labels"]
        item = {"protocol": key[0], "variant": key[1], "target_ghz": key[2],
                "seeds": seeds, "count": len(y), "epoch": first["epoch"],
                "target_manifest_sha256": first["target_manifest_sha256"],
                "recipe_sha256": hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest(),
                "runs": [x[3] for x in ordered], "class_order": reference["classes"].tolist(),
                "recipe": signature}
        for metric in ("acc", "macro_f1"):
            values = np.array([x[0][metric] for x in ordered])
            item[metric] = {"per_seed": values.tolist(), "mean": float(values.mean()),
                            "std_population": float(values.std(ddof=0)),
                            "std_sample": float(values.std(ddof=1)) if len(seeds) > 1 else None}
        item["ensemble_logit_average"] = scores(logits.mean(0).argmax(1), y, len(reference["classes"]))
        item["ensemble_posterior_average"] = scores(probability.mean(0).argmax(1), y, len(reference["classes"]))
        results.append(item)
    if not results:
        raise ValueError(f"No completed non-smoke seed groups found; excluded {len(ignored)} smoke reports")
    comparisons = []
    for proposed in results:
        if proposed["variant"] != "proposed":
            continue
        control = next((r for r in results if r["protocol"] == proposed["protocol"] and r["target_ghz"] == proposed["target_ghz"] and r["variant"] == "no_das_acr"), None)
        if control is None:
            continue
        recipes = []
        for row in (proposed, control):
            recipe = json.loads(json.dumps(row["recipe"]))
            for field in ("USE_DAS", "V13_GRL_WEIGHT", "V13_FREQ_WEIGHT"):
                recipe["effective_config"].pop(field, None)
            recipes.append(recipe)
        if recipes[0] != recipes[1] or proposed["target_manifest_sha256"] != control["target_manifest_sha256"]:
            raise ValueError(f"Proposed/control differ beyond intended DAS/ACR toggles for target {proposed['target_ghz']}")
        delta = {"protocol": proposed["protocol"], "target_ghz": proposed["target_ghz"]}
        for metric in ("acc", "macro_f1"):
            values = 100 * (np.asarray(proposed[metric]["per_seed"]) - np.asarray(control[metric]["per_seed"]))
            delta[metric + "_gain_percentage_points"] = {"per_seed": values.tolist(), "mean": float(values.mean())}
        comparisons.append(delta)
    return {"results": results, "paired_proposed_minus_control": comparisons, "ignored_smoke_reports": ignored,
            "note": "Single-model mean/std and ensemble scores are separate. No selection by target. Population std matches original paper convention; sample std also supplied."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1234, 31415])
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("Seeds must be unique")
    report = aggregate(sorted(args.runs_root.rglob("final_target_metrics.json")), args.seeds)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Cross-frequency final-EMA results", "", "Single model mean ± population std across the requested seeds; ensemble is a separate deployment result.", "", "| Protocol | Variant | Target GHz | N | Accuracy | Macro-F1 | Logit ensemble F1 |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in report["results"]:
        a, f = row["acc"], row["macro_f1"]
        lines.append(f"| {row['protocol']} | {row['variant']} | {row['target_ghz']} | {row['count']} | {a['mean']:.4f} ± {a['std_population']:.4f} | {f['mean']:.4f} ± {f['std_population']:.4f} | {row['ensemble_logit_average']['macro_f1']:.4f} |")
    for row in report["paired_proposed_minus_control"]:
        lines.extend(["", f"{row['protocol']} target {row['target_ghz']} GHz proposed-minus-control: accuracy {row['acc_gain_percentage_points']['mean']:+.2f} pp; macro-F1 {row['macro_f1_gain_percentage_points']['mean']:+.2f} pp."])
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
