"""Synthetic reports exercise scientific aggregation without a GPU or model."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aggregate_cross_frequency import aggregate

SEEDS = [42, 1234, 31415]
CLASSES = ["Away", "Bend", "Kneel", "Pick", "SStep", "Sit", "Towards"]
PROPOSED_PREDICTIONS = [list(range(7)), [0, 1, 2, 3, 4, 6, 5], [0, 1, 2, 3, 5, 6, 4]]
CONTROL_PREDICTIONS = [[1, 0, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 6, 5], [1, 2, 0, 3, 4, 5, 6]]


class AggregateCrossFrequencyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_run(self, index, variant="proposed", tag=None):
        seed = SEEDS[index]
        run = self.root / (tag or f"{variant}_seed{seed}")
        reports = run / "reports"
        reports.mkdir(parents=True)
        predictions = np.asarray((PROPOSED_PREDICTIONS if variant == "proposed"
                                  else CONTROL_PREDICTIONS)[index])
        # These are permutations: each wrong class has zero precision/recall,
        # so independently known macro-F1 equals accuracy for every single run.
        correct = int(sum(predictions[i] == i for i in range(7)))
        scientific_hash = f"scientific_{variant}"
        metrics = {
            "protocol": "grouped", "variant": variant, "target_ghz": 77,
            "source_ghz": [10, 24], "seed": seed, "epoch": 100,
            "selection": "fixed_final_epoch_ema", "smoke_only": False,
            "target_manifest_sha256": "target_manifest_same_for_all",
            "scientific_config_sha256": scientific_hash,
            "count": 7, "acc": correct / 7, "macro_f1": correct / 7,
        }
        cfg = {
            "epochs": 100, "batch_size": 16, "num_workers": 0,
            "precision_resolved": "bf16", "smoke_only": False,
            "scientific_config_sha256": scientific_hash,
            "sources": {"train": {"rows": 634, "sha256": "source_train"},
                        "val": {"rows": 153, "sha256": "source_val"}},
            "effective_config": {
                "SEED": seed, "EXPERIMENT_NAME": f"{variant}_{seed}",
                "TRAIN_FREQS": ["10GHz", "24GHz"], "TEST_FREQS": ["77GHz"],
                "FREQ_TO_IDX": {"10GHz": 0, "24GHz": 1},
                "HEAD_HIDDEN": 512, "LR": 0.0003,
                "USE_DAS": variant == "proposed",
                "V13_GRL_WEIGHT": .3 if variant == "proposed" else 0,
                "V13_FREQ_WEIGHT": .05 if variant == "proposed" else 0,
            },
        }
        (run / "run_config.json").write_text(json.dumps(cfg), encoding="utf-8")
        path = reports / "final_target_metrics.json"
        path.write_text(json.dumps(metrics), encoding="utf-8")
        logits = np.zeros((7, 7), dtype=np.float64)
        confidence = [100, 3, 4][index] if variant == "proposed" else 4
        logits[np.arange(7), predictions] = confidence
        exponent = np.exp(logits - logits.max(1, keepdims=True))
        probability = exponent / exponent.sum(1, keepdims=True)
        np.savez_compressed(reports / "final_target_predictions.npz",
                            logits=logits, probabilities=probability,
                            predictions=predictions, labels=np.arange(7),
                            paths=np.asarray([f"dataset/77GHz/{name}/record.png" for name in CLASSES]),
                            classes=np.asarray(CLASSES))
        return path

    def make_group(self, variant="proposed"):
        return [self.make_run(i, variant) for i in range(3)]

    @staticmethod
    def change_json(path, mutate):
        row = json.loads(path.read_text(encoding="utf-8"))
        mutate(row)
        path.write_text(json.dumps(row), encoding="utf-8")

    @staticmethod
    def change_predictions(path, mutate):
        npz_path = path.with_name("final_target_predictions.npz")
        with np.load(npz_path, allow_pickle=False) as source:
            arrays = {key: source[key].copy() for key in source.files}
        mutate(arrays)
        np.savez_compressed(npz_path, **arrays)

    def test_mean_both_standard_deviations_and_distinct_ensembles(self):
        paths = self.make_group()
        result = aggregate(paths[::-1], SEEDS)["results"][0]
        for field in ("acc", "macro_f1"):
            self.assertEqual(result[field]["per_seed"], [1.0, 5 / 7, 4 / 7])
            self.assertAlmostEqual(result[field]["mean"], 16 / 21)
            self.assertAlmostEqual(result[field]["std_population"], (2 / 63) ** .5)
            self.assertAlmostEqual(result[field]["std_sample"], (1 / 21) ** .5)
        self.assertEqual(result["ensemble_logit_average"]["acc"], 1.0)
        self.assertEqual(result["ensemble_logit_average"]["macro_f1"], 1.0)
        self.assertAlmostEqual(result["ensemble_posterior_average"]["acc"], 6 / 7)
        self.assertAlmostEqual(result["ensemble_posterior_average"]["macro_f1"], 17 / 21)
        self.assertEqual(result["seeds"], SEEDS)

    def test_paired_control_deltas_follow_seed_order(self):
        paths = self.make_group() + self.make_group("no_das_acr")
        report = aggregate(paths[::-1], SEEDS)
        self.assertEqual(len(report["results"]), 2)
        delta = report["paired_proposed_minus_control"][0]
        for field in ("acc", "macro_f1"):
            gains = delta[field + "_gain_percentage_points"]
            np.testing.assert_allclose(gains["per_seed"], [200 / 7, 0, 0])
            self.assertAlmostEqual(gains["mean"], 200 / 21)

    def test_duplicate_seed_rejected(self):
        paths = self.make_group()
        duplicate = self.make_run(0, tag="duplicate_seed42")
        with self.assertRaisesRegex(ValueError, "Duplicate runs"):
            aggregate(paths + [duplicate], SEEDS)

    def test_missing_seed_rejected(self):
        paths = self.make_group()
        with self.assertRaisesRegex(ValueError, "Incomplete seeds"):
            aggregate(paths[:2], SEEDS)

    def test_unexpected_seed_rejected(self):
        paths = self.make_group()
        self.change_json(paths[0], lambda row: row.update(seed=99))
        with self.assertRaisesRegex(ValueError, "Unexpected seed 99"):
            aggregate(paths, SEEDS)

    def test_mixed_training_configuration_rejected(self):
        paths = self.make_group()
        self.change_json(paths[1].parents[1] / "run_config.json", lambda cfg: cfg.update(batch_size=32))
        with self.assertRaisesRegex(ValueError, "Mixed training recipes"):
            aggregate(paths, SEEDS)

    def test_source_manifest_change_rejected(self):
        paths = self.make_group()
        self.change_json(paths[1].parents[1] / "run_config.json",
                         lambda cfg: cfg["sources"]["train"].update(sha256="changed"))
        with self.assertRaisesRegex(ValueError, "Mixed training recipes"):
            aggregate(paths, SEEDS)

    def test_scientific_hash_mismatch_rejected(self):
        paths = self.make_group()
        self.change_json(paths[1], lambda row: row.update(scientific_config_sha256="different"))
        with self.assertRaisesRegex(ValueError, "Scientific configuration hash mismatch"):
            aggregate(paths, SEEDS)

    def test_target_manifest_change_rejected(self):
        paths = self.make_group()
        self.change_json(paths[1], lambda row: row.update(target_manifest_sha256="different"))
        with self.assertRaisesRegex(ValueError, "Mixed target manifests"):
            aggregate(paths, SEEDS)

    def test_target_path_misalignment_rejected(self):
        paths = self.make_group()
        self.change_predictions(paths[1], lambda arrays: arrays.update(paths=arrays["paths"][::-1]))
        with self.assertRaisesRegex(ValueError, "Prediction alignment mismatch for paths"):
            aggregate(paths, SEEDS)

    def test_metric_prediction_disagreement_rejected(self):
        paths = self.make_group()
        self.change_json(paths[1], lambda row: row.update(acc=1.0))
        with self.assertRaisesRegex(ValueError, "Saved metric/prediction inconsistency"):
            aggregate(paths, SEEDS)

    def test_control_with_unintended_difference_rejected(self):
        proposed = self.make_group()
        control = self.make_group("no_das_acr")
        for path in control:
            self.change_json(path.parents[1] / "run_config.json",
                             lambda cfg: cfg["effective_config"].update(LR=.001))
        with self.assertRaisesRegex(ValueError, "differ beyond intended DAS/ACR toggles"):
            aggregate(proposed + control, SEEDS)

    def test_smoke_report_skipped_before_config_or_predictions_read(self):
        paths = self.make_group()
        smoke = self.root / "smoke.json"
        smoke.write_text(json.dumps({"smoke_only": True}), encoding="utf-8")
        report = aggregate([smoke] + paths, SEEDS)
        self.assertEqual(len(report["results"]), 1)
        self.assertEqual(report["ignored_smoke_reports"], [str(smoke)])

    def test_only_smoke_or_no_reports_produce_clear_error(self):
        smoke = self.root / "smoke.json"
        smoke.write_text(json.dumps({"smoke_only": True}), encoding="utf-8")
        for reports in ([], [smoke]):
            with self.subTest(reports=reports):
                with self.assertRaisesRegex(ValueError, "No completed non-smoke seed groups"):
                    aggregate(reports, SEEDS)


if __name__ == "__main__":
    unittest.main()
