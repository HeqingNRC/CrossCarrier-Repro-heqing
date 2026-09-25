"""Source isolation and hardware policy tests; no neural dependencies needed."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cross_frequency as cf


class CrossFrequencyTests(unittest.TestCase):
    def args(self, target=77, protocol="grouped", *extra):
        return cf.parse_args(["train", "--target", str(target), "--protocol", protocol,
                              "--run-dir", str(ROOT / "unused_test_run"), *extra])

    def test_three_physical_source_pairs(self):
        self.assertEqual(self.args(10).sources, [24, 77])
        self.assertEqual(self.args(24).sources, [10, 77])
        self.assertEqual(self.args(77).sources, [10, 24])

    def test_native_precision_policy(self):
        self.assertEqual(cf.resolve_precision("auto", "cuda", False), "fp16")
        self.assertEqual(cf.resolve_precision("auto", "cuda", True), "bf16")
        self.assertEqual(cf.resolve_precision("auto", "cpu"), "fp32")
        self.assertEqual(cf.resolve_precision("fp32", "cuda"), "fp32")
        with self.assertRaises(ValueError):
            cf.resolve_precision("bf16", "cuda", False)
        with self.assertRaises(ValueError):
            cf.resolve_precision("fp16", "cpu")

    def test_legacy_new_direction_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args(10, "legacy")

    def test_short_run_requires_smoke_marker(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args(77, "grouped", "--epochs", "1")
        args = self.args(77, "grouped", "--epochs", "1", "--max-steps", "1")
        self.assertEqual(args.max_steps, 1)

    def test_preflight_never_reads_target(self):
        original_open = Path.open
        def guarded_open(path, *args, **kwargs):
            if path.name == "test.csv":
                raise AssertionError("Target manifest was opened during source preflight")
            return original_open(path, *args, **kwargs)
        with patch.object(Path, "open", guarded_open):
            for target, protocol, expected in ((10, "grouped", 672), (24, "grouped", 620),
                                               (77, "grouped", 634), (77, "legacy", 644)):
                with self.subTest(target=target, protocol=protocol):
                    counts = cf.source_preflight(self.args(target, protocol))
                    self.assertEqual(counts["train"]["rows"], expected)

    def test_grouped_cannot_mislabel_legacy_task_override(self):
        args = self.args(77, "grouped", "--task-root", str(ROOT / "tasks/cross_frequency/target77"))
        with self.assertRaisesRegex(ValueError, "recording overlap"):
            cf.source_preflight(args)

    def test_target_frequency_contamination_rejected_before_image_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            task = Path(temp)
            (task / "manifest").mkdir()
            original = ROOT / "tasks/cross_frequency_grouped/target77/manifest/train.csv"
            (task / "manifest/train.csv").write_text(
                original.read_text(encoding="utf-8").replace("10GHz", "77GHz"), encoding="utf-8")
            args = self.args(77, "grouped", "--task-root", str(task))
            with self.assertRaisesRegex(ValueError, "frequencies"):
                cf.source_preflight(args)

    def test_multiple_configurations_fail_before_importing_torch(self):
        with patch.dict(sys.modules, {"config": object()}):
            with self.assertRaisesRegex(RuntimeError, "fresh Python process"):
                cf.configure(self.args())


if __name__ == "__main__":
    unittest.main()
