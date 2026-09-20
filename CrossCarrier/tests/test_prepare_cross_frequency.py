"""Protocol tests: prevent target contamination and recording-level leakage."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_cross_frequency import (CLASSES, FREQUENCIES, build_grouped,
    deduplicate, make_group_assignments, recording_key, validate_direction)


def fixture_rows():
    rows = []
    for class_idx, name in enumerate(CLASSES):
        for capture in range(10):
            timestamp = str(1574000000 + 100 * class_idx + capture)
            for frequency in FREQUENCIES:
                filename = f"06060010_{timestamp}.png"
                rows.append({"frequency": frequency, "class": name,
                    "class_idx": str(class_idx), "subject": "ua_0010",
                    "path": f"dataset/{frequency}/{name}/{filename}",
                    "source_file": filename,
                    "recording_id": f"ua_0010|{name}|{timestamp}",
                    "sha256": f"{frequency}|{name}|{timestamp}"})
    return rows


class CrossFrequencyProtocolTests(unittest.TestCase):
    def test_copy_suffix_keeps_recording_identity(self):
        row = {"subject": "ua_0029", "class": "Kneel",
               "source_file": "06100029_1574628503 (copy).png"}
        self.assertEqual(recording_key(row), ("ua_0029", "Kneel", "1574628503"))
        row["source_file"] = "04100029_1574628503_Raw_1.png"
        self.assertEqual(recording_key(row), ("ua_0029", "Kneel", "1574628503"))

    def test_three_directions_grouped_and_target_excluded(self):
        rows = fixture_rows()
        assignments = make_group_assignments(rows, 42, 0.20)
        self.assertEqual(sum(v == "val" for v in assignments.values()), 14)
        for target in FREQUENCIES:
            splits = build_grouped(rows, target, assignments)
            report = validate_direction(splits, target, strict=True)
            self.assertEqual(len(splits["train"]), 112)
            self.assertEqual(len(splits["val"]), 28)
            self.assertEqual(len(splits["test"]), 70)
            self.assertTrue(report["recording_disjoint_source_validation"])

    def test_assignment_deterministic_across_order_and_frequency_direction(self):
        rows = fixture_rows()
        self.assertEqual(make_group_assignments(rows, 42, .20),
                         make_group_assignments(rows[::-1], 42, .20))
        self.assertNotEqual(make_group_assignments(rows, 42, .20),
                            make_group_assignments(rows, 7, .20))

    def test_deduplication_prefers_original_file(self):
        rows = fixture_rows()
        duplicate = dict(rows[0])
        duplicate["path"] = duplicate["path"].replace(".png", " (copy).png")
        duplicate["source_file"] = duplicate["source_file"].replace(".png", " (copy).png")
        unique, removed = deduplicate([duplicate] + rows)
        self.assertEqual(len(unique), len(rows))
        self.assertEqual(removed[0]["retained_path"], rows[0]["path"])

    def test_hash_links_join_different_timestamps(self):
        rows = fixture_rows()
        a, b = rows[0], rows[3]
        b["sha256"] = a["sha256"]
        assignments = make_group_assignments(rows, 42, .20)
        self.assertEqual(assignments[a["recording_id"]], assignments[b["recording_id"]])

    def test_tampered_target_training_fails(self):
        rows = fixture_rows()
        splits = build_grouped(rows, "77GHz", make_group_assignments(rows, 42, .20))
        splits["train"].append(splits["test"][0])
        with self.assertRaisesRegex(ValueError, "Target frequency leaked"):
            validate_direction(splits, "77GHz", strict=True)

    def test_shared_source_recording_fails_even_with_different_images(self):
        rows = fixture_rows()
        splits = build_grouped(rows, "77GHz", make_group_assignments(rows, 42, .20))
        original = splits["train"].pop(0)
        splits["val"].append(original)
        with self.assertRaisesRegex(ValueError, "recording or identical-image leakage"):
            validate_direction(splits, "77GHz", strict=True)

    def test_conflicting_duplicate_labels_fail(self):
        rows = fixture_rows()
        rows[30]["sha256"] = rows[0]["sha256"]
        with self.assertRaisesRegex(ValueError, "conflicting subject/class"):
            make_group_assignments(rows, 42, .20)


if __name__ == "__main__":
    unittest.main()
