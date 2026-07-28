import unittest

import numpy as np

from src import detect_AC


class DetectACObjectsTests(unittest.TestCase):
    def test_hysteresis_keeps_low_component_with_high_seed(self):
        image = np.zeros((7, 11), dtype=np.float32)
        image[1:4, 1:4] = 120
        image[2, 2] = 300
        image[1:4, 7:10] = 120

        labels, props = detect_AC.detect_ac_objects_hysteresis(
            image,
            low_threshold=100,
            high_threshold=250,
            min_object_area_px=1,
        )

        self.assertEqual(int(labels.max()), 1)
        self.assertEqual(len(props), 1)
        self.assertEqual(float(props.loc[0, 'area']), 9)
        self.assertTrue(np.all(labels[1:4, 1:4] > 0))
        self.assertTrue(np.all(labels[1:4, 7:10] == 0))

    def test_equal_hysteresis_thresholds_match_single_threshold(self):
        image = np.array(
            [
                [0, 0, 0, 0],
                [0, 150, 150, 0],
                [0, 150, 300, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.float32,
        )

        single_labels, single_props = detect_AC.detect_ac_objects(
            image,
            threshold=100,
            min_object_area_px=1,
        )
        hysteresis_labels, hysteresis_props = (
            detect_AC.detect_ac_objects_hysteresis(
                image,
                low_threshold=100,
                high_threshold=100,
                min_object_area_px=1,
            )
        )

        np.testing.assert_array_equal(single_labels, hysteresis_labels)
        self.assertEqual(
            single_props.to_dict(orient='records'),
            hysteresis_props.to_dict(orient='records'),
        )

    def test_minimum_area_applies_after_hysteresis_growth(self):
        image = np.zeros((5, 8), dtype=np.float32)
        image[1:3, 1:3] = 120
        image[1, 1] = 300
        image[1, 6] = 300

        labels, props = detect_AC.detect_ac_objects_hysteresis(
            image,
            low_threshold=100,
            high_threshold=250,
            min_object_area_px=3,
        )

        self.assertEqual(int(labels.max()), 1)
        self.assertEqual(len(props), 1)
        self.assertEqual(float(props.loc[0, 'area']), 4)
        self.assertEqual(int(labels[1, 6]), 0)

    def test_invalid_hysteresis_threshold_order_raises(self):
        with self.assertRaises(ValueError):
            detect_AC.detect_ac_objects_hysteresis(
                np.zeros((3, 3), dtype=np.float32),
                low_threshold=250,
                high_threshold=100,
                min_object_area_px=1,
            )


if __name__ == '__main__':
    unittest.main()
