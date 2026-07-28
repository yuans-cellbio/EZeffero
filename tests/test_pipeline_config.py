import unittest

from src.pipeline import resolve_ac_detection_thresholds


class ResolveACDetectionThresholdsTests(unittest.TestCase):
    def test_single_is_backward_compatible_default(self):
        resolved = resolve_ac_detection_thresholds({'threshold': 78})

        self.assertEqual(resolved['strategy'], 'single')
        self.assertEqual(resolved['threshold'], 78)
        self.assertEqual(resolved['low_threshold'], 78)
        self.assertEqual(resolved['high_threshold'], 78)

    def test_hysteresis_resolves_low_and_high(self):
        resolved = resolve_ac_detection_thresholds(
            {
                'threshold_strategy': 'hysteresis',
                'hysteresis': {
                    'low_threshold': 150,
                    'high_threshold': 250,
                },
            }
        )

        self.assertEqual(resolved['strategy'], 'hysteresis')
        self.assertEqual(resolved['threshold'], 250)
        self.assertEqual(resolved['low_threshold'], 150)
        self.assertEqual(resolved['high_threshold'], 250)

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            resolve_ac_detection_thresholds(
                {'threshold_strategy': 'adaptive', 'threshold': 100}
            )

    def test_hysteresis_requires_both_thresholds(self):
        with self.assertRaises(ValueError):
            resolve_ac_detection_thresholds(
                {
                    'threshold_strategy': 'hysteresis',
                    'hysteresis': {'low_threshold': 100},
                }
            )


if __name__ == '__main__':
    unittest.main()
