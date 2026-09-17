import unittest

from src.algorithms.equal_weight import create_equal_weights


class EqualWeightTests(unittest.TestCase):
    def test_creates_one_equal_weight_per_asset(self):
        result = create_equal_weights(5)

        self.assertEqual(
            result,
            [0.20, 0.20, 0.20, 0.20, 0.20],
        )
        self.assertAlmostEqual(sum(result), 1.0)

    def test_rejects_non_positive_asset_count(self):
        with self.assertRaises(ValueError):
            create_equal_weights(0)


if __name__ == "__main__":
    unittest.main()
