import unittest
import numpy as np
import geospatial_airquality as aq

class TestBasicMetrics(unittest.TestCase):

    def test_summary_stats(self):
        stats = aq.summary_stats([1, 2, 3, np.nan])
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 3.0)
        self.assertAlmostEqual(stats["mean"], 2.0, places=6)

    def test_exceedance_ratio(self):
        ratio = aq.exceedance_ratio([0, 10, 20, 30], threshold=15)
        self.assertAlmostEqual(ratio, 0.5, places=6)

if __name__ == "__main__":
    unittest.main()
