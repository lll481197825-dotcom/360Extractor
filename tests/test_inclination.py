import unittest
import numpy as np
from src.core.geometry import GeometryProcessor

class TestGeometryInclination(unittest.TestCase):

    def test_ring_inclination(self):
        """Test n < 6 (Ring) with pitch offset"""
        # Case: High/Perch (-20)
        views = GeometryProcessor.generate_views(4, pitch_offset=-20)
        self.assertEqual(len(views), 4)
        for name, yaw, pitch, roll in views:
            self.assertEqual(pitch, -20, f"View {name} should have pitch -20")

        # Case: Low/Ground (+20)
        views = GeometryProcessor.generate_views(4, pitch_offset=20)
        self.assertEqual(len(views), 4)
        for name, yaw, pitch, roll in views:
            self.assertEqual(pitch, 20, f"View {name} should have pitch 20")

    def test_large_ring_inclination(self):
        """Test n >= 6 (Previously Cube/Fibonacci) now uses Ring with pitch offset"""
        
        # Test n=6 (Previously Cube)
        views_6 = GeometryProcessor.generate_views(6, pitch_offset=-20)
        self.assertEqual(len(views_6), 6)
        for i, (name, yaw, pitch, roll) in enumerate(views_6):
            self.assertEqual(pitch, -20, f"View {name} (n=6) should have pitch -20")
            expected_yaw = (i * 360.0) / 6
            self.assertAlmostEqual(yaw, expected_yaw, places=5, msg=f"View {name} (n=6) yaw mismatch")

        # Test n=10 (Previously Fibonacci)
        views_10 = GeometryProcessor.generate_views(10, pitch_offset=20)
        self.assertEqual(len(views_10), 10)
        for i, (name, yaw, pitch, roll) in enumerate(views_10):
            self.assertEqual(pitch, 20, f"View {name} (n=10) should have pitch 20")
            expected_yaw = (i * 360.0) / 10
            self.assertAlmostEqual(yaw, expected_yaw, places=5, msg=f"View {name} (n=10) yaw mismatch")

if __name__ == '__main__':
    unittest.main()