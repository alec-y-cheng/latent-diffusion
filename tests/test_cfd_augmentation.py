import unittest

import numpy as np

from ldm.data.cfd_transforms import transform_uk_roof_sample


class UKRoofAugmentationTest(unittest.TestCase):
    def setUp(self):
        base = np.arange(12, dtype=np.float32).reshape(3, 4)
        self.x = np.stack([base + 100 * channel for channel in range(8)])
        self.y = np.stack([base + 1000, base + 2000])

    def test_transpose_swaps_coordinates_and_direction_components(self):
        x, y = transform_uk_roof_sample(self.x.copy(), self.y.copy(), transpose=True)

        np.testing.assert_array_equal(x[0], self.x[1].T)
        np.testing.assert_array_equal(x[1], self.x[0].T)
        np.testing.assert_array_equal(x[6], self.x[7].T)
        np.testing.assert_array_equal(x[7], self.x[6].T)
        np.testing.assert_array_equal(x[4], self.x[4].T)
        np.testing.assert_array_equal(y, self.y.transpose(0, 2, 1))

    def test_horizontal_flip_negates_x_and_direction_sin_only(self):
        x, y = transform_uk_roof_sample(self.x.copy(), self.y.copy(), hflip=True)

        np.testing.assert_array_equal(x[0], -self.x[0, :, ::-1])
        np.testing.assert_array_equal(x[6], -self.x[6, :, ::-1])
        np.testing.assert_array_equal(x[1], self.x[1, :, ::-1])
        np.testing.assert_array_equal(x[4], self.x[4, :, ::-1])
        np.testing.assert_array_equal(y, self.y[:, :, ::-1])

    def test_vertical_flip_negates_y_and_direction_cos_only(self):
        x, y = transform_uk_roof_sample(self.x.copy(), self.y.copy(), vflip=True)

        np.testing.assert_array_equal(x[1], -self.x[1, ::-1, :])
        np.testing.assert_array_equal(x[7], -self.x[7, ::-1, :])
        np.testing.assert_array_equal(x[0], self.x[0, ::-1, :])
        np.testing.assert_array_equal(x[4], self.x[4, ::-1, :])
        np.testing.assert_array_equal(y, self.y[:, ::-1, :])

    def test_composed_transform_matches_sequential_operations(self):
        actual_x, actual_y = transform_uk_roof_sample(
            self.x.copy(), self.y.copy(), transpose=True, hflip=True, vflip=True
        )
        expected_x, expected_y = transform_uk_roof_sample(
            self.x.copy(), self.y.copy(), transpose=True
        )
        expected_x, expected_y = transform_uk_roof_sample(
            expected_x, expected_y, hflip=True
        )
        expected_x, expected_y = transform_uk_roof_sample(
            expected_x, expected_y, vflip=True
        )

        np.testing.assert_array_equal(actual_x, expected_x)
        np.testing.assert_array_equal(actual_y, expected_y)
        self.assertTrue(actual_x.flags.c_contiguous)
        self.assertTrue(actual_y.flags.c_contiguous)


if __name__ == "__main__":
    unittest.main()
