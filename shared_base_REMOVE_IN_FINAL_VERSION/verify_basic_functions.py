"""Numerical checks for the three implemented HW2 helpers (no pytest needed).

Run: python verify_basic_functions.py --data-dir /path/to/original/ex2
Omit --data-dir to run only the self-contained synthetic checks.
"""
import argparse
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from scipy import signal, ndimage

import lucas_kanade as lk


class BasicFunctionChecks(unittest.TestCase):
    def test_pyramid_level_count_and_odd_dimensions(self):
        p = lk.build_pyramid(np.zeros((9, 13)), 3)
        self.assertEqual([a.shape for a in p], [(9, 13), (5, 7), (3, 4), (2, 2)])

    def test_pyramid_constant_image_and_symmetric_edges(self):
        for level in lk.build_pyramid(np.full((11, 15), 173, dtype=np.uint8), 4):
            np.testing.assert_allclose(level, 173)

    def test_pyramid_impulse_uses_supplied_filter(self):
        image = np.zeros((9, 13))
        image[4, 6] = 256
        p = lk.build_pyramid(image, 1)
        np.testing.assert_array_equal(p[1][1:4, 2:5], [[1, 6, 1], [6, 36, 6], [1, 6, 1]])

    def test_pyramid_zero_levels_is_an_independent_copy(self):
        image = np.arange(24).reshape(4, 6)
        p = lk.build_pyramid(image, 0)
        self.assertEqual(len(p), 1)
        p[0][0, 0] = -100
        self.assertEqual(image[0, 0], 0)

    def test_pyramid_does_not_call_opencv(self):
        with patch.object(cv2, 'pyrDown', side_effect=AssertionError('Forbidden')):
            lk.build_pyramid(np.ones((9, 11)), 2)

    def test_step_recovers_manufactured_linearized_motion(self):
        # Manufacture the brightness equation with a known exact solution.
        image = np.random.default_rng(9).normal(100, 10, size=(19, 27))
        ix = signal.convolve2d(image, lk.X_DERIVATIVE_FILTER, mode='same')
        iy = signal.convolve2d(image, lk.Y_DERIVATIVE_FILTER, mode='same')
        expected_u, expected_v = 0.12, -0.08
        first = image + expected_u * ix + expected_v * iy
        u, v = lk.lucas_kanade_step(first, image, 5)
        np.testing.assert_allclose(u[2:-2, 2:-2], expected_u, atol=1e-12)
        np.testing.assert_allclose(v[2:-2, 2:-2], expected_v, atol=1e-12)
        self.assertTrue(np.all(u[:2] == 0) and np.all(u[-2:] == 0))
        self.assertTrue(np.all(v[:, :2] == 0) and np.all(v[:, -2:] == 0))

    def test_step_identical_images_give_zero(self):
        image = np.random.default_rng(10).normal(size=(15, 21))
        for flow in lk.lucas_kanade_step(image, image, 5):
            np.testing.assert_array_equal(flow, 0)

    def test_step_flat_and_aperture_windows_give_zero(self):
        # At the center of a ramp only one gradient direction is available.
        for second in [np.full((15, 21), 100.0), np.tile(np.arange(21.0), (15, 1))]:
            u, v = lk.lucas_kanade_step(second + 1, second, 5)
            self.assertEqual(u[7, 10], 0)
            self.assertEqual(v[7, 10], 0)

    def test_step_keeps_processing_after_a_singular_window(self):
        second = np.zeros((15, 25))
        second[:, 12:] = np.random.default_rng(3).normal(80, 12, (15, 13))
        ix = signal.convolve2d(second, lk.X_DERIVATIVE_FILTER, mode='same')
        first = second + 0.2 * ix
        u, _ = lk.lucas_kanade_step(first, second, 5)
        self.assertEqual(u[7, 2], 0)
        self.assertAlmostEqual(u[7, 19], 0.2, places=12)

    def test_step_uint8_subtraction_matches_float(self):
        rng = np.random.default_rng(5)
        first = rng.integers(0, 256, size=(15, 21), dtype=np.uint8)
        second = rng.integers(0, 256, size=(15, 21), dtype=np.uint8)
        actual = lk.lucas_kanade_step(first, second, 5)
        expected = lk.lucas_kanade_step(first.astype(float), second.astype(float), 5)
        for a, e in zip(actual, expected):
            np.testing.assert_allclose(a, e)

    def test_warp_identity(self):
        image = np.arange(15 * 21, dtype=float).reshape(15, 21)
        np.testing.assert_allclose(lk.warp_image(image, np.zeros_like(image), np.zeros_like(image)), image)

    def test_warp_sampling_direction_and_hole_fill(self):
        image = np.arange(9 * 15, dtype=float).reshape(9, 15)
        u, v = np.ones_like(image), np.full_like(image, -2)
        expected = image.copy()
        expected[2:, :-1] = image[:-2, 1:]
        np.testing.assert_allclose(lk.warp_image(image, u, v), expected)

    def test_warp_fractional_interpolation(self):
        x, y = np.meshgrid(np.arange(15), np.arange(9))
        image = 2.0 * x + 3.0 * y
        actual = lk.warp_image(image, np.full(image.shape, 0.25), np.full(image.shape, 0.5))
        np.testing.assert_allclose(actual[:-1, :-1], image[:-1, :-1] + 2.0, atol=1e-12)
        np.testing.assert_array_equal(actual[-1], image[-1])
        np.testing.assert_array_equal(actual[:, -1], image[:, -1])

    def test_warp_resizes_and_scales_axes_independently(self):
        x, y = np.meshgrid(np.arange(18), np.arange(12))
        image = 2.0 * x + 3.0 * y
        # Both maps start on a (4, 9) grid: horizontal x2, vertical x3.
        actual = lk.warp_image(image, np.ones((4, 9)), np.ones((4, 9)))
        np.testing.assert_allclose(actual[:-3, :-2], image[3:, 2:], atol=1e-12)
        # Also exercise different input resolutions for the two components.
        actual = lk.warp_image(image, np.ones((6, 9)), np.ones((4, 6)))
        np.testing.assert_allclose(actual[:-3, :-2], image[3:, 2:], atol=1e-12)

    def test_helpers_do_not_modify_inputs(self):
        image = np.random.default_rng(7).integers(0, 256, (15, 21), dtype=np.uint8)
        second = image[::-1].copy()
        u = np.ones((5, 7))
        v = -np.ones((5, 7))
        copies = [a.copy() for a in [image, second, u, v]]
        lk.build_pyramid(image, 2)
        lk.lucas_kanade_step(image, second, 5)
        lk.warp_image(image, u, v)
        for actual, expected in zip([image, second, u, v], copies):
            np.testing.assert_array_equal(actual, expected)

    def test_step_and_warp_reduce_small_translation_error(self):
        first = ndimage.gaussian_filter(np.random.default_rng(11).normal(size=(49, 65)), 2)
        for dx, dy in [(0.25, 0), (0, -0.25), (0.2, -0.15)]:
            with self.subTest(dx=dx, dy=dy):
                second = ndimage.shift(first, (dy, dx), order=3, mode='reflect')
                u, v = lk.lucas_kanade_step(first, second, 7)
                warped = lk.warp_image(second, u, v)
                inner = np.s_[8:-8, 8:-8]
                before = np.mean((first[inner] - second[inner]) ** 2)
                after = np.mean((first[inner] - warped[inner]) ** 2)
                self.assertLess(after, before)


def check_river(data_dir):
    first = cv2.imread(str(data_dir / 'river1.png'), cv2.IMREAD_COLOR)
    second = cv2.imread(str(data_dir / 'river2.png'), cv2.IMREAD_COLOR)
    if first is None or second is None:
        raise FileNotFoundError('Expected river1.png and river2.png in --data-dir')
    # Match the supplied main_river.py color conversion exactly.
    first = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY)
    second = cv2.cvtColor(second, cv2.COLOR_RGB2GRAY)
    u, v = lk.lucas_kanade_step(first, second, 5)
    warped = lk.warp_image(second, u, v)
    crop = np.s_[2:-2, 2:-2]
    before = float(np.mean((first[crop].astype(float) - second[crop].astype(float)) ** 2))
    after = float(np.mean((first[crop].astype(float) - warped[crop]) ** 2))
    if not np.isfinite(warped).all() or not after < before:
        raise AssertionError(f'River one-step alignment failed: {before=} {after=}')
    print('River validation (float MSE, window_size=5):')
    print(json.dumps({'input_shape': list(first.shape), 'before': before, 'after': after,
                      'reduction_percent': 100 * (1 - after / before)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(BasicFunctionChecks))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.data_dir is not None:
        check_river(args.data_dir)
