"""Run: .venv/bin/python -m unittest camera_accuracy.test_coarse -v"""
import unittest

from aiohttp.test_utils import TestClient, TestServer
import numpy as np

from .gaze import (POINTS, VALIDATION_POINTS, RegionTracker, fit_calibration, make_points, predict,
                   validation_passed)
from .server import RUNTIME_KEY, create_app
from .test_accuracy import raw_for, result


def weak_vertical(point):
    """Real 10/2 recording shape: plenty of horizontal signal, little vertical."""
    return [(point[0] - .5) * .5, (.5 - point[1]) * .08]


class CoarseMapping(unittest.TestCase):
    def test_affine_recovers_linear_mapping_and_offset_shifts_prediction(self):
        model = fit_calibration([raw_for(p) for p in POINTS], POINTS, mapping='affine')
        self.assertEqual(model['mapping'], 'affine')
        self.assertEqual(len(model['weights']), 3)
        for point in VALIDATION_POINTS:
            np.testing.assert_allclose(predict(model, raw_for(point)), point, atol=1e-9)
        shifted = dict(model, offset=[.05, -.02])
        np.testing.assert_allclose(predict(shifted, raw_for((.5, .5))), [.55, .48], atol=1e-9)

    def test_region_tolerance_is_half_a_cell_per_axis(self):
        self.assertTrue(validation_passed([.16, .16], 'region', 3))
        self.assertFalse(validation_passed([.16, .18], 'region', 3))
        self.assertTrue(validation_passed([.16, .24], 'region', (3, 2)))   # 3 columns x 2 rows
        self.assertFalse(validation_passed([.13, 0.], 'distance'))          # 0.13 > 0.12

    def test_robust_fit_drops_one_inconsistent_point_only(self):
        raw = [raw_for(p) for p in POINTS]
        raw[7] = [raw[7][0], raw[4][1]]          # bottom-middle reads like the middle row (lid)
        with self.assertRaises(ValueError):
            fit_calibration(raw, POINTS, mapping='affine', validation='region')
        model = fit_calibration(raw, POINTS, mapping='affine', validation='region', robust=True)
        self.assertEqual(model['excluded_point'], 7)
        raw[2] = [raw[0][0], raw[2][1]]          # a second bad point is not silently dropped
        with self.assertRaises(ValueError):
            fit_calibration(raw, POINTS, mapping='affine', validation='region', robust=True)

    def test_noisy_weak_vertical_signal_affine_beats_quadratic_out_of_sample(self):
        rng = np.random.default_rng(3)
        wins = 0
        for _ in range(200):
            raw = np.array([weak_vertical(p) for p in POINTS]) + rng.normal(0, .004, (9, 2))
            test = np.array([weak_vertical(p) for p in VALIDATION_POINTS]) + rng.normal(0, .004, (4, 2))
            errors = {}
            for mapping in ('quadratic', 'affine'):
                try:
                    model = fit_calibration(raw, POINTS, mapping=mapping, validation='region', grid=1)
                except ValueError:
                    errors[mapping] = np.inf
                    continue
                errors[mapping] = np.mean([np.linalg.norm(predict(model, r) - t) for r, t in zip(test, VALIDATION_POINTS)])
            wins += errors['affine'] <= errors['quadratic']
        self.assertGreater(wins, 110)   # affine is better in most trials (measured ~64%)

    def test_region_hysteresis_and_bottom_row(self):
        tracker = RegionTracker(3, margin=.04)
        self.assertEqual(tracker.update((.30, .5)), [0, 1])
        self.assertEqual(tracker.update((.35, .5)), [0, 1])   # just over the border: stay
        self.assertEqual(tracker.update((.40, .5)), [1, 1])
        self.assertIsNone(tracker.update(None))
        self.assertEqual(make_points(.8)[-1], (.9, .8))


class CoarseServer(unittest.IsolatedAsyncioTestCase):
    async def test_coarse_calibration_region_output_and_drift_correction(self):
        config = dict(mapping='affine', validation='region', grid=[3, 3], bottom_row=.8, robust_fit=True)
        client = TestClient(TestServer(create_app(config=config, camera_enabled=False)))
        await client.start_server()
        runtime = client.app[RUNTIME_KEY]
        origin = str(client.make_url('')).rstrip('/')

        async def post(data):
            return await client.post('/api/calibration', json=data, headers={'Origin': origin})

        async def capture():
            runtime.capture_info = dict(valid_frames=24, total_frames=24, raw_p90=0., samples=[runtime.raw] * 24)
            return runtime.raw.copy()

        runtime.capture = capture
        try:
            runtime.accept(result())
            plan = await (await client.get('/api/plan')).json()
            self.assertEqual(plan['points'][-1], [.9, .8])
            self.assertEqual(plan['grid'], [3, 3])
            response = await post({'action': 'drift', 'target': [.5, .5]})
            self.assertEqual(response.status, 400)                      # needs a finished calibration
            session = (await (await post({'action': 'begin', 'viewport': {'width': 1600, 'height': 900}})).json())['session_id']
            for i, point in enumerate(plan['points']):
                runtime.accept(result(raw_for(point)))
                self.assertEqual((await post({'action': 'sample', 'index': i, 'session_id': session})).status, 200)
            self.assertEqual(runtime.candidate['mapping'], 'affine')
            for i, point in enumerate(VALIDATION_POINTS):
                # 0.15 off horizontally: fails the 0.12 distance rule, passes a 3x3 region rule.
                runtime.accept(result(raw_for((point[0] + .15, point[1]))))
                response = await post({'action': 'validate', 'validation_index': i, 'session_id': session})
                self.assertEqual(response.status, 200, await response.text())
            summary = await response.json()
            self.assertTrue(summary['calibrated'])
            self.assertEqual(summary['validation_passes'], [True] * 4)
            # Wearer drifted: looking at the centre now reads as (0.6, 0.45).
            runtime.accept(result(raw_for((.6, .45))))
            response = await post({'action': 'drift', 'target': [.5, .5]})
            self.assertEqual(response.status, 200, await response.text())
            np.testing.assert_allclose((await response.json())['offset'], [-.1, .05], atol=1e-9)
            for _ in range(30):
                runtime.accept(result(raw_for((.6, .45))))
            packet = runtime.packet()
            np.testing.assert_allclose([packet['x'], packet['y']], [.5, .5], atol=.02)
            self.assertEqual(packet['region'], [1, 1])
            runtime.accept(result(raw_for((.95, .5))))
            response = await post({'action': 'drift', 'target': [.5, .5]})
            self.assertEqual(response.status, 400)                      # too large: recalibrate instead
            np.testing.assert_allclose(runtime.model['offset'], [-.1, .05], atol=1e-9)  # previous offset kept
        finally:
            await client.close()


if __name__ == '__main__':
    unittest.main()
