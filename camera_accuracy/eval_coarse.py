"""Coarse-mode evidence from the 10/2 real recording plus a labelled simulation.

    python -m camera_accuracy.eval_coarse --output camera_accuracy/diagnostics/coarse/eval-20261004.json

Part A (real): the nine pupil centres stored in
diagnostics/20261002T071853Z-0058e7f6/summary.json. That session has no independent
validation points, so each target is predicted from the other eight (leave-one-out),
which over-states error at the grid edges (extrapolation).

Part B (simulation, labelled as such): a pupil whose centre moves 56 px horizontally
and 10 px vertically across the 0.1-0.9 screen span (measured above, 320x240 px), with
Gaussian centre noise. 640x480 capture is modelled as doubling both spans in pixels
while the detector noise in pixels stays the same - an assumption, not a measurement.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .gaze import POINTS, VALIDATION_POINTS, fit_calibration, make_points, predict, validation_passed

HERE = Path(__file__).parent
SUMMARY = HERE / 'diagnostics/20261002T071853Z-0058e7f6/summary.json'
MODES = {
    'original: quadratic + distance 0.12': dict(mapping='quadratic', validation='distance', grid=3),
    'coarse: affine + 3x3 region': dict(mapping='affine', validation='region', grid=3),
    'coarse: affine + 3x2 region': dict(mapping='affine', validation='region', grid=(3, 2)),
}


def leave_one_out(centres, targets, mode):
    hits, errors = 0, []
    for index in range(9):
        keep = np.arange(9) != index
        # Fit without gates: we want the prediction even when the training fit would be rejected.
        from .gaze import _solve, features
        mu, sd, _, weights = _solve(centres[keep], targets[keep], mode['mapping'])
        delta = features([(centres[index] - mu) / sd], mode['mapping'])[0] @ weights - targets[index]
        errors.append(np.abs(delta).tolist())
        hits += validation_passed(delta, mode['validation'], mode['grid'])
    return dict(loo_hits=int(hits), loo_abs_error_mean=np.mean(errors, axis=0).round(3).tolist())


def real_part():
    centres = np.array(json.loads(SUMMARY.read_text())['original_centers'], dtype=float)
    raw = centres / [160, -120]   # any fixed scaling; the regression normalises inputs
    targets = np.array(POINTS)
    span = dict(horizontal_px=float(np.ptp([centres[[0, 3, 6], 0].mean(), centres[[2, 5, 8], 0].mean()])),
                vertical_px=float(np.ptp([centres[:3, 1].mean(), centres[6:, 1].mean()])))
    out = dict(source=str(SUMMARY.relative_to(HERE)), pupil_span_320x240=span, modes={})
    for name, mode in MODES.items():
        row = leave_one_out(raw, targets, mode)
        for robust in (False, True):
            try:
                model = fit_calibration(raw, targets, robust=robust, **mode)
                row['training_fit' + ('_robust' if robust else '')] = dict(
                    passed=True, excluded_point=model['excluded_point'], fit_error=round(model['fit_error'], 3))
            except ValueError as error:
                row['training_fit' + ('_robust' if robust else '')] = dict(passed=False, error=str(error))
        out['modes'][name] = row
    return out


def simulated_part(trials=2000, seed=7):
    rng = np.random.default_rng(seed)
    out = {}
    for capture, scale in (('320x240', 1), ('640x480', 2)):
        for noise in (1., 2.):
            for bottom in (.9, .8):
                points = np.array(make_points(bottom))

                def pupil(p):
                    # 56 px / 10 px over the 0.8 screen span; screen y down = pupil y down.
                    return np.array([(.5 - p[0]) * 70, (p[1] - .5) * 12.5]) * scale

                for name, mode in MODES.items():
                    passed = 0
                    for _ in range(trials):
                        raw = np.array([pupil(p) for p in points]) + rng.normal(0, noise, (9, 2))
                        try:
                            model = fit_calibration(raw, points, robust=mode['mapping'] == 'affine', **mode)
                        except ValueError:
                            continue
                        test = np.array([pupil(p) for p in VALIDATION_POINTS]) + rng.normal(0, noise, (4, 2))
                        passed += all(validation_passed(predict(model, r) - t, mode['validation'], mode['grid'])
                                      for r, t in zip(test, VALIDATION_POINTS))
                    out.setdefault(name, []).append(dict(capture=capture, noise_px=noise, bottom_row=bottom,
                                                         full_1_9_4_pass_rate=round(passed / trials, 3)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output')
    args = parser.parse_args()
    result = dict(real=real_part(), simulation=simulated_part(),
                  scope='Part A: real centres, leave-one-out only (no independent validation in that session). '
                        'Part B: simulation with stated assumptions; not a measurement.')
    print(json.dumps(result['real'], ensure_ascii=False, indent=1))
    for name, rows in result['simulation'].items():
        print(name)
        for row in rows:
            print('  ', row)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
