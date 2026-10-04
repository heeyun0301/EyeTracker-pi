"""Screen calibration and time-based filtering; no camera or network dependencies."""
from collections import deque
import math
import numpy as np

POINTS = [(x, y) for y in (.1, .5, .9) for x in (.1, .5, .9)]
VALIDATION_POINTS = [(.3, .3), (.7, .3), (.3, .7), (.7, .7)]
MAPPINGS = ('quadratic', 'affine')
VALIDATIONS = ('distance', 'region')


def make_points(bottom=.9):
    """3x3 calibration grid; a higher bottom row keeps the pupil clear of the upper lid."""
    return [(x, y) for y in (.1, .5, float(bottom)) for x in (.1, .5, .9)]


def features(points, mapping='quadratic'):
    x, y = np.asarray(points).T
    if mapping == 'affine':
        return np.column_stack((np.ones_like(x), x, y))
    return np.column_stack((np.ones_like(x), x, y, x * y, x * x, y * y))


def grid_shape(grid):
    """grid: 3 -> (3, 3); (3, 2) -> 3 columns x 2 rows."""
    cols, rows = (grid, grid) if np.isscalar(grid) else grid
    return int(cols), int(rows)


def validation_passed(delta, mode='distance', grid=3):
    """distance: Euclidean error <= 0.12. region: a screen cell (1/cols x 1/rows) centred
    on the point would be hit, i.e. |dx| <= 1/(2*cols) and |dy| <= 1/(2*rows)."""
    delta = np.asarray(delta, dtype=float)
    if mode == 'region':
        cols, rows = grid_shape(grid)
        return bool(abs(delta[0]) <= 1 / (2 * cols) and abs(delta[1]) <= 1 / (2 * rows))
    return bool(np.linalg.norm(delta) <= .12)


def tolerance_text(mode='distance', grid=3):
    if mode != 'region':
        return '거리 0.120'
    cols, rows = grid_shape(grid)
    return f'가로 {1 / (2 * cols):.3f} · 세로 {1 / (2 * rows):.3f} ({cols}×{rows} 영역)'


def _solve(inputs, targets, mapping):
    mu, sd = inputs.mean(axis=0), inputs.std(axis=0)
    design = features((inputs - mu) / sd, mapping)
    terms = design.shape[1]
    if np.linalg.matrix_rank(design) < terms or np.linalg.cond(design) > 100:
        raise ValueError('지점 데이터가 겹칩니다. 다시 보정하세요')
    weights = np.linalg.lstsq(design, targets, rcond=None)[0]
    return mu, sd, design, weights


def fit_calibration(inputs, targets, mapping='quadratic', validation='distance', grid=3, robust=False):
    """Fit screen mapping. ``robust`` (coarse mode) may drop the single worst of the
    nine points when the rest are consistent, e.g. one lid-occluded bottom target."""
    inputs, targets = np.asarray(inputs, dtype=float), np.asarray(targets, dtype=float)
    if mapping not in MAPPINGS or validation not in VALIDATIONS:
        raise ValueError('지원하지 않는 보정 방식입니다')
    if inputs.shape != (9, 2) or targets.shape != (9, 2) or not np.isfinite(inputs).all():
        raise ValueError('9개 지점을 모두 수집하세요')
    if np.any(inputs.std(axis=0) < .015):
        raise ValueError('눈의 이동 범위가 너무 작습니다. 카메라 위치를 확인하세요')
    keep = np.ones(9, bool)
    mu, sd, design, weights = _solve(inputs, targets, mapping)
    residuals = design @ weights - targets
    passed = [validation_passed(r, validation, grid) for r in residuals]
    excluded = None
    if not all(passed) and robust:
        worst = int(np.argmax(np.linalg.norm(residuals, axis=1)))
        keep[worst] = False
        try:
            mu2, sd2, design2, weights2 = _solve(inputs[keep], targets[keep], mapping)
            design_all = features((inputs - mu2) / sd2, mapping)
            residuals2 = design_all @ weights2 - targets
            if all(validation_passed(r, validation, grid) for r in residuals2[keep]):
                mu, sd, weights, residuals, excluded = mu2, sd2, weights2, residuals2, worst
        except ValueError:
            pass
        if excluded is None:
            keep[:] = True
    errors = np.linalg.norm(residuals, axis=1)
    error = float(np.max(errors[keep]))
    if not all(validation_passed(r, validation, grid) for r in residuals[keep]):
        raise ValueError(f'전체 9점 보정 오차 {error:.3f} / 기준 {tolerance_text(validation, grid)} · 가장 큰 오차는 '
                         f'{int(np.argmax(errors)) + 1}번 목표입니다. 실패가 반복되면 취소하고 9점부터 다시 보정하세요')
    # Leave-one-point-out checks expose a flexible polynomial fitting inconsistent observations.
    leave_one_out = []
    for index in range(9):
        rest = keep & (np.arange(9) != index)
        try:
            m, s, _, w = _solve(inputs[rest], targets[rest], mapping)
            leave_one_out.append(float(np.linalg.norm(features([(inputs[index] - m) / s], mapping)[0] @ w - targets[index])))
        except (ValueError, np.linalg.LinAlgError):
            leave_one_out.append(float('nan'))
    return {'mu': mu.tolist(), 'sd': sd.tolist(), 'weights': np.asarray(weights).tolist(), 'fit_error': error,
            'mapping': mapping, 'validation': validation, 'grid': list(grid_shape(grid)), 'excluded_point': excluded,
            'offset': [0., 0.], 'loo_errors': leave_one_out, 'loo_max_error': float(np.nanmax(leave_one_out))}


def predict(model, raw):
    mapping = model.get('mapping', 'quadratic')
    value = (features([(np.asarray(raw) - model['mu']) / model['sd']], mapping) @ model['weights'])[0]
    return value + np.asarray(model.get('offset', (0., 0.)), dtype=float)


class RegionTracker:
    """Grid cell under the gaze with hysteresis, so the highlight does not flicker at borders."""

    def __init__(self, grid=3, margin=.04):
        self.grid, self.margin, self.cell = np.array(grid_shape(grid)), margin, None

    def reset(self):
        self.cell = None

    def update(self, xy):
        if xy is None:
            self.cell = None
            return None
        x, y = np.clip(xy, 0, .9999)
        cols, rows = self.grid
        cell = (int(x * cols), int(y * rows))
        if self.cell is not None and cell != self.cell:
            # Stay in the old cell until the gaze is clearly past its border.
            left, top = self.cell[0] / cols, self.cell[1] / rows
            right, bottom = left + 1 / cols, top + 1 / rows
            if left - self.margin <= x <= right + self.margin and top - self.margin <= y <= bottom + self.margin:
                return list(self.cell)
        self.cell = cell
        return list(cell)


class Stabilizer:
    def __init__(self):
        self.history = deque(maxlen=3)
        self.position = None
        self.last_time = None

    def reset(self):
        self.history.clear()
        self.position = self.last_time = None

    def update(self, point, now, tau=.08, max_speed=4.):
        point = np.asarray(point, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all():
            self.reset()
            return None
        if self.last_time is None or now - self.last_time > .35:
            self.reset()
        self.history.append(point)
        target = np.median(self.history, axis=0)
        if self.position is None:
            self.position = target
        else:
            dt = max(0., now - self.last_time)
            delta = target - self.position
            distance = np.linalg.norm(delta)
            # ponytail: speed cap trades saccade latency for spike rejection; tune in UI.
            delta *= min(1., max_speed * dt / max(distance, 1e-9))
            self.position += delta * (1 - math.exp(-dt / tau))
        self.last_time = now
        return np.clip(self.position, 0, 1).tolist()
