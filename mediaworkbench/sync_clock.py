"""Invert calibrated camera clocks on the programme's continuous sample axis."""
from fractions import Fraction
import math

RATE = 48000
FRAME = 1920


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('clock values must be finite numbers')
    return Fraction(str(value))


def calibrate(anchors, duration_ms):
    if not isinstance(anchors, list) or not 2 <= len(anchors) <= 64:
        raise ValueError('a camera needs two to sixty-four clock observations')
    points = [[number(p['programme_ms']) * 48, number(p['source_ms']) * 48]
              for p in anchors]
    for i, (programme, source) in enumerate(points):
        if programme < 0 or not 0 <= source <= number(duration_ms) * 48:
            raise ValueError('clock observation outside media extent')
        if i and (programme <= points[i-1][0] or source <= points[i-1][1]):
            raise ValueError('clock observations must increase on both clocks')
    return [[str(a), str(b)] for a, b in points]


def _points(camera):
    return [(Fraction(a), Fraction(b)) for a, b in camera['clock']]


def bounds(camera):
    """Whole-sample programme extent covered by the observations."""
    clock = camera['clock']
    return int(Fraction(clock[0][0])), int(Fraction(clock[-1][0]))


def position(camera, programme_sample):
    """Exact piecewise-linear programme->source mapping, snapped to a source frame.

    Each pair of adjacent alignment observations is its own linear segment, so
    every anchor is honoured exactly and drift never accumulates between
    alignment points. No extrapolation: programme samples outside the observed
    clock range return None instead of inventing coverage.
    """
    points = _points(camera)
    t = Fraction(programme_sample)
    first_programme, first_source = points[0]
    last_programme, last_source = points[-1]
    if t < first_programme or t > last_programme:
        return None
    for index in range(len(points) - 1):
        programme, source = points[index]
        next_programme, next_source = points[index + 1]
        if programme <= t <= next_programme:
            span = next_programme - programme
            if span <= 0 or next_source <= source:
                raise ValueError('invalid camera calibration segment')
            exact = source + (t - programme) * (next_source - source) / span
            nearest_frame = int(exact / FRAME + Fraction(1, 2))
            return Fraction(nearest_frame * FRAME)
    return last_source


def covers(camera, start, end):
    """A frame is covered only when its whole span sits inside the observations."""
    clock = camera['clock']
    return Fraction(clock[0][0]) <= start and end <= Fraction(clock[-1][0])


def score(camera, programme_sample):
    sample = int(programme_sample)
    for interval in camera['quality']:
        if interval['start_sample'] <= sample < interval['end_sample']:
            return interval['score']
    return 0
