"""static_lock: the still interval must end before the object starts moving."""
import sys
import unittest
from pathlib import Path

import numpy as np

T1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T1))
from static_lock import still_until  # noqa: E402


class StaticLock(unittest.TestCase):
    def test_onset_ignores_jitter_and_short_glitches(self):
        rng = np.random.default_rng(0)
        t = rng.normal(scale=.005, size=(300, 3)) + [0, 0, 3]
        t[120:123] += [.1, 0, 0]                          # 3-frame tracker glitch: not motion
        t[200:] += np.linspace(0, .6, 100)[:, None] * [1, 0, 0]
        last = still_until(t, onset_cm=4, hold=10)
        self.assertTrue(200 <= last <= 210, last)
        self.assertEqual(still_until(t[:150], 4, 10), 149)   # never moves: whole interval


if __name__ == '__main__':
    unittest.main()
