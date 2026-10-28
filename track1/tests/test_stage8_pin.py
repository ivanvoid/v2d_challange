"""stage 8 --pin keeps an earlier depth placement exactly; malformed pins fail loudly."""
import sys
import unittest
from pathlib import Path

T1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T1))
from stage8_scale import parse_pin  # noqa: E402


class Stage8Pin(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_pin('follow:1.125'), ('follow', 1.125))
        self.assertEqual(parse_pin('fixed:1.0249999999'), ('fixed', 1.025))
        for bad in ('global:1.0', 'fixed', 'fixed:-1', 'follow:'):
            with self.assertRaises(ValueError):
                parse_pin(bad)

    def test_pinned_choice_overrides_the_search_but_keeps_k1_for_the_legacy_log(self):
        src = (T1 / 'stage8_scale.py').read_text()
        self.assertIn('np.array(sorted({1.0, pin[1]}))', src)
        self.assertIn('if pin:\n        best = next(', src)


if __name__ == '__main__':
    unittest.main()
