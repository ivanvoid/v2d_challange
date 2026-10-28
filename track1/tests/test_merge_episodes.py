"""merge_episodes: each episode comes from exactly one experiment."""
import sys
import unittest
from pathlib import Path

T1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T1))
from merge_episodes import parse_take  # noqa: E402


class MergeEpisodes(unittest.TestCase):
    def test_parse_take(self):
        o = parse_take(['/x/a/episodes:15,16', '/x/b/episodes:7'])
        self.assertEqual(o[16], Path('/x/a/episodes'))
        self.assertEqual(o[7], Path('/x/b/episodes'))
        with self.assertRaises(ValueError):
            parse_take(['/x/a:1,2', '/x/b:2'])


if __name__ == '__main__':
    unittest.main()
