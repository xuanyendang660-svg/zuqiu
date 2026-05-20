import unittest

from score_model.cli import _matches_from_payload


class CliBatchInputTests(unittest.TestCase):
    def test_loads_matches_from_matches_key(self):
        payload = {
            "matches": [
                {
                    "match_id": "a",
                    "league": "League",
                    "home_team": "Home A",
                    "away_team": "Away A",
                },
                {
                    "match_id": "b",
                    "league": "League",
                    "home_team": "Home B",
                    "away_team": "Away B",
                },
            ]
        }

        matches = _matches_from_payload(payload)

        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0].home_team, "Home A")
        self.assertEqual(matches[1].away_team, "Away B")

    def test_loads_single_match_for_backward_compatibility(self):
        payload = {
            "match_id": "single",
            "league": "League",
            "home_team": "Home",
            "away_team": "Away",
        }

        matches = _matches_from_payload(payload)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].match_id, "single")


if __name__ == "__main__":
    unittest.main()

