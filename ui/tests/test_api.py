import unittest

from ui.backend.app import validate_profile


class ValidateProfileTests(unittest.TestCase):
    def test_validate_profile_errors_for_empty_profile(self):
        result = validate_profile({})
        self.assertFalse(result["valid"])
        self.assertGreater(len(result["errors"]), 0)


if __name__ == "__main__":
    unittest.main()
