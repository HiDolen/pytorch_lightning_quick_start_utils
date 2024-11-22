import unittest
from unittest.mock import patch
from pl_utils.lightning import format_next_version_name

class TestLogger(unittest.TestCase):

    @patch("pl_utils.lightning.logger.os.listdir")
    @patch("pl_utils.lightning.logger.os.path.exists")
    @patch("pl_utils.lightning.logger.os.path.isdir")
    def test_format_next_version_name(self, mock_isdir, mock_exists, mock_listdir):
        mock_exists.return_value = True
        mock_listdir.return_value = ["version_000", "version_001", "version_005_test"]
        mock_isdir.return_value = True
        self.assertEqual(format_next_version_name(), "version_006")
        self.assertEqual(format_next_version_name("test"), "version_006_test")
        self.assertEqual(format_next_version_name("test", 10), "version_010_test")


if __name__ == '__main__':
    unittest.main()
