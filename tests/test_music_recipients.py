import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import daily_music as dm


class RecipientTests(unittest.TestCase):
    @patch.dict(os.environ, {'MUSIC_CHANNEL_ID': '123', 'MUSIC_ADDITIONAL_CHAT_IDS': '456, 123,456'}, clear=True)
    def test_preserve_existing_and_deduplicate(self):
        self.assertEqual(dm.recipients(), [123, 456])

    @patch.dict(os.environ, {'MUSIC_ADDITIONAL_CHAT_IDS': 'bot-token:invalid'}, clear=True)
    def test_invalid_recipient(self):
        with self.assertRaises(ValueError):
            dm.recipients()

    @patch.dict(os.environ, {'MUSIC_CHANNEL_ID': '123', 'MUSIC_ADDITIONAL_CHAT_IDS': '456'}, clear=True)
    def test_independent_schedules_and_history(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'ollama': SimpleNamespace(Client=Mock())}):
            db = str(Path(tmp) / 'state.db')
            app = SimpleNamespace(job_queue=Mock())
            dm.install(app, None, db, 'embed', 'host')
            calls = app.job_queue.run_repeating.call_args_list
            self.assertEqual(len(calls), 2)
            first, second = [call.args[0].__self__ for call in calls]
            self.assertEqual([first.channel, second.channel], [123, 456])
            self.assertIsNot(first.lock, second.lock)
            self.assertTrue(first.reserve_delivery('2026-09-28', '오전', '/music/test.mp3'))
            self.assertEqual(second.delivery_progress('2026-09-28', '오전')[0], set())
            self.assertTrue(second.reserve_delivery('2026-09-28', '오전', '/music/test.mp3'))
            self.assertFalse(first.reserve_delivery('2026-09-28', '오전', '/music/test.mp3'))
            self.assertEqual(app.job_queue.run_daily.call_count, 4)


if __name__ == '__main__':
    unittest.main()
