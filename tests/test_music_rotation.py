import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import daily_music as dm


class RotationTests(unittest.IsolatedAsyncioTestCase):
    async def test_folder_limit_restart_and_next_day_rotation(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            'MUSIC_PATH': tmp, 'RECOMMEND_REPEAT_DAYS': '7',
            'RECOMMEND_LOW_PRIORITY_DIRECTORY': '잡다한', 'RECOMMEND_LOW_PRIORITY_MAX': '1',
            'RECOMMEND_MELON_COUNT': '4', 'RECOMMEND_OTHER_COUNT': '6'}, clear=True), \
            patch.dict(sys.modules, {'ollama': SimpleNamespace(Client=Mock()),
                'telegram.error': SimpleNamespace(BadRequest=type('BadRequest', (Exception,), {}),
                    Forbidden=type('Forbidden', (Exception,), {}), RetryAfter=type('RetryAfter', (Exception,), {}))}), \
            patch.object(dm, 'datetime') as clock, patch.object(dm.asyncio, 'sleep', new=AsyncMock()):
            clock.strptime = datetime.strptime
            tracks = []
            for folder, n in [('잡다한/가요모음', 10), ('other', 12), ('melon_top100', 8)]:
                for i in range(n):
                    path = Path(tmp) / folder / f'{i}.mp3'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b'test')
                    tracks.append((str(path), {}))
            def runner(channel=123):
                item = dm.DailyMusic(None, None, str(Path(tmp)/'state.db'), 'embed', 'host', channel=channel)
                item.recommendation_query = AsyncMock(return_value=dm.fallback_intent('가을', '오전'))
                item.candidates = Mock(return_value=tracks)
                return item
            bot = SimpleNamespace(get_chat=AsyncMock(), send_audio=AsyncMock())
            clock.now.return_value = datetime(2026, 10, 1, 5)
            await runner().tick(SimpleNamespace(bot=bot))
            self.assertEqual(bot.send_audio.await_count, 10)
            await runner().tick(SimpleNamespace(bot=bot))
            self.assertEqual(bot.send_audio.await_count, 10)
            clock.now.return_value = datetime(2026, 10, 2, 5)
            await runner().tick(SimpleNamespace(bot=bot))
            names = [call.kwargs['audio'].name for call in bot.send_audio.await_args_list]
            self.assertEqual(len(names), 20)
            self.assertEqual(len(set(names)), 20)
            for batch in (names[:10], names[10:]):
                self.assertEqual(sum(runner().is_low_priority(p) for p in batch), 1)
                self.assertEqual(sum(dm.track_group(p) == 'melon' for p in batch), 4)
            used, counts = runner().delivery_progress('2026-10-08', '오전')
            self.assertIn(names[0], used)  # October 1 still inside prior seven days.
            self.assertEqual(counts, {'melon': 0, 'other': 0})
            used, _ = runner().delivery_progress('2026-10-09', '오전')
            self.assertNotIn(names[0], used)
            self.assertIn(names[10], used)
            self.assertEqual(runner(456).delivery_progress('2026-10-02', '오전')[0], set())
            with patch.dict(os.environ, {'RECOMMEND_LOW_PRIORITY_MAX': '0'}):
                item = runner(789)
                await item.tick(SimpleNamespace(bot=bot))
                batch = [c.kwargs['audio'].name for c in bot.send_audio.await_args_list[20:]]
                self.assertEqual(len(batch), 10)
                self.assertFalse(any(item.is_low_priority(p) for p in batch))


if __name__ == '__main__':
    unittest.main()
