import asyncio
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import daily_music as dm

class ContextTests(unittest.TestCase):
    def test_weekday_regions(self):
        now = datetime(2026, 9, 28)
        self.assertEqual([x[0] for x in dm.locations_for(now, "오전")], ["군포", "광명"])
        self.assertEqual([x[0] for x in dm.locations_for(now, "오후")], ["화성"])

    def test_weekend_national(self):
        for day in (26, 27):
            now = datetime(2026, 9, day)
            self.assertEqual(dm.locations_for(now, "오전"), dm.locations_for(now, "오후"))
            self.assertEqual(len(dm.locations_for(now, "오전")), 7)

    @patch.dict(os.environ, {}, clear=True)
    def test_default_schedule(self):
        self.assertEqual([t.hour for _, t in dm.parse_schedule()], [5, 15])

    @patch.dict(os.environ, {"RECOMMEND_MORNING": "13:00"})
    def test_invalid_schedule(self):
        with self.assertRaises(ValueError):
            dm.parse_schedule()

    @patch.object(dm, "fetch", side_effect=OSError("offline"))
    def test_context_outage(self, fetch):
        result = dm.collect_context(datetime(2026, 9, 28), "오전")
        self.assertEqual(result["season"], "가을")
        self.assertEqual(result["weather_군포"], {"unavailable": True})
        self.assertEqual(result["market"], {"unavailable": True})

class DeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_restart_no_duplicates_and_afternoon_ten_new_tracks(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracks = []
            for i in range(25):
                p = Path(tmp) / f"{i}.mp3"
                p.write_bytes(b"test")
                tracks.append((str(p), {"title": str(i)}))
            bot = SimpleNamespace(send_audio=AsyncMock(), get_chat=AsyncMock())
            context = SimpleNamespace(bot=bot)
            with patch.dict(os.environ, {"MUSIC_CHANNEL_ID": "-100123", "MUSIC_PATH": tmp}), \
                 patch.dict(sys.modules, {"ollama": SimpleNamespace(Client=Mock()),
                     "telegram.error": SimpleNamespace(BadRequest=type("BadRequest", (Exception,), {}),
                         Forbidden=type("Forbidden", (Exception,), {}), RetryAfter=type("RetryAfter", (Exception,), {}))}), \
                 patch.object(dm, "collect_context", return_value={"season": "가을"}), \
                 patch.object(dm, "choose_query", return_value="잔잔한 음악"), \
                 patch.object(dm.DailyMusic, "candidates", return_value=tracks), \
                 patch.object(dm.asyncio, "sleep", new=AsyncMock()), \
                 patch.object(dm, "datetime") as clock:
                clock.strptime = datetime.strptime
                clock.now.return_value = datetime(2026, 9, 28, 5, 0)
                def runner():
                    return dm.DailyMusic(None, None, str(Path(tmp)/"state.db"), "embed", "host")
                bad_request = sys.modules["telegram.error"].BadRequest
                bot.get_chat.side_effect = bad_request("Chat not found")
                await runner().tick(context)
                self.assertEqual(bot.send_audio.await_count, 0)
                bot.get_chat.side_effect = None
                bot.send_audio.side_effect = bad_request("Chat not found")
                await runner().tick(context)
                with runner().db() as db:
                    self.assertEqual(db.execute("SELECT count(*) FROM daily_deliveries").fetchone()[0], 0)
                bot.send_audio.reset_mock(side_effect=True)
                await runner().tick(context)
                self.assertEqual(bot.send_audio.await_count, 10)
                await runner().tick(context)
                self.assertEqual(bot.send_audio.await_count, 10)
                clock.now.return_value = datetime(2026, 9, 28, 15, 0)
                await runner().tick(context)
                self.assertEqual(bot.send_audio.await_count, 20)
                names = [c.kwargs["audio"].name for c in bot.send_audio.await_args_list]
                self.assertEqual(len(set(names)), 20)
                clock.now.return_value = datetime(2026, 9, 29, 4, 59)
                await runner().tick(context)
                self.assertEqual(bot.send_audio.await_count, 20)

if __name__ == "__main__":
    unittest.main()
