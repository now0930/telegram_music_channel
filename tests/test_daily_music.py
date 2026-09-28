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
    @patch.dict(os.environ, {"RECOMMEND_MELON_COUNT": "5", "RECOMMEND_OTHER_COUNT": "15"})
    def test_custom_counts(self):
        self.assertEqual(dm.read_quotas(), {"melon": 5, "other": 15})

    @patch.dict(os.environ, {"RECOMMEND_MELON_COUNT": "-1"})
    def test_negative_counts_rejected(self):
        with self.assertRaises(ValueError):
            dm.read_quotas()

    def test_directory_matching(self):
        for name in ("melon top 100", "melon_top100", "Melon-Top-100"):
            self.assertEqual(dm.track_group("/music/" + name + "/album/song.mp3"), "melon")
        self.assertEqual(dm.track_group("/music/not_melon_top100/song.mp3"), "other")
        self.assertEqual(dm.track_group("/music/other/melon_top100.mp3"), "other")

    @patch.dict(os.environ, {}, clear=True)
    def test_default_counts(self):
        self.assertEqual(dm.read_quotas(), {"melon": 4, "other": 6})

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
    async def test_restart_no_duplicates_and_twenty_ten_per_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracks = []
            for i in range(75):
                p = Path(tmp) / ("melon top 100" if i < 50 else "other") / f"{i}.mp3"
                p.parent.mkdir(exist_ok=True)
                p.write_bytes(b"test")
                tracks.append((str(p), {"title": str(i)}))
            bot = SimpleNamespace(send_audio=AsyncMock(), get_chat=AsyncMock())
            context = SimpleNamespace(bot=bot)
            with patch.dict(os.environ, {"MUSIC_CHANNEL_ID": "-100123", "MUSIC_PATH": tmp, "RECOMMEND_MELON_COUNT": "20",
                                        "RECOMMEND_OTHER_COUNT": "10", "RECOMMEND_MELON_DIRECTORY": "melon_top100"}), \
                 patch.dict(sys.modules, {"ollama": SimpleNamespace(Client=Mock()),
                     "telegram.error": SimpleNamespace(BadRequest=type("BadRequest", (Exception,), {}),
                         Forbidden=type("Forbidden", (Exception,), {}), RetryAfter=type("RetryAfter", (Exception,), {}))}), \
                 patch.object(dm, "collect_context", return_value={"season": "가을"}), \
                 patch.object(dm.DailyMusic, "recommendation_query", new=AsyncMock(return_value=dm.fallback_intent("가을", "오전"))), \
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
                self.assertEqual(bot.send_audio.await_count, 30)
                await runner().tick(context)
                self.assertEqual(bot.send_audio.await_count, 30)
                clock.now.return_value = datetime(2026, 9, 28, 15, 0)
                await runner().tick(context)
                self.assertEqual(bot.send_audio.await_count, 60)
                names = [c.kwargs["audio"].name for c in bot.send_audio.await_args_list]
                self.assertEqual(len(set(names)), 60)
                for batch in (names[:30], names[30:]):
                    self.assertEqual(sum(dm.track_group(p) == "melon" for p in batch), 20)
                    self.assertEqual(sum(dm.track_group(p) == "other" for p in batch), 10)
                clock.now.return_value = datetime(2026, 9, 29, 4, 59)
                await runner().tick(context)
                self.assertEqual(bot.send_audio.await_count, 60)
                # Only 10 melon + 5 other tracks remain today; do not fill either shortage
                # from the opposite group, even though more tracks exist there.
                clock.now.return_value = datetime(2026, 9, 28, 15, 0)
                with patch.dict(os.environ, {"RECOMMEND_MELON_COUNT": "35", "RECOMMEND_OTHER_COUNT": "10"}):
                    await runner().tick(context)
                self.assertEqual(bot.send_audio.await_count, 70)
                extras = bot.send_audio.await_args_list[60:]
                self.assertTrue(all(dm.track_group(c.kwargs["audio"].name) == "melon" for c in extras))

if __name__ == "__main__":
    unittest.main()
