"""Exercise real tick time conversion without Telegram calls or production state."""
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import daily_music as dm


class ScheduleTests(unittest.IsolatedAsyncioTestCase):
    async def test_utc_boundaries_and_local_calendar(self):
        cases = [
            ('2026-09-27T19:59:59+00:00', None, None),
            ('2026-09-27T20:00:00+00:00', '2026-09-28', '오전'),
            ('2026-09-28T05:59:59+00:00', '2026-09-28', '오전'),
            ('2026-09-28T06:00:00+00:00', '2026-09-28', '오후'),
            ('2026-09-28T14:59:59+00:00', '2026-09-28', '오후'),
            ('2026-09-28T15:00:00+00:00', None, None),
            ('2026-09-28T20:00:00+00:00', '2026-09-29', '오전'),
            ('2026-09-25T20:00:00+00:00', '2026-09-26', '오전'),
            ('2026-09-26T06:00:00+00:00', '2026-09-26', '오후'),
            ('2026-09-26T20:00:00+00:00', '2026-09-27', '오전'),
        ]
        for instant, day, slot in cases:
            with self.subTest(utc=instant), tempfile.TemporaryDirectory() as tmp, \
                 patch.dict(os.environ, {'MUSIC_CHANNEL_ID': '123', 'RECOMMEND_TIMEZONE': 'Asia/Seoul'}, clear=True), \
                 patch.dict(sys.modules, {'ollama': SimpleNamespace(Client=Mock()),
                     'telegram.error': SimpleNamespace(BadRequest=type('BadRequest', (Exception,), {}),
                         Forbidden=type('Forbidden', (Exception,), {}), RetryAfter=type('RetryAfter', (Exception,), {}))}), \
                 patch.object(dm, 'datetime') as clock:
                clock.strptime = datetime.strptime
                utc = datetime.fromisoformat(instant)
                clock.now.side_effect = lambda tz: utc.astimezone(tz)
                runner = dm.DailyMusic(None, None, str(Path(tmp)/'test.db'), 'embed', 'host')
                runner.delivery_progress = Mock(return_value=(set(), {'melon': 4, 'other': 6}))
                runner.log_delivery_status = Mock()
                context = SimpleNamespace(bot=SimpleNamespace(send_audio=AsyncMock()))
                await runner.tick(context)
                if slot is None:
                    runner.delivery_progress.assert_not_called()
                else:
                    runner.delivery_progress.assert_called_once_with(day, slot)
                    local = utc.astimezone(runner.timezone)
                    locations = [city for city, _, _ in dm.locations_for(local, slot)]
                    expected = (['서울', '강릉', '대전', '광주', '대구', '부산', '제주']
                                if local.weekday() >= 5 else ['군포', '광명'] if slot == '오전' else ['화성'])
                    self.assertEqual(locations, expected)
                context.bot.send_audio.assert_not_awaited()

    @patch.dict(os.environ, {'MUSIC_CHANNEL_ID': '123'}, clear=True)
    def test_registration_retains_seoul_timezone(self):
        runner = SimpleNamespace(schedule=dm.parse_schedule(), timezone=ZoneInfo('Asia/Seoul'),
                                 tick=AsyncMock(), quotas={'melon': 4, 'other': 6})
        app = SimpleNamespace(job_queue=Mock())
        with patch.object(dm, 'DailyMusic', return_value=runner):
            dm.install(app, None, 'unused', 'embed', 'host')
        calls = app.job_queue.run_daily.call_args_list
        self.assertEqual(len(calls), 2)
        for call, hour, utc_hour in zip(calls, [5, 15], [20, 6]):
            at = call.kwargs['time']
            self.assertEqual(at.hour, hour)
            self.assertEqual(at.tzinfo.key, 'Asia/Seoul')
            local = datetime.combine(datetime(2026, 9, 28).date(), at)
            self.assertEqual(local.astimezone(timezone.utc).hour, utc_hour)
        self.assertEqual(app.job_queue.run_repeating.call_args.kwargs['interval'], 300)

if __name__ == '__main__':
    unittest.main()
