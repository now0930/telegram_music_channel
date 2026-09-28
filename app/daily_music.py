"""Context-aware channel delivery; persistent per-day/per-slot send journal."""
import asyncio
import json
import logging
import os
import re
import sqlite3
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo
from music_intent import vocabulary, choose_intent, fallback_intent, rank_candidates, describe_intent

log = logging.getLogger(__name__)
MODEL = "hf.co/sky7350/Mica-v0.1-4B:Q5_K_M"


def fetch(url):
    with urlopen(Request(url, headers={"User-Agent": "MusicChannel/1.0"}), timeout=15) as response:
        return response.read(2_000_000)


def locations_for(now, slot):
    if now.weekday() >= 5:
        return [("서울", 37.5665, 126.9780), ("강릉", 37.7519, 128.8761),
                ("대전", 36.3504, 127.3845), ("광주", 35.1595, 126.8526),
                ("대구", 35.8714, 128.6014), ("부산", 35.1796, 129.0756),
                ("제주", 33.4996, 126.5312)]
    if slot == "오전":
        return [("군포", 37.3617, 126.9352), ("광명", 37.4785, 126.8643)]
    return [("화성", 37.1995, 126.8312)]


def collect_context(now, slot):
    month = now.month
    data = {"date": now.isoformat(), "season": ("겨울" if month in (12, 1, 2) else
            "봄" if month < 6 else "여름" if month < 9 else "가을"),
            "scope": "전국 주요 지역" if now.weekday() >= 5 else slot}
    sources = {}
    for city, latitude, longitude in locations_for(now, slot):
        url = "https://api.open-meteo.com/v1/forecast?" + urlencode({
            "latitude": latitude, "longitude": longitude,
            "current": "temperature_2m,weather_code", "timezone": "Asia/Seoul"})
        sources["weather_" + city] = lambda url=url: json.loads(fetch(url))["current"]
    sources["issues"] = lambda: [{"title": item.findtext("title"), "published": item.findtext("pubDate")}
        for item in ET.fromstring(fetch(os.getenv("RECOMMEND_NEWS_RSS",
        "https://news.google.com/rss?hl=ko&gl=KR&ceid=KR:ko"))).findall("./channel/item")[:8]]
    def market():
        symbol = os.getenv("RECOMMEND_MARKET_SYMBOL", "^KS11")
        result = json.loads(fetch("https://query1.finance.yahoo.com/v8/finance/chart/" +
            quote(symbol, safe="") + "?interval=1d&range=1d"))["chart"]["result"][0]["meta"]
        return {"symbol": symbol, "price": result.get("regularMarketPrice"),
                "previous_close": result.get("previousClose", result.get("chartPreviousClose")),
                "as_of": result.get("regularMarketTime")}
    sources["market"] = market
    for name, provider in sources.items():
        try:
            data[name] = provider()
        except Exception as exc:
            log.warning("Context source unavailable: %s (%s)", name, type(exc).__name__)
            data[name] = {"unavailable": True}
    return data


def parse_schedule():
    morning = datetime.strptime(os.getenv("RECOMMEND_MORNING", "05:00"), "%H:%M").time()
    afternoon = datetime.strptime(os.getenv("RECOMMEND_AFTERNOON", "15:00"), "%H:%M").time()
    if not morning.hour < 12 <= afternoon.hour:
        raise ValueError("Morning must be before noon and afternoon after noon")
    return [("오전", morning), ("오후", afternoon)]


def directory_key(name):
    return re.sub(r"[\s_-]+", "", name).casefold()


def track_group(path):
    target = directory_key(os.getenv("RECOMMEND_MELON_DIRECTORY", "melon_top100"))
    return "melon" if any(directory_key(part) == target for part in Path(path).parts[:-1]) else "other"


def read_quotas():
    quotas = {"melon": int(os.getenv("RECOMMEND_MELON_COUNT", "4")),
              "other": int(os.getenv("RECOMMEND_OTHER_COUNT", "6"))}
    if any(n < 0 for n in quotas.values()) or not 1 <= sum(quotas.values()) <= 100:
        raise ValueError("Recommendation counts must be nonnegative and total between 1 and 100")
    if not directory_key(os.getenv("RECOMMEND_MELON_DIRECTORY", "melon_top100")):
        raise ValueError("RECOMMEND_MELON_DIRECTORY must not be empty")
    return quotas


class DailyMusic:
    def __init__(self, application, collection, db_path, embed_model, host):
        import ollama
        self.application, self.collection = application, collection
        self.client = ollama.Client(host=host, timeout=120)
        self.embed_model = embed_model
        channel = os.getenv("MUSIC_CHANNEL_ID", "").strip()
        self.channel = int(channel) if channel.lstrip("-").isdigit() else channel
        self.timezone = ZoneInfo(os.getenv("RECOMMEND_TIMEZONE", "Asia/Seoul"))
        self.schedule = parse_schedule()
        self.quotas = read_quotas()
        self.total = sum(self.quotas.values())
        self.lock = asyncio.Lock()
        self.db_path = db_path
        with self.db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS daily_deliveries (channel TEXT, day TEXT, "
                "slot TEXT, path TEXT, state TEXT, PRIMARY KEY(channel, day, path))")

    @contextmanager
    def db(self):
        connection = sqlite3.connect(self.db_path, timeout=10)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def delivery_progress(self, day, slot):
        """Pending sends count toward quotas to avoid ambiguous duplicate delivery."""
        with self.db() as db:
            rows = db.execute(
                "SELECT path, slot FROM daily_deliveries WHERE channel=? AND day=?",
                (str(self.channel), day),
            ).fetchall()
        used = {path for path, _ in rows}
        counts = {group: 0 for group in self.quotas}
        for path, sent_slot in rows:
            if sent_slot == slot:
                counts[track_group(path)] += 1
        return used, counts

    def log_delivery_status(self, day, slot):
        with self.db() as db:
            rows = db.execute(
                "SELECT state, COUNT(*) FROM daily_deliveries WHERE channel=? AND day=? AND slot=? GROUP BY state",
                (str(self.channel), day, slot),
            ).fetchall()
        states = dict(rows)
        log.info("%s %s delivery status: sent=%d pending=%d; target melon=%d other=%d",
                 day, slot, states.get("sent", 0), states.get("pending", 0),
                 self.quotas["melon"], self.quotas["other"])

    def quota_reached(self, counts):
        return all(counts[group] >= quota for group, quota in self.quotas.items())

    def reserve_delivery(self, day, slot, path):
        with self.db() as db:
            return bool(db.execute(
                "INSERT OR IGNORE INTO daily_deliveries VALUES (?, ?, ?, ?, 'pending')",
                (str(self.channel), day, slot, path),
            ).rowcount)

    def finish_delivery(self, day, path, *, rejected=False):
        query = (
            "DELETE FROM daily_deliveries WHERE channel=? AND day=? AND path=? AND state='pending'"
            if rejected else
            "UPDATE daily_deliveries SET state='sent' WHERE channel=? AND day=? AND path=?"
        )
        with self.db() as db:
            db.execute(query, (str(self.channel), day, path))

    async def recommendation_query(self, now, slot):
        data = await asyncio.to_thread(collect_context, now, slot)
        try:
            values = await asyncio.to_thread(vocabulary, self.collection)
            intent = await asyncio.to_thread(choose_intent, self.client,
                os.getenv("OLLAMA_MODEL", MODEL), data, slot, values)
            log.info("Mica metadata preferences: %s", json.dumps(intent, ensure_ascii=False))
            return intent
        except Exception:
            log.exception("Mica unavailable; using seasonal music query")
            return fallback_intent(data["season"], slot)

    def candidates(self, query):
        count = self.collection.count()
        if not count:
            log.error("Music library is empty; check MUSIC_DB_PATH and Docker mounts")
            return []
        terms = " ".join(str(value) for key, value in query.items() if value is not None and key != "reason")
        embedding = self.client.embeddings(model=self.embed_model, prompt=terms).embedding
        result = self.collection.query(query_embeddings=[embedding], n_results=count, include=["metadatas"])
        candidates = [(meta.get("path") or ident, meta) for ident, meta in
                      zip(result["ids"][0], result["metadatas"][0]) if meta]
        return rank_candidates(candidates, query)

    async def tick(self, context):
        from telegram.error import BadRequest, Forbidden, RetryAfter
        if self.lock.locked():
            return
        async with self.lock:
            now = datetime.now(self.timezone)
            # Catch up only the latest due slot, never both batches on startup.
            due = [(name, at) for name, at in self.schedule if at <= now.time()]
            if not due:
                return
            slot = due[-1][0]
            day = now.date().isoformat()
            used, counts = self.delivery_progress(day, slot)
            completed = sum(counts.values())
            if self.quota_reached(counts):
                self.log_delivery_status(day, slot)
                return
            try:
                # Validate the destination before reserving tracks or querying Mica.
                try:
                    await context.bot.get_chat(self.channel)
                except (BadRequest, Forbidden):
                    log.error("Channel inaccessible: check MUSIC_CHANNEL_ID and bot channel membership/posting permissions")
                    return
                query = await self.recommendation_query(now, slot)
                candidates = await asyncio.to_thread(self.candidates, query)
                music_root = Path(os.getenv("MUSIC_PATH", "/music")).resolve()
                for path, meta in candidates:
                    path = str(Path(path).resolve())
                    if path in used or not Path(path).is_relative_to(music_root) or not Path(path).is_file():
                        continue
                    if self.quota_reached(counts):
                        break
                    group = track_group(path)
                    if counts[group] >= self.quotas[group]:
                        continue
                    # Reserve before sending: ambiguous timeouts/crashes must not resend audio.
                    if not self.reserve_delivery(day, slot, path):
                        continue
                    used.add(path)
                    completed += 1
                    counts[group] += 1
                    try:
                        with open(path, "rb") as audio:
                            await context.bot.send_audio(chat_id=self.channel, audio=audio,
                                caption=(f"🎵 {day} {slot} 추천 {completed}/{self.total}\n"
                                    f"{meta.get('artist', '')} - {meta.get('title', '')}\n{describe_intent(query)}")[:1024],
                                write_timeout=120, read_timeout=120, connect_timeout=30)
                        self.finish_delivery(day, path)
                    except (BadRequest, Forbidden, RetryAfter) as exc:
                        # Telegram explicitly rejected this request: nothing was delivered.
                        self.finish_delivery(day, path, rejected=True)
                        log.error("Telegram rejected delivery (%s): %s; stopping this batch", type(exc).__name__, exc)
                        return
                    except Exception:
                        log.exception("Delivery outcome uncertain; reserved track will not be retried: %s", path)
                    await asyncio.sleep(2)
                for group, quota in self.quotas.items():
                    if counts[group] < quota:
                        log.warning("%s %s: %s only %d/%d tracks available; no cross-directory substitution",
                                    day, slot, group, counts[group], quota)
                self.log_delivery_status(day, slot)
            except Exception:
                log.exception("Scheduled recommendation failed")


def install(application, collection, db_path, embed_model, host):
    if os.getenv("RECOMMEND_ENABLED", "true").lower() != "true" or not os.getenv("MUSIC_CHANNEL_ID", "").strip():
        log.info("Daily recommendations disabled (set MUSIC_CHANNEL_ID to enable)")
        return
    runner = DailyMusic(application, collection, db_path, embed_model, host)
    for slot, at in runner.schedule:
        application.job_queue.run_daily(runner.tick, time=at.replace(tzinfo=runner.timezone),
                                        name="music-" + slot)
    application.job_queue.run_repeating(runner.tick, interval=300, first=5,
                                       job_kwargs={"max_instances": 1, "coalesce": True})
    log.info("Daily recommendations enabled: %s (%s)", runner.schedule, runner.timezone)
    log.info("Per batch quotas: melon=%d other=%d", runner.quotas["melon"], runner.quotas["other"])
