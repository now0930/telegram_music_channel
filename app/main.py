"""
music_bot.py — Telegram Music Bot (안정성 강화 + 하이브리드 지능 판)
주요 수정 사항:
  - 구글 최신 표준 라이브러리(google-genai) 적용 (Deprecated 문제 해결)
  - Gemini API (1순위) + Ollama gemma4:26b (2순위 폴백) 하이브리드 의도 추출 도입
  - 0단계: 하드 텍스트 매칭으로 벡터 검색의 한계 극복 (정확도 대폭 향상)
  - ChromaDB InternalError 방지를 위해 where_document($contains) 필터 제거
  - 네트워크 Bad Gateway 대응을 위한 run_polling 타임아웃 설정 제거 (최신 라이브러리 호환)
"""

import os
import json
import asyncio
import re
import sys
import random
import time
import logging
import aiosqlite

import ollama
import chromadb
from chromadb.errors import InternalError as ChromaInternalError
from telegram import Update
from telegram.error import TelegramError
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

# 최신 구글 생성형 AI 라이브러리 (2026년 표준)
try:
    from google import genai
except ImportError:
    genai = None

# ═════════════════════════════════════════════════════════════════════════════════════
#  📋  로깅 설정
# ═════════════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("music_bot")

# ═════════════════════════════════════════════════════════════════════════════════════
#  ⚙️  설정
# ═════════════════════════════════════════════════════════════════════════════════════
MUSIC_PATH   = "/music"
DB_PATH      = "./music_vector_db"
TEXT_MODEL   = "gemma4:26b"  # 내부 폴백용 모델
EMBED_MODEL  = "mxbai-embed-large"
MAX_RESULTS  = 10   
CAPTION_LIMIT = 1024  

BOT_DB_PATH = "./bot_data.db"
MAX_USERS = 10
TTL_SECONDS = 7 * 24 * 60 * 60  

OLLAMA_HOST      = os.getenv("OLLAMA_HOST", "http://ollama:11434")
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")  # Gemini API 키 추가

if not TELEGRAM_TOKEN:
    logger.critical("❌ 필수 환경 변수 미설정: TELEGRAM_TOKEN")
    sys.exit(1)

# ⚙️ 설정 부분 수정
# 2026년 5월 기준 안정적인 상용 모델로 설정
GEMINI_MODEL_NAME = "gemini-3.1-flash-lite-preview"


# Gemini 초기화 (새로운 클라이언트 방식)
client_gemini = None
if genai and GEMINI_API_KEY:
    try:
        client_gemini = genai.Client(api_key=GEMINI_API_KEY)
        logger.info(f"✅ 최신 Gemini API 설정 완료. (모델: {GEMINI_MODEL_NAME}) 하이브리드 엔진 모드로 동작합니다.")
    except Exception as e:
        logger.error(f"❌ Gemini 클라이언트 초기화 실패: {e}")
else:
    logger.warning("⚠️ GEMINI_API_KEY가 없거나 라이브러리가 설치되지 않아 Ollama 단독 모드로 동작합니다.")

# ═════════════════════════════════════════════════════════════════════════════════════
#  🗄️  DB 연결
# ═════════════════════════════════════════════════════════════════════════════════════
try:
    _chroma    = chromadb.PersistentClient(path=DB_PATH)
    collection = _chroma.get_or_create_collection(name="music_library")
except ChromaInternalError as e:
    logger.error("ChromaDB 컬렉션 로드 중 에러 발생. 재구성을 시도합니다: %s", e)
    _chroma.delete_collection("music_library")
    collection = _chroma.get_or_create_collection(name="music_library")

# ═════════════════════════════════════════════════════════════════════════════════════
#  🛠️  유틸리티
# ═════════════════════════════════════════════════════════════════════════════════════
async def init_db():
    async with aiosqlite.connect(BOT_DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                chat_id INTEGER PRIMARY KEY,
                last_active REAL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sent_messages (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER,
                message_id INTEGER,
                sent_at REAL
            )
        """)
        await db.commit()

async def update_user_access(chat_id: int):
    async with aiosqlite.connect(BOT_DB_PATH) as db:
        await db.execute("INSERT OR REPLACE INTO users (chat_id, last_active) VALUES (?, ?)", (chat_id, time.time()))
        cursor = await db.execute(
            "SELECT chat_id FROM users WHERE chat_id NOT IN (SELECT chat_id FROM users ORDER BY last_active DESC LIMIT ?)",
            (MAX_USERS,)
        )
        evicted_users = await cursor.fetchall()
        for row in evicted_users:
            await db.execute("DELETE FROM sent_messages WHERE chat_id = ?", (row[0],))
            await db.execute("DELETE FROM users WHERE chat_id = ?", (row[0],))
        await db.commit()

async def ttl_cleanup_task(application):
    await asyncio.sleep(10)
    while True:
        try:
            cutoff_time = time.time() - TTL_SECONDS
            async with aiosqlite.connect(BOT_DB_PATH) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute("SELECT rowid, chat_id, message_id FROM sent_messages WHERE sent_at < ?", (cutoff_time,))
                rows = await cursor.fetchall()
                for row in rows:
                    try:
                        await application.bot.delete_message(row["chat_id"], row["message_id"])
                    except: pass
                    await db.execute("DELETE FROM sent_messages WHERE rowid = ?", (row["rowid"],))
                await db.commit()
        except Exception as e:
            logger.error("TTL 정리 루프 에러: %s", e)
        await asyncio.sleep(3600)

async def safe_edit(message, text: str) -> None:
    try:
        await message.edit_text(text)
    except: pass

def truncate_caption(text: str, limit: int = CAPTION_LIMIT) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"

# ═════════════════════════════════════════════════════════════════════════════════════
#  🤖  LLM 의도 추출 (Gemini + Ollama 하이브리드)
# ═════════════════════════════════════════════════════════════════════════════════════

def _extract_intent_sync(user_query: str) -> dict:
    intent_prompt = (
        f"당신은 전문 음악 큐레이터입니다. 사용자의 요청에서 검색 핵심 정보를 추출하세요.\n\n"
        f"요청: \"{user_query}\"\n\n"
        f"── [추출 규칙] ──\n"
        f"1. '노래', '들려줘', '찾아줘' 같은 무의미한 단어는 제외하고 핵심 제목과 가수만 남기세요.\n"
        f"2. 사용자가 말한 단어가 곡 제목 같으면 반드시 'title' 필드에 넣으세요. (예: '나는 반디불' -> title: '나는 반디불')\n"
        f"3. 가수 이름이 명시되었다면 'artist' 필드에 넣으세요.\n"
        f"4. 결과는 반드시 순수한 JSON 형식으로만 응답하세요.\n\n"
        f"── [출력 형식] ──\n"
        f"{{\"artist\": \"\", \"title\": \"\", \"year\": null, \"bpm_range\": null, \"count\": 5}}"
    )

    # 1. Gemini 우선 시도 (최신 방식 적용)
    if client_gemini:
        try:
            logger.info(f"✨ {GEMINI_MODEL_NAME} 엔진으로 의도 분석 중...")
            response = client_gemini.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=intent_prompt
            )
            match = re.search(r"\{.*?\}", response.text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.warning(f"⚠️ Gemini 오류 발생, Ollama({TEXT_MODEL})로 전환: {e}")

    # 2. Ollama(gemma4:26b)로 폴백
    try:
        resp = ollama.generate(
            model=TEXT_MODEL, 
            prompt=intent_prompt, 
            stream=False, 
            options={"temperature": 0}
        )
        raw = resp.get("response", "").strip()
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        logger.error(f"❌ 모델({TEXT_MODEL}) 의도 추출 실패: {e}")

    return {"keyword": user_query, "count": 5}

async def extract_intent(user_query: str) -> dict:
    return await asyncio.to_thread(_extract_intent_sync, user_query)

# ═════════════════════════════════════════════════════════════════════════════════════
#  🔍  벡터 검색 (0단계 하드 텍스트 매칭 포함)
# ═════════════════════════════════════════════════════════════════════════════════════
def _search_music_sync(intent: dict) -> tuple[list[str], list[dict]]:
    target_count = intent.get("count", 5)
    db_count = collection.count()
    def safe_n(n): return max(1, min(n, db_count))

    # [0단계] 텍스트 기반 직접 필터링 시도 (정확도 향상의 핵심)
    search_keyword = intent.get("title") or intent.get("artist") or intent.get("keyword")
    
    if search_keyword:
        logger.info(f"[0단계] 키워드 직접 매칭 시도: {search_keyword}")
        all_data = collection.get(include=["metadatas"])
        match_ids = []
        match_metas = []
        
        for i, meta in enumerate(all_data["metadatas"]):
            title = meta.get("title", "")
            artist = meta.get("artist", "")
            # 띄어쓰기를 무시하고 포함 여부 검사
            if search_keyword.replace(" ", "") in title.replace(" ", "") or \
               search_keyword.replace(" ", "") in artist.replace(" ", ""):
                match_ids.append(all_data["ids"][i])
                match_metas.append(meta)
        
        if match_ids:
            logger.info(f"✅ 직접 매칭 성공! {len(match_ids)}곡 발견")
            return match_ids[:target_count], match_metas[:target_count]

    # [1단계] 필터 생성 및 벡터 검색 준비
    def _build_where(current_intent):
        filter_list = []
        if current_intent.get("bpm_range") == "high": filter_list.append({"bpm": {"$gte": 120}})
        elif current_intent.get("bpm_range") == "low": filter_list.append({"bpm": {"$lte": 100}})
        if current_intent.get("year"):
            try: filter_list.append({"year": int(current_intent["year"])})
            except: pass
        
        if current_intent.get("era") and not current_intent.get("year"):
            nums = re.sub(r'[^0-9]', '', str(current_intent["era"]))
            if nums:
                decade = int(nums)
                start = (1900 + decade if decade >= 50 else 2000 + decade) if decade < 100 else decade
                filter_list.append({"year": {"$gte": start, "$lte": start + 9}})

        if not filter_list: return None
        return {"$and": filter_list} if len(filter_list) > 1 else filter_list[0]

    # 임베딩 생성 (Ollama 사용)
    search_terms = [str(v) for k, v in intent.items() if v and k not in ['count', 'keyword', 'bpm_range']]
    query_text = " ".join(search_terms) if search_terms else (intent.get("keyword") or "music")
    embed_resp = ollama.embeddings(model=EMBED_MODEL, prompt=query_text)
    query_embed = embed_resp.get("embedding") if isinstance(embed_resp, dict) else embed_resp.embedding

    def _do_query(n: int, where_c=None):
        try:
            return collection.query(query_embeddings=[query_embed], n_results=safe_n(n), where=where_c)
        except Exception as e:
            logger.error("쿼리 오류로 필터 없이 재시도: %s", e)
            return collection.query(query_embeddings=[query_embed], n_results=safe_n(n))

    def _unpack(res): return res.get("ids", [[]])[0], res.get("metadatas", [[]])[0]

    logger.info("[1단계] 필터 적용 검색")
    ids, metas = _unpack(_do_query(40, _build_where(intent)))

    if not ids:
        logger.info("[2단계] 순수 벡터 검색")
        ids, metas = _unpack(_do_query(target_count, None))

    if not ids: return [], []

    combined = list(zip(ids, metas))
    sampled = random.sample(combined, min(len(combined), target_count))
    ids_out, meta_out = zip(*sampled)
    return list(ids_out), list(meta_out)

async def search_music(intent: dict) -> tuple[list[str], list[dict]]:
    return await asyncio.to_thread(_search_music_sync, intent)

# ═════════════════════════════════════════════════════════════════════════════════════
#  📨  텔레그램 전송 및 핸들러
# ═════════════════════════════════════════════════════════════════════════════════════
async def send_music_files(context, chat_id, update, file_paths, metadatas, user_query, status_msg):
    success = 0
    total = len(file_paths)
    for i, (path, meta) in enumerate(zip(file_paths, metadatas), start=1):
        display = f"{meta.get('artist', 'Unknown')} - {meta.get('title', 'Unknown')}"
        if not os.path.exists(path): continue
        await safe_edit(status_msg, f"🎧 [{i}/{total}] 전송 중: {display}")
        try:
            with open(path, "rb") as audio:
                msg = await context.bot.send_audio(
                    chat_id=chat_id, audio=audio, 
                    caption=truncate_caption(f"[{i}/{total}] {display}\n🔍 요청: {user_query}"),
                    write_timeout=120, connect_timeout=120
                )
            success += 1
            async with aiosqlite.connect(BOT_DB_PATH) as db:
                await db.execute("INSERT INTO sent_messages (chat_id, message_id, sent_at) VALUES (?, ?, ?)", (chat_id, msg.message_id, time.time()))
                await db.commit()
            if i < total: await asyncio.sleep(2)
        except Exception as e:
            logger.error("전송 실패: %s", e)
    return success

async def handle_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text: return
    user_query = update.message.text.strip()
    chat_id = update.message.chat.id

    if user_query == "ID": return await update.message.reply_text(f"ID: `{chat_id}`", parse_mode="Markdown")
    if user_query == "DB": return await update.message.reply_text(f"DB: {collection.count()}곡")

    await update_user_access(chat_id)
    status_msg = await update.message.reply_text("🤔 요청 분석 및 검색 중...")

    try:
        intent = await extract_intent(user_query)
        file_paths, metadatas = await search_music(intent)
        if not file_paths: return await safe_edit(status_msg, "😢 결과가 없습니다.")
        
        await safe_edit(status_msg, f"🎧 {len(file_paths)}곡을 찾았습니다. 전송 시작!")
        success = await send_music_files(context, chat_id, update, file_paths, metadatas, user_query, status_msg)
        await update.message.reply_text(f"✅ 완료: {success}/{len(file_paths)}곡")
    except Exception as e:
        logger.exception("에러: %s", e)
        await safe_edit(status_msg, "❌ 처리 중 오류가 발생했습니다.")

# ═════════════════════════════════════════════════════════════════════════════════════
#  🚀  진입점
# ═════════════════════════════════════════════════════════════════════════════════════
async def post_init(application):
    await init_db()
    asyncio.create_task(ttl_cleanup_task(application))

if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).post_init(post_init).build()
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_query))
    
    logger.info("🤖 봇 가동 중...")
    app.run_polling()
