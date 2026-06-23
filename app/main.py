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
from agent_tools import music_tools
from artist_aliases import ARTIST_ALIASES  

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
#TEXT_MODEL   = "gemma4:26b"  # 내부 폴백용 모델
TEXT_MODEL   = os.getenv("OLLAMA_MODEL", "hermes3")
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
# Gemini를 끄고 로컬 Hermes 모델만 사용하도록 강제 (False 처리)
if False:
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

    # 2. Hermes 에이전트를 통한 Function Calling (도구 호출)
    try:
        response = ollama.chat(
            model=TEXT_MODEL,
            messages=[
                # 챗봇 자아를 완전히 삭제하는 아주 강력한 시스템 프롬프트
                {
                    'role': 'system',
                    'content': '당신은 시스템 백엔드 API입니다. 절대로 사용자와 대화하거나 인삿말, 설명을 출력하지 마세요. 입력이 단어 1개이든 불완전한 문장이든, 무조건 `search_local_music` 도구(Function)만 호출하여 JSON을 반환해야 합니다.'
                },
                {'role': 'user', 'content': user_query}
            ],
            tools=music_tools,
            options={"temperature": 0}
        )

        # 모델이 도구를 정상적으로 호출한 경우
        if response.message.tool_calls:
            tool = response.message.tool_calls[0]
            function_name = tool.function.name
            arguments = tool.function.arguments

            logger.info(f"🛠️ 도구 선택됨: {function_name} | 추출된 파라미터: {arguments}")

            if function_name == 'search_local_music':
                if 'count' not in arguments or not arguments['count']:
                    arguments['count'] = 5
                return arguments

        # 도구를 호출하지 않고 텍스트로 대답해버린 경우 (원인 파악용 로그 추가)
        else:
            logger.warning(f"⚠️ 에이전트가 도구를 무시함. 모델의 텍스트 대답: {response.message.content}")

    except Exception as e:
        logger.error(f"❌ 에이전트 실행 실패: {e}")

    # 최종 실패 시 폴백
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

    # ═════════════════════════════════════════════════════════════════════════
    # [0단계] 폴더(Alias 포함) 및 텍스트 기반 직접 하드 매칭
    # ═════════════════════════════════════════════════════════════════════════
    search_keyword = intent.get("title") or intent.get("artist") or intent.get("keyword")
    raw_dir = intent.get("directory", "")
    
    # 동의어 사전을 이용한 디렉토리 타겟 리스트 생성
    search_dir_clean = raw_dir.replace(" ", "").lower() if raw_dir else ""
    target_dirs = ARTIST_ALIASES.get(search_dir_clean, [search_dir_clean]) if search_dir_clean else []

    if search_keyword or target_dirs:
        logger.info(f"[0단계] 하드 매칭 시도: 키워드({search_keyword}), 폴더({target_dirs})")
        all_data = collection.get(include=["metadatas"])
        match_ids, match_metas = [], []
        
        for i, meta in enumerate(all_data["metadatas"]):
            title = meta.get("title", "")
            artist = meta.get("artist", "")
            path = meta.get("path", "")
            
            path_clean = path.replace(" ", "").lower()
            title_clean = title.replace(" ", "").lower()
            artist_clean = artist.replace(" ", "").lower()
            
            is_match = False
            
            # 1. 폴더 조건 확인
            if target_dirs:
                dir_matched = any(d in path_clean for d in target_dirs if d)
                if not dir_matched:
                    continue # 폴더 조건이 있는데 안 맞으면 무조건 스킵
                    
            # 2. 키워드 조건 확인
            if not search_keyword:
                is_match = True
            else:
                keyword_clean = search_keyword.replace(" ", "").lower()
                if keyword_clean in title_clean or keyword_clean in artist_clean:
                    is_match = True

            if is_match:
                match_ids.append(all_data["ids"][i])
                match_metas.append(meta)
        
        if match_ids:
            logger.info(f"✅ 직접 매칭 성공! {len(match_ids)}곡 발견")
            return match_ids[:target_count], match_metas[:target_count]

    # ═════════════════════════════════════════════════════════════════════════
    # [1단계] ChromaDB 메타데이터 필터(where) 생성
    # ═════════════════════════════════════════════════════════════════════════
    def _build_where(current_intent):
        filter_list = []
        if current_intent.get("year"):
            try: filter_list.append({"year": int(current_intent["year"])})
            except: pass
        elif current_intent.get("era"):
            filter_list.append({"era": current_intent["era"]})
            
        if current_intent.get("mood"):
            filter_list.append({"mood": current_intent["mood"]})
        if current_intent.get("genre_fixed"):
            filter_list.append({"genre_fixed": current_intent["genre_fixed"]})
        if current_intent.get("is_instrumental"):
            filter_list.append({"is_instrumental": current_intent["is_instrumental"]})
        if current_intent.get("directory"):
            filter_list.append({"path": {"$contains": current_intent["directory"]}})

        if not filter_list: return None
        return {"$and": filter_list} if len(filter_list) > 1 else filter_list[0]

    # ═════════════════════════════════════════════════════════════════════════
    # [2단계] 임베딩 생성 및 벡터 검색 (의미 기반)
    # ═════════════════════════════════════════════════════════════════════════
    # 벡터 검색에 방해되는 시스템 변수(True/False, 숫자 등)는 제외하고 순수 검색어만 뭉침
    ignore_keys = ['count', 'is_instrumental', 'directory', 'era', 'year']
    search_terms = [str(v) for k, v in intent.items() if v and k not in ignore_keys]
    
    # 추출된 검색어가 없으면 일반 키워드 사용
    query_text = " ".join(search_terms) if search_terms else (intent.get("keyword") or "음악")
    logger.info(f"🧠 벡터 임베딩 텍스트: {query_text}")
    
    embed_resp = ollama.embeddings(model=EMBED_MODEL, prompt=query_text)
    query_embed = embed_resp.get("embedding") if isinstance(embed_resp, dict) else embed_resp.embedding

    def _do_query(n: int, where_c=None):
        try:
            return collection.query(query_embeddings=[query_embed], n_results=safe_n(n), where=where_c)
        except Exception as e:
            logger.error(f"쿼리 오류로 필터 없이 재시도: {e}")
            return collection.query(query_embeddings=[query_embed], n_results=safe_n(n))

    def _unpack(res): return res.get("ids", [[]])[0], res.get("metadatas", [[]])[0]

    logger.info("[1단계] 필터 적용 벡터 검색")
    ids, metas = _unpack(_do_query(40, _build_where(intent)))

    if not ids:
        logger.info("[2단계] 순수 벡터 검색 (필터 해제)")
        ids, metas = _unpack(_do_query(target_count, None))

    if not ids: return [], []

    # 랜덤 샘플링 (매번 똑같은 곡만 나오는 것 방지)
    combined = list(zip(ids, metas))
    sampled = random.sample(combined, min(len(combined), target_count))
    ids_out, meta_out = zip(*sampled)
    
    return list(ids_out), list(meta_out)

    # [1단계] 필터 생성 및 벡터 검색 준비
    def _build_where(current_intent):
        filter_list = []
        
        # 1. 연도 필터 (Integer)
        if current_intent.get("year"):
            try: filter_list.append({"year": int(current_intent["year"])})
            except: pass
            
        # 2. 연대 필터 (String - DB에 "1990년대" 형태로 저장되어 있으므로 직접 매칭)
        elif current_intent.get("era"):
            filter_list.append({"era": current_intent["era"]})
            
        # 3. 분위기 필터 (String)
        if current_intent.get("mood"):
            filter_list.append({"mood": current_intent["mood"]})
            
        # 4. 장르 필터 (String)
        if current_intent.get("genre_fixed"):
            filter_list.append({"genre_fixed": current_intent["genre_fixed"]})
            
        # 5. 연주곡 필터 (String "True" / "False")
        if current_intent.get("is_instrumental"):
            filter_list.append({"is_instrumental": current_intent["is_instrumental"]})

        # 기존 _build_where 함수 내부에 아래 3줄을 추가
        if current_intent.get("directory"):
            # 경로 문자열 어딘가에 해당 디렉토리명이 포함되어 있는지 검사
            filter_list.append({"path": {"$contains": current_intent["directory"]}})

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
