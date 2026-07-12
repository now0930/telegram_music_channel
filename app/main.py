"""
music_bot.py — Telegram Music Bot

핵심:
  - agent_tools.py의 search_local_music 스키마 기준으로 동작
  - Hermes가 artist/title을 잘못 추출해도 사용자 원문에서 만든 keyword를 함께 검색
  - 입력이 항상 가수는 아니므로 artist로 강제 추론하지 않음
  - year: -1 제거
  - is_instrumental: "False" 제거
  - alias에 오타를 계속 추가하지 않도록 fuzzy match 적용
  - ChromaDB where_document 사용 안 함
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
from difflib import SequenceMatcher

import ollama
import chromadb
from chromadb.errors import InternalError as ChromaInternalError
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

from agent_tools import music_tools
from artist_aliases import ARTIST_ALIASES

# 최신 구글 생성형 AI 라이브러리
try:
    from google import genai
except ImportError:
    genai = None

# ═════════════════════════════════════════════════════════════════════════════════════
#  📋 로깅 설정
# ═════════════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("music_bot")

# ═════════════════════════════════════════════════════════════════════════════════════
#  ⚙️ 설정
# ═════════════════════════════════════════════════════════════════════════════════════
MUSIC_PATH = "/music"
DB_PATH = "./music_vector_db"

TEXT_MODEL = os.getenv("OLLAMA_MODEL", "hermes3")
EMBED_MODEL = "mxbai-embed-large"

MAX_RESULTS = 10
CAPTION_LIMIT = 1024

BOT_DB_PATH = "./bot_data.db"
MAX_USERS = 10
TTL_SECONDS = 7 * 24 * 60 * 60

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not TELEGRAM_TOKEN:
    logger.critical("❌ 필수 환경 변수 미설정: TELEGRAM_TOKEN")
    sys.exit(1)

GEMINI_MODEL_NAME = "gemini-3.1-flash-lite-preview"

# Gemini 초기화
client_gemini = None

# 현재는 로컬 Hermes 모델만 사용하도록 강제
if False:
    try:
        client_gemini = genai.Client(api_key=GEMINI_API_KEY)
        logger.info(
            f"✅ 최신 Gemini API 설정 완료. "
            f"(모델: {GEMINI_MODEL_NAME}) 하이브리드 엔진 모드로 동작합니다."
        )
    except Exception as e:
        logger.error(f"❌ Gemini 클라이언트 초기화 실패: {e}")
else:
    logger.warning("⚠️ GEMINI_API_KEY가 없거나 라이브러리가 설치되지 않아 Ollama 단독 모드로 동작합니다.")

# ═════════════════════════════════════════════════════════════════════════════════════
#  🗄️ ChromaDB 연결
# ═════════════════════════════════════════════════════════════════════════════════════
try:
    _chroma = chromadb.PersistentClient(path=DB_PATH)
    collection = _chroma.get_or_create_collection(name="music_library")
except ChromaInternalError as e:
    logger.error("ChromaDB 컬렉션 로드 중 에러 발생. 재구성을 시도합니다: %s", e)
    _chroma.delete_collection("music_library")
    collection = _chroma.get_or_create_collection(name="music_library")

# ═════════════════════════════════════════════════════════════════════════════════════
#  🛠️ 유틸리티
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
        await db.execute(
            "INSERT OR REPLACE INTO users (chat_id, last_active) VALUES (?, ?)",
            (chat_id, time.time()),
        )
        cursor = await db.execute(
            """
            SELECT chat_id
            FROM users
            WHERE chat_id NOT IN (
                SELECT chat_id
                FROM users
                ORDER BY last_active DESC
                LIMIT ?
            )
            """,
            (MAX_USERS,),
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
                cursor = await db.execute(
                    """
                    SELECT rowid, chat_id, message_id
                    FROM sent_messages
                    WHERE sent_at < ?
                    """,
                    (cutoff_time,),
                )
                rows = await cursor.fetchall()
                for row in rows:
                    try:
                        await application.bot.delete_message(
                            row["chat_id"],
                            row["message_id"],
                        )
                    except Exception:
                        pass
                    await db.execute(
                        "DELETE FROM sent_messages WHERE rowid = ?",
                        (row["rowid"],),
                    )
                await db.commit()
        except Exception as e:
            logger.error("TTL 정리 루프 에러: %s", e)
        await asyncio.sleep(3600)

async def safe_edit(message, text: str) -> None:
    try:
        await message.edit_text(text)
    except Exception:
        pass

def truncate_caption(text: str, limit: int = CAPTION_LIMIT) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"

def _to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip()

def _to_int(value, default=None):
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        pass
    match = re.search(r"-?\d+", str(value))
    if match:
        try:
            return int(match.group())
        except Exception:
            return default
    return default

def _extract_count_from_query(user_query: str, default=5) -> int:
    match = re.search(r"(\d+)\s*(개|곡|개만|곡만)", user_query)
    if match:
        return int(match.group(1))
    return default

def _clean_keyword_from_query(user_query: str) -> str:
    text = user_query.strip()
    text = re.sub(r"(19\d{2}|20\d{2})\s*년(?:도)?", " ", text)
    text = re.sub(r"\d+\s*(개|곡|개만|곡만)", " ", text)
    remove_words = [
        "노래", "곡", "음악", "들려줘", "찾아줘", "검색", "추천", "추천해줘", "틀어줘", "보내줘", "좀", "만",
    ]
    for word in remove_words:
        text = text.replace(word, " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_intent(intent: dict, user_query: str) -> dict:
    if not isinstance(intent, dict):
        intent = {}
    fixed = dict(intent)

    for key in [
        "title", "artist", "era", "mood", "genre_fixed", "is_instrumental", "keyword", "directory",
    ]:
        if key in fixed:
            fixed[key] = _to_text(fixed.get(key))

    count = _to_int(fixed.get("count"), default=None)
    if not count or count <= 0:
        count = _extract_count_from_query(user_query, default=5)
    fixed["count"] = count

    year = _to_int(fixed.get("year"), default=None)
    if year and year > 0:
        fixed["year"] = year
    else:
        fixed.pop("year", None)

    if fixed.get("is_instrumental") != "True":
        fixed.pop("is_instrumental", None)

    cleaned_keyword = _clean_keyword_from_query(user_query)
    
    _KEYWORD_TO_DIR = {
        "melontop":     "melon_top100",
        "melontop100":  "melon_top100",
        "melon_top100": "melon_top100",
        "melon":        "melon_top100",
        "멜론":          "melon_top100",
        "멜론탑":        "melon_top100",
        "멜론차트":      "melon_top100",
        "탑100":        "melon_top100",
        "top100":       "melon_top100",
        "최신곡":        "melon_top100",
        "인기곡":        "melon_top100",
    }

    # 💡 [수정 포인트 1] LLM이 이미 directory를 찾았더라도 매핑 테이블을 거치게 강제
    current_dir = fixed.get("directory", "").replace(" ", "").lower()
    if current_dir in _KEYWORD_TO_DIR:
        fixed["directory"] = _KEYWORD_TO_DIR[current_dir]

    # LLM이 directory를 못 찾았을 경우 keyword 등에서 유추
    if not fixed.get("directory"):
        for field in ["keyword", "artist", "title"]:
            val = fixed.get(field, "").replace(" ", "").lower()
            if val in _KEYWORD_TO_DIR:
                fixed["directory"] = _KEYWORD_TO_DIR[val]
                fixed.pop(field, None)
                logger.info(f"📁 '{val}' → directory: '{fixed['directory']}' 강제 매핑")
                break

    # 💡 [수정 포인트 2] 폴더나 가수가 명확히 지정되었으면 '멜롬 5' 같은 오타성 키워드는 제거
    if not fixed.get("keyword"):
        # 다른 조건이 아예 없을 때만 원문을 키워드로 사용
        if not fixed.get("directory") and not fixed.get("artist") and not fixed.get("title"):
            fixed["keyword"] = cleaned_keyword or user_query
    else:
        # 만약 디렉토리가 확실한데 keyword에 쓸데없는 오타가 들어있다면 제거
        if fixed.get("directory") and fixed.get("keyword") == user_query:
            fixed.pop("keyword", None)

    cleaned = {}
    for key, value in fixed.items():
        if value is None or value == "" or value == []:
            continue
        cleaned[key] = value

    logger.info(f"🧩 보정된 intent: {cleaned}")
    return cleaned


# ═════════════════════════════════════════════════════════════════════════════════════
#  🤖 LLM 의도 추출
# ═════════════════════════════════════════════════════════════════════════════════════
def _extract_intent_sync(user_query: str) -> dict:
    intent_prompt = (
        f"당신은 전문 음악 큐레이터입니다. 사용자의 요청에서 검색 핵심 정보를 추출하세요.\n\n"
        f"요청: \"{user_query}\"\n\n"
        f"── [추출 규칙] ──\n"
        f"1. '노래', '들려줘', '찾아줘' 같은 무의미한 단어는 제외하고 핵심 검색어만 남기세요.\n"
        f"2. 사용자가 제목이라고 명시하면 title 필드에 넣으세요.\n"
        f"3. 사용자가 가수라고 명시하면 artist 필드에 넣으세요.\n"
        f"4. 가수인지 제목인지 애매한 단어는 keyword 필드에 넣으세요.\n"
        f"5. 사용자가 말하지 않은 가수/제목은 추측하지 마세요.\n"
        f"6. 결과는 반드시 순수한 JSON 형식으로만 응답하세요.\n\n"
        f"── [출력 형식] ──\n"
        f"{{"
        f"\"artist\": \"\", "
        f"\"title\": \"\", "
        f"\"keyword\": \"\", "
        f"\"year\": null, "
        f"\"era\": \"\", "
        f"\"mood\": \"\", "
        f"\"genre_fixed\": \"\", "
        f"\"is_instrumental\": \"\", "
        f"\"directory\": \"\", "
        f"\"count\": 5"
        f"}}"
    )

    if client_gemini:
        try:
            logger.info(f"✨ {GEMINI_MODEL_NAME} 엔진으로 의도 분석 중...")
            response = client_gemini.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=intent_prompt,
            )
            match = re.search(r"\{.*?\}", response.text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.warning(f"⚠️ Gemini 오류 발생, Ollama({TEXT_MODEL})로 전환: {e}")

    try:
        response = ollama.chat(
            model=TEXT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "당신은 시스템 백엔드 API입니다. "
                        "절대로 사용자와 대화하거나 인삿말, 설명을 출력하지 마세요. "
                        "입력이 단어 1개이든 불완전한 문장이든, 무조건 `search_local_music` 도구(Function)만 호출해야 합니다. "
                        "가수인지 곡 제목인지 확실하지 않은 단어는 artist나 title로 단정하지 말고 keyword에 넣으세요. "
                        "사용자가 연주곡을 명시하지 않으면 is_instrumental은 비워두세요. "
                        "연도가 없으면 year를 -1로 넣지 말고 비워두세요."
                    ),
                },
                {
                    "role": "user",
                    "content": user_query,
                },
            ],
            tools=music_tools,
            options={"temperature": 0},
        )

        if response.message.tool_calls:
            tool = response.message.tool_calls[0]
            function_name = tool.function.name
            arguments = tool.function.arguments

            logger.info(f"🛠️ 도구 선택됨: {function_name} | 추출된 파라미터: {arguments}")

            if function_name == "search_local_music":
                if "count" not in arguments or not arguments["count"]:
                    arguments["count"] = 5
                return arguments

        logger.warning(f"⚠️ 에이전트가 도구를 무시함. 모델의 텍스트 대답: {response.message.content}")

    except Exception as e:
        logger.error(f"❌ 에이전트 실행 실패: {e}")

    logger.warning("⚠️ Function Calling 실패. JSON 직접 추출 모드로 전환...")

    try:
        json_response = ollama.chat(
            model=TEXT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a JSON extractor. "
                        "Output ONLY a single JSON object. No text before or after. "
                        "Extract search conditions from the user query."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f'Extract from: "{user_query}"\n\n'
                        f'Output format (JSON only):\n'
                        f'{{"artist":"","title":"","keyword":"{user_query}","year":null,'
                        f'"era":"","mood":"","genre_fixed":"","is_instrumental":"",'
                        f'"count":5,"directory":""}}'
                    ),
                },
            ],
            options={"temperature": 0},
        )

        text = json_response.message.content or ""
        match = re.search(r"\{.*?\}", text, re.DOTALL)

        if match:
            parsed = json.loads(match.group())
            logger.info(f"✅ JSON 직접 추출 성공: {parsed}")
            return parsed

    except Exception as e:
        logger.error(f"❌ JSON 직접 추출도 실패: {e}")

    return {
        "keyword": user_query,
        "count": _extract_count_from_query(user_query, default=5),
    }

async def extract_intent(user_query: str) -> dict:
    raw_intent = await asyncio.to_thread(_extract_intent_sync, user_query)
    return normalize_intent(raw_intent, user_query)

# ═════════════════════════════════════════════════════════════════════════════════════
#  🔍 벡터 검색
# ═════════════════════════════════════════════════════════════════════════════════════
def _search_music_sync(intent: dict) -> tuple[list[str], list[dict]]:
    target_count = intent.get("count", 5)
    db_count = collection.count()

    if db_count <= 0:
        return [], []

    def safe_n(n):
        return max(1, min(n, db_count))

    def _clean_for_match(text: str) -> str:
        return _to_text(text).replace(" ", "").lower()

    def _loose_match(keyword: str, candidates: list[str]) -> bool:
        keyword = _clean_for_match(keyword)
        if not keyword:
            return False

        for candidate in candidates:
            candidate = _clean_for_match(candidate)
            if not candidate:
                continue

            if keyword in candidate or candidate in keyword:
                return True

            if len(keyword) < 4 or len(candidate) < 4:
                continue

            score = SequenceMatcher(None, keyword, candidate).ratio()
            if score >= 0.78:
                return True

        return False

    # ═════════════════════════════════════════════════════════════════════════
    # [0단계] 폴더 alias 포함 직접 하드 매칭
    # ═════════════════════════════════════════════════════════════════════════
    raw_dir = _to_text(intent.get("directory"))

    search_dir_clean = raw_dir.replace(" ", "").lower() if raw_dir else ""
    target_dirs = ARTIST_ALIASES.get(search_dir_clean, [search_dir_clean]) if search_dir_clean else []

    raw_keywords = [
        _to_text(intent.get("title")),
        _to_text(intent.get("artist")),
        _to_text(intent.get("keyword")),
    ]

    search_keywords = []

    for kw in raw_keywords:
        kw_clean = _clean_for_match(kw)
        if not kw_clean:
            continue
        if kw_clean not in search_keywords:
            search_keywords.append(kw_clean)
        for alias in ARTIST_ALIASES.get(kw_clean, []):
            alias_clean = _clean_for_match(alias)
            if alias_clean and alias_clean not in search_keywords:
                search_keywords.append(alias_clean)

    if search_keywords or target_dirs:
        logger.info(f"[0단계] 하드 매칭 시도: 키워드({search_keywords}), 폴더({target_dirs})")

        all_data = collection.get(include=["metadatas"])
        match_ids = []
        match_metas = []

        for i, meta in enumerate(all_data["metadatas"]):
            title = _to_text(meta.get("title"))
            artist = _to_text(meta.get("artist"))
            path = _to_text(meta.get("path"))
            path_clean = _clean_for_match(path)

            is_match = False

            if target_dirs:
                dir_matched = any(d in path_clean for d in target_dirs if d)
                if not dir_matched:
                    continue

            if not search_keywords:
                is_match = True
            else:
                path_parts = [
                    _clean_for_match(part)
                    for part in path.replace("\\", "/").split("/")
                    if part.strip()
                ]
                candidates = [
                    title,
                    artist,
                    path,
                    *path_parts,
                ]
                if any(_loose_match(kw, candidates) for kw in search_keywords):
                    is_match = True

            if is_match:
                match_ids.append(all_data["ids"][i])
                match_metas.append(meta)
        if match_ids:
            logger.info(f"✅ 직접 매칭 성공! {len(match_ids)}곡 발견")
            # 매번 같은 곡만 나오지 않도록 무작위 셔플 후 추출
            combined_match = list(zip(match_ids, match_metas))
            sampled_match = random.sample(combined_match, min(len(combined_match), target_count))
            
            ids_out, meta_out = zip(*sampled_match)
            return list(ids_out), list(meta_out)

    # ═════════════════════════════════════════════════════════════════════════
    # [1단계] ChromaDB 메타데이터 필터 생성 (수정본)
    # ═════════════════════════════════════════════════════════════════════════
    def _build_where(current_intent):
        filter_list = []

        if current_intent.get("year"):
            try:
                year = int(current_intent["year"])
                if year > 0:
                    filter_list.append({"year": year})
            except Exception:
                pass
        elif current_intent.get("era"):
            filter_list.append({"era": current_intent["era"]})

        if current_intent.get("mood"):
            filter_list.append({"mood": current_intent["mood"]})

        if current_intent.get("genre_fixed"):
            filter_list.append({"genre_fixed": current_intent["genre_fixed"]})

        if current_intent.get("is_instrumental") == "True":
            filter_list.append({"is_instrumental": "True"})

        if not filter_list:
            return None

        return {"$and": filter_list} if len(filter_list) > 1 else filter_list[0]

    # ═════════════════════════════════════════════════════════════════════════
    # [2단계] 임베딩 생성 및 벡터 검색
    # ═════════════════════════════════════════════════════════════════════════
    ignore_keys = ["count", "is_instrumental", "directory", "era", "year"]
    search_terms = []

    for key, value in intent.items():
        if not value or key in ignore_keys:
            continue
        text = _to_text(value)
        if text:
            search_terms.append(text)

    query_text = " ".join(search_terms).strip()
    if not query_text:
        query_text = intent.get("keyword") or "음악"

    logger.info(f"🧠 벡터 임베딩 텍스트: {query_text}")
    embed_resp = ollama.embeddings(model=EMBED_MODEL, prompt=query_text)

    query_embed = (
        embed_resp.get("embedding")
        if isinstance(embed_resp, dict)
        else embed_resp.embedding
    )

    def _do_query(n: int, where_c=None):
        try:
            if where_c:
                return collection.query(
                    query_embeddings=[query_embed],
                    n_results=safe_n(n),
                    where=where_c,
                )
            return collection.query(
                query_embeddings=[query_embed],
                n_results=safe_n(n),
            )
        except Exception as e:
            logger.error(f"쿼리 오류로 필터 없이 재시도: {e}")
            return collection.query(
                query_embeddings=[query_embed],
                n_results=safe_n(n),
            )

    def _unpack(res):
        return res.get("ids", [[]])[0], res.get("metadatas", [[]])[0]

    logger.info("[1단계] 필터 적용 벡터 검색")
    ids, metas = _unpack(_do_query(40, _build_where(intent)))

    # 💡 [핵심 수정] 파이썬 메모리 단에서 디렉토리 경로 조건 강제 필터링
    if ids and intent.get("directory"):
        target_dir = intent["directory"].lower()
        filtered_combined = [
            (idx, meta) for idx, meta in zip(ids, metas)
            if target_dir in _to_text(meta.get("path")).lower()
        ]
        
        if filtered_combined:
            ids, metas = zip(*filtered_combined)
            ids, metas = list(ids), list(metas)
        else:
            # 매칭되는 폴더 곡이 하나도 없으면 2단계 전체 검색으로 넘어가기 위해 초기화
            ids, metas = [], []

    if not ids:
        logger.info("[2단계] 순수 벡터 검색 (필터 해제)")
        ids, metas = _unpack(_do_query(target_count, None))

    if not ids:
        return [], []

    # 매번 완전히 같은 곡만 나오는 것 방지
    combined = list(zip(ids, metas))
    sampled = random.sample(combined, min(len(combined), target_count))

    ids_out, meta_out = zip(*sampled)
    return list(ids_out), list(meta_out)

async def search_music(intent: dict) -> tuple[list[str], list[dict]]:
    return await asyncio.to_thread(_search_music_sync, intent)

# ═════════════════════════════════════════════════════════════════════════════════════
#  📨 텔레그램 전송 및 핸들러
# ═════════════════════════════════════════════════════════════════════════════════════
async def send_music_files(context, chat_id, update, file_paths, metadatas, user_query, status_msg):
    success = 0
    total = len(file_paths)

    for i, (path, meta) in enumerate(zip(file_paths, metadatas), start=1):
        display = f"{meta.get('artist', 'Unknown')} - {meta.get('title', 'Unknown')}"

        if not os.path.exists(path):
            logger.warning(f"파일 없음: {path}")
            continue

        await safe_edit(status_msg, f"🎧 [{i}/{total}] 전송 중: {display}")

        try:
            with open(path, "rb") as audio:
                msg = await context.bot.send_audio(
                    chat_id=chat_id,
                    audio=audio,
                    caption=truncate_caption(f"[{i}/{total}] {display}\n🔍 요청: {user_query}"),
                    write_timeout=120,
                    connect_timeout=120,
                )

            success += 1

            async with aiosqlite.connect(BOT_DB_PATH) as db:
                await db.execute(
                    """
                    INSERT INTO sent_messages (chat_id, message_id, sent_at)
                    VALUES (?, ?, ?)
                    """,
                    (chat_id, msg.message_id, time.time()),
                )
                await db.commit()

            if i < total:
                await asyncio.sleep(2)

        except Exception as e:
            logger.error("전송 실패: %s", e)

    return success

async def handle_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    user_query = update.message.text.strip()
    chat_id = update.message.chat.id

    if user_query == "ID":
        return await update.message.reply_text(
            f"ID: `{chat_id}`",
            parse_mode="Markdown",
        )

    if user_query == "DB":
        return await update.message.reply_text(f"DB: {collection.count()}곡")

    await update_user_access(chat_id)

    status_msg = await update.message.reply_text("🤔 요청 분석 및 검색 중...")

    try:
        intent = await extract_intent(user_query)
        file_paths, metadatas = await search_music(intent)

        if not file_paths:
            return await safe_edit(status_msg, "😢 결과가 없습니다.")

        await safe_edit(status_msg, f"🎧 {len(file_paths)}곡을 찾았습니다. 전송 시작!")

        success = await send_music_files(
            context=context,
            chat_id=chat_id,
            update=update,
            file_paths=file_paths,
            metadatas=metadatas,
            user_query=user_query,
            status_msg=status_msg,
        )

        await update.message.reply_text(f"✅ 완료: {success}/{len(file_paths)}곡")

    except Exception as e:
        logger.exception("에러: %s", e)
        await safe_edit(status_msg, "❌ 처리 중 오류가 발생했습니다.")

# ═════════════════════════════════════════════════════════════════════════════════════
#  🚀 진입점
# ═════════════════════════════════════════════════════════════════════════════════════
async def post_init(application):
    await init_db()
    asyncio.create_task(ttl_cleanup_task(application))

if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).post_init(post_init).build()

    app.add_handler(
        MessageHandler(
            filters.TEXT & (~filters.COMMAND),
            handle_query,
        )
    )

    logger.info("🤖 봇 가동 중...")
    app.run_polling()
