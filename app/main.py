"""
music_bot.py — Telegram Music Bot (개선판)
변경 사항:
  - 동기 ollama 호출을 asyncio.to_thread() 로 래핑 → 이벤트 루프 블로킹 제거
  - 전역 예외 핸들러 및 상태 메시지 안전 편집 헬퍼 추가
  - ChromaDB n_results 를 collection.count() 와 비교하여 크래시 방지
  - zip(*sampled) 빈 시퀀스 처리
  - TELEGRAM_TOKEN 환경 변수 검증
  - print() → logging 모듈로 통일
  - Telegram caption 길이 2048 자 제한 대응
  - 검색 재시도(fallback) 단계별 로그 추가
"""

import os
import json
import asyncio
import re
import sys
import random
import logging

import ollama
import chromadb
from telegram import Update
from telegram.error import TelegramError
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

# ══════════════════════════════════════════════════════════════════
#  📋  로깅 설정
# ══════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("music_bot")

# ══════════════════════════════════════════════════════════════════
#  ⚙️  설정
# ══════════════════════════════════════════════════════════════════
MUSIC_PATH   = "/music"
DB_PATH      = "./music_vector_db"
TEXT_MODEL   = "gemma4:e4b"
EMBED_MODEL  = "mxbai-embed-large"
MAX_RESULTS  = 10   # 최대 검색 결과 수
CAPTION_LIMIT = 1024  # Telegram caption 최대 길이 (여유 있게 제한)

OLLAMA_HOST      = os.getenv("OLLAMA_HOST", "http://ollama:11434")
MUSIC_CHANNEL_ID = os.getenv("MUSIC_CHANNEL_ID")
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN")

# 필수 환경 변수 검증
_missing = [name for name, val in [
    ("MUSIC_CHANNEL_ID", MUSIC_CHANNEL_ID),
    ("TELEGRAM_TOKEN",   TELEGRAM_TOKEN),
] if not val]

if _missing:
    logger.critical("❌ 필수 환경 변수 미설정: %s", ", ".join(_missing))
    sys.exit(1)

try:
    MUSIC_CHANNEL_ID = int(MUSIC_CHANNEL_ID)
except ValueError:
    pass  # 문자열 채널 ID 도 허용 (예: @channelname)

# ══════════════════════════════════════════════════════════════════
#  🗄️  DB 연결
# ══════════════════════════════════════════════════════════════════
_chroma    = chromadb.PersistentClient(path=DB_PATH)
collection = _chroma.get_or_create_collection(name="music_library")

# ══════════════════════════════════════════════════════════════════
#  🛠️  유틸리티
# ══════════════════════════════════════════════════════════════════
async def safe_edit(message, text: str) -> None:
    """status 메시지 편집 실패 시 예외를 무시하고 로그만 남긴다."""
    try:
        await message.edit_text(text)
    except TelegramError as e:
        logger.warning("status 메시지 편집 실패 (무시): %s", e)


def truncate_caption(text: str, limit: int = CAPTION_LIMIT) -> str:
    """Telegram caption 길이 제한을 초과하면 말줄임표를 붙여 자른다."""
    return text if len(text) <= limit else text[: limit - 1] + "…"

# ══════════════════════════════════════════════════════════════════
#  🤖  LLM 의도 추출 (비동기 래핑)
# ══════════════════════════════════════════════════════════════════
def _extract_intent_sync(user_query: str) -> dict:
    """
    동기 함수. asyncio.to_thread() 를 통해 호출된다.
    사용자 요청을 분석하여 DB 쿼리용 JSON 구조로 변환한다.
    """
    intent_prompt = (
        f'사용자의 음악 검색 요청을 분석하여 JSON 으로 변환하세요.\n'
        f'요청: "{user_query}"\n\n'
        f'── [필드 지침] ──\n'
        f'1. artist, title, album, genre: 명시된 경우만 추출\n'
        f'2. year: 연도 숫자만 추출 (예: 2020)\n'
        f'3. era: "90 년대", "2000 년대" 등 시대적 배경\n'
        f'4. bpm_range: "빠른", "신나는" -> "high" / "느린", "잔잔한" -> "low"\n'
        f'5. count: 결과 개수 (기본 5)\n\n'
        f'── [주의] ──\n'
        f'- 질문에 없는 가수는 절대 추측하지 마세요.\n'
        f'출력 (JSON 한 줄):'
    )

    try:
        resp = ollama.generate(model=TEXT_MODEL, prompt=intent_prompt, stream=False, options={"temperature": 0})
        raw = resp.get("response", "").strip()
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            intent = json.loads(match.group())
            intent["count"] = min(max(int(intent.get("count", 5)), 1), MAX_RESULTS)
            
            # 🖍️ 연도 데이터 타입 보정 및 추출
            if intent.get("year"):
                try: intent["year"] = int(re.sub(r'[^0-9]', '', str(intent["year"])))
                except: pass
                
            return intent
        logger.warning("LLM 응답에서 JSON 을 찾을 수 없음: %s", raw[:200])
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("LLM 응답 파싱 실패: %s", e)
    except Exception as e:
        logger.error("LLM 호출 에러: %s", e)

    return {"keyword": user_query, "count": 5}


async def extract_intent(user_query: str) -> dict:
    """동기 LLM 호출을 별도 스레드에서 실행하여 이벤트 루프를 블로킹하지 않는다."""
    return await asyncio.to_thread(_extract_intent_sync, user_query)

# ══════════════════════════════════════════════════════════════════
#  🔍  폴백 로직 포함 벡터 검색 (비동기 래핑)
# ══════════════════════════════════════════════════════════════════
def _search_music_sync(intent: dict) -> tuple[list[str], list[dict]]:
    """
    동기 함수. asyncio.to_thread() 를 통해 호출된다.
    단계적 필터 완화 폴백 및 문서 필터링 포함 벡터 검색.
    """
    target_count = intent.get("count", 5)
    db_count     = collection.count()

    def safe_n(n): return max(1, min(n, db_count))

    def _build_where(current_intent):
        filter_list = []
        
        # ✅ bpm_range 논리를 DB 의 실제 키 'bpm' 과 숫자 범위로 변환
        if current_intent.get("bpm_range") == "high":
            filter_list.append({"bpm": {"$gte": 120}}) # 120 이상
        elif current_intent.get("bpm_range") == "low":
            filter_list.append({"bpm": {"$lte": 100}}) # 100 이하
            
        # 연도(year) 필터링
        if current_intent.get("year"):
            filter_list.append({"year": current_intent["year"]})

        # 🖍️ 시대(era) 필터링 (예: "90 년대" -> 1990 ~ 1999, "2010 년대" -> 2010 ~ 2019)
        if current_intent.get("era") and not current_intent.get("year"):
            era_str = current_intent.get("era")
            numbers = re.sub(r'[^0-9]', '', str(era_str))
            if numbers:
                decade = int(numbers)
                # 2 자리 숫자이면 1900/2000 을 더하고, 4 자리 숫자이면 그대로 사용
                if decade < 100:
                    start_year = 1900 + decade if decade >= 50 else 2000 + decade
                else:
                    start_year = decade
                
                end_year = start_year + 9
                filter_list.append({"year": {"$gte": start_year}})
                filter_list.append({"year": {"$lte": end_year}})

        # 부분 일치 검색 ($contains)
        for key in ["title", "album", "genre"]:
            if current_intent.get(key):
                filter_list.append({key: {"$contains": current_intent[key]}})

        if not filter_list: return None
        return {"$and": filter_list} if len(filter_list) > 1 else filter_list[0]

    # 임베딩 쿼리 생성
    search_terms = [str(v) for k, v in intent.items() if v and k not in ['count', 'keyword', 'bpm_range']]
    if intent.get("bpm_range") == "high": search_terms.append("fast exciting dance")
    if intent.get("bpm_range") == "low": search_terms.append("slow calm ballad")
    
    query_text = " ".join(search_terms) if search_terms else (intent.get("keyword") or "music")
    embed_resp  = ollama.embeddings(model=EMBED_MODEL, prompt=query_text)
    query_embed = (
        embed_resp.get("embedding")
        if isinstance(embed_resp, dict)
        else embed_resp.embedding
    )

    # 🖍️ 핵심 키워드(아티스트명, 제목) 가 문서에 포함된 것만 찾는 필터 생성 (오타 방지)
    doc_search_keyword = intent.get("artist") or intent.get("title") or query_text
    keywords = re.findall(r'[가-힣a-zA-Z]{2,}', doc_search_keyword)
    doc_filter = {"$contains": keywords[0]} if keywords else None

    def _do_query(n: int, where_c=None, where_doc=None):
        return collection.query(
            query_embeddings=[query_embed],
            n_results=safe_n(n),
            where=where_c,
            where_document=where_doc
        )

    def _unpack(results: dict) -> tuple[list, list]:
        return results.get("ids", [[]])[0], results.get("metadatas", [[]])[0]

    # 1 단계: 전체 필터 적용
    logger.info("[검색 1 단계] 전체 필터 적용")
    ids, metadatas = _unpack(_do_query(40, _build_where(intent), doc_filter))

    # 2 단계: 결과 부족 시 조건 완화 (BPM, Year 제외)
    if not ids and (intent.get("bpm_range") or intent.get("year")):
        logger.info("[검색 2 단계] 조건 완화 후 재검색")
        relaxed_intent = {k: v for k, v in intent.items() if k in ["artist", "title", "era"]}
        ids, metadatas = _unpack(_do_query(40, _build_where(relaxed_intent), doc_filter))

    # 3 단계: 연도/시대 필터가 있었는데 결과가 없다면, 관련 없는 시대의 노래 추천 방지
    if not ids and (intent.get("year") or intent.get("era")):
        logger.info("[검색 3 단계] 요청하신 연도/시대 곡 없음")
        return [], []

    # 4 단계: 순수 벡터 검색 (문서 필터만 유지)
    if not ids:
        logger.info("[검색 4 단계] 순수 벡터 검색 (최종 폴백)")
        ids, metadatas = _unpack(_do_query(target_count, None, doc_filter))

    if not ids:
        logger.warning("모든 검색 단계에서 결과 없음")
        return [], []

    # 결과 다양성을 위한 랜덤 샘플링
    if len(ids) > target_count:
        combined = list(zip(ids, metadatas))
        sampled  = random.sample(combined, target_count)
        ids_out, meta_out = zip(*sampled)
        return list(ids_out), list(meta_out)

    return list(ids), list(metadatas)


async def search_music(intent: dict) -> tuple[list[str], list[dict]]:
    """동기 검색 함수를 별도 스레드에서 실행한다."""
    return await asyncio.to_thread(_search_music_sync, intent)

# ══════════════════════════════════════════════════════════════════
#  📨  텔레그램 전송
# ══════════════════════════════════════════════════════════════════
async def send_music_files(
    context: ContextTypes.DEFAULT_TYPE,
    update: Update,
    file_paths: list[str],
    metadatas: list[dict],
    user_query: str,
) -> int:
    """음악 파일을 MUSIC_CHANNEL_ID 채널로 전송하고 성공 건수를 반환한다."""
    success = 0
    errors  = []
    total   = len(file_paths)

    for i, (path, meta) in enumerate(zip(file_paths, metadatas), start=1):
        artist  = meta.get("artist", "Unknown")
        title   = meta.get("title",  "Unknown")
        display = f"{artist} - {title}"

        if not os.path.exists(path):
            logger.warning("파일 없음: %s", path)
            errors.append(display)
            continue

        caption = truncate_caption(
            f"[{i}/{total}] {display}\n🔍 요청: {user_query}"
        )

        try:
            with open(path, "rb") as audio:
                await context.bot.send_audio(
                    chat_id=MUSIC_CHANNEL_ID,
                    audio=audio,
                    caption=caption,
                )
            success += 1
            logger.info("[%d/%d] 전송 완료: %s", i, total, display)
            await asyncio.sleep(1.0)
        except TelegramError as e:
            logger.error("전송 실패 [%s]: %s", display, e)
            errors.append(display)
        except OSError as e:
            logger.error("파일 읽기 오류 [%s]: %s", path, e)
            errors.append(display)

    if errors:
        preview = ", ".join(errors[:3])
        suffix  = f" 외 {len(errors) - 3}곡" if len(errors) > 3 else ""
        await update.message.reply_text(f"⚠️ 전송 실패: {preview}{suffix}")

    return success

# ══════════════════════════════════════════════════════════════════
#  🤖  텔레그램 메시지 핸들러
# ══════════════════════════════════════════════════════════════════
async def handle_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    user_query = update.message.text.strip()
    chat_id    = update.message.chat.id

    # ── 특수 명령어 ──
    if user_query == "ID":
        await update.message.reply_text(f"ID: `{chat_id}`", parse_mode="Markdown")
        return
    if user_query == "DB":
        await update.message.reply_text(f"DB: {collection.count()}곡")
        return

    status_msg = await update.message.reply_text("🤔 요청 분석 및 검색 중. ..")

    try:
        # 1 단계: LLM 의도 추출
        intent = await extract_intent(user_query)
        logger.info("추출된 intent: %s", intent)

        # 2 단계: 벡터 검색
        file_paths, metadatas = await search_music(intent)

        if not file_paths:
            await safe_edit(status_msg, "😢 검색 결과가 없습니다.")
            return

        await safe_edit(status_msg, f"🎧 {len(file_paths)}곡을 찾았습니다. 전송을 시작합니다!")

        # 3 단계: 파일 전송
        success = await send_music_files(context, update, file_paths, metadatas, user_query)
        await update.message.reply_text(f"✅ 전송 완료: {success}/{len(file_paths)}곡")

    except Exception as e:
        logger.exception("handle_query 처리 중 예외 발생: %s", e)
        await safe_edit(status_msg, "❌ 처리 중 오류가 발생했습니다. 다시 시도해 주세요.")

# ══════════════════════════════════════════════════════════════════
#  🚀  진입점
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_query))
    logger.info("🤖 봇 가동 중... (DB: %d곡)", collection.count())
    app.run_polling()
