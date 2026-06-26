import asyncio
import json
import traceback

from main import collection, extract_intent, search_music


async def run_one_query(user_query: str):
    print("\n" + "━" * 75)
    print(f"💬 질문: {user_query}")

    try:
        # 텔레그램 handle_query()와 동일한 의도 추출
        intent = await extract_intent(user_query)

        print(f"🤖 분석: {json.dumps(intent, ensure_ascii=False)}")

        # 텔레그램 handle_query()와 동일한 검색 함수 호출
        ids, metas = await search_music(intent)

        print(f"🔎 검색 결과 ({len(ids)} 곡):")

        if not ids:
            print("   ❌ 결과 없음")
            return

        for i, (music_id, meta) in enumerate(zip(ids, metas), start=1):
            artist = meta.get("artist", "Unknown")
            title = meta.get("title", "Unknown")
            year = meta.get("year", "?")
            bpm = meta.get("bpm", "?")
            genre = meta.get("genre_fixed") or meta.get("genre") or "?"
            mood = meta.get("mood", "?")
            path = meta.get("path") or music_id

            print(
                f"   [{i}] {artist} - {title} "
                f"({year}년 | BPM: {bpm} | 장르: {genre} | 분위기: {mood})"
            )
            print(f"       path/id: {path}")

    except Exception:
        print("❌ 테스트 중 오류 발생")
        traceback.print_exc()


async def main():
    print("🎵 Telegram 동일 방식 검색 테스트")
    print(f"📦 ChromaDB 곡 수: {collection.count()}")

    test_queries = [
        "비트 빠른 노래 2020년도 4개",
        "90년대 잔잔한 노래 3개",
        "Imagine Dragons 2010 년 노래",
        "제목이 봄날인 노래",
        "록 장르 10 개",
        "악동 뮤직션 노래 3개",
        "아이유 노래 5개",
        "한로로 노래 10개",
        "멜론 10개"
    ]
    test_queries = [
        "멜론 10개"
    ]



    for query in test_queries:
        await run_one_query(query)


if __name__ == "__main__":
    asyncio.run(main())
