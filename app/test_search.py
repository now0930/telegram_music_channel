import os
import json
import re
import ollama
from main import collection, EMBED_MODEL, TEXT_MODEL

# 1. AI 의도 추출: 추상적 요청을 논리적 카테고리로 변환
def extract_intent_refined(user_query: str):
    intent_prompt = (
        f'사용자의 음악 검색 요청을 분석하여 JSON으로 변환하세요.\n'
        f'요청: "{user_query}"\n\n'
        f'── [필드 지침] ──\n'
        f'1. artist, title, album, genre: 명시된 경우만 추출\n'
        f'2. year: 연도 숫자만 추출 (예: 2020)\n'
        f'3. era: "90년대", "2000년대" 등 시대적 배경\n'
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
            # 연도 데이터 타입 보정 (Integer)
            if intent.get("year"):
                try: intent["year"] = int(re.sub(r'[^0-9]', '', str(intent["year"])))
                except: pass
                
            if not intent.get("count"): intent["count"] = 5
            return intent
    except Exception as e:
        print(f"❌ AI 분석 에러: {e}")
    
    return {"keyword": user_query, "count": 5}

# 2. 최적화된 검색 로직 (bpm_range -> bpm 숫자 변환 포함)
def search_music_optimized(intent: dict):
    target_count = intent.get("count", 5)
    
    def _build_where(current_intent):
        filter_list = []
        
        # ✅ 핵심: bpm_range 논리를 DB 의 실제 키 'bpm' 과 숫자 범위로 변환
        if current_intent.get("bpm_range") == "high":
            filter_list.append({"bpm": {"$gte": 120}}) # 120 이상
        elif current_intent.get("bpm_range") == "low":
            filter_list.append({"bpm": {"$lte": 100}}) # 100 이하
            
        # 연도(year) 필터링
        if current_intent.get("year"):
            filter_list.append({"year": current_intent["year"]})

        # 🖍️ 시대(era) 필터링 (예: "90년대" -> 1990 ~ 1999, "2010년대" -> 2010 ~ 2019)
        if current_intent.get("era") and not current_intent.get("year"):
            era_str = current_intent.get("era")
            numbers = re.sub(r'[^0-9]', '', str(era_str))
            if numbers:
                decade = int(numbers)
                
                # 2자리 숫자(예: 90)이면 1900이나 2000을 더하고, 4자리 숫자(예: 2010)이면 그대로 사용
                if decade < 100:
                    start_year = 1900 + decade if decade >= 50 else 2000 + decade
                else:
                    start_year = decade
                
                end_year = start_year + 9
                
                filter_list.append({"year": {"$gte": start_year}})
                filter_list.append({"year": {"$lte": end_year}})

        # 🖍️ 아티스트 필터링 (SQL 의 LIKE 처럼 부분 일치 검색)
        # (아티스트 검색은 아래 where_document 를 통해 오타에 강하게 처리합니다)
        # 부분 일치 검색 ($contains)
        # 🖍️ "era"를 목록에서 제거하세요. (DB 에 era 컬럼이 없으므로 오류/무효화 방지)
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
    embed_resp = ollama.embeddings(model=EMBED_MODEL, prompt=query_text)

    # 🖍️ 핵심 키워드(아티스트명, 제목)가 문서에 포함된 것만 찾는 필터 생성 (오타 방지)
    doc_search_keyword = intent.get("artist") or intent.get("title") or query_text
    # 2 글자 이상의 한글/영어 단어만 추출 (예: "악동 뮤직션" -> "악동")
    keywords = re.findall(r'[가-힣a-zA-Z]{2,}', doc_search_keyword)
    
    doc_filter = None
    if keywords:
        doc_filter = {"$contains": keywords[0]} # 첫 번째 핵심 키워드 사용

    # [1 단계] 필터 적용 검색
    print(f"   🔍 필터 검색 시도 (범위: {intent.get('bpm_range', 'ALL')})...")
    where_clause = _build_where(intent)
    results = collection.query(
        query_embeddings=[embed_resp.embedding], 
        n_results=40, 
        where=where_clause, 
        where_document=doc_filter # 🖍️ 문서 필터 추가
    )
    ids, metas = results.get("ids", [[]])[0], results.get("metadatas", [[]])[0]

    # [2 단계] 결과 부족 시 조건 완화 (BPM, Year 제외)
    if not ids and (intent.get("bpm_range") or intent.get("year")):
        print(f"   ⚠️ 조건 완화 후 재검색...")
        relaxed_intent = {k: v for k, v in intent.items() if k in ["artist", "title", "era"]}
        where_clause = _build_where(relaxed_intent)
        results = collection.query(
            query_embeddings=[embed_resp.embedding], 
            n_results=40, 
            where=where_clause, 
            where_document=doc_filter # 🖍️ 문서 필터 유지
        )
        ids, metas = results.get("ids", [[]])[0], results.get("metadatas", [[]])[0]

    # [3 단계] 최종 폴백: 벡터 검색
    # 연도/시대 필터가 있었는데 결과가 없다면, 관련 없는 시대의 노래를 추천하지 않도록 폴백을 중단합니다.
    if not ids and (intent.get("year") or intent.get("era")):
        print("   ⚠️ 요청하신 연도/시대의 곡이 데이터베이스에 없습니다.")
        return [], []
        
    if not ids:
        print(f"   ⚠️ 벡터 검색 수행 (키워드: {query_text})...")
        # 이때도 doc_filter 를 적용하여 최소한 가수나 제목이 언급된 문서 내에서 찾도록 합니다.
        results = collection.query(
            query_embeddings=[embed_resp.embedding], 
            n_results=target_count, 
            where_document=doc_filter # 🖍️ 문서 필터 유지
        )
        ids, metas = results.get("ids", [[]])[0], results.get("metadatas", [[]])[0]

    return ids[:target_count], metas[:target_count]

# 3. 테스트 실행
if __name__ == "__main__":
    test_queries = [
        "비트 빠른 노래 2020년도 4개",
        "90년대 잔잔한 노래 3개",
        "Imagine Dragons 2010 년 노래",
        "제목이 봄날인 노래",
        "록 장르 10 개",
        "악동 뮤직션 노래 3개",
        "아이유 노래 5개"

    ]
    
    for q in test_queries:
        print(f"\n" + "━"*75)
        print(f"💬 질문: {q}")
        intent = extract_intent_refined(q)
        print(f"🤖 분석: {json.dumps(intent, ensure_ascii=False)}")
        
        ids, metas = search_music_optimized(intent)
        
        print(f"🔎 검색 결과 ({len(ids)} 곡):")
        if not ids:
            print("   ❌ 결과 없음")
        else:
            for i, (p, m) in enumerate(zip(ids, metas)):
                print(f"   [{i+1}] {m.get('artist')} - {m.get('title')} ({m.get('year')}년 | BPM: {m.get('bpm')})")
