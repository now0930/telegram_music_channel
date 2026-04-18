"""
┌──────────────────────────────────────────────────────────────────────────┐
│  🤖  ai_inferrer.py  —  LLM 기반 메타데이터 추론 도구                    │
│                                                                            │
│  목적                                                                      │
│    metadata_fixer.py에서 해결하지 못한 누락 데이터(year='0', genre='')를   │
│    로컬 Ollama LLM을 사용하여 배치 단위로 추론합니다.                      │
│                                                                            │
│  실행 예시                                                                 │
│    python ai_inferrer.py                   # 추론 및 DB 업데이트 실행      │
│    python ai_inferrer.py --dry-run         # 미리보기 (DB 미수정)          │
│    python ai_inferrer.py --batch-size 10   # 배치 사이즈 변경 (기본 15)    │
└──────────────────────────────────────────────────────────────────────────┐
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Generator, Optional

import chromadb
import requests
import xml.etree.ElementTree as ET

# ══════════════════════════════════════════════════════════════════════════
#  ⚙️  전역 설정
# ══════════════════════════════════════════════════════════════════════════
DB_PATH    = "./music_vector_db"
COLLECTION = "music_library"

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
FALLBACK_HOSTS = ["http://host.docker.internal:11434", "http://localhost:11434"]
ACTIVE_OLLAMA_URL = None  # Will be set after health check

# Ollama에 설치된 모델명
OLLAMA_MODEL   = "gemma4:e4b" 

# ══════════════════════════════════════════════════════════════════════════
#  🎨  컬러 출력 헬퍼
# ══════════════════════════════════════════════════════════════════════════
GREEN  = "\033[92m"; YELLOW = "\033[93m"; RED    = "\033[91m"
RESET  = "\033[0m";  BOLD   = "\033[1m";  DIM    = "\033[2m"

def ok(msg: str)      -> None: print(f"  {GREEN}✅  {msg}{RESET}")
def warn(msg: str)    -> None: print(f"  {YELLOW}⚠️   {msg}{RESET}")
def err(msg: str)     -> None: print(f"  {RED}❌  {msg}{RESET}")
def info(msg: str)    -> None: print(f"  {DIM}ℹ️   {msg}{RESET}")

# ══════════════════════════════════════════════════════════════════════════
#  🗄️  DB 연결 및 조회
# ══════════════════════════════════════════════════════════════════════════
def check_ollama_status() -> None:
    """Ollama 서버 연결 상태를 확인하고 활성화된 API URL을 설정합니다."""
    global ACTIVE_OLLAMA_URL
    candidates = [OLLAMA_HOST] + FALLBACK_HOSTS
    
    for host in candidates:
        url = f"{host}/api/tags"
        info(f"Ollama 서버 연결 시도 중: {host} ...")
        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                ok(f"Ollama 서버 연결 성공: {host}")
                ACTIVE_OLLAMA_URL = f"{host}/api/generate"
                return
        except requests.exceptions.RequestException:
            continue
            
    err("Ollama 서버를 찾을 수 없습니다. 주소와 컨테이너 상태를 확인하세요.")
    sys.exit(1)

def connect_db() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(name=COLLECTION)

def get_unresolved_tracks(col: chromadb.Collection) -> list[dict]:
    """year가 '0'(또는 비어있거나 None) 이거나 genre가 비어있는 트랙 조회"""
    info("DB에서 미해결 트랙을 조회하는 중...")
    all_data = col.get(include=["metadatas"])
    
    unresolved = []
    for doc_id, meta in zip(all_data["ids"], all_data["metadatas"]):
        year_str = str(meta.get("year", "0")).strip().lower()
        genre_str = str(meta.get("genre", "")).strip().lower()
        
        is_year_bad = (year_str in ("0", "", "none", "unknown"))
        is_genre_bad = (genre_str in ("", "none", "unknown"))
        
        if is_year_bad or is_genre_bad:
            unresolved.append({
                "id": doc_id,
                "artist": meta.get("artist", "Unknown Artist"),
                "title": meta.get("title", "Unknown Title"),
                "needs_year": is_year_bad,
                "needs_genre": is_genre_bad
            })
            
    info(f"총 {len(unresolved)}곡의 미해결 데이터 발견.")
    return unresolved

# ══════════════════════════════════════════════════════════════════════════
#  📦  배치 제너레이터
# ══════════════════════════════════════════════════════════════════════════
def batch_generator(data: list[Any], batch_size: int = 15) -> Generator[list[Any], None, None]:
    """데이터를 배치 사이즈만큼 묶어서 반환하는 제너레이터"""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

# ══════════════════════════════════════════════════════════════════════════
#  🧠  Ollama LLM 추론
# ══════════════════════════════════════════════════════════════════════════
def fetch_musicbrainz(artist: str, title: str) -> dict:
    """MusicBrainz API에서 메타데이터 검색"""
    try:
        url = f"https://musicbrainz.org/ws/2/recording/?query=artist:{artist}+recording:{title}&fmt=json"
        headers = {"User-Agent": "MetadataFixer/1.0 (ai-inferrer-script)"}
        res = requests.get(url, timeout=10, headers=headers)
        if res.status_code == 200:
            data = res.json().get("recordings", [])
            if data:
                result = {}
                first_hit = data[0]
                # 연도 추출
                if first_hit.get("first-release-date"):
                    result["year"] = first_hit["first-release-date"][:4]
                # 장르 추출 (tags 사용)
                tags = first_hit.get("tags", [])
                if tags:
                    result["genre"] = tags[0].get("name", "").capitalize()
                return result
    except Exception:
        pass
    return {}

def fetch_maniadb(artist: str, title: str) -> dict:
    """ManiaDB API에서 메타데이터 검색"""
    try:
        query = requests.utils.quote(f"{artist} {title}")
        url = f"http://www.maniadb.com/api/search/{query}/?sr=recording&display=10&v=0.5"
        res = requests.get(url, timeout=10)
        if res.status_code == 200 and res.text:
            root = ET.fromstring(res.text)
            item = root.find(".//item")
            if item is not None:
                result = {}
                # ManiaDB XML 파싱
                release_group = item.find("releasegroup")
                if release_group is not None and release_group.text:
                    result["year"] = release_group.text[:4]
                
                genres = item.findall(".//genre")
                if genres:
                    result["genre"] = genres[0].text.capitalize()
                return result
    except Exception:
        pass
    return {}

def search_track_info(artist: str, title: str) -> dict:
    """ManiaDB 및 MusicBrainz에서 메타데이터 검색하여 가능한 결과 반환"""
    result = {}
    
    # ManiaDB 시도
    try:
        maniadb_data = fetch_maniadb(artist, title) # metadata_fixer의 함수명에 맞게 수정
        if maniadb_data:
            if not result.get('year') and maniadb_data.get('year'): result['year'] = maniadb_data['year']
            if not result.get('genre') and maniadb_data.get('genre'): result['genre'] = maniadb_data['genre']
    except Exception:
        pass
        
    if result.get('year') and result.get('genre'):
        return result # 둘 다 찾았으면 조기 반환
        
    # MusicBrainz 시도
    try:
        mb_data = fetch_musicbrainz(artist, title) # metadata_fixer의 함수명에 맞게 수정
        if mb_data:
            if not result.get('year') and mb_data.get('year'): result['year'] = mb_data['year']
            if not result.get('genre') and mb_data.get('genre'): result['genre'] = mb_data['genre']
    except Exception:
        pass
        
    return result

def build_prompt(batch: list[dict]) -> str:
    """LLM에게 전달할 프롬프트 생성 (검색으로 알아낸 일부 정보 포함)"""
    track_list_str = ""
    for i, track in enumerate(batch, 1):
        partial = track.get('partial_info', {})
        partial_str = ""
        if partial:
            parts = []
            if 'year' in partial: parts.append(f"연도: {partial['year']}")
            if 'genre' in partial: parts.append(f"장르: {partial['genre']}")
            if parts:
                partial_str = f" (검색 결과: {', '.join(parts)} - 나머지 추론 필요)"
            else:
                partial_str = " (검색결과 없음 -> 추론 필요)"
        else:
             partial_str = " (검색결과 없음 -> 추론 필요)"
             
        track_list_str += f"{i}. ID: {track['id']}, Artist: {track['artist']}, Title: {track['title']}{partial_str}\n"
    
    prompt = f"""인터넷 검색으로 찾지 못한 곡들입니다. 아티스트의 스타일을 고려해 추론해 주세요.
아래 제공된 {len(batch)}곡의 리스트를 보고 각각의 '발매연도(YYYY)'와 '대표 장르'를 추론해서 JSON 배열로만 응답해. 다른 설명은 생략해.
리스트:
{track_list_str}
출력 형식 (반드시 아래와 같은 JSON 배열 형식만 출력할 것):
[
  {{"id": "입력된ID", "year": "추론된연도", "genre": "추론된장르"}}
]
JSON 출력:"""
    return prompt

def query_ollama(prompt: str) -> Optional[list[dict]]:
    """Ollama API를 호출하여 JSON 형태의 추론 결과 반환"""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "format": "json",
        "stream": False
    }
    
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(ACTIVE_OLLAMA_URL, json=payload, timeout=180)
            response.raise_for_status()
            
            # Ollama 응답 파싱
            raw_response = response.json().get("response", "")
            
            # JSON 문자열 파싱 (마크다운 코드블럭 ```json ... ``` 제거 로직 포함)
            # gemma 모델 등에서 마크다운을 포함할 수 있으므로 정규식으로 추출
            json_match = re.search(r'```json\s*(\[[\s\S]*?\])\s*```', raw_response)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 마크다운이 없는 경우 첫 번째 '[' 와 마지막 ']' 사이의 문자열 추출
                start_idx = raw_response.find('[')
                end_idx = raw_response.rfind(']')
                if start_idx != -1 and end_idx != -1:
                    json_str = raw_response[start_idx:end_idx+1]
                else:
                    json_str = ""
                    
            if json_str:
                return json.loads(json_str)
            else:
                warn("LLM 응답에서 유효한 JSON 배열을 찾을 수 없습니다.")
                warn(f"원본 응답: {raw_response[:200]}")
                return None
                
        except requests.exceptions.ConnectionError as e:
            err(f"연결 거부/오류 발생 (시도 {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                info("5초 후 재시도합니다...")
                time.sleep(5)
            else:
                err("최대 재시도 횟수를 초과했습니다.")
                return None
                
        except requests.exceptions.RequestException as e:
            err(f"Ollama API 호출 실패: {e}")
            return None
            
        except json.JSONDecodeError:
            err("LLM 응답을 JSON으로 파싱할 수 없습니다.")
            return None
            
    return None

# ══════════════════════════════════════════════════════════════════════════
#  ✅  데이터 검증 및 DB 업데이트
# ══════════════════════════════════════════════════════════════════════════
def is_valid_year(year_str: str) -> bool:
    """추론된 연도가 유효한 4자리 숫자인지 검증"""
    return bool(re.match(r"^(19|20)\d{2}$", str(year_str)))

def is_valid_genre(genre_str: str) -> bool:
    """추론된 장르가 유효한 문자열인지 검증"""
    return bool(genre_str and len(genre_str) > 1 and genre_str.lower() not in ("unknown", "none", "n/a"))

def process_and_update(col: chromadb.Collection, batch: list[dict], inferred_data: list[dict], dry_run: bool, source: str = "추론성공") -> int:
    """추론/검색 결과를 검증하고 DB를 업데이트"""
    inferred_map = {item["id"]: item for item in inferred_data}
    emoji = "🔍" if source == "검색성공" else "🧠"
    
    updated_count = 0
    for track in batch:
        track_id = track["id"]
        inference = inferred_map.get(track_id)
        
        if not inference:
            continue
            
        update_payload = {}
        
        if track["needs_year"]:
            y = str(inference.get("year", ""))
            if is_valid_year(y):
                update_payload["year"] = y
                update_payload["era"] = f"{(int(y) // 10) * 10}년대" 
      
    
        if track["needs_genre"]:
            g = str(inference.get("genre", ""))
            if is_valid_genre(g):
                update_payload["genre"] = g
                update_payload["genre_fixed"] = g
                
        if update_payload:
            print(f"  {BOLD}{emoji} [{source}] {track['artist']} - {track['title']}{RESET}")
            print(f"     업데이트값: {update_payload}")
            if not dry_run:
                col.update(ids=[track_id], metadatas=[update_payload])
            updated_count += 1
            
    return updated_count

# ══════════════════════════════════════════════════════════════════════════
#  🚀  메인 실행
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 기반 메타데이터 추론 도구")
    parser.add_argument("--dry-run", action="store_true", help="미리보기 (DB 미수정)")
    parser.add_argument("--batch-size", type=int, default=15, help="배치 처리 사이즈 (기본값: 15)")
    args = parser.parse_args()

    print(f"\n{BOLD}{'═'*68}{RESET}")
    print(f"{BOLD}  🤖 LLM 기반 메타데이터 추론 시작{RESET}")
    print(f"{BOLD}{'═'*68}{RESET}")
    
    check_ollama_status() # 사전 점검 및 ACTIVE_OLLAMA_URL 설정
    
    info(f"사용 모델: {OLLAMA_MODEL}")
    info(f"활성 Ollama API: {ACTIVE_OLLAMA_URL}")

    col = connect_db()
    unresolved_tracks = get_unresolved_tracks(col)
    
    if not unresolved_tracks:
        ok("해결할 누락 데이터가 없습니다. 종료합니다.")
        sys.exit(0)

    total_tracks = len(unresolved_tracks)
    processed_count = 0
    success_count = 0

    for batch in batch_generator(unresolved_tracks, args.batch_size):
        batch_num = processed_count + len(batch)
        info(f"\n배치 처리 중... ({batch_num}/{total_tracks})")
        
        search_resolved_batch = []
        llm_required_batch = []
        
        # 1단계: 검색 우선 시도 (Search First)
        for track in batch:
            search_data = search_track_info(track['artist'], track['title'])
            
            fully_resolved = True
            partial_info = {}
            
            if track['needs_year']:
                if search_data.get('year') and is_valid_year(search_data['year']):
                    partial_info['year'] = search_data['year']
                else:
                    fully_resolved = False
                    
            if track['needs_genre']:
                if search_data.get('genre') and is_valid_genre(search_data['genre']):
                    partial_info['genre'] = search_data['genre']
                else:
                    fully_resolved = False
                    
            if fully_resolved and (track['needs_year'] or track['needs_genre']):
                # 검색으로 모든 누락 데이터를 찾음
                search_resolved_batch.append({**track, 'inference': partial_info})
            else:
                # 검색으로 못 찾거나 일부만 찾은 데이터 (LLM 추론 필요)
                track['partial_info'] = partial_info
                llm_required_batch.append(track)
                
        # 검색으로 해결된 곡 즉시 업데이트
        if search_resolved_batch:
            for track in search_resolved_batch:
                inference_with_id = track['inference']
                inference_with_id['id'] = track['id']  # 누락된 'id' 키 추가
                process_and_update(col, [track], [inference_with_id], args.dry_run, source="검색성공")
                success_count += 1
                
        # 2단계: 추론 후순위 (Inference Later)
        if llm_required_batch:
            info(f"🔍 검색 완료. {len(llm_required_batch)}곡 LLM 추론 시작...")
            prompt = build_prompt(llm_required_batch)
            
            inferred = query_ollama(prompt)
            
            if inferred:
                # 검색으로 찾은 부분 정보와 LLM 결과 병합
                for item in inferred:
                    item_id = item.get("id")
                    track_obj = next((t for t in llm_required_batch if t['id'] == item_id), None)
                    if track_obj and 'partial_info' in track_obj:
                        for k, v in track_obj['partial_info'].items():
                            if k not in item or not item[k]:
                                item[k] = v
   
            
                updated = process_and_update(col, llm_required_batch, inferred, args.dry_run, source="추론성공")
                success_count += updated
            else:
                warn("이 배치에 대한 추론 실패. 다음 배치로 넘어갑니다.")
                
        processed_count += len(batch)
        # Ollama 서버 부하 방지
        time.sleep(2)

    print(f"\n{BOLD}{'═'*68}{RESET}")
    if not args.dry_run:
        ok(f"완료되었습니다. 총 {success_count}곡 업데이트됨.")
    else:
        info(f"Dry-run 모드 완료. 총 {success_count}곡 업데이트 가능.")
    print(f"{BOLD}{'═'*68}{RESET}\n")
