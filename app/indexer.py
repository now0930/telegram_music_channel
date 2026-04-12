"""
┌─────────────────────────────────────────────────────────────────┐
│  🎵  음악 인덱서  (Gemma4:e4b + ChromaDB)  ─ v2 리팩토링판      │
│                                                                   │
│  동작 모드                                                        │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ --artist / --album / --title 지정                          │  │
│  │   → 해당 파일만 필터 + 지정 필드 가중치 3× 부여            │  │
│  │   → AI 태그 생성 안 함                                     │  │
│  ├────────────────────────────────────────────────────────────┤  │
│  │ 인자 없이 실행                                              │  │
│  │   → 전체 신규 파일 색인                                    │  │
│  │   → AI 태그 자동 생성 (Gemma4:e4b)                        │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ── v2 주요 변경사항 ──────────────────────────────────────────  │
│  ✅ year / bpm → int 타입 저장 (숫자 범위 필터 $gte/$lte 가능)  │
│  ✅ ai_tags 파싱 → mood / era / bpm_tag / genre_fixed 독립 필드  │
│  ✅ era 자동 보정: "90년대" → "1990년대" (XXXX년대 형식 통일)    │
│  ✅ build_document: ai_tags 전체 내용 임베딩 텍스트에 포함        │
│                                                                   │
│  ⚠️  기존 DB 호환 주의사항 (하단 주석 참고)                       │
│                                                                   │
│  실행 예시                                                        │
│    python indexer.py                          # 전체 신규 색인   │
│    python indexer.py --artist "아이유"         # 가수 지정       │
│    python indexer.py --artist "BTS" --title "Dynamite"          │
│    python indexer.py --reset                  # DB 초기화 후 색인│
└─────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⚠️  기존 DB와의 호환성 경고
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ChromaDB는 컬렉션 내 동일 ID(파일 경로)에 upsert하면 메타데이터
  전체를 덮어씁니다. 따라서:

  1) 전체 재색인 (권장):
     --reset 플래그를 사용하거나, 실행 전 아래 명령으로 컬렉션을
     삭제하면 기존 문자열 year/bpm 데이터가 완전히 초기화됩니다.
       python indexer.py --reset

  2) 부분 업데이트:
     --artist / --album / --title 로 범위를 지정해 실행하면
     해당 파일들의 메타데이터만 int 형식으로 갱신됩니다.
     단, AI 태그 필드(mood, era, bpm_tag, genre_fixed)는
     지정 모드에서는 생성되지 않으므로 빈 문자열로 저장됩니다.
     이 경우 별도로 --no-ai 없이 재실행하는 것이 좋습니다.

  3) 주의: 기존 DB에 year/bpm이 문자열로 남아 있는 항목과
     새로 저장되는 int 항목이 혼재하면 $gte/$lte 필터 시
     타입 불일치 오류가 발생할 수 있습니다.
     → 전체 재색인을 강력히 권장합니다.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import chromadb
import librosa
import ollama
import requests
from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3

warnings.filterwarnings("ignore", category=UserWarning)  # librosa 경고 억제


import os
from dotenv import load_dotenv

# 1. 경로 설정 (현재 indexer.py 위치 기준)
current_dir = os.path.dirname(os.path.abspath(__file__)) # ~/telegram_music_channel/app
root_dir = os.path.join(current_dir, "..")               # ~/telegram_music_channel

# 2. .env 로드 (상위 폴더의 .env 가리킴)
load_dotenv(os.path.join(current_dir, ".env"))


# ══════════════════════════════════════════════════════════════════
#  ⚙️  전역 설정
# ══════════════════════════════════════════════════════════════════
MUSIC_PATH  = "/music"
DB_PATH     = "./music_vector_db"
EMBED_MODEL = "mxbai-embed-large:latest"
TEXT_MODEL  = "gemma4:e4b"               # ← 버전 변경 시 여기만 수정

# API 키 — 환경변수로 주입하거나 아래에 직접 입력
# MusicBrainz는 키 불필요
AUDD_API_KEY          = os.getenv("AUDD_API_KEY",          "")
MELON_API_KEY         = os.getenv("MELON_API_KEY",         "")
GENIE_APP_ID          = os.getenv("GENIE_APP_ID",          "")
GENIE_APP_SECRET      = os.getenv("GENIE_APP_SECRET",      "")

# ChromaDB — 전역 초기화 (reset 처리는 run_indexing 내부에서 수행)
_chroma    = chromadb.PersistentClient(path=DB_PATH)
collection = _chroma.get_or_create_collection(name="music_library")


# ══════════════════════════════════════════════════════════════════
#  🔢  타입 캐스팅 유틸리티
# ══════════════════════════════════════════════════════════════════

def safe_int(value, default: int = 0) -> int:
    """
    문자열·숫자·None 등 어떤 값이든 안전하게 int로 변환.
    변환 불가 시 default(기본 0) 반환.
    ChromaDB 숫자 범위 필터($gte/$lte)를 위해 반드시 int 타입이어야 함.
    """
    if value is None or value == "":
        return default
    try:
        # float 경유: "136.7" 같은 소수 문자열도 처리
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        return default


# ══════════════════════════════════════════════════════════════════
#  📅  era 정규화 유틸리티
# ══════════════════════════════════════════════════════════════════

# 두 자리 연도 약칭 → 네 자리 연도 매핑 (1900년대 한정; 필요 시 확장)
_ERA_SHORT_MAP = {
    "10": "1910", "20": "1920", "30": "1930", "40": "1940",
    "50": "1950", "60": "1960", "70": "1970", "80": "1980",
    "90": "1990", "00": "2000", "2000": "2000",
    # 두 자리이지만 2000년대로 해석해야 하는 경우
    "10s_2k": "2010", "20s_2k": "2020",
}

def normalize_era(era_raw: str) -> str:
    """
    다양한 형식의 era 문자열을 'XXXX년대' 형식으로 통일.

    처리 예시:
      "90년대"        → "1990년대"
      "90s"           → "1990년대"
      "1990년대"      → "1990년대"  (변환 없음)
      "2000년대 초반" → "2000년대"  (세부 수식어 제거)
      "2020년대"      → "2020년대"
      "미상" / ""     → 그대로 반환

    규칙:
      - 네 자리 숫자가 있으면 그 숫자의 십 단위 내림 → XXXX년대
      - 두 자리 숫자만 있으면:
          50~99  → 1900년대 계열 (1950년대 ~ 1990년대)
          00~49  → 2000년대 계열 (2000년대 ~ 2040년대)
    """
    if not era_raw or era_raw in ("미상", "unknown", "N/A"):
        return era_raw

    s = era_raw.strip()

    # ── 네 자리 연도가 포함된 경우 ────────────────────────────
    m4 = re.search(r"(1[89]\d{2}|20\d{2})", s)
    if m4:
        year4 = int(m4.group(1))
        decade = (year4 // 10) * 10
        return f"{decade}년대"

    # ── 두 자리 숫자 + '년대' 또는 's' 패턴 ─────────────────
    m2 = re.search(r"\b(\d{2})[년s]", s)
    if m2:
        two = int(m2.group(1))
        if 50 <= two <= 99:
            return f"{1900 + two}년대"
        else:
            return f"{2000 + two}년대"

    # ── 순수 두 자리 숫자만 ───────────────────────────────────
    m_num = re.fullmatch(r"\d{2}", s.strip())
    if m_num:
        two = int(s.strip())
        if 50 <= two <= 99:
            return f"{1900 + two}년대"
        else:
            return f"{2000 + two}년대"

    # ── 변환 불가: 원본 반환 ──────────────────────────────────
    return s


# ══════════════════════════════════════════════════════════════════
#  🧩  ai_tags 문자열 파싱 → 독립 필드 추출
# ══════════════════════════════════════════════════════════════════

def parse_ai_tags(ai_tags: str) -> dict:
    """
    ai_tags 문자열에서 mood / era / bpm_tag / genre_fixed 를 추출해
    독립 필드 딕셔너리로 반환.

    지원 형식 예시:
      " / AI_태그: [mood:감성적, era:1990년대, bpm_tag:업비트, ...]"
      또는 generate_ai_tags 가 반환하는 동일 포맷

    반환 키:
      mood        - 감성적, 밝음 등
      era         - normalize_era() 적용 후 'XXXX년대' 형식
      bpm_tag     - 업비트, 슬로우 등
      genre_fixed - AI 추론 장르명 (etc 또는 별도 장르 필드에서 추출)
    """
    result = {
        "mood":        "",
        "era":         "",
        "bpm_tag":     "",
        "genre_fixed": "",
    }

    if not ai_tags:
        return result

    # ── 대괄호 내부 k:v 쌍 파싱 ──────────────────────────────
    bracket_match = re.search(r"\[(.+?)\]", ai_tags, re.DOTALL)
    if bracket_match:
        inner = bracket_match.group(1)
    else:
        inner = ai_tags

    # "key:value" 패턴 전체 추출
    pairs = re.findall(r"(\w+)\s*:\s*([^,\]]+)", inner)
    tag_dict: dict[str, str] = {}
    for k, v in pairs:
        tag_dict[k.strip().lower()] = v.strip()

    # ── 필드별 매핑 ───────────────────────────────────────────
    result["mood"]    = tag_dict.get("mood", "")
    result["bpm_tag"] = tag_dict.get("bpm_tag", "")

    # era: 추출 후 정규화
    raw_era = tag_dict.get("era", "")
    result["era"] = normalize_era(raw_era) if raw_era else ""

    # genre_fixed: 'genre' 또는 'etc' 에서 장르 키워드 탐색
    genre_candidate = tag_dict.get("genre", "") or tag_dict.get("genre_fixed", "")
    if not genre_candidate:
        # etc 에서 장르 관련 키워드 추출 시도
        etc_val = tag_dict.get("etc", "")
        genre_keywords = ["팝", "록", "R&B", "힙합", "재즈", "클래식", "발라드",
                          "댄스", "인디", "포크", "일렉트로닉", "소울", "컨트리",
                          "pop", "rock", "jazz", "hiphop", "dance", "indie"]
        for kw in genre_keywords:
            if kw.lower() in etc_val.lower():
                genre_candidate = kw
                break
    result["genre_fixed"] = genre_candidate

    return result


# ══════════════════════════════════════════════════════════════════
#  🧹  DB 정리
# ══════════════════════════════════════════════════════════════════
def cleanup_deleted_files() -> None:
    """실제 파일이 없는 DB 항목을 삭제한다."""
    print("🧹 DB 전수 조사 중...")
    if collection.count() == 0:
        print("   DB가 비어 있습니다.")
        return

    all_ids = collection.get()["ids"]
    removed = 0
    for path in all_ids:
        if not os.path.exists(path):
            collection.delete(ids=[path])
            print(f"   🗑️  삭제: {os.path.basename(path)}")
            removed += 1

    if removed:
        print(f"   ✅ {removed}개 유령 데이터 삭제 완료")
    else:
        print("   ✅ 유령 데이터 없음")


# ══════════════════════════════════════════════════════════════════
#  📂  로컬 태그 + 파일명 파싱
# ══════════════════════════════════════════════════════════════════
def extract_local_info(file_path: str) -> tuple[dict, str]:
    """
    우선순위: ID3 태그 → 파일명 파싱 → 상위 디렉토리명
    반환: (메타딕셔너리, 확장자 제거 파일명)
    """
    info: dict = {
        "artist":   "",
        "title":    "",
        "album":    "",
        "lyrics":   "",
        "year":     "",
        "dir_name": "",   # 상위 디렉토리명 (항상 보존)
        "stem":     "",   # 확장자 제거 파일명 (항상 보존)
    }

    # ── EasyID3 기본 태그 ──────────────────────────────────────
    try:
        audio          = EasyID3(file_path)
        info["artist"] = audio.get("artist", [""])[0].strip()
        info["title"]  = audio.get("title",  [""])[0].strip()
        info["album"]  = audio.get("album",  [""])[0].strip()
        info["year"]   = audio.get("date",   [""])[0][:4].strip()
    except Exception:
        pass

    # ── ID3 가사(USLT) ─────────────────────────────────────────
    try:
        tags   = ID3(file_path)
        frames = tags.getall("USLT")
        if frames:
            info["lyrics"] = frames[0].text.strip()
    except Exception:
        pass

    # ── 파일명·디렉토리 보정 ───────────────────────────────────
    stem     = os.path.splitext(os.path.basename(file_path))[0]
    dir_name = os.path.basename(os.path.dirname(file_path))

    if not info["title"] or not info["artist"]:
        for sep in (" - ", " – ", "_-_", "_"):
            if sep in stem:
                left, right    = stem.split(sep, 1)
                info["artist"] = info["artist"] or left.strip()
                info["title"]  = info["title"]  or right.strip()
                break
        info["title"] = info["title"] or stem

    if not info["album"] and dir_name and dir_name != os.path.basename(MUSIC_PATH):
        info["album"] = dir_name

    # 항상 보존
    info["dir_name"] = dir_name
    info["stem"]     = stem

    return info, stem


# ══════════════════════════════════════════════════════════════════
#  🎵  BPM 분석 — librosa 로컬 전용
# ══════════════════════════════════════════════════════════════════
def analyze_bpm(file_path: str) -> str:
    """
    librosa로 직접 BPM을 분석한다.
    30초만 로드하여 속도와 정확도를 균형 있게 유지.
    실패 시 빈 문자열 반환.
    """
    try:
        y, sr    = librosa.load(file_path, sr=22050, mono=True, offset=30, duration=30)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm      = int(round(float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)))
        return str(bpm)
    except Exception as e:
        print(f"    [⚠️ BPM 분석 실패]: {e}")
        return ""


def bpm_to_range(bpm_str: str) -> str:
    """BPM 숫자 → 한국어 템포 범주"""
    if not bpm_str:
        return "미상"
    try:
        b = int(float(bpm_str))
    except (ValueError, TypeError):
        return "미상"
    if   b < 60:  return "매우 느린 템포 (Largo)"
    elif b < 80:  return "느린 템포 (Andante)"
    elif b < 110: return "미디엄 템포 (Moderato)"
    elif b < 140: return "업비트 (Allegro)"
    else:          return "하이템포/댄스 (Presto)"


# ══════════════════════════════════════════════════════════════════
#  🌐  MusicBrainz API  (Spotify 대체 — 무료·무키·장르 풍부)
#
#  Spotify Web API는 2024-11-27부터 신규 앱의 검색 엔드포인트를
#  차단(403)하여 MusicBrainz로 완전 교체.
#
#  MusicBrainz 제공 정보:
#    - 트랙 제목 / 아티스트 / 앨범(릴리즈) / 발매연도
#    - 장르(태그) — 커뮤니티 투표 기반, 팝·록·R&B 등 상세
#    - MusicBrainz ID (mbid)
#  Rate limit: 1 req/sec — User-Agent 필수
# ══════════════════════════════════════════════════════════════════
_MB_HEADERS = {
    "User-Agent": "MusicIndexer/2.0 (local-home-server)",
    "Accept":     "application/json",
}

def search_musicbrainz(artist: str, title: str, album: str = "") -> dict:
    """
    MusicBrainz recordings 검색.
    반환 키는 enrich_metadata fill_fields 와 일치:
      title, artist, album, year, genre, mb_id
    """
    try:
        # ── 1단계: recording 검색 ─────────────────────────────
        query_parts = [f'recording:"{title}"', f'artist:"{artist}"']
        if album:
            query_parts.append(f'release:"{album}"')

        r = requests.get(
            "https://musicbrainz.org/ws/2/recording",
            params  = {
                "query": " AND ".join(query_parts),
                "limit": 1,
                "fmt":   "json",
                "inc":   "releases+artist-credits+tags",
            },
            headers = _MB_HEADERS,
            timeout = 10,
        )
        if r.status_code != 200:
            print(f"    [MusicBrainz Search Error]: {r.status_code}")
            return {}

        recordings = r.json().get("recordings", [])
        if not recordings:
            return {}

        rec = recordings[0]

        # ── 제목 / 아티스트 ───────────────────────────────────
        title_val  = rec.get("title", "")
        credits    = rec.get("artist-credit", [])
        artist_val = credits[0].get("artist", {}).get("name", "") if credits else ""

        # ── 앨범 / 발매연도 ───────────────────────────────────
        releases   = rec.get("releases", [])
        album_val  = ""
        year_val   = ""
        if releases:
            rel       = releases[0]
            album_val = rel.get("title", "")
            date_str  = rel.get("date", "")          # "1980-10-08" or "1980"
            year_val  = date_str[:4] if date_str else ""

        # ── 장르 (태그) ───────────────────────────────────────
        # tags는 [{"name": "pop", "count": 5}, ...] 형식, count 내림차순 정렬
        tags = sorted(rec.get("tags", []), key=lambda t: t.get("count", 0), reverse=True)
        genre_val = tags[0]["name"] if tags else ""

        # ── MusicBrainz ID ────────────────────────────────────
        mb_id = rec.get("id", "")

        result = {
            "title":   title_val,
            "artist":  artist_val,
            "album":   album_val,
            "year":    year_val,
            "genre":   genre_val,
            "mb_id":   mb_id,
        }
        # 비어있는 필드 제거 (fill_fields 로직과 일관성 유지)
        return {k: v for k, v in result.items() if v}

    except Exception as e:
        print(f"    [MusicBrainz Exception]: {e}")
        return {}


# ══════════════════════════════════════════════════════════════════
#  🌐  Melon Open API
# ══════════════════════════════════════════════════════════════════
def search_melon(artist: str, title: str) -> dict:
    if not MELON_API_KEY:
        return {}
    try:
        r = requests.get(
            "https://openapi.melon.com/v1/search/track",
            params  = {"query": f"{artist} {title}", "count": 1},
            headers = {"appKey": MELON_API_KEY},
            timeout = 10,
        )
        tracks = (
            r.json()
            .get("response", {})
            .get("trackList", {})
            .get("track", [])
        )
        if not tracks:
            return {}
        t = tracks[0]
        return {
            "title":  t.get("trackName",   ""),
            "artist": t.get("artistList",  [{}])[0].get("artistName", ""),
            "album":  t.get("albumName",   ""),
            "year":   t.get("issueDate",   "")[:4],
            "genre":  t.get("genreCode",   ""),
            "lyrics": t.get("lyrics",      ""),
        }
    except Exception as e:
        print(f"    [Melon 오류]: {e}")
        return {}


# ══════════════════════════════════════════════════════════════════
#  🌐  Genie Music API
# ══════════════════════════════════════════════════════════════════
def search_genie(artist: str, title: str) -> dict:
    if not (GENIE_APP_ID and GENIE_APP_SECRET):
        return {}
    try:
        r = requests.get(
            "https://api.genie.co.kr/song/getSongInfo",
            params  = {
                "appId":     GENIE_APP_ID,
                "appSecret": GENIE_APP_SECRET,
                "query":     f"{artist} {title}",
                "count":     1,
            },
            timeout = 10,
        )
        songs = r.json().get("response", {}).get("songList", [])
        if not songs:
            return {}
        s = songs[0]
        return {
            "title":  s.get("songTitle",   ""),
            "artist": s.get("artistName",  ""),
            "album":  s.get("albumTitle",  ""),
            "year":   s.get("releaseDate", "")[:4],
            "lyrics": s.get("lyrics",      ""),
        }
    except Exception as e:
        print(f"    [Genie 오류]: {e}")
        return {}


# ══════════════════════════════════════════════════════════════════
#  🌐  AudD — 오디오 핑거프린팅
# ══════════════════════════════════════════════════════════════════
def recognize_audd(file_path: str) -> dict:
    if not AUDD_API_KEY:
        return {}
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(500_000)
        r = requests.post(
            "https://api.audd.io/",
            data  = {"api_token": AUDD_API_KEY, "return": "lyrics,spotify"},
            files = {"file": ("audio.mp3", chunk, "audio/mpeg")},
            timeout = 30,
        )
        result = r.json().get("result") or {}
        if not result:
            return {}
        lyrics_field = result.get("lyrics", "")
        if isinstance(lyrics_field, dict):
            lyrics_field = lyrics_field.get("lyrics", "")
        return {
            "title":  result.get("title",        ""),
            "artist": result.get("artist",        ""),
            "album":  result.get("album",         ""),
            "year":   result.get("release_date",  "")[:4],
            "lyrics": lyrics_field,
        }
    except Exception as e:
        print(f"    [AudD 오류]: {e}")
        return {}


# ══════════════════════════════════════════════════════════════════
#  🔀  메타데이터 통합 보강
# ══════════════════════════════════════════════════════════════════
def enrich_metadata(local: dict, file_path: str) -> dict:
    """
    로컬 태그 → MusicBrainz → Melon → Genie → AudD 순으로 보강.
    BPM은 librosa로 항상 직접 분석.
    """
    meta   = dict(local)
    artist = meta.get("artist", "")
    title  = meta.get("title",  "")

    api_results: list[tuple[str, dict]] = []

    if artist and title:
        mb = search_musicbrainz(artist, title, meta.get("album", ""))
        if mb:
            api_results.append(("MusicBrainz", mb))

    if artist and title:
        ml = search_melon(artist, title)
        if ml:
            api_results.append(("Melon", ml))

    if artist and title:
        gn = search_genie(artist, title)
        if gn:
            api_results.append(("Genie", gn))

    if not (artist and title):
        ad = recognize_audd(file_path)
        if ad:
            api_results.append(("AudD", ad))

    fill_fields = ("title", "artist", "album", "year", "genre", "lyrics", "mb_id")
    for source, data in api_results:
        filled = []
        for k in fill_fields:
            if data.get(k) and not meta.get(k):
                meta[k] = data[k]
                filled.append(k)
        if filled:
            print(f"    ✅ {source}: {', '.join(filled)} 보강")
        else:
            print(f"    ℹ️  {source}: 응답 받았으나 보강할 신규 필드 없음")

    # BPM — librosa 로컬 분석
    print(f"    🎵 librosa BPM 분석 중...", end=" ", flush=True)
    bpm        = analyze_bpm(file_path)
    meta["bpm"] = bpm
    print(f"{bpm} BPM ({bpm_to_range(bpm)})" if bpm else "분석 실패")

    # 연주곡 판별
    title_lower    = (meta.get("title") or "").lower()
    instr_keywords = ("inst", "instrumental", "mr", "karaoke", "반주", "연주")
    meta["is_instrumental"] = (
        any(kw in title_lower for kw in instr_keywords)
        or not bool((meta.get("lyrics") or "").strip())
    )

    return meta


# ══════════════════════════════════════════════════════════════════
#  🌐  웹 검색 — Wikipedia API (무료·무키·안정적)
#  DDG Instant Answer API는 빈 응답이 잦고 음악 정보가 빈약하여 교체.
#  Wikipedia는 아티스트·앨범 장르, 분위기, 배경 정보가 풍부함.
# ══════════════════════════════════════════════════════════════════
def search_web(query: str, max_results: int = 3) -> str:
    """
    Wikipedia API로 음악 관련 컨텍스트를 검색.
    1차: Wikipedia search → extract 순으로 시도.
    2차: 실패 시 DuckDuckGo HTML fallback.
    """
    snippets: list[str] = []

    # ── 1차: Wikipedia search API ─────────────────────────────
    try:
        # 검색어로 관련 페이지 제목 탐색
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params = {
                "action":   "query",
                "list":     "search",
                "srsearch": query,
                "srlimit":  2,
                "format":   "json",
            },
            headers = {"User-Agent": "MusicIndexer/1.0"},
            timeout = 6,
        )
        if r.text.strip():
            results = r.json().get("query", {}).get("search", [])
            for item in results[:2]:
                title = item.get("title", "")
                if not title:
                    continue
                # 페이지 요약(extract) 가져오기
                r2 = requests.get(
                    "https://en.wikipedia.org/w/api.php",
                    params = {
                        "action":   "query",
                        "prop":     "extracts",
                        "exintro":  True,
                        "explaintext": True,
                        "titles":   title,
                        "format":   "json",
                    },
                    headers = {"User-Agent": "MusicIndexer/1.0"},
                    timeout = 6,
                )
                pages = r2.json().get("query", {}).get("pages", {})
                for page in pages.values():
                    extract = (page.get("extract") or "").strip()
                    if extract:
                        snippets.append(extract[:300])
            if snippets:
                return "\n".join(snippets)
    except Exception as e:
        print(f"    [🌐 Wikipedia 오류]: {e}")

    # ── 2차: DuckDuckGo HTML fallback ────────────────────────
    try:
        r = requests.get(
            "https://html.duckduckgo.com/html/",
            params  = {"q": query},
            headers = {"User-Agent": "Mozilla/5.0"},
            timeout = 6,
        )
        if r.text.strip():
            matches = re.findall(
                r'class="result__snippet"[^>]*>(.*?)</a>',
                r.text, re.DOTALL,
            )
            for m in matches[:max_results]:
                clean = re.sub(r"<[^>]+>", "", m).strip()
                if clean:
                    snippets.append(clean[:150])
        return "\n".join(snippets)
    except Exception as e:
        print(f"    [🌐 DDG HTML 오류]: {e}")
        return ""


def search_web_parallel(queries: list[str]) -> str:
    """여러 쿼리를 병렬 실행 후 결과 합산."""
    results: list[str] = []
    with ThreadPoolExecutor(max_workers=len(queries)) as ex:
        futures = {ex.submit(search_web, q): q for q in queries}
        for future in as_completed(futures):
            q = futures[future]
            r = future.result()
            if r:
                results.append(r)
                print(f"    [🌐 검색 완료]: {q}")
    return "\n".join(results)


# ══════════════════════════════════════════════════════════════════
#  🤖  Gemma4 AI 태그 생성
# ══════════════════════════════════════════════════════════════════
_AI_TAG_SCHEMA = {
    "mood":         "가사·음악의 분위기 (예: 밝음, 슬픔, 설렘, 신남, 잔잔함, 몽환적)",
    "era":          "출시 시기 추정 — 반드시 'XXXX년대' 형식 (예: 1990년대, 2000년대, 2020년대)",
    "bpm_tag":      "BPM 범주 (예: 슬로우, 미디엄, 업비트, 하이템포/댄스)",
    "instrumental": "연주곡 여부 (연주곡 / 보컬곡)",
    "etc":          "기타 특징 키워드 (상황·계절·장르 특성, 예: 드라이브, 새벽감성, 재즈풍)",
}

_GENERIC_PATTERN = re.compile(
    r"^(track\d*|cd\d*|various\s*artist[s]?|미상|unknown).*$",
    re.IGNORECASE,
)


def generate_ai_tags(meta: dict) -> str:
    """
    웹 검색 (병렬) → 200자 컨텍스트 → Gemma JSON 태그 생성.

    변경 사항 (v2):
      - era 응답을 normalize_era()로 자동 보정 ('90년대' → '1990년대')
      - 프롬프트에 era 형식 규칙 명시 (XXXX년대)
      - JSON 파싱 실패 시 fallback 태그도 era 정규화 적용
    """
    bpm_str   = meta.get("bpm", "")
    bpm_range = bpm_to_range(bpm_str)
    instr     = "연주곡" if meta.get("is_instrumental") else "보컬곡"

    artist   = (meta.get("artist") or "").strip()
    title    = (meta.get("title")  or "").strip()
    album    = (meta.get("album")  or "").strip()
    year     = (meta.get("year")   or "").strip()
    dir_name = meta.get("dir_name", "")
    stem     = meta.get("stem",     "")

    def _is_generic(s: str) -> bool:
        return not s or bool(_GENERIC_PATTERN.match(s.strip()))

    no_id          = _is_generic(artist) and _is_generic(title)
    display_artist = artist or dir_name or "미상"
    display_title  = title  or stem     or "미상"
    display_album  = album  or dir_name or "미상"

    # ── 웹 검색 쿼리 구성 ────────────────────────────────────
    queries: list[str] = []
    if not no_id:
        queries.append(f"{artist} {title} 노래 장르 분위기")
        queries.append(f"{artist} {album or '음악'} 장르 스타일")
    else:
        if dir_name: queries.append(f"{dir_name} OST 음악 분위기 장르")
        if stem:     queries.append(f"{stem} 음악 정보")

    # ── 병렬 웹 검색 → 200자 컨텍스트 ───────────────────────
    web_context = ""
    if queries:
        raw_web = search_web_parallel(queries[:2])
        if raw_web:
            web_context = raw_web[:200].replace("\n", " ").strip()
            print(f"    [🌐 컨텍스트]: {web_context[:80]}...")

    # ── 한 줄 압축 프롬프트 (era 형식 규칙 강조) ─────────────
    example = (
        '{"mood": "신남", "era": "2010년대", '
        '"bpm_tag": "업비트", "instrumental": "보컬곡", "etc": "파티, 댄스"}'
    )
    info = (
        f"아티스트:{display_artist} | 제목:{display_title} | 앨범:{display_album} | "
        f"연도:{year or '미상'} | BPM:{bpm_str}({bpm_range}) | 유형:{instr} | "
        f"참고:{web_context[:200] if web_context else '없음'}"
    )
    prompt = (
        f"음악정보: {info}\n"
        f"위 음악의 태그를 예시와 똑같은 JSON 형식 한 줄로 출력하세요.\n"
        f"규칙: era는 반드시 '1990년대', '2000년대' 처럼 네 자리 숫자+'년대' 형식으로.\n"
        f"예시: {example}\n"
        f"출력:"
    )

    try:
        resp = ollama.generate(
            model   = TEXT_MODEL,
            prompt  = prompt,
            stream  = False,
            think   = False,
            options = {"temperature": 0.05, "num_predict": 300},
        )

        if isinstance(resp, dict):
            raw = (resp.get("response") or "").strip()
        else:
            raw = (resp.response or "").strip()

        print(f"    [🔍 Gemma raw]: {repr(raw[:200])}")

        # ── 빈 응답 → 최소 태그 자동 생성 ───────────────────
        if not raw:
            print("    [⚠️ 빈 응답 → 최소 태그 자동 생성]")
            raw_era = (year[:4] + "년대") if year else "미상"
            fallback = {
                "mood":         "미상",
                "era":          normalize_era(raw_era),   # ← 정규화 적용
                "bpm_tag":      bpm_range,
                "instrumental": instr,
                "etc":          ", ".join(filter(None, [
                                    f"파일명:{stem}",
                                    f"디렉토리:{dir_name}",
                                ])),
            }
            parts = [f"{k}:{v}" for k, v in fallback.items() if v]
            return " / AI_태그: [" + ", ".join(parts) + "]"

        # ── JSON 파싱 ─────────────────────────────────────────
        raw_clean = re.sub(r"```(?:json)?", "", raw).strip("` \n")
        match     = re.search(r"\{.*?\}", raw_clean, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                print(parsed)

                # era 자동 보정 (핵심 수정)
                if "era" in parsed and parsed["era"]:
                    parsed["era"] = normalize_era(str(parsed["era"]))

                parts = [f"{k}:{v}" for k, v in parsed.items() if v]
                return " / AI_태그: [" + ", ".join(parts) + "]"
            except json.JSONDecodeError:
                pass  # 아래 clean fallback으로 이어짐

        clean = re.sub(r"\s+", " ", raw_clean).strip()
        return f" / AI_태그: [{clean}]"

    except Exception as e:
        print(f"    [⚠️ AI 태그 생성 실패]: {e}")
        return ""


# ══════════════════════════════════════════════════════════════════
#  📝  임베딩용 텍스트 구성
# ══════════════════════════════════════════════════════════════════
def build_document(
    meta:          dict,
    stem:          str,
    ai_tags:       str,
    target_artist: str,
    target_album:  str,
    target_title:  str,
) -> str:
    """
    지정 필드는 3회 반복 → 벡터 공간에서 가중치 효과.

    v2 변경:
      - ai_tags의 모든 내용을 document 텍스트 말미에 완전히 포함.
      - parse_ai_tags()로 추출한 mood / era / bpm_tag / genre_fixed 도
        개별 라벨로 추가하여 의미 검색 정확도 향상.
    """
    parts: list[str] = []

    def add(label: str, value: str, boost: bool = False) -> None:
        if not value:
            return
        entry = f"{label}: {value}"
        parts.extend([entry] * 3 if boost else [entry])

    add("제목",     meta.get("title",  ""), boost=bool(target_title))
    add("가수",     meta.get("artist", ""), boost=bool(target_artist))
    add("앨범",     meta.get("album",  ""), boost=bool(target_album))
    add("파일명",   stem)
    add("발매연도", str(safe_int(meta.get("year",  ""), 0)) if meta.get("year") else "")
    add("BPM",      str(safe_int(meta.get("bpm",   ""), 0)) if meta.get("bpm")  else "")
    add("BPM범주",  bpm_to_range(meta.get("bpm", "")))
    add("장르",     meta.get("genre",  ""))
    add("연주곡",   "연주곡" if meta.get("is_instrumental") else "보컬곡")

    if meta.get("lyrics"):
        parts.append(f"가사: {meta['lyrics'][:500]}")

    # ── AI 태그: 원본 문자열 전체 포함 ─────────────────────────
    if ai_tags:
        parts.append(ai_tags.lstrip(" /"))

    # ── AI 태그 파싱 필드 개별 추가 (의미 검색 강화) ────────────
    parsed_tags = parse_ai_tags(ai_tags)
    if parsed_tags.get("mood"):
        parts.append(f"분위기: {parsed_tags['mood']}")
    if parsed_tags.get("era"):
        parts.append(f"출시시기: {parsed_tags['era']}")
    if parsed_tags.get("bpm_tag"):
        parts.append(f"BPM태그: {parsed_tags['bpm_tag']}")
    if parsed_tags.get("genre_fixed"):
        parts.append(f"장르추론: {parsed_tags['genre_fixed']}")

    return " / ".join(parts)


# ══════════════════════════════════════════════════════════════════
#  🚀  메인 인덱싱
# ══════════════════════════════════════════════════════════════════
def run_indexing(
    target_artist: str = "",
    target_album:  str = "",
    target_title:  str = "",
    reset_db:      bool = False,
) -> None:
    """
    target_* 중 하나라도 지정되면 → 지정 모드 (AI 태그 없음)
    모두 비어있으면              → 전체 신규 색인 (AI 태그 생성)
    reset_db=True               → 컬렉션 삭제 후 전체 재색인

    ⚠️  reset_db 주의:
        기존 DB의 모든 데이터가 영구 삭제됩니다.
        year/bpm 타입 혼재 문제를 해결하려면 이 옵션을 사용하세요.
    """
    global collection

    # ── DB 초기화 (--reset 플래그) ────────────────────────────
    if reset_db:
        print("⚠️  DB 초기화 모드: 기존 컬렉션을 삭제하고 새로 생성합니다.")
        try:
            _chroma.delete_collection(name="music_library")
            print("   🗑️  기존 컬렉션 삭제 완료")
        except Exception as e:
            print(f"   [컬렉션 삭제 오류 (무시)]: {e}")
        collection = _chroma.get_or_create_collection(name="music_library")
        print("   ✅ 새 컬렉션 생성 완료\n")
    else:
        cleanup_deleted_files()

    filter_mode = bool(target_artist or target_album or target_title)
    use_ai_tags = not filter_mode

    # ── ollama 버전 확인 ──────────────────────────────────────
    try:
        import importlib.metadata
        ver = importlib.metadata.version("ollama")
    except Exception:
        ver = "알 수 없음"
    print(f"[ollama 라이브러리 버전]: {ver}")

    # ── 모델 동작 사전 점검 ───────────────────────────────────
    print(f"🤖 모델 사전 점검: {TEXT_MODEL}")
    try:
        test     = ollama.generate(
            model   = TEXT_MODEL,
            prompt  = '{"test": "ok"} 위 JSON을 그대로 한 줄 출력:',
            stream  = False,
            think   = False,
            options = {"temperature": 0, "num_predict": 50},
        )
        test_raw = (
            test.get("response", "") if isinstance(test, dict) else (test.response or "")
        ).strip()
        print(f"   응답: {repr(test_raw)}")
        print(f"   {'✅ 정상' if test_raw else '❌ 빈 응답 — TEXT_MODEL 확인 필요'}")
        if not test_raw:
            return
    except Exception as e:
        print(f"   ❌ 모델 오류: {e}")
        return

    print()
    if filter_mode:
        parts = []
        if target_artist: parts.append(f"가수='{target_artist}'")
        if target_album:  parts.append(f"앨범='{target_album}'")
        if target_title:  parts.append(f"곡='{target_title}'")
        print(f"🎯 지정 검색 모드 | {' + '.join(parts)}")
        print("   ※ 지정 필드 가중치 3× 적용 / AI 태그 생성 안 함")
    else:
        print(f"🔄 전체 신규 색인 모드 | AI 태그 자동 생성 (모델: {TEXT_MODEL})")
    print("─" * 60)

    # ── 처리 대상 파일 수집 ───────────────────────────────────
    targets: list[str] = []
    for root, _dirs, files in os.walk(MUSIC_PATH):
        for file in files:
            if not file.lower().endswith((".mp3", ".flac", ".m4a")):
                continue
            path = os.path.join(root, file)
            if filter_mode:
                hay = (file + root).lower()
                if target_title  and target_title.lower()  not in hay: continue
                if target_artist and target_artist.lower() not in hay: continue
                if target_album  and target_album.lower()  not in hay: continue
            elif collection.get(ids=[path])["ids"]:
                continue
            targets.append(path)

    print(f"📋 처리 대상: {len(targets)}곡\n")
    new_count = 0

    for path in targets:
        try:
            local_meta, stem = extract_local_info(path)
            file             = os.path.basename(path)
            display          = f"{local_meta['artist'] or '?'} - {local_meta['title'] or file}"
            print(f"\n📀 {display}")

            meta    = enrich_metadata(local_meta, path)
            ai_tags = ""
            if use_ai_tags:
                print("    🤖 Gemma4 AI 태그 생성 중...")
                ai_tags = generate_ai_tags(meta)
                if ai_tags:
                    print(f"    {ai_tags.lstrip()}")

            doc = build_document(
                meta, stem, ai_tags,
                target_artist, target_album, target_title,
            )

            # ── 임베딩 ────────────────────────────────────────
            embed_resp = ollama.embeddings(model=EMBED_MODEL, prompt=doc)
            embedding  = (
                embed_resp.get("embedding")
                if isinstance(embed_resp, dict)
                else embed_resp.embedding
            )

            # ── ai_tags 파싱 필드 추출 ────────────────────────
            parsed_tags = parse_ai_tags(ai_tags)

            # ──────────────────────────────────────────────────────
            # ✅ 핵심 수정: year / bpm을 반드시 int로 저장
            #    - ChromaDB 숫자 범위 필터($gte/$lte)는 타입이 int여야 동작
            #    - 데이터 없거나 변환 실패 시 0으로 저장 (필터 오류 방지)
            # ──────────────────────────────────────────────────────
            year_int = safe_int(meta.get("year", ""), default=0)
            bpm_int  = safe_int(meta.get("bpm",  ""), default=0)

            collection.upsert(
                ids        = [path],
                embeddings = [embedding],
                metadatas  = [{
                    # ── 기본 식별 필드 ────────────────────────
                    "path":            path,
                    "filename":        file,
                    "artist":          meta.get("artist",          ""),
                    "title":           meta.get("title",           ""),
                    "album":           meta.get("album",           ""),

                    # ── 숫자 필드: int 타입으로 저장 ─────────
                    #    (문자열 저장 시 $gte/$lte 필터 동작 안 함)
                    "year":            year_int,
                    "bpm":             bpm_int,

                    # ── 기타 메타 필드 ────────────────────────
                    "bpm_range":       bpm_to_range(meta.get("bpm", "")),
                    "genre":           meta.get("genre",           ""),
                    "is_instrumental": str(meta.get("is_instrumental", False)),

                    # ── MusicBrainz 필드 ────────────────────────
                    "mb_id":           meta.get("mb_id", ""),

                    # ── AI 태그 원본 문자열 ───────────────────
                    "ai_tags":         ai_tags,

                    # ── AI 태그 독립 필드 (v2 신규) ───────────
                    #    각 필드를 독립 메타데이터로 분리하여 정밀 필터링 가능
                    "mood":            parsed_tags.get("mood",        ""),
                    "era":             parsed_tags.get("era",         ""),  # 'XXXX년대' 형식
                    "bpm_tag":         parsed_tags.get("bpm_tag",     ""),
                    "genre_fixed":     parsed_tags.get("genre_fixed", ""),
                }],
                documents  = [doc],
            )
            new_count += 1
            print(f"  └─ ✅ 색인 완료  (year={year_int}, bpm={bpm_int}, era={parsed_tags.get('era','')})")

        except Exception as e:
            import traceback
            print(f"  ⚠️  오류 ({os.path.basename(path)}): {e}")
            traceback.print_exc()   # 상세 스택 출력으로 디버깅 용이

    print("\n" + "═" * 60)
    print(f"✅ 완료!  {new_count}개 곡 색인됨.")
    print("═" * 60)


# ══════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gemma4:e4b 기반 음악 인덱서 v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python indexer.py                              # 전체 신규 색인 (AI 태그 포함)
  python indexer.py --reset                      # DB 초기화 후 전체 재색인 (권장)
  python indexer.py --artist "아이유"             # 아이유 곡만 재색인
  python indexer.py --artist "BTS" --album "Map of the Soul"
  python indexer.py --title "Dynamite"           # 제목으로 검색 색인

⚠️  기존 DB와 타입 혼재 문제 해결:
  python indexer.py --reset                      # 모든 데이터 삭제 후 재색인
        """,
    )
    parser.add_argument("--artist", default="", metavar="가수명",  help="가수 이름 지정")
    parser.add_argument("--album",  default="", metavar="앨범명",  help="앨범 이름 지정")
    parser.add_argument("--title",  default="", metavar="곡명",    help="곡 이름 지정")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="기존 DB 컬렉션을 완전 초기화 후 전체 재색인 (year/bpm 타입 혼재 해결)",
    )
    args = parser.parse_args()

    run_indexing(
        target_artist = args.artist,
        target_album  = args.album,
        target_title  = args.title,
        reset_db      = args.reset,
    )
