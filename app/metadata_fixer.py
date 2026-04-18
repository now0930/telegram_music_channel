"""
┌──────────────────────────────────────────────────────────────────────────┐
│  🔧  metadata_fixer.py  —  ChromaDB 메타데이터 수정 도구  v2             │
│                                                                            │
│  목적                                                                      │
│    ChromaDB(SQLite embedding_metadata) 에서 누락·오류 항목을 찾아         │
│    외부 API 를 통해 올바른 값으로 업데이트.                               │
│    ※ 새 색인(indexing/embedding) 은 절대 수행하지 않음.                   │
│                                                                            │
│  지원 필드 (--field / --fields / --all-fields)                             │
│  ┌──────────────────┬────────────────────────────────────────────────┐    │
│  │ 필드             │ 탐색 순서                                      │    │
│  ├──────────────────┼────────────────────────────────────────────────┤    │
│  │ year             │ MusicBrainz → Melon → Genie → Wikipedia → AudD│    │
│  │ genre            │ MusicBrainz → Melon → LastFM → Wikipedia       │    │
│  │ genre_fixed      │ genre 와 동일 소스 (AI태그 파생 필드)          │    │
│  │ album            │ MusicBrainz → Melon → Genie → AudD             │    │
│  │ artist           │ MusicBrainz(역조회) → AudD                     │    │
│  │ title            │ MusicBrainz(역조회) → AudD                     │    │
│  │ bpm              │ librosa 로컬 분석 (파일 직접 필요)              │    │
│  │ bpm_range        │ bpm 값에서 자동 계산 (API 불필요)               │    │
│  │ era              │ year 값에서 자동 계산 (API 불필요)              │    │
│  │ mood             │ LastFM + Wikipedia 텍스트 분석                  │    │
│  │ is_instrumental  │ title 키워드 + AudD 보컬 감지                  │    │
│  │ mb_id            │ MusicBrainz                                     │    │
│  │ lyrics           │ Melon → Genie → AudD                           │    │
│  └──────────────────┴────────────────────────────────────────────────┘    │
│                                                                            │
│  실행 예시                                                                 │
│    python metadata_fixer.py --status              # 전체 현황만 출력      │
│    python metadata_fixer.py                       # year 누락 전체 수정   │
│    python metadata_fixer.py --field genre         # genre 누락 수정       │
│    python metadata_fixer.py --all-fields          # 모든 필드 순서대로    │
│    python metadata_fixer.py --fields year,genre,album  # 복수 필드        │
│    python metadata_fixer.py --dry-run             # 미리보기 (DB 미수정)  │
│    python metadata_fixer.py --artist "아이유"     # 특정 아티스트만       │
│    python metadata_fixer.py --limit 50            # 최대 50곡             │
└──────────────────────────────────────────────────────────────────────────┐
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Optional

import chromadb
import requests
from dotenv import load_dotenv

import logging
import traceback
import difflib

# ══════════════════════════════════════════════════════════════════════════
#  ⚙️  전역 설정
# ══════════════════════════════════════════════════════════════════════════
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

DB_PATH    = "./music_vector_db"
COLLECTION = "music_library"

AUDD_API_KEY     = os.getenv("AUDD_API_KEY",     "")
MELON_API_KEY    = os.getenv("MELON_API_KEY",    "")
GENIE_APP_ID     = os.getenv("GENIE_APP_ID",     "")
GENIE_APP_SECRET = os.getenv("GENIE_APP_SECRET", "")
LASTFM_API_KEY   = os.getenv("LASTFM_API_KEY",   "")   # Last.fm (무료 가입)

# ══════════════════════════════════════════════════════════════════════════
#  🔑  API 키 매핑 사전 (아키텍처 개선)
# ══════════════════════════════════════════════════════════════════════════
SOURCE_KEYS = {
    "Melon": MELON_API_KEY,
    "Genie": GENIE_APP_ID,
    "AudD": AUDD_API_KEY,
    "LastFM": LASTFM_API_KEY,
    "LastFMArtist": LASTFM_API_KEY,
}

_MB_HEADERS = {
    "User-Agent": "MusicMetaFixer/2.0 (local-home-server)",
    "Accept":     "application/json",
}
_MB_RATE_LIMIT = 1.1   # MusicBrainz 1 req/sec

# ══════════════════════════════════════════════════════════════════════════
#  🎨  컬러 출력 헬퍼
# ══════════════════════════════════════════════════════════════════════════
CYAN    = "\033[96m";  GREEN   = "\033[92m";  YELLOW  = "\033[93m"
RED     = "\033[91m";  RESET   = "\033[0m";   BOLD    = "\033[1m"
DIM     = "\033[2m";   MAGENTA = "\033[95m"

def header(msg: str)  -> None:
    print(f"\n{BOLD}{'═'*68}{RESET}\n{BOLD}  {msg}{RESET}\n{BOLD}{'═'*68}{RESET}")

def section(msg: str) -> None:
    print(f"\n{CYAN}── {msg}{RESET}")

def ok(msg: str)      -> None: print(f"  {GREEN}✅  {msg}{RESET}")
def warn(msg: str)    -> None: print(f"  {YELLOW}⚠️   {msg}{RESET}")
def err(msg: str)     -> None: print(f"  {RED}❌  {msg}{RESET}")
def info(msg: str)    -> None: print(f"  {DIM}ℹ️   {msg}{RESET}")

def found(src: str, field: str, val: Any) -> None:
    print(f"  {GREEN}◆ {src:>16}{RESET}  {field} = {BOLD}{val!r}{RESET}")

# ══════════════════════════════════════════════════════════════════════════
#  📋  필드 메타 정보
#  형식: 필드명 → (빈값판단함수, 저장타입("int"|"str"), 설명)
# ══════════════════════════════════════════════════════════════════════════
def _is_empty_str(v: Any)  -> bool:
    return v is None or str(v).strip() == ""

def _is_empty_year(v: Any) -> bool:
    if v is None:
        return True
    s = str(v).strip().lower()
    return s in ("0", "", "none", "unknown")

def _is_empty_bpm(v: Any)  -> bool:
    return v is None or v == 0 or str(v).strip() in ("0", "")

def _is_empty_bool_str(v: Any) -> bool:
    return v is None or str(v).strip().lower() in ("", "none", "unknown")

FIELD_META: dict[str, tuple] = {
    "year":            (_is_empty_year,     "int", "발매연도"),
    "genre":           (_is_empty_str,      "str", "장르"),
    "genre_fixed":     (_is_empty_str,      "str", "AI태그 장르"),
    "album":           (_is_empty_str,      "str", "앨범명"),
    "artist":          (_is_empty_str,      "str", "아티스트"),
    "title":           (_is_empty_str,      "str", "곡 제목"),
    "bpm":             (_is_empty_bpm,      "int", "BPM"),
    "bpm_range":       (_is_empty_str,      "str", "BPM 범주"),
    "era":             (_is_empty_str,      "str", "시대 (XXXX년대)"),
    "mood":            (_is_empty_str,      "str", "분위기"),
    "is_instrumental": (_is_empty_bool_str, "str", "연주곡 여부"),
    "mb_id":           (_is_empty_str,      "str", "MusicBrainz ID"),
    "lyrics":          (_is_empty_str,      "str", "가사"),
}

# 전체 자동 수정 시 처리 순서
ALL_FIELDS_ORDER = [
    "year", "genre", "genre_fixed", "album", "artist",
    "title", "bpm", "bpm_range", "era", "mood",
    "is_instrumental", "mb_id", "lyrics",
]

# ══════════════════════════════════════════════════════════════════════════
#  🔌  DB 연결
# ══════════════════════════════════════════════════════════════════════════
def connect_db() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(name=COLLECTION)

# ══════════════════════════════════════════════════════════════════════════
#  📊  상태 출력 헬퍼
# ══════════════════════════════════════════════════════════════════════════
def print_status(col: chromadb.Collection) -> None:
    """DB 컬렉션 상태 정보를 출력."""
    section("📊 DB 상태")
    count = len(col)
    print(f"  {BOLD}총 곡 수:{RESET} {count}")
    print(f"  {BOLD}컬렉션 이름:{RESET} {col.name}")
    print(f"  {BOLD}DB 경로:{RESET} {DB_PATH}")
    print(f"  {BOLD}API 키 상태:{RESET}")
    print(f"    AUDD:   {'✅' if AUDD_API_KEY else '❌'}")
    print(f"    Melon:  {'✅' if MELON_API_KEY else '❌'}")
    print(f"    Genie:  {'✅' if GENIE_APP_ID else '❌'}")
    print(f"    Last.fm: {'✅' if LASTFM_API_KEY else '❌'}")

# ══════════════════════════════════════════════════════════════════════════
#  🔍  누락 항목 탐색
# ══════════════════════════════════════════════════════════════════════════
def find_missing(
    col:           chromadb.Collection,
    field:         str,
    artist_filter: str = "",
    limit:         int = 0,
) -> list[dict]:
    """field 값이 비어있는 항목 목록 반환."""
    check, _, _ = FIELD_META[field]
    all_data     = col.get(include=["metadatas"])
    missing: list[dict] = []
    for doc_id, meta in zip(all_data["ids"], all_data["metadatas"]):
        if artist_filter and artist_filter.lower() not in (meta.get("artist") or "").lower():
            continue
        if check(meta.get(field)):
            missing.append({"id": doc_id, "meta": meta})
        if limit and len(missing) >= limit:
            break
    return missing

# ══════════════════════════════════════════════════════════════════════════
#  🛠️  유틸리티
# ══════════════════════════════════════════════════════════════════════════
def _clean_year(raw: str) -> str:
    m = re.search(r"(1[89]\d{2}|20\d{2})", str(raw))
    return m.group(1) if m else ""

def _normalize_era(year_str: str) -> str:
    """'1998' → '1990년대'"""
    m = re.search(r"(1[89]\d{2}|20\d{2})", str(year_str))
    if not m:
        return ""
    y = int(m.group(1))
    return f"{(y // 10) * 10}년대"

def _bpm_to_range(bpm: Any) -> str:
    try:
        b = int(float(str(bpm)))
    except (ValueError, TypeError):
        return ""
    if   b < 60:  return "매우 느린 템포 (Largo)"
    elif b < 80:  return "느린 템포 (Andante)"
    elif b < 110: return "미디엄 템포 (Moderato)"
    elif b < 140: return "업비트 (Allegro)"
    else:          return "하이템포/댄스 (Presto)"

def _is_instrumental_by_title(title: str) -> bool:
    kw = ("inst", "instrumental", " mr", "(mr)", "karaoke", "반주", "연주")
    return any(k in title.lower() for k in kw)

# ══════════════════════════════════════════════════════════════════════════
#  🔑  API 키 진단 및 유틸리티
# ══════════════════════════════════════════════════════════════════════════
def check_api_keys() -> None:
    """현재 로드된 .env 를 바탕으로 활성화/비활성화 소스 요약 테이블 출력"""
    section("🔑 API 소스 활성화 상태 요약")
    
    # API 키가 필요한 소스 출력
    for name, key in SOURCE_KEYS.items():
        is_active = bool(key)
        status_str = f"{GREEN}✅ Active{RESET}" if is_active else f"{RED}❌ Disabled (키 미설정){RESET}"
        print(f"  {BOLD}{name:<15}{RESET} : {status_str}")
        
    # API 키가 필요 없는 소스 출력
    print(f"  {BOLD}{'MusicBrainz':<15}{RESET} : {GREEN}✅ Active (키 불필요){RESET}")
    print(f"  {BOLD}{'Wikipedia':<15}{RESET} : {GREEN}✅ Active (키 불필요){RESET}")
    print(f"  {BOLD}{'Librosa':<15}{RESET} : {GREEN}✅ Active (키 불필요){RESET}")

def normalize_query(text: str) -> str:
    """Removes special tags like Feat., (Prod. by), &, etc., to clean search queries."""
    if not text:
        return ""
    # Remove contents inside parentheses/brackets starting with feat, prod, etc.
    text = re.sub(r'\(?\s*(feat|feat\.|featuring|prod|prod\.|produced by)\s*[^)]*\)?', '', text, flags=re.IGNORECASE)
    # Remove special characters except alphanumeric, spaces, and Hangul
    text = re.sub(r'[^a-zA-Z0-9가-힣\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def is_similar_artist(original: str, fetched: str, threshold: float = 0.8) -> bool:
    """Checks if the fetched artist name is similar enough to the original."""
    if not original or not fetched:
        return False
    ratio = difflib.SequenceMatcher(None, original.lower().strip(), fetched.lower().strip()).ratio()
    logging.info(f"    {DIM}[유사도 검사] 원본: '{original}' vs 결과: '{fetched}' → {ratio:.2f}{RESET}")
    return ratio >= threshold

# ══════════════════════════════════════════════════════════════════════════
#  🌐  외부 API 함수들
# ══════════════════════════════════════════════════════════════════════════

def fetch_musicbrainz(artist: str, title: str, album: str = "") -> dict:
    """MusicBrainz recordings: year / genre / album / artist / mb_id 반환."""
    try:
        url = "https://musicbrainz.org/ws/2/recording"
        query_parts = [f'recording:"{title}"', f'artist:"{artist}"']
        if album:
            query_parts.append(f'release:"{album}"')
        params = {
            "query": " AND ".join(query_parts),
            "limit": 1, "fmt": "json",
            "inc":   "releases+artist-credits+tags",
        }
        logging.info(f"  {CYAN}[MusicBrainz 요청] URL: {url} | Params: {params}{RESET}")
        
        r = requests.get(
            url,
            params=params,
            headers=_MB_HEADERS, timeout=10,
        )
        logging.info(f"  {CYAN}[MusicBrainz 응답] Status: {r.status_code} | Body: {r.text[:200]}{RESET}")
        
        time.sleep(_MB_RATE_LIMIT)
        if r.status_code != 200:
            return {}
        recs = r.json().get("recordings", [])
        if not recs:
            return {}
        rec       = recs[0]
        releases  = rec.get("releases", [])
        year_val  = album_val = ""
        if releases:
            rel       = releases[0]
            album_val = rel.get("title", "")
            year_val  = _clean_year(rel.get("date", ""))
        tags      = sorted(rec.get("tags", []), key=lambda t: t.get("count", 0), reverse=True)
        genre_val = tags[0]["name"].title() if tags else ""
        credits   = rec.get("artist-credit", [])
        artist_val= credits[0].get("artist", {}).get("name", "") if credits else ""
        return {k: v for k, v in {
            "year":        year_val,
            "genre":       genre_val,
            "genre_fixed": genre_val,
            "album":       album_val,
            "artist":      artist_val,
            "mb_id":       rec.get("id", ""),
        }.items() if v}
    except Exception as e:
        logging.error(f"MusicBrainz 오류:\n{traceback.format_exc()}")
        return {}


def fetch_musicbrainz_by_mbid(mb_id: str) -> dict:
    """mb_id 로 직접 recording 조회 (역방향 보강)."""
    if not mb_id:
        return {}
    try:
        url = f"https://musicbrainz.org/ws/2/recording/{mb_id}"
        params = {"fmt": "json", "inc": "releases+artist-credits+tags"}
        logging.info(f"  {CYAN}[MusicBrainz MBID 요청] URL: {url} | Params: {params}{RESET}")
        
        r = requests.get(
            url,
            params=params,
            headers=_MB_HEADERS, timeout=10,
        )
        logging.info(f"  {CYAN}[MusicBrainz MBID 응답] Status: {r.status_code} | Body: {r.text[:200]}{RESET}")
        
        time.sleep(_MB_RATE_LIMIT)
        if r.status_code != 200:
            return {}
        rec       = r.json()
        releases  = rec.get("releases", [])
        year_val  = album_val = ""
        if releases:
            rel       = releases[0]
            album_val = rel.get("title", "")
            year_val  = _clean_year(rel.get("date", ""))
        tags      = sorted(rec.get("tags", []), key=lambda t: t.get("count", 0), reverse=True)
        genre_val = tags[0]["name"].title() if tags else ""
        credits   = rec.get("artist-credit", [])
        artist_val= credits[0].get("artist", {}).get("name", "") if credits else ""
        return {k: v for k, v in {
            "year":        year_val,
            "genre":       genre_val,
            "genre_fixed": genre_val,
            "album":       album_val,
            "artist":      artist_val,
            "title":       rec.get("title", ""),
        }.items() if v}
    except Exception as e:
        logging.error(f"MusicBrainz MBID 오류:\n{traceback.format_exc()}")
        return {}


def fetch_melon(artist: str, title: str) -> dict:
    """Melon Open API: year / genre / album / artist / lyrics 반환."""
    if not MELON_API_KEY:
        return {}
    try:
        url = "https://openapi.melon.com/v1/search/track"
        params = {"query": f"{artist} {title}", "count": 1}
        logging.info(f"  {CYAN}[Melon 요청] URL: {url} | Params: {params}{RESET}")
        
        r = requests.get(
            url,
            params=params,
            headers={"appKey": MELON_API_KEY}, timeout=10,
        )
        logging.info(f"  {CYAN}[Melon 응답] Status: {r.status_code} | Body: {r.text[:200]}{RESET}")
        
        tracks = r.json().get("response", {}).get("trackList", {}).get("track", [])
        if not tracks:
            return {}
        t = tracks[0]
        return {k: v for k, v in {
            "year":        _clean_year(t.get("issueDate", "")),
            "genre":       t.get("genreCode", ""),
            "genre_fixed": t.get("genreCode", ""),
            "album":       t.get("albumName", ""),
            "artist":      t.get("artistList", [{}])[0].get("artistName", ""),
            "lyrics":      t.get("lyrics", ""),
        }.items() if v}
    except Exception as e:
        logging.error(f"Melon 오류:\n{traceback.format_exc()}")
        return {}


def fetch_genie(artist: str, title: str) -> dict:
    """Genie Music API: year / album / artist / lyrics 반환."""
    if not (GENIE_APP_ID and GENIE_APP_SECRET):
        return {}
    try:
        url = "https://api.genie.co.kr/song/getSongInfo"
        params = {
            "appId":     GENIE_APP_ID,
            "appSecret": GENIE_APP_SECRET,
            "query":     f"{artist} {title}",
            "count":     1,
        }
        logging.info(f"  {CYAN}[Genie 요청] URL: {url} | Params: {params}{RESET}")
        
        r = requests.get(
            url,
            params=params,
            timeout=10,
        )
        logging.info(f"  {CYAN}[Genie 응답] Status: {r.status_code} | Body: {r.text[:200]}{RESET}")
        
        songs = r.json().get("response", {}).get("songList", [])
        if not songs:
            return {}
        s = songs[0]
        return {k: v for k, v in {
            "year":   _clean_year(s.get("releaseDate", "")),
            "album":  s.get("albumTitle",  ""),
            "artist": s.get("artistName",  ""),
            "lyrics": s.get("lyrics",      ""),
        }.items() if v}
    except Exception as e:
        logging.error(f"Genie 오류:\n{traceback.format_exc()}")
        return {}


def fetch_lastfm(artist: str, title: str) -> dict:
    """Last.fm track.getInfo: genre(tag) / year / album + summary 반환."""
    if not LASTFM_API_KEY:
        return {}
    try:
        url = "https://ws.audioscrobbler.com/2.0/"
        params = {
            "method":  "track.getInfo",
            "api_key": LASTFM_API_KEY,
            "artist":  artist,
            "track":   title,
            "format":  "json",
        }
        logging.info(f"  {CYAN}[Last.fm 요청] URL: {url} | Params: {params}{RESET}")
        
        r = requests.get(
            url,
            params=params,
            timeout=10,
        )
        logging.info(f"  {CYAN}[Last.fm 응답] Status: {r.status_code} | Body: {r.text[:200]}{RESET}")
        
        track = r.json().get("track", {})
        if not track:
            return {}
        tags_raw  = track.get("toptags", {}).get("tag", [])
        genre_val = tags_raw[0].get("name", "").title() if tags_raw else ""
        alb       = track.get("album", {})
        album_val = alb.get("title", "")
        wiki      = track.get("wiki", {})
        year_val  = _clean_year(wiki.get("published", ""))
        summary   = wiki.get("summary", "")
        return {k: v for k, v in {
            "year":             year_val,
            "genre":            genre_val,
            "genre_fixed":      genre_val,
            "album":            album_val,
            "_lastfm_summary":  summary[:400],
        }.items() if v}
    except Exception as e:
        logging.error(f"Last.fm 오류:\n{traceback.format_exc()}")
        return {}


def fetch_lastfm_artist(artist: str) -> dict:
    """Last.fm artist.getInfo: genre(tag) + bio 반환."""
    if not LASTFM_API_KEY:
        return {}
    try:
        url = "https://ws.audioscrobbler.com/2.0/"
        params = {
            "method":  "artist.getInfo",
            "api_key": LASTFM_API_KEY,
            "artist":  artist,
            "format":  "json",
        }
        logging.info(f"  {CYAN}[Last.fm Artist 요청] URL: {url} | Params: {params}{RESET}")
        
        r = requests.get(
            url,
            params=params,
            timeout=10,
        )
        logging.info(f"  {CYAN}[Last.fm Artist 응답] Status: {r.status_code} | Body: {r.text[:200]}{RESET}")
        
        art_data  = r.json().get("artist", {})
        if not art_data:
            return {}
        tags_raw  = art_data.get("tags", {}).get("tag", [])
        genre_val = tags_raw[0].get("name", "").title() if tags_raw else ""
        bio       = art_data.get("bio", {}).get("summary", "")
        return {k: v for k, v in {
            "genre":         genre_val,
            "genre_fixed":   genre_val,
            "_lastfm_bio":   bio[:400],
        }.items() if v}
    except Exception as e:
        logging.error(f"Last.fm artist 오류:\n{traceback.format_exc()}")
        return {}


def fetch_audd(file_path: str) -> dict:
    """AudD 오디오 인식: year / title / artist / album / lyrics 반환."""
    if not AUDD_API_KEY:
        return {}
    if not os.path.exists(file_path):
        return {}
    try:
        url = "https://api.audd.io/"
        params = {"api_token": AUDD_API_KEY, "return": "lyrics,spotify"}
        logging.info(f"  {CYAN}[AudD 요청] URL: {url} | Params: {params}{RESET}")
        
        with open(file_path, "rb") as f:
            chunk = f.read(500_000)
        
        r = requests.post(
            url,
            data=params,
            files={"file": ("audio.mp3", chunk, "audio/mpeg")},
            timeout=30,
        )
        logging.info(f"  {CYAN}[AudD 응답] Status: {r.status_code} | Body: {r.text[:200]}{RESET}")
        
        result = r.json().get("result") or {}
        if not result:
            return {}
        lyrics_field = result.get("lyrics", "")
        if isinstance(lyrics_field, dict):
            lyrics_field = lyrics_field.get("lyrics", "")
        return {k: v for k, v in {
            "year":   _clean_year(result.get("release_date", "")),
            "title":  result.get("title",  ""),
            "artist": result.get("artist", ""),
            "album":  result.get("album",  ""),
            "lyrics": lyrics_field,
        }.items() if v}
    except Exception as e:
        logging.error(f"AudD 오류:\n{traceback.format_exc()}")
        return {}


def fetch_wikipedia(artist: str, title: str) -> dict:
    """Wikipedia: year / genre 텍스트 파싱 + extract 반환."""
    result: dict = {}
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action":   "query",
            "list":     "search",
            "srsearch": f"{artist} {title} song",
            "srlimit":  2,
            "format":   "json",
        }
        logging.info(f"  {CYAN}[Wikipedia 요청] URL: {url} | Params: {params}{RESET}")
        
        r = requests.get(
            url,
            params=params,
            headers={"User-Agent": "MusicMetaFixer/2.0"}, timeout=6,
        )
        logging.info(f"  {CYAN}[Wikipedia 응답] Status: {r.status_code} | Body: {r.text[:200]}{RESET}")
        
        items = r.json().get("query", {}).get("search", [])
        for item in items[:2]:
            pg_title = item.get("title", "")
            if not pg_title:
                continue
            r2 = requests.get(
                url,
                params={
                    "action":      "query",
                    "prop":        "extracts",
                    "exintro":     True,
                    "explaintext": True,
                    "titles":      pg_title,
                    "format":      "json",
                },
                headers={"User-Agent": "MusicMetaFixer/2.0"}, timeout=6,
            )
            logging.info(f"  {CYAN}[Wikipedia Extract 요청] Status: {r2.status_code} | Body: {r2.text[:200]}{RESET}")
            
            pages = r2.json().get("query", {}).get("pages", {})
            for page in pages.values():
                extract = (page.get("extract") or "").strip()
                if not extract:
                    continue
                if not result.get("year"):
                    ym = re.search(
                        r"(?:released?|issued?|debut)[^\d]{0,20}(1[89]\d{2}|20\d{2})",
                        extract, re.IGNORECASE,
                    ) or re.search(r"\((1[89]\d{2}|20\d{2})\)", extract)
                    if ym:
                        result["year"] = ym.group(1)
                if not result.get("genre"):
                    gm = re.search(
                        r"genre[s]?\s*(?:include[s]?|:|is|are)\s*([\w\s/&-]{3,30})",
                        extract, re.IGNORECASE,
                    )
                    if gm:
                        g = gm.group(1).strip().title()
                        result["genre"]       = g
                        result["genre_fixed"] = g
                if not result.get("_wiki_extract"):
                    result["_wiki_extract"] = extract[:400]
    except Exception as e:
        logging.error(f"Wikipedia 오류:\n{traceback.format_exc()}")
    return result


# ══════════════════════════════════════════════════════════════════════════
#  🎭  mood 추론 (텍스트 키워드 기반)
# ══════════════════════════════════════════════════════════════════════════
_MOOD_KW: list[tuple[str, list[str]]] = [
    ("슬픔",   ["sad", "melancholy", "sorrow", "heartbreak", "tear", "grief", "슬픔", "눈물"]),
    ("밝음",   ["happy", "cheerful", "upbeat", "bright", "joyful", "밝음", "기쁨"]),
    ("잔잔함", ["calm", "peaceful", "quiet", "gentle", "soothing", "잔잔", "편안"]),
    ("감성적", ["emotional", "touching", "moving", "romantic", "nostalgic", "감성", "설렘"]),
    ("에너지", ["energetic", "powerful", "intense", "dynamic", "에너지", "강렬"]),
    ("몽환적", ["dreamy", "ethereal", "ambient", "hazy", "몽환", "신비"]),
    ("신남",   ["exciting", "fun", "dance", "party", "groove", "신나", "파티"]),
]

def infer_mood(text: str) -> str:
    if not text:
        return ""
    tl = text.lower()
    scores = {mood: sum(tl.count(k) for k in kws) for mood, kws in _MOOD_KW}
    scores = {m: s for m, s in scores.items() if s}
    return max(scores, key=scores.__getitem__) if scores else ""


# ══════════════════════════════════════════════════════════════════════════
#  🔑  필드별 탐색 전략
# ══════════════════════════════════════════════════════════════════════════
def lookup_field(field: str, meta: dict, file_path: str) -> Optional[Any]:
    """
    field 에 맞는 외부 API 를 순서대로 호출하여 새 값을 반환.
    찾지 못하면 None 반환. embedding 은 절대 건드리지 않음.
    """
    artist_raw = (meta.get("artist")  or "").strip()
    title_raw  = (meta.get("title")   or "").strip()
    
    # Normalized queries for better API hits
    artist  = normalize_query(artist_raw)
    title   = normalize_query(title_raw)
    
    album   = (meta.get("album")   or "").strip()
    mb_id   = (meta.get("mb_id")   or "").strip()
    
    # 동적 소스 필터링을 위한 헬퍼 함수
    def is_enabled(src_name: str) -> bool:
        """해당 소스의 API 키가 존재할 때만 True 반환"""
        return bool(SOURCE_KEYS.get(src_name))

    # 1. MusicBrainz (mb_id 역조회 우선)
    if mb_id:
        mb_by_id = fetch_musicbrainz_by_mbid(mb_id)
        if mb_by_id:
            sources: list[tuple[str, dict]] = [("MusicBrainz(mbid)", mb_by_id)]
        else:
            sources = []
    else:
        sources = []
    
    if artist and title:
        # MusicBrainz (키 불필요)
        if not any(s[0].startswith("MusicBrainz") for s in sources):
            mb = fetch_musicbrainz(artist, title, album)
            if mb:
                sources.append(("MusicBrainz", mb))

        # Melon (필터링 적용)
        if field in ("year", "genre", "genre_fixed", "album", "artist", "lyrics") and is_enabled("Melon"):
            ml = fetch_melon(artist, title)
            if ml:
                sources.append(("Melon", ml))

        # Genie (필터링 적용)
        if field in ("year", "album", "artist", "lyrics") and is_enabled("Genie"):
            gn = fetch_genie(artist, title)
            if gn:
                sources.append(("Genie", gn))

        # Last.fm (track) (필터링 적용)
        if field in ("genre", "genre_fixed", "year", "album") and is_enabled("LastFM"):
            lf = fetch_lastfm(artist, title)
            if lf:
                sources.append(("Last.fm", lf))

        # Last.fm (artist) — genre 보조 (필터링 적용)
        if field in ("genre", "genre_fixed") and is_enabled("LastFMArtist"):
            if not any(s[1].get(field) for s in sources):
                lfa = fetch_lastfm_artist(artist)
                if lfa:
                    sources.append(("Last.fm(artist)", lfa))

        # Wikipedia (키 불필요)
        if field in ("year", "genre", "genre_fixed"):
            wp = fetch_wikipedia(artist, title)
            if wp:
                sources.append(("Wikipedia", wp))
    
    # 2. AudD (파일 기반, 필터링 적용)
    if file_path and field in ("year", "album", "artist", "title", "lyrics") and is_enabled("AudD"):
        au = fetch_audd(file_path)
        if au:
            sources.append(("AudD", au))
    
    # 3. Mood 추론 (텍스트 기반)
    if field == "mood":
        text = (meta.get("lyrics") or meta.get("summary") or "").strip()
        if text:
            mood = infer_mood(text)
            if mood:
                sources.append(("MoodInfer", {"mood": mood}))
    
    # 4. BPM 계산 (파일 기반)
    if field == "bpm":
        if file_path and os.path.exists(file_path):
            try:
                import librosa
                y, sr = librosa.load(file_path, duration=30)
                tempo = librosa.beat.beat_track(y=y, sr=sr)
                bpm = int(round(tempo))
                if bpm:
                    sources.append(("Librosa", {"bpm": bpm}))
            except Exception:
                pass
    
    # 5. BPM Range 자동 계산
    if field == "bpm_range":
        bpm_val = meta.get("bpm")
        if bpm_val:
            sources.append(("AutoCalc", {"bpm_range": _bpm_to_range(bpm_val)}) if bpm_val else None)
    
    # 6. Era 자동 계산
    if field == "era":
        year_val = meta.get("year")
        if year_val:
            sources.append(("AutoCalc", {"era": _normalize_era(year_val)}) if year_val else None)
    
    # 7. Instrumental 자동 판정
    if field == "is_instrumental":
        if _is_instrumental_by_title(title_raw):
            sources.append(("AutoCalc", {"is_instrumental": "true"}))
        elif meta.get("is_instrumental"):
            sources.append(("AutoCalc", {"is_instrumental": "true"}))
    
    # 8. 첫 번째로 찾은 유효값 반환 (데이터 검증 로직 포함)
    for src_name, data in sources:
        val = data.get(field, "")
        if val and str(val).strip() not in ("", "0", "none", "None"):
            # If the field is 'year' or 'album', validate the artist similarity
            if field in ("year", "album") and src_name not in ("자동계산", "AudD"):
                fetched_artist = data.get("artist", "")
                if fetched_artist and not is_similar_artist(artist_raw, fetched_artist):
                    warn(f"[{src_name}] 아티스트 유사도 미달로 {field}={val} 폐기")
                    continue 
            
            found(src_name, field, val)
            return str(val).strip()

    return None


# ══════════════════════════════════════════════════════════════════════════
#  🚀  메인 실행
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChromaDB 메타데이터 수정 도구")
    parser.add_argument("--status", action="store_true", help="전체 상태만 출력")
    parser.add_argument("--field",  type=str, default="", help="수정할 필드명 (예: year)")
    parser.add_argument("--fields", type=str, default="", help="수정할 필드명 목록 (예: year,genre)")
    parser.add_argument("--all-fields", action="store_true", help="모든 필드 순서대로 수정")
    parser.add_argument("--artist", type=str, default="", help="아티스트 필터 (예: 아이유)")
    parser.add_argument("--limit",   type=int, default=0, help="최대 수정 개수 (0=전체)")
    parser.add_argument("--dry-run", action="store_true", help="미리보기 (DB 미수정)")
    args = parser.parse_args()

    # API 키 로드 확인
    check_api_keys()

    if args.status:
        col = connect_db()
        print_status(col)
        sys.exit(0)

    col = connect_db()
    if not col:
        err("DB 연결 실패")
        sys.exit(1)

    # 필드 목록 결정
    fields_to_fix: list[str] = []
    if args.all_fields:
        fields_to_fix = ALL_FIELDS_ORDER
    elif args.fields:
        fields_to_fix = [f.strip() for f in args.fields.split(",") if f.strip()]
    elif args.field:
        fields_to_fix = [args.field]
    else:
        fields_to_fix = ["year"]

    if not fields_to_fix:
        err("필드명 지정 없음 (--field, --fields, --all-fields 중 하나 필요)")
        sys.exit(1)

    # 전체 누락 항목 탐색
    all_missing: list[dict] = []
    for field in fields_to_fix:
        missing = find_missing(col, field, artist_filter=args.artist, limit=args.limit)
        all_missing.extend(missing)

    if not all_missing:
        ok("수정할 누락 항목 없음")
        sys.exit(0)

    # 수정 로직
    for item in all_missing:
        doc_id = item["id"]
        meta   = item["meta"]
        file_path = meta.get("file_path", "")
        
        for field in fields_to_fix:
            if field not in meta:
                continue
            
            new_val = lookup_field(field, meta, file_path)
            if new_val is not None:
                if not args.dry_run:
                    col.update(ids=[doc_id], metadatas=[{field: new_val}])
                ok(f"{doc_id[:8]}... {field} → {new_val}")
            else:
                info(f"{doc_id[:8]}... {field} → (찾을 수 없음)")

    if not args.dry_run:
        ok("수정이 완료되었습니다.")
    else:
        info("Dry-run 모드입니다. DB 는 수정되지 않았습니다.")
