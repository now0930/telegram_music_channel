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
└──────────────────────────────────────────────────────────────────────────┘
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
    return v is None or v == 0 or str(v).strip() in ("0", "")

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
#  🌐  외부 API 함수들
# ══════════════════════════════════════════════════════════════════════════

def fetch_musicbrainz(artist: str, title: str, album: str = "") -> dict:
    """MusicBrainz recordings: year / genre / album / artist / mb_id 반환."""
    try:
        query_parts = [f'recording:"{title}"', f'artist:"{artist}"']
        if album:
            query_parts.append(f'release:"{album}"')
        r = requests.get(
            "https://musicbrainz.org/ws/2/recording",
            params={
                "query": " AND ".join(query_parts),
                "limit": 1, "fmt": "json",
                "inc":   "releases+artist-credits+tags",
            },
            headers=_MB_HEADERS, timeout=10,
        )
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
        warn(f"MusicBrainz 오류: {e}")
        return {}


def fetch_musicbrainz_by_mbid(mb_id: str) -> dict:
    """mb_id 로 직접 recording 조회 (역방향 보강)."""
    if not mb_id:
        return {}
    try:
        r = requests.get(
            f"https://musicbrainz.org/ws/2/recording/{mb_id}",
            params={"fmt": "json", "inc": "releases+artist-credits+tags"},
            headers=_MB_HEADERS, timeout=10,
        )
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
        warn(f"MusicBrainz MBID 오류: {e}")
        return {}


def fetch_melon(artist: str, title: str) -> dict:
    """Melon Open API: year / genre / album / artist / lyrics 반환."""
    if not MELON_API_KEY:
        return {}
    try:
        r = requests.get(
            "https://openapi.melon.com/v1/search/track",
            params={"query": f"{artist} {title}", "count": 1},
            headers={"appKey": MELON_API_KEY}, timeout=10,
        )
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
        warn(f"Melon 오류: {e}")
        return {}


def fetch_genie(artist: str, title: str) -> dict:
    """Genie Music API: year / album / artist / lyrics 반환."""
    if not (GENIE_APP_ID and GENIE_APP_SECRET):
        return {}
    try:
        r = requests.get(
            "https://api.genie.co.kr/song/getSongInfo",
            params={
                "appId":     GENIE_APP_ID,
                "appSecret": GENIE_APP_SECRET,
                "query":     f"{artist} {title}",
                "count":     1,
            },
            timeout=10,
        )
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
        warn(f"Genie 오류: {e}")
        return {}


def fetch_lastfm(artist: str, title: str) -> dict:
    """Last.fm track.getInfo: genre(tag) / year / album + summary 반환."""
    if not LASTFM_API_KEY:
        return {}
    try:
        r = requests.get(
            "https://ws.audioscrobbler.com/2.0/",
            params={
                "method":  "track.getInfo",
                "api_key": LASTFM_API_KEY,
                "artist":  artist,
                "track":   title,
                "format":  "json",
            },
            timeout=10,
        )
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
        warn(f"Last.fm 오류: {e}")
        return {}


def fetch_lastfm_artist(artist: str) -> dict:
    """Last.fm artist.getInfo: genre(tag) + bio 반환."""
    if not LASTFM_API_KEY:
        return {}
    try:
        r = requests.get(
            "https://ws.audioscrobbler.com/2.0/",
            params={
                "method":  "artist.getInfo",
                "api_key": LASTFM_API_KEY,
                "artist":  artist,
                "format":  "json",
            },
            timeout=10,
        )
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
        warn(f"Last.fm artist 오류: {e}")
        return {}


def fetch_audd(file_path: str) -> dict:
    """AudD 오디오 인식: year / title / artist / album / lyrics 반환."""
    if not AUDD_API_KEY:
        return {}
    if not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(500_000)
        r = requests.post(
            "https://api.audd.io/",
            data={"api_token": AUDD_API_KEY, "return": "lyrics,spotify"},
            files={"file": ("audio.mp3", chunk, "audio/mpeg")},
            timeout=30,
        )
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
        warn(f"AudD 오류: {e}")
        return {}


def fetch_wikipedia(artist: str, title: str) -> dict:
    """Wikipedia: year / genre 텍스트 파싱 + extract 반환."""
    result: dict = {}
    try:
        query = f"{artist} {title} song"
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action":   "query",
                "list":     "search",
                "srsearch": query,
                "srlimit":  2,
                "format":   "json",
            },
            headers={"User-Agent": "MusicMetaFixer/2.0"}, timeout=6,
        )
        items = r.json().get("query", {}).get("search", [])
        for item in items[:2]:
            pg_title = item.get("title", "")
            if not pg_title:
                continue
            r2 = requests.get(
                "https://en.wikipedia.org/w/api.php",
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
        warn(f"Wikipedia 오류: {e}")
    return result


def analyze_bpm_local(file_path: str) -> dict:
    """librosa 로컬 BPM 분석 (파일 필요)."""
    if not os.path.exists(file_path):
        return {}
    try:
        import warnings
        import librosa
        warnings.filterwarnings("ignore")
        y, sr    = librosa.load(file_path, sr=22050, mono=True, offset=30, duration=30)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm      = int(round(float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)))
        return {"bpm": str(bpm)} if bpm > 0 else {}
    except ImportError:
        warn("librosa 미설치 — pip install librosa")
        return {}
    except Exception as e:
        warn(f"BPM 분석 오류: {e}")
        return {}


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
    artist  = (meta.get("artist")  or "").strip()
    title   = (meta.get("title")   or "").strip()
    album   = (meta.get("album")   or "").strip()
    mb_id   = (meta.get("mb_id")   or "").strip()
    bpm_cur = meta.get("bpm",  0)
    year_cur= str(meta.get("year", "") or "")

    # ── 자동 계산 필드 (API 불필요) ───────────────────────────────

    if field == "bpm_range":
        if not _is_empty_bpm(bpm_cur):
            val = _bpm_to_range(bpm_cur)
            if val:
                found("자동계산", field, val)
                return val
        return None

    if field == "era":
        if not _is_empty_year(year_cur):
            val = _normalize_era(year_cur)
            if val:
                found("자동계산", field, val)
                return val
        return None

    # ── BPM : librosa ────────────────────────────────────────────
    if field == "bpm":
        res = analyze_bpm_local(file_path)
        if res.get("bpm"):
            found("librosa", field, res["bpm"])
            return res["bpm"]
        return None

    # ── is_instrumental ───────────────────────────────────────────
    if field == "is_instrumental":
        if title and _is_instrumental_by_title(title):
            found("title키워드", field, "True")
            return "True"
        ad = fetch_audd(file_path)
        if ad.get("title"):
            val = str(_is_instrumental_by_title(ad["title"]))
            found("AudD", field, val)
            return val
        return None

    # ── mb_id ─────────────────────────────────────────────────────
    if field == "mb_id":
        if artist and title:
            mb = fetch_musicbrainz(artist, title, album)
            if mb.get("mb_id"):
                found("MusicBrainz", field, mb["mb_id"])
                return mb["mb_id"]
        return None

    # ── lyrics ────────────────────────────────────────────────────
    if field == "lyrics":
        if artist and title:
            for src_name, fn in [
                ("Melon", lambda: fetch_melon(artist, title)),
                ("Genie", lambda: fetch_genie(artist, title)),
            ]:
                res = fn()
                if res.get("lyrics"):
                    found(src_name, field, res["lyrics"][:30] + "…")
                    return res["lyrics"]
        ad = fetch_audd(file_path)
        if ad.get("lyrics"):
            found("AudD", field, ad["lyrics"][:30] + "…")
            return ad["lyrics"]
        return None

    # ── mood ──────────────────────────────────────────────────────
    if field == "mood":
        combined = ""
        if artist and title:
            combined += fetch_lastfm(artist, title).get("_lastfm_summary", "")
            combined += fetch_lastfm_artist(artist).get("_lastfm_bio", "")
            combined += fetch_wikipedia(artist, title).get("_wiki_extract", "")
        val = infer_mood(combined)
        if val:
            found("텍스트분석", field, val)
            return val
        return None

    # ── 공통 소스 풀 탐색: year / genre / genre_fixed / album / artist / title
    sources: list[tuple[str, dict]] = []

    # mb_id 역조회 우선
    if mb_id:
        mb_by_id = fetch_musicbrainz_by_mbid(mb_id)
        if mb_by_id:
            sources.append(("MusicBrainz(mbid)", mb_by_id))

    if artist and title:
        # MusicBrainz (중복 방지)
        if not any(s[0].startswith("MusicBrainz") for s in sources):
            mb = fetch_musicbrainz(artist, title, album)
            if mb:
                sources.append(("MusicBrainz", mb))

        # Melon
        if field in ("year", "genre", "genre_fixed", "album", "artist", "lyrics"):
            ml = fetch_melon(artist, title)
            if ml:
                sources.append(("Melon", ml))

        # Genie
        if field in ("year", "album", "artist", "lyrics"):
            gn = fetch_genie(artist, title)
            if gn:
                sources.append(("Genie", gn))

        # Last.fm (track)
        if field in ("genre", "genre_fixed", "year", "album"):
            lf = fetch_lastfm(artist, title)
            if lf:
                sources.append(("Last.fm", lf))

        # Last.fm (artist) — genre 보조
        if field in ("genre", "genre_fixed"):
            if not any(s[1].get(field) for s in sources):
                lfa = fetch_lastfm_artist(artist)
                if lfa:
                    sources.append(("Last.fm(artist)", lfa))

        # Wikipedia
        if field in ("year", "genre", "genre_fixed"):
            wp = fetch_wikipedia(artist, title)
            if wp:
                sources.append(("Wikipedia", wp))

    # artist / title 없을 때 → AudD
    if (not artist or not title) and field in (
        "year", "genre", "genre_fixed", "album", "artist", "title", "lyrics"
    ):
        ad = fetch_audd(file_path)
        if ad:
            sources.append(("AudD", ad))

    # 첫 번째로 찾은 유효값 반환
    for src_name, data in sources:
        val = data.get(field, "")
        if val and str(val).strip() not in ("", "0", "none", "None"):
            found(src_name, field, val)
            return str(val).strip()

    return None


# ══════════════════════════════════════════════════════════════════════════
#  💾  DB 업데이트 (embedding 보존)
# ══════════════════════════════════════════════════════════════════════════
def update_field(
    col:     chromadb.Collection,
    doc_id:  str,
    meta:    dict,
    field:   str,
    new_val: Any,
) -> bool:
    """embedding / document 는 그대로, 해당 field 만 수정 후 upsert."""
    try:
        existing     = col.get(ids=[doc_id], include=["embeddings", "documents"])
        existing_emb = existing["embeddings"][0] if existing.get("embeddings") else None
        existing_doc = existing["documents"][0]  if existing.get("documents")  else ""

        updated_meta = dict(meta)
        _, store_type, _ = FIELD_META[field]
        if store_type == "int":
            try:
                updated_meta[field] = int(float(str(new_val)))
            except (ValueError, TypeError):
                updated_meta[field] = 0
        else:
            updated_meta[field] = str(new_val)

        kwargs: dict = {
            "ids":       [doc_id],
            "metadatas": [updated_meta],
            "documents": [existing_doc],
        }
        if existing_emb is not None:
            kwargs["embeddings"] = [existing_emb]

        col.upsert(**kwargs)
        return True
    except Exception as e:
        err(f"DB 업데이트 실패: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════
#  📊  현황 통계 출력
# ══════════════════════════════════════════════════════════════════════════
def print_status(col: chromadb.Collection) -> None:
    header("📊 메타데이터 누락 현황")
    all_data = col.get(include=["metadatas"])
    total    = len(all_data["ids"])
    metas    = all_data["metadatas"]

    if total == 0:
        warn("DB 가 비어있습니다.")
        return

    print(f"\n  총 곡 수: {BOLD}{total}{RESET}\n")
    print(f"  {'필드':<16} {'설명':<16} {'누락':>5} {'비율':>7}   {'진행바 (20칸)'}")
    print(f"  {'─'*16} {'─'*16} {'─'*5} {'─'*7}   {'─'*22}")

    for field in ALL_FIELDS_ORDER:
        check, _, desc = FIELD_META[field]
        miss  = sum(1 for m in metas if check(m.get(field)))
        pct   = miss / total * 100
        bar_n = int(pct / 5)
        bar   = "█" * bar_n + "░" * (20 - bar_n)
        color = RED if pct > 50 else YELLOW if pct > 10 else GREEN
        print(
            f"  {field:<16} {desc:<16} {color}{miss:>5}{RESET} "
            f"{pct:>6.1f}%   [{color}{bar}{RESET}]"
        )

    # ── year 연도 분포 ──────────────────────────────────────────
    section("year 연도 분포  (빨강 = 0/누락)")
    year_dist: dict[str, int] = {}
    for m in metas:
        y = str(m.get("year", 0))
        year_dist[y] = year_dist.get(y, 0) + 1
    for yr, cnt in sorted(year_dist.items())[:25]:
        bar = "█" * min(cnt, 50)
        clr = RED if yr in ("0", "") else RESET
        print(f"  {clr}{yr:>6}{RESET}  {bar}  ({cnt})")
    if len(year_dist) > 25:
        print(f"  {DIM}... 외 {len(year_dist)-25}개 연도{RESET}")

    # ── genre 분포 (상위 15) ────────────────────────────────────
    section("genre 분포  (상위 15)")
    genre_dist: dict[str, int] = {}
    for m in metas:
        g = str(m.get("genre") or "").strip() or "(없음)"
        genre_dist[g] = genre_dist.get(g, 0) + 1
    for g, cnt in sorted(genre_dist.items(), key=lambda x: -x[1])[:15]:
        bar = "█" * min(cnt, 50)
        clr = RED if g == "(없음)" else RESET
        print(f"  {clr}{g:<22}{RESET}  {bar}  ({cnt})")
    print()


# ══════════════════════════════════════════════════════════════════════════
#  🔧  단일 필드 수정 루프
# ══════════════════════════════════════════════════════════════════════════
def run_fixer(
    col:           chromadb.Collection,
    field:         str,
    artist_filter: str  = "",
    limit:         int  = 0,
    dry_run:       bool = False,
) -> tuple[int, int]:
    """누락 항목을 찾아 수정. (fixed, failed) 반환."""
    _, _, desc = FIELD_META[field]

    section(f"누락 탐색  [{field}]  {desc}")
    missing = find_missing(col, field, artist_filter, limit)
    total   = len(missing)

    if not total:
        ok(f"누락 없음  (field={field})")
        return 0, 0

    print(f"\n  수정 대상: {BOLD}{RED}{total}{RESET}곡")
    print(f"\n  {'#':<5} {'아티스트':<24} {'제목':<32} {'현재값'}")
    print(f"  {'─'*5} {'─'*24} {'─'*32} {'─'*14}")
    for i, item in enumerate(missing[:10], 1):
        m   = item["meta"]
        cur = str(m.get(field, "-"))[:14]
        print(
            f"  {i:<5} {(m.get('artist') or '?')[:23]:<24} "
            f"{(m.get('title') or m.get('filename') or '?')[:31]:<32} {cur}"
        )
    if total > 10:
        print(f"  {DIM}... 외 {total-10}곡{RESET}")

    fixed = failed = 0
    start = datetime.now()

    for idx, item in enumerate(missing, 1):
        doc_id = item["id"]
        meta   = item["meta"]
        artist = (meta.get("artist") or "").strip()
        titl   = (meta.get("title")  or "").strip()
        fname  = meta.get("filename", os.path.basename(doc_id))
        disp   = f"{artist or '?'} - {titl or fname}"
        elapsed= int((datetime.now() - start).total_seconds())

        print(
            f"\n  [{idx:>4}/{total}]  {disp[:62]}"
            f"  {DIM}(경과 {elapsed}s){RESET}"
        )

        new_val = lookup_field(field, meta, doc_id)

        if new_val is not None:
            if dry_run:
                ok(f"[DRY-RUN]  {field} = {str(new_val)[:60]!r}  (미적용)")
                fixed += 1
            else:
                if update_field(col, doc_id, meta, field, new_val):
                    old = str(meta.get(field, "?"))[:20]
                    ok(f"{field}: {old!r} → {str(new_val)[:40]!r}")
                    fixed += 1
                else:
                    failed += 1
        else:
            warn(f"탐색 실패 — {field} 값을 찾을 수 없음")
            failed += 1

    return fixed, failed


# ══════════════════════════════════════════════════════════════════════════
#  🚀  전체 실행 (단일 / 복수 필드)
# ══════════════════════════════════════════════════════════════════════════
def run(
    fields:        list[str],
    artist_filter: str  = "",
    limit:         int  = 0,
    dry_run:       bool = False,
) -> None:
    col = connect_db()

    header("🔧 메타데이터 수정기  v2")
    print(f"  DB 경로      : {DB_PATH}")
    print(f"  총 항목 수   : {col.count()}")
    print(f"  대상 필드    : {BOLD}{', '.join(fields)}{RESET}")
    if artist_filter: print(f"  아티스트 필터: {artist_filter}")
    if limit:         print(f"  처리 한도    : {limit}곡")
    if dry_run:       print(f"  {YELLOW}** DRY-RUN 모드 — DB 미수정 **{RESET}")

    # 실행 전 현황
    print_status(col)

    grand_fixed = grand_failed = 0
    start_all   = datetime.now()

    for field in fields:
        header(f"📝 처리 중: {field}  ({FIELD_META[field][2]})")
        fx, fl = run_fixer(col, field, artist_filter, limit, dry_run)
        grand_fixed  += fx
        grand_failed += fl

    # 최종 리포트
    elapsed = int((datetime.now() - start_all).total_seconds())
    header("📋 최종 완료 리포트")
    print(f"""
  처리 필드    : {', '.join(fields)}
  ✅ 수정 성공 : {GREEN}{grand_fixed}{RESET}
  ❌ 탐색 실패 : {RED}{grand_failed}{RESET}
  소요 시간    : {elapsed}초
  완료 시각    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  {'[DRY-RUN — 실제 DB 변경 없음]' if dry_run else ''}
""")

    if not dry_run:
        section("수정 후 현황")
        print_status(col)


# ══════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    SUPPORTED = list(FIELD_META.keys())

    parser = argparse.ArgumentParser(
        description="ChromaDB 음악 메타데이터 수정 도구 v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "지원 필드:\n" +
            "".join(f"  {f:<16} — {FIELD_META[f][2]}\n" for f in ALL_FIELDS_ORDER) +
            """
예시:
  python metadata_fixer.py --status
  python metadata_fixer.py                              # year 누락 수정
  python metadata_fixer.py --field genre                # genre 수정
  python metadata_fixer.py --fields year,genre,album    # 복수 필드
  python metadata_fixer.py --all-fields                 # 모든 필드 순서대로
  python metadata_fixer.py --dry-run --fields year,bpm  # 미리보기
  python metadata_fixer.py --artist "아이유" --limit 20
"""
        ),
    )
    parser.add_argument(
        "--field",  default="year", choices=SUPPORTED, metavar="FIELD",
        help=f"수정할 필드 1개 (기본: year). 선택: {', '.join(SUPPORTED)}",
    )
    parser.add_argument(
        "--fields", default="", metavar="F1,F2,...",
        help="콤마 구분 복수 필드 (--field 보다 우선)",
    )
    parser.add_argument(
        "--all-fields", action="store_true",
        help="모든 필드를 순서대로 처리",
    )
    parser.add_argument("--artist",  default="", metavar="가수명",
                        help="특정 아티스트만")
    parser.add_argument("--limit",   type=int, default=0, metavar="N",
                        help="최대 처리 곡 수 (0=무제한)")
    parser.add_argument("--dry-run", action="store_true",
                        help="탐색만, DB 미수정")
    parser.add_argument("--status",  action="store_true",
                        help="현황 통계만 출력 후 종료")
    args = parser.parse_args()

    if args.status:
        print_status(connect_db())
        sys.exit(0)

    # 필드 목록 결정
    if args.all_fields:
        target_fields = ALL_FIELDS_ORDER
    elif args.fields:
        raw = [f.strip() for f in args.fields.split(",") if f.strip()]
        bad = [f for f in raw if f not in SUPPORTED]
        if bad:
            print(f"[오류] 지원하지 않는 필드: {bad}\n지원 목록: {SUPPORTED}")
            sys.exit(1)
        target_fields = raw
    else:
        target_fields = [args.field]

    run(
        fields        = target_fields,
        artist_filter = args.artist,
        limit         = args.limit,
        dry_run       = args.dry_run,
    )
