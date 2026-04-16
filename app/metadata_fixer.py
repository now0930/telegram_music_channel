"""
┌──────────────────────────────────────────────────────────────────────┐
│  🔧  metadata_fixer.py  —  ChromaDB 메타데이터 수정 도구             │
│                                                                        │
│  목적                                                                  │
│    ChromaDB / SQLite embedding_metadata 에서 특정 필드가 누락·오류인  │
│    항목을 찾아 외부 API 를 통해 올바른 값으로 업데이트.               │
│    ※ 새로운 색인(indexing)은 절대 수행하지 않음.                       │
│                                                                        │
│  현재 지원 필드                                                        │
│    ✅ year   — MusicBrainz → Melon → Genie → AudD → Wikipedia 순 탐색 │
│    (향후 확장: genre / album / artist 등 --field 옵션으로 추가 가능)   │
│                                                                        │
│  실행 예시                                                             │
│    python metadata_fixer.py                    # year 0인 항목 전체 수정│
│    python metadata_fixer.py --field year       # 명시적으로 year 지정  │
│    python metadata_fixer.py --dry-run          # 실제 수정 없이 미리 보기│
│    python metadata_fixer.py --artist "아이유"  # 특정 아티스트만       │
│    python metadata_fixer.py --limit 50         # 최대 50곡만           │
│    python metadata_fixer.py --status           # 전체 현황 요약만 출력 │
└──────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

import chromadb
import requests

# ══════════════════════════════════════════════════════════════════════
#  ⚙️  설정
# ══════════════════════════════════════════════════════════════════════
DB_PATH    = "./music_vector_db"
COLLECTION = "music_library"

# API 키 — 환경변수 또는 직접 입력
import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

AUDD_API_KEY     = os.getenv("AUDD_API_KEY",     "")
MELON_API_KEY    = os.getenv("MELON_API_KEY",    "")
GENIE_APP_ID     = os.getenv("GENIE_APP_ID",     "")
GENIE_APP_SECRET = os.getenv("GENIE_APP_SECRET", "")

_MB_HEADERS = {
    "User-Agent": "MusicMetaFixer/1.0 (local-home-server)",
    "Accept":     "application/json",
}

# MusicBrainz rate limit (1 req/sec)
_MB_RATE_LIMIT = 1.1

# ══════════════════════════════════════════════════════════════════════
#  🖨️  출력 헬퍼
# ══════════════════════════════════════════════════════════════════════
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"

def _c(color: str, text: str) -> str:
    return f"{color}{text}{RESET}"

def header(msg: str) -> None:
    print(f"\n{BOLD}{'═' * 64}{RESET}")
    print(f"{BOLD}  {msg}{RESET}")
    print(f"{BOLD}{'═' * 64}{RESET}")

def section(msg: str) -> None:
    print(f"\n{CYAN}── {msg}{RESET}")

def ok(msg: str)   -> None: print(f"  {GREEN}✅  {msg}{RESET}")
def warn(msg: str) -> None: print(f"  {YELLOW}⚠️   {msg}{RESET}")
def err(msg: str)  -> None: print(f"  {RED}❌  {msg}{RESET}")
def info(msg: str) -> None: print(f"  {DIM}ℹ️   {msg}{RESET}")

# ══════════════════════════════════════════════════════════════════════
#  🔌  ChromaDB 연결
# ══════════════════════════════════════════════════════════════════════
def connect_db() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=DB_PATH)
    col    = client.get_or_create_collection(name=COLLECTION)
    return col

# ══════════════════════════════════════════════════════════════════════
#  🔍  누락 항목 탐색
# ══════════════════════════════════════════════════════════════════════

# 각 필드별 "비어있는(수정 대상)" 판단 기준
FIELD_EMPTY_CHECK: dict[str, callable] = {
    "year":   lambda v: v is None or v == "" or v == 0 or str(v).strip() in ("0", ""),
    "genre":  lambda v: v is None or str(v).strip() == "",
    "album":  lambda v: v is None or str(v).strip() == "",
    "artist": lambda v: v is None or str(v).strip() == "",
}

def find_missing(
    col:         chromadb.Collection,
    field:       str,
    artist_filter: str = "",
    limit:       int   = 0,
) -> list[dict]:
    """
    field 값이 비어있는 ChromaDB 항목 목록을 반환.
    반환 형식: [{"id": path, "meta": {...}}, ...]
    """
    check = FIELD_EMPTY_CHECK.get(field)
    if check is None:
        raise ValueError(f"지원하지 않는 필드: {field!r}. 지원 목록: {list(FIELD_EMPTY_CHECK)}")

    all_data = col.get(include=["metadatas"])
    ids      = all_data["ids"]
    metas    = all_data["metadatas"]

    missing = []
    for doc_id, meta in zip(ids, metas):
        if artist_filter and artist_filter.lower() not in (meta.get("artist") or "").lower():
            continue
        if check(meta.get(field)):
            missing.append({"id": doc_id, "meta": meta})
        if limit and len(missing) >= limit:
            break

    return missing

# ══════════════════════════════════════════════════════════════════════
#  🌐  외부 API 조회 함수들 (year 전용이지만 구조는 범용)
# ══════════════════════════════════════════════════════════════════════

def _clean_year(raw: str) -> str:
    """연도 문자열에서 4자리 숫자만 추출."""
    m = re.search(r"(1[89]\d{2}|20\d{2})", str(raw))
    return m.group(1) if m else ""


def fetch_musicbrainz(artist: str, title: str, album: str = "") -> dict:
    """MusicBrainz 에서 year / genre / mb_id 조회."""
    try:
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
        time.sleep(_MB_RATE_LIMIT)   # rate limit

        if r.status_code != 200:
            return {}

        recs = r.json().get("recordings", [])
        if not recs:
            return {}

        rec     = recs[0]
        releases = rec.get("releases", [])
        year_val = ""
        album_val = ""
        if releases:
            rel       = releases[0]
            album_val = rel.get("title", "")
            date_str  = rel.get("date", "")
            year_val  = _clean_year(date_str)

        tags     = sorted(rec.get("tags", []), key=lambda t: t.get("count", 0), reverse=True)
        genre_val = tags[0]["name"] if tags else ""

        return {k: v for k, v in {
            "year":  year_val,
            "genre": genre_val,
            "album": album_val,
            "mb_id": rec.get("id", ""),
        }.items() if v}

    except Exception as e:
        warn(f"MusicBrainz 오류: {e}")
        return {}


def fetch_melon(artist: str, title: str) -> dict:
    """Melon Open API 에서 year 조회."""
    if not MELON_API_KEY:
        return {}
    try:
        r = requests.get(
            "https://openapi.melon.com/v1/search/track",
            params  = {"query": f"{artist} {title}", "count": 1},
            headers = {"appKey": MELON_API_KEY},
            timeout = 10,
        )
        tracks = r.json().get("response", {}).get("trackList", {}).get("track", [])
        if not tracks:
            return {}
        t = tracks[0]
        return {k: v for k, v in {
            "year":  _clean_year(t.get("issueDate", "")),
            "genre": t.get("genreCode", ""),
            "album": t.get("albumName", ""),
        }.items() if v}
    except Exception as e:
        warn(f"Melon 오류: {e}")
        return {}


def fetch_genie(artist: str, title: str) -> dict:
    """Genie Music API 에서 year 조회."""
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
        return {k: v for k, v in {
            "year":  _clean_year(s.get("releaseDate", "")),
            "album": s.get("albumTitle", ""),
        }.items() if v}
    except Exception as e:
        warn(f"Genie 오류: {e}")
        return {}


def fetch_audd(file_path: str) -> dict:
    """AudD 오디오 핑거프린팅으로 year 조회 (artist/title 없을 때 사용)."""
    if not AUDD_API_KEY:
        return {}
    if not os.path.exists(file_path):
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
        return {k: v for k, v in {
            "year":   _clean_year(result.get("release_date", "")),
            "title":  result.get("title", ""),
            "artist": result.get("artist", ""),
            "album":  result.get("album", ""),
        }.items() if v}
    except Exception as e:
        warn(f"AudD 오류: {e}")
        return {}


def fetch_wikipedia(artist: str, title: str) -> dict:
    """Wikipedia API 에서 연도 텍스트 파싱으로 year 보조 탐색."""
    try:
        query = f"{artist} {title} song"
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params  = {
                "action":   "query",
                "list":     "search",
                "srsearch": query,
                "srlimit":  2,
                "format":   "json",
            },
            headers = {"User-Agent": "MusicMetaFixer/1.0"},
            timeout = 6,
        )
        results = r.json().get("query", {}).get("search", [])
        for item in results[:2]:
            pg_title = item.get("title", "")
            if not pg_title:
                continue
            r2 = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params  = {
                    "action":      "query",
                    "prop":        "extracts",
                    "exintro":     True,
                    "explaintext": True,
                    "titles":      pg_title,
                    "format":      "json",
                },
                headers = {"User-Agent": "MusicMetaFixer/1.0"},
                timeout = 6,
            )
            pages = r2.json().get("query", {}).get("pages", {})
            for page in pages.values():
                extract = (page.get("extract") or "").strip()
                if not extract:
                    continue
                # 연도 패턴 탐색: "released in 1998", "(1998)", "released: 1998" 등
                year_m = re.search(
                    r"(?:released?|issued?|debut)[^\d]{0,20}(1[89]\d{2}|20\d{2})",
                    extract, re.IGNORECASE
                ) or re.search(r"\((1[89]\d{2}|20\d{2})\)", extract)
                if year_m:
                    return {"year": year_m.group(1)}
    except Exception as e:
        warn(f"Wikipedia 오류: {e}")
    return {}


# ══════════════════════════════════════════════════════════════════════
#  🔑  필드별 외부 API 탐색 전략 (확장 가능)
# ══════════════════════════════════════════════════════════════════════

def lookup_field(field: str, meta: dict, file_path: str) -> Optional[str]:
    """
    주어진 field 를 외부 API 에서 조회하여 새 값을 반환.
    찾지 못하면 None 반환.

    현재 지원:
      - year : MusicBrainz → Melon → Genie → AudD → Wikipedia 순
      - genre: MusicBrainz
      - album: MusicBrainz → Melon → Genie → AudD
      - artist: AudD (artist/title 없을 때)

    향후 확장 시 이 함수에 elif 블록만 추가하면 됩니다.
    """
    artist = (meta.get("artist") or "").strip()
    title  = (meta.get("title")  or "").strip()
    album  = (meta.get("album")  or "").strip()

    sources: list[tuple[str, dict]] = []

    # ── year ──────────────────────────────────────────────────────
    if field == "year":
        if artist and title:
            mb = fetch_musicbrainz(artist, title, album)
            if mb:
                sources.append(("MusicBrainz", mb))

            ml = fetch_melon(artist, title)
            if ml:
                sources.append(("Melon", ml))

            gn = fetch_genie(artist, title)
            if gn:
                sources.append(("Genie", gn))

            wp = fetch_wikipedia(artist, title)
            if wp:
                sources.append(("Wikipedia", wp))
        else:
            ad = fetch_audd(file_path)
            if ad:
                sources.append(("AudD", ad))

    # ── genre ─────────────────────────────────────────────────────
    elif field == "genre":
        if artist and title:
            mb = fetch_musicbrainz(artist, title, album)
            if mb:
                sources.append(("MusicBrainz", mb))

    # ── album ─────────────────────────────────────────────────────
    elif field == "album":
        if artist and title:
            mb = fetch_musicbrainz(artist, title, album)
            if mb:
                sources.append(("MusicBrainz", mb))
            ml = fetch_melon(artist, title)
            if ml:
                sources.append(("Melon", ml))
            gn = fetch_genie(artist, title)
            if gn:
                sources.append(("Genie", gn))
        else:
            ad = fetch_audd(file_path)
            if ad:
                sources.append(("AudD", ad))

    # ── artist ────────────────────────────────────────────────────
    elif field == "artist":
        if not artist:
            ad = fetch_audd(file_path)
            if ad:
                sources.append(("AudD", ad))

    # ── 새 필드 추가 예시 ─────────────────────────────────────────
    # elif field == "new_field":
    #     ...

    # ── 결과에서 field 값 추출 ─────────────────────────────────────
    for source_name, data in sources:
        val = data.get(field, "")
        if val and str(val).strip() not in ("", "0"):
            info(f"{source_name} → {field} = {val!r}")
            return str(val).strip()

    return None


# ══════════════════════════════════════════════════════════════════════
#  📊  현황 통계
# ══════════════════════════════════════════════════════════════════════

def print_status(col: chromadb.Collection) -> None:
    """전체 DB 의 필드별 누락 현황을 테이블로 출력."""
    header("📊 메타데이터 현황 요약")

    all_data = col.get(include=["metadatas"])
    total    = len(all_data["ids"])
    metas    = all_data["metadatas"]

    if total == 0:
        warn("DB 가 비어있습니다.")
        return

    print(f"\n  총 곡 수: {BOLD}{total}{RESET}\n")

    rows = []
    for field, check in FIELD_EMPTY_CHECK.items():
        missing_cnt = sum(1 for m in metas if check(m.get(field)))
        pct         = missing_cnt / total * 100
        bar_len     = int(pct / 5)
        bar         = "█" * bar_len + "░" * (20 - bar_len)
        color       = RED if pct > 50 else YELLOW if pct > 10 else GREEN
        rows.append((field, missing_cnt, total, pct, bar, color))

    print(f"  {'필드':<12} {'누락':<8} {'전체':<8} {'비율':<8}  {'[        20칸 바       ]'}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8}  {'─'*22}")
    for field, miss, tot, pct, bar, color in rows:
        print(f"  {field:<12} {color}{miss:<8}{RESET} {tot:<8} {pct:>5.1f}%   [{color}{bar}{RESET}]")

    print()

    # year 상세: 0값 vs 실제 연도 분포 (상위 10개)
    section("year 분포 (0 = 누락)")
    year_dist: dict[str, int] = {}
    for m in metas:
        y = str(m.get("year", 0))
        year_dist[y] = year_dist.get(y, 0) + 1

    sorted_years = sorted(year_dist.items(), key=lambda x: x[0])
    for yr, cnt in sorted_years[:20]:
        bar = "█" * min(cnt, 40)
        color = RED if yr in ("0", "") else RESET
        print(f"  {color}{yr:>6}{RESET}  {bar}  ({cnt})")
    if len(sorted_years) > 20:
        print(f"  ... 외 {len(sorted_years) - 20}개 연도")


# ══════════════════════════════════════════════════════════════════════
#  🔧  메인 수정 루프
# ══════════════════════════════════════════════════════════════════════

def run_fixer(
    field:         str  = "year",
    artist_filter: str  = "",
    limit:         int  = 0,
    dry_run:       bool = False,
) -> None:

    col = connect_db()

    header(f"🔧 메타데이터 수정기  —  필드: {field}")
    print(f"  DB 경로    : {DB_PATH}")
    print(f"  총 항목 수 : {col.count()}")
    print(f"  대상 필드  : {BOLD}{field}{RESET}")
    if artist_filter:
        print(f"  아티스트 필터: {artist_filter}")
    if limit:
        print(f"  처리 한도  : {limit}곡")
    if dry_run:
        print(f"  {YELLOW}** DRY-RUN 모드 — 실제 DB 는 수정하지 않습니다 **{RESET}")

    # ── 누락 항목 탐색 ────────────────────────────────────────────
    section(f"누락 항목 탐색 중 (field={field}) ...")
    missing = find_missing(col, field, artist_filter, limit)
    total_missing = len(missing)

    if not missing:
        ok(f"누락 항목이 없습니다. (field={field})")
        return

    print(f"\n  수정 대상: {BOLD}{RED}{total_missing}{RESET}곡\n")
    print(f"  {'#':<5} {'아티스트':<25} {'제목':<35} {'현재값'}")
    print(f"  {'─'*5} {'─'*25} {'─'*35} {'─'*10}")
    for i, item in enumerate(missing[:10], 1):
        m = item["meta"]
        print(f"  {i:<5} {(m.get('artist') or '?')[:24]:<25} {(m.get('title') or m.get('filename') or '?')[:34]:<35} {str(m.get(field, '-'))}")
    if total_missing > 10:
        print(f"  ... 외 {total_missing - 10}곡")

    # ── 진행 루프 ─────────────────────────────────────────────────
    section("외부 API 탐색 및 수정 시작")

    fixed_count   = 0
    failed_count  = 0
    skip_count    = 0
    start_time    = datetime.now()

    for idx, item in enumerate(missing, 1):
        doc_id    = item["id"]
        meta      = item["meta"]
        artist    = (meta.get("artist") or "").strip()
        title_val = (meta.get("title")  or "").strip()
        file_path = doc_id   # ChromaDB id == 파일 경로

        display = f"{artist or '?'} - {title_val or meta.get('filename', '?')}"
        elapsed = (datetime.now() - start_time).seconds
        print(
            f"\n  [{idx:>4}/{total_missing}] {display[:60]}"
            f"  {DIM}(경과: {elapsed}s){RESET}"
        )

        new_val = lookup_field(field, meta, file_path)

        if new_val:
            if dry_run:
                ok(f"[DRY-RUN] {field} = {new_val!r}  (미적용)")
                fixed_count += 1
            else:
                # ── ChromaDB upsert (embedding 은 건드리지 않음) ──
                # get 으로 기존 embedding 가져와서 그대로 재사용
                try:
                    existing = col.get(ids=[doc_id], include=["embeddings", "documents"])
                    existing_emb = existing["embeddings"][0] if existing["embeddings"] else None
                    existing_doc = existing["documents"][0]  if existing["documents"]  else ""

                    # 메타데이터 복사 후 해당 필드만 업데이트
                    updated_meta = dict(meta)
                    updated_meta[field] = int(new_val) if field in ("year", "bpm") else new_val

                    upsert_kwargs: dict = {
                        "ids":       [doc_id],
                        "metadatas": [updated_meta],
                        "documents": [existing_doc],
                    }
                    if existing_emb is not None:
                        upsert_kwargs["embeddings"] = [existing_emb]

                    col.upsert(**upsert_kwargs)
                    ok(f"{field}: {meta.get(field, '?')} → {new_val!r}")
                    fixed_count += 1

                except Exception as e:
                    err(f"DB 업데이트 실패: {e}")
                    failed_count += 1
        else:
            warn(f"탐색 실패 — {field} 값을 찾지 못함")
            failed_count += 1

    # ── 최종 리포트 ───────────────────────────────────────────────
    header("📋 완료 리포트")
    total_elapsed = (datetime.now() - start_time).seconds

    print(f"""
  대상 필드    : {BOLD}{field}{RESET}
  총 대상      : {total_missing}곡
  ✅ 수정 성공 : {GREEN}{fixed_count}{RESET}
  ❌ 탐색 실패 : {RED}{failed_count}{RESET}
  소요 시간    : {total_elapsed}초
  완료 시각    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  {'[DRY-RUN — 실제 DB 변경 없음]' if dry_run else ''}
""")

    if not dry_run:
        section("수정 후 현황")
        still_missing = find_missing(col, field, artist_filter)
        print(f"  수정 후 누락: {BOLD}{len(still_missing)}{RESET}곡")


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ChromaDB 음악 메타데이터 수정 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python metadata_fixer.py                      # year 0인 항목 전체 수정
  python metadata_fixer.py --field year         # year 명시 (기본값과 동일)
  python metadata_fixer.py --dry-run            # 실제 수정 없이 탐색만
  python metadata_fixer.py --status             # 전체 현황 테이블 출력
  python metadata_fixer.py --artist "아이유"    # 특정 아티스트만
  python metadata_fixer.py --limit 20           # 최대 20곡

지원 필드 (--field):
  year    — 발매연도 (현재 주력 지원)
  genre   — 장르 (MusicBrainz 한정)
  album   — 앨범명
  artist  — 아티스트 (AudD 오디오 인식)
        """,
    )
    parser.add_argument(
        "--field",   default="year",
        help="수정할 메타데이터 필드 (기본: year)",
        choices=list(FIELD_EMPTY_CHECK.keys()),
    )
    parser.add_argument(
        "--artist",  default="", metavar="가수명",
        help="특정 아티스트만 처리",
    )
    parser.add_argument(
        "--limit",   type=int, default=0, metavar="N",
        help="최대 처리 곡 수 (0=무제한)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="실제 DB 수정 없이 탐색 결과만 확인",
    )
    parser.add_argument(
        "--status",  action="store_true",
        help="전체 현황 통계만 출력하고 종료",
    )
    args = parser.parse_args()

    col = connect_db()

    if args.status:
        print_status(col)
        sys.exit(0)

    # 실행 전 현황 출력
    print_status(col)

    run_fixer(
        field         = args.field,
        artist_filter = args.artist,
        limit         = args.limit,
        dry_run       = args.dry_run,
    )
