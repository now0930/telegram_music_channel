# Telegram Music Bot

로컬 음원을 검색해서 Telegram으로 보내는 봇입니다. 자연어 요청에 응답하며,
사용자 요청 없이도 Mica가 날씨·온도·계절·주가·뉴스를 참고해 매일 음악을 추천합니다.
음악 파일은 기존 로컬 라이브러리에서 선택합니다.

## 1. 자동 추천 일정

기본 시간대는 `Asia/Seoul`입니다. 매 회차 **Melon Top 100 4곡 + 기타 디렉토리 6곡**,
하루 총 20곡을 선택합니다. 개수는 아래 환경 설정으로 변경할 수 있습니다.

| 시간 | 평일 날씨·온도 기준 | 주말 기준 |
| --- | --- | --- |
| 오전 05:00 | 경기도 군포·광명 | 전국 주요 지역 |
| 오후 15:00 | 경기도 화성 | 전국 주요 지역 |

주말 대표 지점은 서울·강릉·대전·광주·대구·부산·제주입니다.
날씨는 Open-Meteo, 주가는 Yahoo Finance의 코스피 최근 값, 이슈는 한국 Google News RSS를 사용합니다.
휴장일 주가는 최근 거래 값이며, 조회 실패 정보는 누락으로 처리합니다.
Mica 호출 실패 시 계절·시간대에 맞는 기본 분위기로 검색합니다.

### Mica 판단과 DB 검색 형식

Mica는 외부 정보를 음악 검색 조건으로 바꿉니다. 판단 함수는
[`app/music_intent.py`](app/music_intent.py)의 `choose_intent()`이며,
[`app/daily_music.py`](app/daily_music.py)의 `recommendation_query()`에서 호출합니다.

선택지는 코드에 고정된 목록이 아닙니다. 추천을 실행할 때마다 `vocabulary()`가 실제 DB의
메타데이터를 읽고 빈 문자열과 `미상`을 제외한 뒤, **필드별 사용 빈도 상위 24개**를 뽑습니다.
값이 24개 미만이면 있는 값을 모두 사용합니다. DB 내용이 달라지면 다음 추천의 선택지도 달라질 수 있습니다.
상위 24개 제한은 현재 Mica 서버의 4,096토큰 컨텍스트를 고려한 것입니다.

| 판단 필드 | 기존 DB 형식과 예시 |
| --- | --- |
| `mood` | 문자열: `신남`, `감성적`, `잔잔함` 등 |
| `genre_fixed` | 문자열: `발라드`, `댄스`, `Rock` 등 |
| `era` | 문자열: `2020년대`, `2010년대` 등 |
| `bpm_range` | 문자열: `업비트 (Allegro)`, `미디엄 템포 (Moderato)` 등 |
| `is_instrumental` | 문자열 `"True"` 또는 `"False"` |
| `keyword` | 자유 작성: 의미 검색용 음악 분위기 문장, 최대 300자 |
| `reason` | 자유 작성: 수집 정보에 근거한 추천 이유, 최대 300자; 발송 설명에 표시 |

앞의 다섯 메타데이터 필드는 제공된 선택지의 정확한 문자열 또는 `null`만 허용합니다.
`null`은 해당 조건을 제한하지 않는다는 뜻입니다. 예를 들어 `is_instrumental=null`이면
보컬곡과 연주곡 모두 후보가 됩니다. `keyword`와 `reason`은 비어 있지 않은 문장을 작성해야 합니다.

`validate_intent()`가 Mica 응답을 다시 검사합니다. 선택지에 `잔잔함`만 있다면
`잔잔한`처럼 바꾼 표현은 거부하고, `is_instrumental`에 문자열 `"True"` 대신
boolean `true`를 넣어도 거부합니다. DB에 존재하더라도 상위 24개 선택지에 포함되지 않은 값은
이번 판단에서 선택할 수 없습니다. 형식·값 검증 실패 시 계절·시간대 기본 추천으로 전환합니다.

실제 테스트에서 반환한 판단의 예시는 다음과 같습니다.

```json
{
  "mood": "잔잔함",
  "genre_fixed": null,
  "era": null,
  "bpm_range": "미디엄 템포 (Moderato)",
  "is_instrumental": null,
  "keyword": "따뜻한 오후의 잔잔함",
  "reason": "맑은 날씨와 18.4도의 가을 오후는 차분하고 따뜻한 분위기를 연출합니다."
}
```

`year`, `bpm` 숫자를 임의로 만들거나 가수·곡명을 추측하지 않습니다.
DB를 수정하거나 다시 색인할 필요는 없습니다. 기존 태그의 정확도는 그대로 영향을 줍니다.

검색은 선택한 메타데이터가 많이 일치하는 곡을 먼저 배치하고, 같은 일치 수에서는
임베딩 유사도 순서를 유지합니다. 일치 곡이 부족하면 부분 일치·의미 검색 결과로 보충하므로
모든 곡이 모든 조건을 만족하는 하드 필터는 아닙니다. 폴더별 4:6 개수와 당일 중복 제외는
Mica가 변경하지 않으며, 기존 사용자 설정대로 적용합니다. 전송 시간과 수신 대화방도
Mica의 판단 대상이 아닙니다.

## 2. 디렉토리와 준비 사항

작업과 실행은 `/home/now0930/telegram_music_channel`에서 진행합니다.
Docker Compose, 접근 가능한 Ollama 서버, Telegram 봇 토큰, 음원과 색인 DB가 필요합니다.
기본 Compose는 기존 외부 Docker 네트워크 `ollama_network`를 사용합니다.

| 호스트 경로 | 컨테이너 경로 | 용도 |
| --- | --- | --- |
| `/mnt/ExtSSD/MP3` | `/music` | 읽기 전용 음원 |
| `app` | `/app` | 코드와 운영 데이터 |
| `app/music_vector_db` | `/app/music_vector_db` | 실제 운영 ChromaDB |
| `app/bot_data.db` | `/app/bot_data.db` | 대화 및 자동 발송 이력 |
| `music_vector_db` | `/app/vector_db` | 기존 별도 마운트 |

저장소 루트의 `music_vector_db`를 `/app/music_vector_db`에 마운트하면
`app` 아래의 실제 운영 DB가 가려집니다. 기존 운영 DB와 백업은 유지하세요.

Ollama 컨테이너 이름이 `ollama`라면 필요한 모델을 다음과 같이 준비합니다.

```bash
docker exec ollama ollama pull hf.co/sky7350/Mica-v0.1-4B:Q5_K_M
docker exec ollama ollama pull mxbai-embed-large
```

## 3. 환경 설정과 수신 대상

처음 설치할 때만 `.env.example`을 `.env`로 복사하고 값을 채웁니다.
기존 `.env`가 있다면 덮어쓰지 말고 필요한 항목만 수정하세요.

```dotenv
TELEGRAM_TOKEN=발급받은_봇_토큰
MUSIC_CHANNEL_ID=받을_대화방_ID
OLLAMA_HOST=http://ollama:11434
OLLAMA_MODEL=hf.co/sky7350/Mica-v0.1-4B:Q5_K_M
RECOMMEND_ENABLED=true
RECOMMEND_TIMEZONE=Asia/Seoul
RECOMMEND_MORNING=05:00
RECOMMEND_AFTERNOON=15:00
RECOMMEND_MELON_COUNT=4
RECOMMEND_OTHER_COUNT=6
RECOMMEND_MELON_DIRECTORY=melon_top100
```

`MUSIC_CHANNEL_ID`는 이름과 달리 **개인 대화방도 지원**합니다.

- 개인 대화방: 봇과 대화를 시작하고 `ID`를 보내면 반환되는 숫자를 입력합니다.
- 채널: 채널의 `-100…` 숫자 ID 또는 공개 `@username`을 입력하고,
  봇을 메시지 게시 권한이 있는 관리자로 추가합니다.

봇의 `@username`은 발신 계정이며 수신 대화방 ID를 대신하지 않습니다.
`TELEGRAM_TOKEN`은 보내는 봇을 인증하는 값입니다. 수신자를 추가해도 같은 토큰을 사용합니다.
기존 `MUSIC_CHANNEL_ID`를 유지하고 `MUSIC_ADDITIONAL_CHAT_IDS`에 추가 수신 ID를 쉼표로 나열합니다.

```dotenv
MUSIC_CHANNEL_ID=기존_숫자_ID
MUSIC_ADDITIONAL_CHAT_IDS=새_숫자_ID,또다른_숫자_ID
```

추가할 개인 계정도 먼저 같은 봇과 대화를 시작해야 합니다. 봇에게 `ID`를 보내 수신 ID를 확인합니다.
시작만으로 자동 구독되지는 않으며 서버 설정에 등록한 대상에게만 예약 발송합니다.
각 수신자는 매 회차 Melon 4곡·기타 6곡을 받으며, 곡 선택과 발송 이력은 수신자별로 관리합니다.
한 수신자의 발송 실패는 다른 수신자의 예약 작업을 중단하지 않습니다.
동일한 ID의 중복 입력은 제거합니다. 공개 채널 별칭과 숫자 ID로 같은 채널을 중복 등록하지 마세요.
환경 설정 변경 후 아래 재시작 명령으로 적용합니다.
`RECOMMEND_ENABLED=false` 또는 모든 수신 ID가 비어 있으면 예약 발송만 중지하고 대화 검색은 유지합니다.

### 추천 개수 변경

`RECOMMEND_MELON_COUNT`와 `RECOMMEND_OTHER_COUNT`는 **매 회차 곡 수**입니다.
예를 들어 `5`, `5`는 1:1, `4`, `6`은 4:6입니다. 한쪽은 0으로 설정할 수 있고 합계는 1~100이어야 합니다.
폴더명은 경로 구성 요소 단위로 비교하며 대소문자·공백·밑줄·하이픈은 무시합니다.
`melon top 100`, `melon_top100`, `Melon-Top-100`은 같은 폴더명으로 취급합니다.
해당 폴더의 하위 폴더도 Melon 그룹에 포함합니다.

주가 종목과 뉴스 주소는 `.env.example`의 `RECOMMEND_MARKET_SYMBOL`, `RECOMMEND_NEWS_RSS`로 변경합니다.

## 4. 실행과 업데이트

```bash
cd /home/now0930/telegram_music_channel
docker compose up -d --force-recreate
docker compose logs --since=2m -f
```

현재 Compose는 컨테이너 시작 시 Python 패키지를 설치하므로 봇 시작까지 시간이 걸릴 수 있습니다.
로그의 `Music DB: ... (N tracks)`로 실제 색인 곡 수를 확인하세요.

GitHub 변경을 받을 때는 위 실행 명령 전에 `git pull --ff-only origin main`을 실행합니다.
서버 파일이나 `.env`를 직접 수정한 경우에는 pull 없이 재시작하면 됩니다.
`.env`는 Git 추적 대상이 아니며 토큰을 저장소에 올리지 않습니다.

기존 색인이 있으면 다시 생성할 필요가 없습니다. 새 음원을 색인할 때는
실행 중인 컨테이너에서 동일한 `/app` 작업 디렉토리와 `/music` 경로를 사용합니다.

```bash
docker compose exec -w /app music-ai-agent python indexer.py
```

## 5. Telegram 사용 방법

봇과의 대화방에 다음과 같이 요청합니다.

- `멜론에서 아이유 노래 3곡 찾아줘`
- `90년대 잔잔한 연주곡 추천해줘`
- `기분 좋은 팝송 5곡 보내줘`
- `ID`: 현재 대화방 ID 확인
- `DB`: 색인 곡 수 확인

대화 요청은 요청한 대화방으로 응답합니다. 예약 발송은 설정한 `MUSIC_CHANNEL_ID`로 보냅니다.
가수·폴더 동의어는 `app/artist_aliases.py`에서 관리합니다.

## 6. 예약 발송과 오류 처리

예약 시각에 실행하고 5분마다 미완료 회차를 확인합니다. 재시작 시 오늘의 가장 최근 회차만
보충하므로 오후에 켜면 오전 분량을 함께 발송하지 않습니다.
당일과 이전 7일 동안 같은 수신자에게 보낸 파일은 재사용하지 않으며 재시작해도 이력을 유지합니다.
`sent`와 결과가 불확실한 `pending` 모두 제외합니다. 기존 발송 이력에도 적용됩니다.
파일 경로를 기준으로 확인하므로 다른 경로에 복사된 동일 음원까지 식별하지는 않습니다.
기본 하루 분량은 Melon 8곡·기타 12곡이며, 반복 제외 기간을 유지할 만큼의 후보 파일이 필요합니다.
그룹별 음원이 부족하면 다른 그룹으로 대체하지 않고 부족분을 로그에 남깁니다.
반복 제외 조건도 자동으로 완화하지 않습니다.

### 반복 제외 기간과 대용량 폴더 제한

```dotenv
RECOMMEND_REPEAT_DAYS=7
RECOMMEND_LOW_PRIORITY_DIRECTORY=잡다한
RECOMMEND_LOW_PRIORITY_MAX=1
```

`RECOMMEND_REPEAT_DAYS`는 오늘 외에 제외할 이전 날짜 수입니다. `7`이면 오늘과 이전 7일,
`0`이면 기존처럼 당일 중복만 제외합니다. 수신자별로 독립 적용합니다.
예를 들어 10월 1일 발송 파일은 10월 8일까지 제외되고 10월 9일부터 다시 후보가 됩니다.

`잡다한`과 그 하위 폴더 전체는 수신자·회차별 최대 1곡으로 제한합니다.
따라서 기타 6곡 중 최소 5곡은 다른 폴더에서 고릅니다. 상한은 목표 개수가 아니므로
추천 순위에 따라 이 폴더에서 0곡을 보낼 수도 있습니다. 재시작 전 발송도 상한에 포함합니다.
폴더는 `/music` 기준 상대 경로이며 `RECOMMEND_LOW_PRIORITY_MAX=0`이면 자동 추천에서 제외합니다.
다른 폴더에 후보가 부족해도 이 상한을 초과해 보충하지 않습니다. 변경 후 컨테이너를 재시작하세요.
설정한 개수를 늘리면 오늘 회차의 부족분을 추가 발송할 수 있으며, 줄여도 이미 보낸 곡을 회수하지 않습니다.

발송 전에 SQLite에 `pending`을 기록하고 성공하면 `sent`로 변경합니다.
Telegram이 명확히 거절하면 해당 예약을 해제하고 회차를 중단합니다.
타임아웃이나 프로세스 종료처럼 결과가 불확실하면 중복 방지를 위해 `pending`을 유지하므로
실제 받은 곡 수가 설정값보다 적을 수 있습니다. 이 경우 대화방과 발송 이력을 대조해야 합니다.
자동 추천 파일은 기존 대화 응답의 TTL 삭제 대상에 포함하지 않습니다.

| 증상 | 확인할 사항 |
| --- | --- |
| `Chat not found` | 수신 ID, 개인 대화 시작 여부, 채널 봇 가입·권한 |
| `Music library is empty` 또는 0곡 | DB 마운트와 `app/music_vector_db`의 실제 색인 |
| 특정 그룹 곡 부족 | 폴더명, 실제 파일 존재 여부, 당일 발송 이력 |
| `NetworkError` / `httpx.ReadError` in `get_updates` | Telegram 메시지 조회 중 네트워크 오류. 라이브러리가 재시도하며, 반복되면 서버 연결 상태 확인 |
| Mica 연결 실패 | Ollama 주소·네트워크·모델 설치 여부 |

## 7. 코드 구성과 검증

| 파일 | 역할 |
| --- | --- |
| `app/main.py` | 대화 요청 분석, 검색, Telegram 봇 실행 |
| `app/daily_music.py` | 외부 정보 수집, Mica 추천, 그룹별 선택, 예약·발송 이력 |
| `app/music_intent.py` | DB 값 기반 판단 스키마, 응답 검증, 메타데이터 일치 우선순위 |
| `app/indexer.py` | 로컬 음원 메타데이터와 임베딩 색인 |
| `app/agent_tools.py` | 대화 검색 도구 스키마 |
| `tests/test_daily_music.py` | 지역·시각·비율·중복 방지·전송 거절 회귀 테스트 |

```bash
python3 -m unittest discover -s tests -v
```

테스트는 외부 요청과 Telegram 전송을 모의 처리합니다. 실제 전송 결과는 운영 로그와 수신 대화방에서 확인합니다.

회차 상태 로그의 `sent`는 성공 기록, `pending`은 전송 결과 미확정 기록입니다.
개수를 줄이기 전에 보낸 곡도 당일 이력에 포함되므로 `sent`가 새 설정 합계보다 클 수 있습니다.
