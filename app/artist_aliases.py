# artist_aliases.py

# 1. 아티스트 동의어 묶음 (여기에 계속 추가하시면 됩니다)
ALIAS_GROUPS = [
    # K-POP 및 국내 아티스트
    ["아이유", "iu"],
    ["방탄소년단", "bts", "방탄", "bangtan"],
    ["블랙핑크", "blackpink", "블핑"],
    ["악동뮤지션", "akmu", "악뮤"],
    ["볼빨간사춘기", "bol4"],
    ["아이들", "여자아이들", "gidle", "g-idle"],
    ["엠씨더맥스", "앰씨더맥스", "mcthemax"],  # 두 폴더 모두 검색됨
    ["싸이", "psy"],                         # 두 폴더 모두 검색됨
    ["십센치", "10cm"],
    ["투애니원", "2ne1"],
    ["마마무", "mamamoo"],
    ["에픽하이", "epikhigh"],
    ["트와이스", "twice"],
    ["자이언티", "ziont", "zion.t"],
    ["위너", "winner"],
    ["뉴진스", "newjeans"],
    ["에스이에스", "ses"],
    ["엔시티드림", "nctdream", "nct"],
    ["잔나비", "jannabi"],
    ["클래지콰이", "clazziquai"],
    
    # 해외 아티스트
    ["마룬5", "마룬파이브", "maroon5"],
    ["찰리푸스", "charlieputh"],
    ["아델", "adele"],
    ["테일러스위프트", "taylorswift"],
    ["에드시런", "edsheeran"],
    ["두아리파", "dualipa"],
    ["션멘데스", "shawnmendes"],
    ["아바", "abba"],
    ["제이슨므라즈", "jasonmraz"],
    ["빌리조엘", "billyjoel"],
    ["이매진드래곤스", "상상용", "imaginedragons"],
    
    # 기타 분류
    ["오스트", "ost", "soundtrack", "사운드트랙"],
    # 🎵 차트 및 최신곡 특별 폴더 매핑
    ["melon_top100", "melontop", "melontop100", "멜론", "melon", "멜론탑", "멜론차트", "탑100", "top100", "최신곡", "인기곡", "인기"]
]

# 2. 로직 처리를 위한 평탄화 딕셔너리 자동 생성 (수정할 필요 없음)
ARTIST_ALIASES = {}
for group in ALIAS_GROUPS:
    # 띄어쓰기 제거 및 소문자화하여 키(Key) 생성
    cleaned_group = [alias.replace(" ", "").lower() for alias in group]
    for alias in cleaned_group:
        ARTIST_ALIASES[alias] = cleaned_group
