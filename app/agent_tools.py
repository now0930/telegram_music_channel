# agent_tools.py

music_tools = [
    {
        'type': 'function',
        'function': {
            'name': 'search_local_music',
            'description': '사용자의 요청을 분석하여 로컬 음악 DB에서 곡을 검색하기 위한 조건을 추출합니다.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'title': {
                        'type': 'string', 
                        'description': '곡 제목'
                    },
                    'artist': {
                        'type': 'string', 
                        'description': '가수 이름'
                    },
                    'year': {
                        'type': 'integer', 
                        'description': '특정 발매 연도 (예: 1998)'
                    },
                    'era': {
                        'type': 'string', 
                        'description': '특정 년대. 반드시 "1990년대", "2000년대", "2010년대" 형식의 문자열로 출력할 것.'
                    },
                    'mood': {
                        'type': 'string',
                        'description': '곡의 분위기 (예: 감성적, 신나는, 잔잔한, 우울한)'
                    },
                    'genre_fixed': {
                        'type': 'string',
                        'description': '음악 장르 (예: 팝 발라드, 댄스, 힙합, R&B)'
                    },
                    'is_instrumental': {
                        'type': 'string',
                        'enum': ['True', 'False'],
                        'description': '가사가 없는 연주곡(보컬 없음)을 원하면 "True", 아니면 "False"'
                    },
                    'count': {
                        'type': 'integer', 
                        'description': '검색할 곡의 수. 언급이 없으면 5'
                    },
                    'keyword': {
                        'type': 'string', 
                        'description': '위 조건들에 명확히 떨어지지 않는 일반적인 검색어 (벡터 검색용)'
                    },
                    'directory': {
                        'type': 'string',
                        'description': '음악이 저장된 폴더나 위치. 사용자가 "A에서 B"라고 말하면 A는 무조건 directory입니다. 멜론, 벅스, 유튜브 같은 스트리밍 앱 이름이 언급되어도 이를 무조건 로컬 폴더(directory) 이름으로 취급하세요. "~에서", "~폴더" 같은 조사는 제외하고 명사만 추출하세요. (예: "멜론에서 아이유" -> directory: "멜론", "방탄 폴더에서" -> directory: "방탄")'
                    }
                },
                # ✅ 핵심 수정: required 추가
                'required': ['count', 'keyword']
            }
        }
    }
]
