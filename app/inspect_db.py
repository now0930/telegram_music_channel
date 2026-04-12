from main import collection
import json

def inspect_samples(limit=10):
    print(f"🔍 DB에서 상위 {limit}개의 데이터를 추출합니다...\n")
    
    # DB에서 데이터 가져오기
    results = collection.get(limit=limit)
    
    ids = results.get('ids', [])
    metadatas = results.get('metadatas', [])
    
    if not metadatas:
        print("❌ DB에 메타데이터가 없습니다.")
        return

    # 헤더 출력
    print(f"{'No':<3} | {'Artist':<20} | {'Title':<20} | {'Year':<6} | {'Genre':<10} | {'Era':<8}")
    print("-" * 80)

    for i, meta in enumerate(metadatas):
        artist = meta.get('artist', 'N/A')
        title = meta.get('title', 'N/A')
        year = meta.get('year', 'N/A')
        genre = meta.get('genre', 'N/A')
        era = meta.get('era', 'N/A')
        
        # 가독성을 위해 긴 문자열 절삭
        print(f"{i+1:<3} | {str(artist)[:18]:<20} | {str(title)[:18]:<20} | {year:<6} | {str(genre)[:8]:<10} | {era:<8}")

    # 전체 필드 구조 한 개만 상세 출력 (숨겨진 필드 확인용)
    print("\n📝 [상세 필드 구조 샘플]")
    print(json.dumps(metadatas[0], indent=4, ensure_ascii=False))

if __name__ == "__main__":
    inspect_samples()
