# check_db_v2.py
from main import collection

print(f"📦 현재 DB 내 총 곡 수: {collection.count()}곡")
print("-" * 50)

# 1. 특정 키워드가 포함된 메타데이터가 있는지 직접 쿼리
target_artist = "Imagine Dragons"
found = collection.get(
    where={"artist": {"$contains": target_artist}}
)

if found['ids']:
    print(f"✅ {target_artist} 관련 곡을 {len(found['ids'])}곡 찾았습니다!")
    print(f"샘플 메타데이터: {found['metadatas'][0]}")
else:
    print(f"❌ {target_artist}라는 이름으로 저장된 곡이 DB에 하나도 없습니다.")
    
    # 2. 혹시 모르니 artist 키가 존재하는지 1개만 확인
    sample = collection.get(limit=1)
    if sample['metadatas']:
        print(f"📍 실제 사용 중인 메타데이터 키 목록: {list(sample['metadatas'][0].keys())}")
